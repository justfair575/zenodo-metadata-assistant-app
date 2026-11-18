import base64
import binascii
import csv
import io
import json
import os
import re

from urllib.parse import quote, urlparse

import httpx

from fastapi import HTTPException
from pydantic import BaseModel, Field, field_validator


GITHUB_API = "https://api.github.com"

MAX_METADATA_FILES = 80
MAX_METADATA_FILE_BYTES = 200_000
MAX_TOTAL_METADATA_BYTES = 4_000_000
DOCUMENTATION_EXTENSIONS = {
    ".adoc",
    ".bib",
    ".cff",
    ".cfg",
    ".html",
    ".ini",
    ".json",
    ".md",
    ".mdx",
    ".org",
    ".rst",
    ".text",
    ".toml",
    ".txt",
    ".xml",
    ".yaml",
    ".yml",
    ".jsonld",
}
DOCUMENTATION_DIRECTORIES = {
    "data",
    "dataset",
    "datasets",
    "doc",
    "docs",
    "documentation",
    "metadata",
}
DOCUMENTATION_NAME_HINTS = (
    "about",
    "author",
    "citation",
    "codemeta",
    "contributors",
    "creator",
    "croissant",
    "config",
    "datacite",
    "datapackage",
    "data-card",
    "data-dict",
    "data-dictionary",
    "dataset",
    "dataset-card",
    "dataset-info",
    "dataset-metadata",
    "dataset-schema",
    "dataset-description",
    "description",
    "info",
    "metadata",
    "readme",
    "schema",
    "ro-crate",
    "schemaorg",
    "subjects",
)

MAX_DATASET_CANDIDATES = 100
MAX_DATASET_FILE_BYTES = 100_000_000

# Limite específico para inspeção de CSV/TSV. Ficheiros maiores que o
# limite inline da API Contents são lidos do URL raw do commit.
MAX_INSPECTION_FILE_BYTES = 25_000_000
GITHUB_CONTENTS_INLINE_LIMIT = 1_000_000

MAX_SAMPLE_ROWS = 5
MAX_TYPE_INFERENCE_ROWS = 1000

METADATA_NAMES = {
    "readme",
    "readme.md",
    "readme.rst",
    "readme.txt",
    "citation.cff",
    "citation.bib",
    "citation.json",
    "codemeta.json",
    "license",
    "license.md",
    "license.txt",
    "licence",
    "licence.md",
    "licence.txt",
    "authors",
    "authors.md",
    "contributors.md",
    "zenodo.json",
    "datacite.json",
    "datapackage.json",
    "ro-crate-metadata.json",
    "ro-crate-metadata.jsonld",
    "pyproject.toml",
    "package.json",
    "dataset-metadata.json",
    "data_dictionary.md",
    "data-dictionary.md",
    "data_dictionary.rst",
    "data-dictionary.rst",
    "dataset_info.json",
    "dataset_infos.json",
    "dataset_description.json",
    "dataset_card.json",
    "metadata.json",
    "citation.json",
    "datacard.md",
    "dataset_card.md",
}

def metadata_candidate_score(
        item: dict,
        dataset_path: str | None = None,
) -> int | None:
    """Rank documentation likely to describe a repository or selected dataset."""

    path = item.get("path")
    if not isinstance(path, str):
        return None

    parts = path.lower().split("/")
    filename = parts[-1]
    stem, extension = os.path.splitext(filename)
    normalized_stem = re.sub(r"[^a-z0-9]", "", stem)
    directories = set(parts[:-1])
    is_readme = stem.startswith("readme")
    is_known_name = filename in METADATA_NAMES or any(
        normalized_stem == re.sub(r"[^a-z0-9]", "", os.path.splitext(name)[0])
        for name in METADATA_NAMES
    )
    has_name_hint = any(
        re.sub(r"[^a-z0-9]", "", hint) in normalized_stem
        for hint in DOCUMENTATION_NAME_HINTS
    )
    in_documentation_directory = bool(
        directories.intersection(DOCUMENTATION_DIRECTORIES)
    )
    in_structured_metadata_directory = bool(
        directories.intersection(
            DOCUMENTATION_DIRECTORIES - {"data"}
        )
    )
    is_text_document = extension in DOCUMENTATION_EXTENSIONS
    root_text_document = (
        len(parts) == 1
        and extension in {".adoc", ".md", ".mdx", ".org", ".rst", ".text", ".txt"}
    )
    root_structured_document = (
        len(parts) == 1
        and extension in {
            ".bib", ".cff", ".cfg", ".ini", ".toml", ".xml", ".yaml", ".yml",
        }
    )
    if not (
        is_readme
        or is_known_name
        or has_name_hint
        or root_structured_document
        or (
            is_text_document
            and (in_documentation_directory or root_text_document)
        )
    ):
        return None
    if not is_text_document and not (
        is_known_name or has_name_hint or root_structured_document
    ):
        return None
    if (
        extension in {".json", ".jsonld", ".toml", ".xml", ".yaml", ".yml"}
        and not (
            is_known_name
            or has_name_hint
            or root_structured_document
            or in_structured_metadata_directory
        )
    ):
        return None

    score = 0
    if is_readme:
        score += 100
    elif filename.startswith(("citation.", "codemeta.", "dataset-", "metadata.")):
        score += 80
    elif is_known_name or has_name_hint:
        score += 60
    if in_documentation_directory:
        score += 20
    if len(parts) == 1:
        score += 15
    elif directories.intersection({"docs", "documentation", "metadata", "dataset", "datasets"}):
        score += 8
    if extension in {".json", ".jsonld", ".yaml", ".yml", ".toml", ".cff", ".xml", ".bib"}:
        score += 5

    if dataset_path:
        dataset_parts = dataset_path.lower().split("/")[:-1]
        shared_directories = 0
        for candidate_part, dataset_part in zip(parts[:-1], dataset_parts):
            if candidate_part != dataset_part:
                break
            shared_directories += 1
        score += shared_directories * 12

    return score

DATA_EXTENSIONS = {
    ".csv",
    ".tsv",
    ".json",
    ".jsonl",
    ".ndjson",
    ".parquet",
    ".arrow",
    ".feather",
    ".xlsx",
    ".xls",
    ".xml",
    ".ttl",
    ".rdf",
    ".nt",
    ".nq",
    ".zip",
    ".tar",
    ".gz",
    ".h5",
    ".hdf5",
    ".nc",
    ".netcdf",
    ".tif",
    ".tiff",
    ".png",
    ".jpg",
    ".jpeg",
    ".wav",
    ".mp3",
    ".txt",
}

MISSING_VALUES = {
    "",
    "na",
    "n/a",
    "nan",
    "null",
    "none",
}

INSPECTABLE_DATA_EXTENSIONS = {
    ".csv", ".tsv", ".json", ".jsonl", ".ndjson", ".txt",
}


def _json_records(text_content: str, extension: str) -> list[dict]:
    """Read JSON records from common dataset export shapes."""
    try:
        if extension in {".jsonl", ".ndjson"}:
            values = [
                json.loads(line)
                for line in text_content.splitlines()
                if line.strip()
            ]
        else:
            document = json.loads(text_content)
            if isinstance(document, list):
                values = document
            elif isinstance(document, dict):
                values = next(
                    (
                        document[key]
                        for key in ("data", "records", "items", "rows", "examples")
                        if isinstance(document.get(key), list)
                    ),
                    [document],
                )
            else:
                values = [{"value": document}]
    except (json.JSONDecodeError, ValueError) as exc:
        raise HTTPException(
            422,
            "O ficheiro não contém JSON/JSONL válido.",
        ) from exc

    return [
        item if isinstance(item, dict) else {"value": item}
        for item in values
    ]


def _inspect_json_records(text_content: str, extension: str) -> dict:
    records = _json_records(text_content, extension)
    columns = list(dict.fromkeys(key for row in records for key in row))
    samples = records[:MAX_SAMPLE_ROWS]

    def cell(value):
        if value is None:
            return ""
        if isinstance(value, (dict, list)):
            return json.dumps(value, ensure_ascii=False)
        return str(value)

    column_info = []
    for column in columns:
        values = [cell(row.get(column, "")) for row in records]
        present = [value for value in values if not is_missing(value)]
        column_info.append({
            "name": str(column),
            "inferred_type": infer_column_type(present[:MAX_TYPE_INFERENCE_ROWS]),
            "missing_values": len(values) - len(present),
        })

    return {
        "number_of_records": len(records),
        "number_of_columns": len(columns),
        "columns": column_info,
        "sample_rows": [
            {str(key): cell(value) for key, value in row.items()}
            for row in samples
        ],
    }


class AnalyzeRequest(BaseModel):

    url: str = Field(
        ...,
        min_length=1,
        max_length=500,
    )

    ref: str | None = Field(
        default=None,
        max_length=200,
    )

    @field_validator("url")
    @classmethod
    def validate_url(cls, value: str) -> str:

        value = value.strip()
        parsed = urlparse(value)

        if (
                parsed.scheme != "https"
                or parsed.hostname != "github.com"
                or parsed.username is not None
                or parsed.password is not None
                or parsed.port not in (None, 443)
                or parsed.query
                or parsed.fragment
        ):
            raise ValueError(
                "Introduz um URL HTTPS válido do GitHub."
            )

        parts = [
            part
            for part in parsed.path.strip("/").split("/")
            if part
        ]

        if len(parts) != 2:
            raise ValueError(
                "Usa o formato https://github.com/owner/repo"
            )

        owner, repo = parts
        repo = repo.removesuffix(".git")

        if not re.fullmatch(r"[A-Za-z0-9_.-]+", owner):
            raise ValueError("Proprietário inválido.")

        if not re.fullmatch(r"[A-Za-z0-9_.-]+", repo):
            raise ValueError("Repositório inválido.")

        return f"https://github.com/{owner}/{repo}"


class InspectDatasetRequest(AnalyzeRequest):

    dataset_path: str = Field(
        ...,
        min_length=1,
        max_length=500,
    )

    @field_validator("dataset_path")
    @classmethod
    def validate_dataset_path(cls, value: str) -> str:

        value = value.strip()

        if (
                value.startswith("/")
                or "\\" in value
                or any(part in {".", "..", ""} for part in value.split("/"))
        ):
            raise ValueError("Caminho de ficheiro inválido.")

        return value


def github_headers() -> dict:

    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }

    token = os.getenv("GITHUB_TOKEN")

    if token:
        headers["Authorization"] = f"Bearer {token}"

    return headers


def decode_metadata_text(raw: bytes) -> tuple[str, str]:
    """Decode common documentation encodings without dropping readable files."""
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        return raw.decode("utf-16"), "utf-16"
    if b"\x00" in raw:
        raise UnicodeDecodeError(
            "metadata",
            raw,
            0,
            min(len(raw), 1),
            "binary content is not documentation text",
        )
    try:
        return raw.decode("utf-8-sig"), "utf-8"
    except UnicodeDecodeError:
        return raw.decode("cp1252", errors="replace"), "cp1252"


async def github_get(
        client: httpx.AsyncClient,
        path: str,
        params: dict | None = None,
):

    try:
        response = await client.get(
            f"{GITHUB_API}{path}",
            params=params,
            headers=github_headers(),
        )

    except httpx.TimeoutException as exc:
        raise HTTPException(
            504,
            "O GitHub demorou demasiado tempo.",
        ) from exc

    except httpx.RequestError as exc:
        raise HTTPException(
            502,
            "Não foi possível contactar o GitHub.",
        ) from exc

    if response.status_code == 404:
        raise HTTPException(
            404,
            "Repositório, referência ou ficheiro não encontrado.",
        )

    if response.status_code in (403, 429):
        raise HTTPException(
            429,
            "Limite da API GitHub atingido ou acesso recusado.",
        )

    if response.status_code >= 400:
        raise HTTPException(
            502,
            f"Erro da API GitHub: HTTP {response.status_code}.",
        )

    try:
        return response.json()

    except ValueError as exc:
        raise HTTPException(
            502,
            "Resposta inválida da API GitHub.",
        ) from exc


async def github_raw_get(
        client: httpx.AsyncClient,
        url: str,
        max_bytes: int,
) -> bytes:
    """Download a commit-pinned public file without spending API rate limit."""
    content = bytearray()
    try:
        async with client.stream("GET", url) as response:
            if response.status_code == 404:
                raise HTTPException(404, "Ficheiro não encontrado no commit selecionado.")
            if not response.is_success:
                raise HTTPException(
                    502,
                    f"Erro ao descarregar um ficheiro do GitHub: HTTP {response.status_code}.",
                )
            async for chunk in response.aiter_bytes():
                if len(content) + len(chunk) > max_bytes:
                    raise HTTPException(413, "O ficheiro excede o limite de leitura.")
                content.extend(chunk)
    except httpx.TimeoutException as exc:
        raise HTTPException(
            504,
            "O GitHub demorou demasiado tempo a devolver um ficheiro.",
        ) from exc
    except httpx.RequestError as exc:
        raise HTTPException(
            502,
            "Não foi possível descarregar um ficheiro do GitHub.",
        ) from exc
    return bytes(content)


def get_repository_parts(url: str) -> tuple[str, str]:

    parsed = urlparse(url)
    owner, repo = parsed.path.strip("/").split("/")[:2]

    return owner, repo.removesuffix(".git")


async def analyze_github_repository(
        request: AnalyzeRequest,
) -> dict:

    owner, repo = get_repository_parts(request.url)

    owner_path = quote(owner, safe="")
    repo_path = quote(repo, safe="")

    async with httpx.AsyncClient(
            timeout=httpx.Timeout(30, connect=5),
            follow_redirects=False,
    ) as client:

        repository = await github_get(
            client,
            f"/repos/{owner_path}/{repo_path}",
        )

        if repository.get("private"):
            raise HTTPException(
                403,
                "Só são suportados repositórios públicos.",
            )

        selected_ref = (
                request.ref or repository.get("default_branch")
        )

        if not selected_ref:
            raise HTTPException(
                422,
                "Não foi encontrada uma branch predefinida.",
            )

        commit = await github_get(
            client,
            f"/repos/{owner_path}/{repo_path}/commits/"
            f"{quote(selected_ref, safe='')}",
        )

        commit_sha = commit.get("sha")

        tree_sha = (
            commit.get("commit", {})
            .get("tree", {})
            .get("sha")
        )

        if not commit_sha or not tree_sha:
            raise HTTPException(
                502,
                "Resposta do GitHub sem SHA válido.",
            )

        tree_response = await github_get(
            client,
            f"/repos/{owner_path}/{repo_path}/git/trees/"
            f"{quote(tree_sha, safe='')}",
            params={"recursive": "1"},
        )

        files = [
            item
            for item in tree_response.get("tree", [])
            if item.get("type") == "blob"
        ]

        warnings = []

        if tree_response.get("truncated"):
            warnings.append(
                "A árvore foi truncada pelo GitHub."
            )

        dataset_path = getattr(request, "dataset_path", None)
        ranked_metadata_candidates = sorted(
            (
                (metadata_candidate_score(item, dataset_path), item)
                for item in files
            ),
            key=lambda candidate: (
                -(candidate[0] or 0),
                candidate[1]["path"].lower(),
            ),
        )
        metadata_candidate_count = sum(
            score is not None
            for score, _ in ranked_metadata_candidates
        )
        metadata_candidates = [
            item
            for score, item in ranked_metadata_candidates
            if score is not None
        ][:MAX_METADATA_FILES]
        if metadata_candidate_count > MAX_METADATA_FILES:
            warnings.append(
                f"Foram encontrados {metadata_candidate_count} ficheiros potencialmente "
                f"úteis; foram selecionados os {MAX_METADATA_FILES} mais relevantes "
                "para respeitar o limite de leitura."
            )

        all_dataset_candidates = [
            item
            for item in files
            if os.path.splitext(item["path"])[1].lower()
               in DATA_EXTENSIONS
            and os.path.basename(item["path"]).lower() not in METADATA_NAMES
        ]

        supported_dataset_candidates = [
            item for item in all_dataset_candidates
            if os.path.splitext(item["path"])[1].lower()
            in INSPECTABLE_DATA_EXTENSIONS
        ]
        unsupported_candidate_count = (
            len(all_dataset_candidates) - len(supported_dataset_candidates)
        )
        if unsupported_candidate_count:
            warnings.append(
                f"{unsupported_candidate_count} ficheiro(s) de dados foram encontrados, "
                "mas não aparecem para seleção porque o conteúdo não é inspecionado "
                "nesta versão (por exemplo, Parquet, Excel ou formatos multimédia)."
            )

        dataset_candidates = [
            {
                "path": item["path"],
                "size": item.get("size", 0),
                "extension": os.path.splitext(
                    item["path"]
                )[1].lower(),
                "sha": item.get("sha"),
                "confidence": (
                    "high"
                    if os.path.splitext(item["path"])[1].lower()
                       in {".csv", ".tsv", ".parquet", ".feather"}
                    else "medium"
                ),
            }
            for item in supported_dataset_candidates
            if item.get("size", 0) <= MAX_DATASET_FILE_BYTES
        ][:MAX_DATASET_CANDIDATES]

        if len(supported_dataset_candidates) > MAX_DATASET_CANDIDATES:
            warnings.append(
                "O número de candidatos ultrapassa o limite "
                "de apresentação."
            )

        metadata_files = []
        total_bytes = 0

        for item in metadata_candidates:

            path = item["path"]
            size = item.get("size", 0)

            if size > MAX_METADATA_FILE_BYTES:
                metadata_files.append({
                    "path": path,
                    "size": size,
                    "status": "skipped_too_large",
                })
                continue

            if total_bytes + size > MAX_TOTAL_METADATA_BYTES:
                if "Limite total de leitura de metadados atingido." not in warnings:
                    warnings.append(
                        "Limite total de leitura de metadados atingido."
                    )
                metadata_files.append({
                    "path": path,
                    "size": size,
                    "status": "skipped_total_limit",
                })
                continue

            raw_url = (
                "https://raw.githubusercontent.com/"
                f"{owner_path}/{repo_path}/"
                f"{quote(commit_sha, safe='')}/"
                f"{quote(path, safe='/')}"
            )
            try:
                raw = await github_raw_get(
                    client,
                    raw_url,
                    MAX_METADATA_FILE_BYTES,
                )
            except HTTPException as exc:
                if exc.status_code == 413:
                    metadata_files.append({
                        "path": path,
                        "size": size,
                        "status": "skipped_too_large",
                    })
                    continue
                raise

            try:
                text_content, encoding = decode_metadata_text(raw)
            except UnicodeDecodeError:
                metadata_files.append({
                    "path": path,
                    "size": size,
                    "status": "decode_error",
                })
                continue

            if total_bytes + len(raw) > MAX_TOTAL_METADATA_BYTES:
                if "Limite total de leitura de metadados atingido." not in warnings:
                    warnings.append("Limite total de leitura de metadados atingido.")
                metadata_files.append({
                    "path": path,
                    "size": len(raw),
                    "status": "skipped_total_limit",
                })
                continue

            total_bytes += len(raw)

            metadata_files.append({
                "path": path,
                "size": len(raw),
                "sha": item.get("sha"),
                "encoding": encoding,
                "status": "read",
                "content": text_content,
            })

        oversized_metadata_count = sum(
            item["status"] == "skipped_too_large"
            for item in metadata_files
        )
        if oversized_metadata_count:
            warnings.append(
                f"{oversized_metadata_count} ficheiro(s) de metadados excedem "
                f"o limite de {MAX_METADATA_FILE_BYTES // 1000} KB por ficheiro."
            )

        unreadable_metadata_count = sum(
            item["status"] in {"unsupported_encoding", "decode_error"}
            for item in metadata_files
        )
        if unreadable_metadata_count:
            warnings.append(
                f"{unreadable_metadata_count} ficheiro(s) não puderam ser "
                "descodificados como texto."
            )

        readme = next(
            (
                item["content"]
                for item in metadata_files
                if item["path"].split("/")[-1].lower()
                   in {
                       "readme",
                       "readme.md",
                       "readme.rst",
                       "readme.txt",
                   }
                   and item["status"] == "read"
            ),
            None,
        )

        if not readme:
            warnings.append("README não encontrado.")

        license_info = repository.get("license")

        return {
            "repository": {
                "full_name": repository.get("full_name"),
                "url": repository.get("html_url"),
                "description": repository.get("description"),
                "default_branch": repository.get("default_branch"),
                "selected_ref": selected_ref,
                "commit_sha": commit_sha,
                "license": (
                    license_info.get("spdx_id")
                    if license_info
                    else None
                ),
            },
            "inventory": {
                "total_files": len(files),
                "metadata_files_found": metadata_candidate_count,
                "metadata_files_read": sum(
                    item["status"] == "read"
                    for item in metadata_files
                ),
                "dataset_candidates_found": len(
                    all_dataset_candidates
                ),
                "dataset_candidates_returned": len(
                    dataset_candidates
                ),
            },
            "metadata_files": metadata_files,
            "dataset_candidates": dataset_candidates,
            "warnings": warnings,
        }


def is_missing(value: str | None) -> bool:

    if value is None:
        return True

    return value.strip().lower() in MISSING_VALUES


def infer_column_type(values: list[str]) -> str:

    values = [
        value.strip()
        for value in values
        if not is_missing(value)
    ]

    if not values:
        return "unknown"

    if all(
            value.lower() in {"true", "false"}
            for value in values
    ):
        return "boolean"

    try:
        for value in values:
            int(value)

        return "integer"

    except ValueError:
        pass

    try:
        for value in values:
            float(value)

        return "number"

    except ValueError:
        pass

    return "string"


async def inspect_dataset_in_repository(
        repository_data: dict,
        request: InspectDatasetRequest,
) -> dict:

    candidates = {
        item["path"]: item
        for item in repository_data["dataset_candidates"]
    }

    dataset = candidates.get(request.dataset_path)

    if not dataset:
        raise HTTPException(
            404,
            "O ficheiro selecionado não foi encontrado "
            "na lista de datasets do repositório.",
        )

    extension = dataset["extension"]

    if extension not in INSPECTABLE_DATA_EXTENSIONS:
        raise HTTPException(
            422,
            "Formato ainda sem inspeção. São suportados CSV, TSV, JSON, JSONL e TXT.",
        )

    if dataset["size"] > MAX_INSPECTION_FILE_BYTES:
        raise HTTPException(
            413,
            "O ficheiro ultrapassa o limite de 25 MB "
            "para inspeção nesta versão.",
        )

    owner, repo = get_repository_parts(request.url)

    owner_path = quote(owner, safe="")
    repo_path = quote(repo, safe="")

    commit_sha = repository_data["repository"]["commit_sha"]

    async with httpx.AsyncClient(
            timeout=httpx.Timeout(60, connect=5),
            follow_redirects=False,
    ) as client:

        if dataset["size"] > GITHUB_CONTENTS_INLINE_LIMIT:
            raw_url = (
                "https://raw.githubusercontent.com/"
                f"{owner_path}/{repo_path}/"
                f"{quote(commit_sha, safe='')}/"
                f"{quote(request.dataset_path, safe='/')}"
            )

            try:
                async with client.stream("GET", raw_url) as response:
                    if response.status_code == 404:
                        raise HTTPException(
                            404,
                            "O ficheiro não foi encontrado no commit selecionado.",
                        )
                    if not response.is_success:
                        raise HTTPException(
                            502,
                            f"Erro ao descarregar o ficheiro: HTTP {response.status_code}.",
                        )

                    raw = bytearray()
                    async for chunk in response.aiter_bytes():
                        if len(raw) + len(chunk) > MAX_INSPECTION_FILE_BYTES:
                            raise HTTPException(
                                413,
                                "O ficheiro ultrapassa o limite de "
                                "25 MB para inspeção nesta versão.",
                            )
                        raw.extend(chunk)

            except httpx.RequestError as exc:
                raise HTTPException(
                    502,
                    "Não foi possível descarregar o ficheiro do GitHub.",
                ) from exc
        else:
            content = await github_get(
                client,
                f"/repos/{owner_path}/{repo_path}/contents/"
                f"{quote(request.dataset_path, safe='/')}",
                params={"ref": commit_sha},
            )

            if content.get("encoding") != "base64":
                raise HTTPException(
                    422,
                    "O GitHub não disponibilizou o ficheiro "
                    "num formato de texto suportado.",
                )

            try:
                encoded = "".join(
                    content.get("content", "").split()
                )

                raw = base64.b64decode(
                    encoded,
                    validate=True,
                )

            except (
                    ValueError,
                    binascii.Error,
            ) as exc:
                raise HTTPException(
                    422,
                    "Não foi possível descodificar o ficheiro.",
                ) from exc

    if len(raw) > MAX_INSPECTION_FILE_BYTES:
        raise HTTPException(
            413,
            "O ficheiro ultrapassa o limite de 25 MB para inspeção.",
        )

    try:
        text_content = raw.decode("utf-8-sig")
        encoding = "utf-8"

    except UnicodeDecodeError:
        try:
            text_content = raw.decode("latin-1")
            encoding = "latin-1"
        except UnicodeDecodeError as exc:
            raise HTTPException(
                422,
                "Não foi possível descodificar o ficheiro.",
            ) from exc

    if extension in {".json", ".jsonl", ".ndjson"}:
        dataset_details = _inspect_json_records(text_content, extension)
        return {
            "repository": repository_data["repository"],
            "dataset": {
                "path": request.dataset_path,
                "size": len(raw),
                "format": extension,
                "encoding": encoding,
                "delimiter": "",
                **dataset_details,
            },
        }

    if extension == ".txt":
        records = [line.strip() for line in text_content.splitlines() if line.strip()]
        return {
            "repository": repository_data["repository"],
            "dataset": {
                "path": request.dataset_path,
                "size": len(raw),
                "format": extension,
                "encoding": encoding,
                "delimiter": "",
                "number_of_records": len(records),
                "number_of_columns": 1,
                "columns": [{
                    "name": "text",
                    "inferred_type": "string",
                    "missing_values": 0,
                }],
                "sample_rows": [{"text": row} for row in records[:MAX_SAMPLE_ROWS]],
            },
        }

    delimiter = "\t" if extension == ".tsv" else ","

    try:
        try:
            dialect = csv.Sniffer().sniff(
                text_content[:10000],
                delimiters=",;\t|",
            )
            delimiter = dialect.delimiter
        except csv.Error:
            pass

        reader = csv.DictReader(
            io.StringIO(text_content),
            delimiter=delimiter,
        )

        if not reader.fieldnames:
            raise HTTPException(
                422,
                "O ficheiro não contém cabeçalhos CSV válidos.",
            )

        reader.fieldnames = [
            column.strip() if column else ""
            for column in reader.fieldnames
        ]

        columns = reader.fieldnames

        if any(not column for column in columns):
            raise HTTPException(
                422,
                "O ficheiro contém nomes de colunas vazios.",
            )

        if len(set(columns)) != len(columns):
            raise HTTPException(
                422,
                "O ficheiro contém nomes de colunas duplicados.",
            )

        row_count = 0
        sample_rows = []

        column_values = {
            column: []
            for column in columns
        }

        missing_counts = {
            column: 0
            for column in columns
        }

        for row in reader:

            row_count += 1
            clean_row = {}

            for column in columns:

                value = row.get(column)

                if value is None:
                    value = ""

                value = str(value).strip()
                clean_row[column] = value

                if is_missing(value):
                    missing_counts[column] += 1

                elif (
                        len(column_values[column])
                        < MAX_TYPE_INFERENCE_ROWS
                ):
                    column_values[column].append(value)

            if len(sample_rows) < MAX_SAMPLE_ROWS:
                sample_rows.append(clean_row)

    except csv.Error as exc:
        raise HTTPException(
            422,
            "Erro ao interpretar o conteúdo CSV.",
        ) from exc

    column_info = [
        {
            "name": column,
            "inferred_type": infer_column_type(
                column_values[column]
            ),
            "missing_values": missing_counts[column],
        }
        for column in columns
    ]

    return {
        "repository": repository_data["repository"],
        "dataset": {
            "path": request.dataset_path,
            "size": len(raw),
            "format": extension,
            "encoding": encoding,
            "delimiter": delimiter,
            "number_of_records": row_count,
            "number_of_columns": len(columns),
            "columns": column_info,
            "sample_rows": sample_rows,
        },
    }


async def inspect_github_dataset(
        request: InspectDatasetRequest,
) -> dict:

    repository_data = await analyze_github_repository(
        AnalyzeRequest(
            url=request.url,
            ref=request.ref,
        )
    )

    return await inspect_dataset_in_repository(
        repository_data,
        request,
    )
