from pathlib import Path
from typing import Any
import json
import re
import tomllib
import unicodedata
from datetime import date

import yaml

from app.ollama_service import extract_with_ollama
from app.metadata_schema import ExtractedMetadata, MetadataField
from app.metadata_validation import KNOWN_SPDX_LICENSES, validate_license


REQUIRED_FIELDS = [
    "title", "description", "creators", "license",
    "date_published", "dataset_type", "data_modality",
]

FIELD_NAMES = [
    "title", "description", "creators", "license", "date_published",
    "keywords", "dataset_type", "machine_learning_task", "data_modality",
    "language", "number_of_records",
]

# Tokens explícitos, não palavras genéricas que possam causar falsos positivos.
LANGUAGE_NAME_TOKENS = {
    "en": {"english", "eng", "anglais"},
    "pt": {"portuguese", "portugues", "português"},
    "es": {"spanish", "espanol", "español", "castilian"},
    "fr": {"french", "francais", "français"},
    "de": {"german", "deutsch"},
    "it": {"italian", "italiano"},
    "nl": {"dutch", "nederlands"},
    "ja": {"japanese", "nihongo"},
    "zh": {"chinese", "mandarin"},
    "ar": {"arabic"},
    "ru": {"russian"},
    "ko": {"korean"},
    "hi": {"hindi"},
}

def has_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, dict)):
        return bool(value)
    return True


def normalize_field(field: Any) -> MetadataField:
    if isinstance(field, MetadataField):
        result = field
    elif isinstance(field, dict):
        result = MetadataField.model_validate(field)
    else:
        result = MetadataField(value=field, suggested_value=field)

    # Campos extraídos devem continuar como sugestões, nunca confirmação.
    if not has_value(result.suggested_value) and has_value(result.value):
        result.suggested_value = result.value
    if not has_value(result.suggested_value):
        result.suggested_value = [] if isinstance(result.value, list) else None
    if result.status == "confirmed":
        result.status = "suggested"
    return result


def field_has_value(field: MetadataField) -> bool:
    return has_value(field.suggested_value)


def set_deterministic_suggestion(
        field: MetadataField,
        value: Any,
        source: str,
        evidence: str,
        confidence: float,
        force: bool = False,
) -> None:
    if has_value(field.suggested_value) and not force:
        return
    field.suggested_value = value
    field.value = None
    field.source = source
    field.evidence = evidence
    field.confidence = confidence
    field.status = "suggested"


def documentation_sources(repository_data: dict) -> list[dict[str, str]]:
    return [
        item
        for item in repository_data.get("metadata_files", [])
        if item.get("status") == "read"
        and isinstance(item.get("content"), str)
    ]


def _metadata_text(value: Any) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    if isinstance(value, list):
        return next(
            (text for item in value if (text := _metadata_text(item))),
            None,
        )
    if isinstance(value, dict):
        preferred_keys = {
            "name",
            "title",
            "description",
            "abstract",
            "value",
            "id",
            "url",
            "path",
            "spdxid",
            "licenseidentifier",
            "rightsuri",
            "identifier",
        }
        for key, item in value.items():
            if _normalized_key(key) not in preferred_keys:
                continue
            result = _metadata_text(item)
            if result:
                return result
    return None


def _normalized_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def _find_metadata_value(document: Any, aliases: set[str]) -> Any:
    if isinstance(document, dict):
        for key, value in document.items():
            if _normalized_key(key) in aliases and has_value(value):
                return value
        for value in document.values():
            found = _find_metadata_value(value, aliases)
            if has_value(found):
                return found
    elif isinstance(document, list):
        for item in document:
            found = _find_metadata_value(item, aliases)
            if has_value(found):
                return found
    return None


def _parse_metadata_document(path: str, content: str) -> Any:
    suffix = Path(path).suffix.lower()
    if suffix in {".yaml", ".yml", ".cff"}:
        return yaml.safe_load(content)
    if suffix == ".toml":
        return tomllib.loads(content)
    if suffix in {".json", ".jsonld"}:
        return json.loads(content)
    return None


def _find_dataset_node(document: Any) -> dict[str, Any] | None:
    if isinstance(document, dict):
        types = document.get("@type") or document.get("type")
        if isinstance(types, str):
            types = [types]
        if isinstance(types, list) and any(
            "dataset" in str(item).lower()
            for item in types
        ):
            return document
        for value in document.values():
            found = _find_dataset_node(value)
            if found:
                return found
    elif isinstance(document, list):
        for item in document:
            found = _find_dataset_node(item)
            if found:
                return found
    return None


def extract_structured_document_metadata(
        repository_data: dict,
) -> dict[str, tuple[Any, str, str]]:
    extracted: dict[str, tuple[Any, str, str]] = {}

    field_keys = {
        "title": {"title", "titles", "datasetname", "datatitle"},
        "description": {"description", "descriptions", "abstract", "datasetdescription", "summary"},
        "keywords": {"keywords", "keyword", "subject", "subjects"},
        "date_published": {
            "datepublished",
            "dateissued",
            "datereleased",
            "publicationdate",
        },
        "language": {"inlanguage", "language", "languages"},
        "creators": {
            "creator",
            "creators",
            "author",
            "authors",
            "creatorname",
            "authorname",
            "contributor",
            "contributors",
            "maintainer",
            "maintainers",
        },
        "license": {
            "license",
            "licenses",
            "licenseurl",
            "licenseid",
            "licensename",
            "rights",
            "rightslist",
        },
    }

    for source in documentation_sources(repository_data):
        path = source["path"]
        if Path(path).suffix.lower() not in {
            ".json",
            ".jsonld",
            ".yaml",
            ".yml",
            ".toml",
            ".cff",
        }:
            continue

        try:
            document = _parse_metadata_document(path, source["content"])
        except (json.JSONDecodeError, tomllib.TOMLDecodeError, yaml.YAMLError):
            continue
        if not isinstance(document, (dict, list)):
            continue

        for field_name, aliases in field_keys.items():
            if field_name in extracted:
                continue
            value = _find_metadata_value(document, aliases)
            if (
                field_name == "title"
                and not has_value(value)
                and isinstance(document, dict)
            ):
                dataset_node = _find_dataset_node(document)
                value = document.get("name") or (
                    dataset_node.get("name")
                    if dataset_node
                    else None
                )
            if field_name in {"title", "description", "date_published", "license"}:
                value = _metadata_text(value)
            elif field_name == "keywords":
                if isinstance(value, str):
                    value = [item.strip() for item in re.split(r"[,;]", value) if item.strip()]
                elif isinstance(value, list):
                    value = [
                        item.strip()
                        for item in value
                        if isinstance(item, str) and item.strip()
                    ]
                else:
                    value = None
            elif field_name == "language":
                if isinstance(value, str):
                    value = [value.strip()]
                elif isinstance(value, dict):
                    language_name = _metadata_text(value)
                    value = [language_name] if language_name else None
                elif isinstance(value, list):
                    value = [
                        _metadata_text(item) if isinstance(item, dict) else item
                        for item in value
                    ]
                    value = [item for item in value if isinstance(item, str) and item.strip()]
                else:
                    value = None
            elif field_name == "creators":
                if isinstance(value, dict):
                    value = [value]
                elif isinstance(value, str):
                    value = [{"name": value.strip()}]
                elif isinstance(value, list):
                    creators = []
                    for item in value:
                        if isinstance(item, str):
                            creators.append({"name": item.strip()})
                        elif isinstance(item, dict):
                            given = _metadata_text(item.get("given-names") or item.get("givenName"))
                            family = _metadata_text(item.get("family-names") or item.get("familyName"))
                            creator = dict(item)
                            if "name" not in creator and (given or family):
                                creator["name"] = " ".join(part for part in (given, family) if part)
                            orcid = _metadata_text(
                                creator.get("orcid_id")
                                or creator.get("orcid")
                                or creator.get("identifier")
                            )
                            if orcid and "orcid_id" not in creator:
                                creator["orcid_id"] = orcid
                            if creator.get("orcid_id") and not creator.get("creator_type"):
                                creator["creator_type"] = "person"
                            if creator.get("name"):
                                creators.append(creator)
                    value = creators
                else:
                    value = None

            if field_name == "license" and value and validate_license(value):
                value = None
            if isinstance(value, date):
                value = value.isoformat()

            if has_value(value):
                extracted[field_name] = (
                    value,
                    path,
                    f"Valor indicado explicitamente em metadados estruturados de {path}.",
                )

    return extracted


def _readme_title_and_description(
        repository_data: dict,
) -> tuple[tuple[str, str] | None, tuple[str, str] | None]:
    readme = next(
        (
            source for source in documentation_sources(repository_data)
            if Path(source["path"]).name.lower().startswith("readme")
        ),
        None,
    )
    if not readme:
        return None, None

    lines = readme["content"].splitlines()
    title = None
    title_index = -1
    for index, line in enumerate(lines):
        heading = re.match(r"^\s*#\s+(.+?)\s*#*\s*$", line)
        if heading:
            title = heading.group(1).strip()
            title_index = index
            break

    description = None
    for line in lines[title_index + 1:]:
        paragraph = line.strip()
        if not paragraph or paragraph.startswith(("#", "[![", "<")):
            continue
        if paragraph.startswith("!["):
            continue
        description = re.sub(r"\s+", " ", paragraph)
        break

    path = readme["path"]
    title_result = (
        (title, path) if title else None
    )
    description_result = (
        (description, path) if description else None
    )
    return title_result, description_result


def generate_title_from_filename(dataset_path: str | None) -> str | None:
    if not dataset_path:
        return None
    stem = Path(dataset_path).stem
    stem = re.sub(r"[_-]+", " ", stem)
    stem = re.sub(r"\s+", " ", stem).strip()
    if not stem:
        return None
    return " ".join(word.capitalize() for word in stem.split())


def infer_language_from_names(repository_data: dict, dataset_inspection: dict | None) -> tuple[str, str] | None:
    repository = repository_data.get("repository", {})
    repository_name = str(repository.get("full_name") or "")
    dataset_path = str((dataset_inspection or {}).get("path") or "")
    combined = f"{repository_name} {dataset_path}".lower()
    normalized_names = unicodedata.normalize("NFKD", combined)
    normalized_names = "".join(
        character for character in normalized_names
        if not unicodedata.combining(character)
    )
    tokens = set(re.findall(r"[a-z]+", normalized_names))

    matches: list[tuple[str, str]] = []
    for code, names in LANGUAGE_NAME_TOKENS.items():
        if tokens.intersection(names):
            matches.append((code, f"encontra o termo explícito '{sorted(tokens.intersection(names))[0]}'"))

    # Sufixos isolados são aceites apenas quando delimitados por separadores.
    suffix_map = {"en": "en", "pt": "pt", "es": "es", "fr": "fr", "de": "de", "it": "it", "nl": "nl", "ja": "ja", "zh": "zh"}
    for code, suffix in suffix_map.items():
        if re.search(rf"(?:^|[/_.-]){suffix}(?:$|[/_.-])", combined):
            matches.append((code, f"encontra o sufixo de idioma '-{suffix}' ou '_{suffix}'"))

    distinct = {code for code, _ in matches}
    if len(distinct) != 1:
        return None
    code = next(iter(distinct))
    reason = next(reason for matched_code, reason in matches if matched_code == code)
    return code, reason


def apply_deterministic_metadata(
        metadata: ExtractedMetadata,
        repository_data: dict,
        dataset_inspection: dict | None = None,
) -> None:
    dataset = (dataset_inspection or {}).get("dataset", dataset_inspection or {})
    dataset_path = dataset.get("path")
    structured = extract_structured_document_metadata(repository_data)
    readme_title, readme_description = _readme_title_and_description(repository_data)

    for field_name, (value, source, evidence) in structured.items():
        set_deterministic_suggestion(
            getattr(metadata, field_name),
            value,
            source,
            evidence,
            0.98,
            force=True,
        )

    if readme_title:
        title, source = readme_title
        set_deterministic_suggestion(
            metadata.title,
            title,
            source,
            f"Título extraído do cabeçalho principal de {source}.",
            0.90,
        )

    if readme_description:
        description, source = readme_description
        set_deterministic_suggestion(
            metadata.description,
            description,
            source,
            f"Descrição extraída do primeiro parágrafo de {source} após o título.",
            0.82,
        )

    license_found = False
    for source in documentation_sources(repository_data):
        if not Path(source["path"]).name.lower().startswith("readme"):
            continue
        for license_id in sorted(KNOWN_SPDX_LICENSES):
            match = re.search(
                rf"(?<![A-Za-z0-9.-]){re.escape(license_id)}(?![A-Za-z0-9.-])",
                source["content"],
                re.IGNORECASE,
            )
            if not match:
                continue
            context = source["content"][
                max(0, match.start() - 180):match.end() + 180
            ]
            if not (
                re.search(r"\b(?:data|dataset)\b", context, re.IGNORECASE)
                and re.search(
                    r"\b(?:license|licen[cs]e|released|open data)\b",
                    context,
                    re.IGNORECASE,
                )
            ):
                continue
            set_deterministic_suggestion(
                metadata.license,
                license_id,
                source["path"],
                f"A documentação associa explicitamente a licença {license_id} aos dados em {source['path']}.",
                0.95,
                force=True,
            )
            license_found = True
            break
        if license_found:
            break

    if dataset_path:
        set_deterministic_suggestion(
            metadata.title,
            generate_title_from_filename(dataset_path),
            "selected_dataset_inspection",
            f"Título construído a partir do nome do ficheiro selecionado: {dataset_path}.",
            0.65,
        )

        file_format = str(dataset.get("format") or Path(dataset_path).suffix).lower()
        if file_format in {".csv", ".tsv", "csv", "tsv"}:
            set_deterministic_suggestion(
                metadata.dataset_type, "Tabular dataset", "selected_dataset_inspection",
                "O ficheiro selecionado é CSV/TSV e a inspeção identificou linhas e colunas.", 0.98,
            )
            set_deterministic_suggestion(
                metadata.data_modality, "Tabular", "selected_dataset_inspection",
                "A inspeção do ficheiro CSV/TSV identificou uma estrutura tabular.", 0.98,
            )

        records = dataset.get("number_of_records")
        if isinstance(records, int):
            set_deterministic_suggestion(
                metadata.number_of_records, records, "selected_dataset_inspection",
                f"Contagem exata indicada pela inspeção: {records} registos.", 1.0,
            )

    # O idioma inferido a partir de amostras pelo modelo tem prioridade.
    if not has_value(metadata.language.suggested_value):
        inferred = infer_language_from_names(repository_data, dataset)
        if inferred:
            code, reason = inferred
            repo_name = repository_data.get("repository", {}).get("full_name") or "(não indicado)"
            filename = dataset_path or "(não indicado)"
            set_deterministic_suggestion(
                metadata.language,
                [code],
                "repository_and_dataset_name",
                f"Previsão de baixa confiança baseada apenas nos nomes: repositório '{repo_name}' e ficheiro '{filename}'; {reason}. O conteúdo textual do dataset não confirmou esta previsão.",
                0.48,
            )


async def extract_repository_metadata(
        repository_data: dict,
        dataset_inspection: dict | None = None,
) -> ExtractedMetadata:
    payload = dict(repository_data)
    if dataset_inspection is not None:
        payload["dataset_inspection"] = dataset_inspection

    metadata = await extract_with_ollama(payload)

    for field_name in FIELD_NAMES:
        setattr(metadata, field_name, normalize_field(getattr(metadata, field_name)))

    metadata.additional_information = [
        normalize_field(item) for item in metadata.additional_information
    ]

    apply_deterministic_metadata(metadata, repository_data, dataset_inspection)

    missing = [
        field_name for field_name in REQUIRED_FIELDS
        if not field_has_value(getattr(metadata, field_name))
    ]
    metadata.missing_information = missing
    return metadata