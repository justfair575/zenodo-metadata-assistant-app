import base64
import binascii
import json
import os
import httpx
import tempfile

from pathlib import Path
from urllib.parse import quote

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, RedirectResponse
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()

from app.github_service import (
    AnalyzeRequest,
    InspectDatasetRequest,
    analyze_github_repository,
    inspect_dataset_in_repository,
    github_get,
    get_repository_parts,
)

from app.metadata_extractor import (
    extract_repository_metadata,
)

from app.ollama_service import (
    OLLAMA_URL,
    OLLAMA_MODEL,
)

from app.metadata_schema import (
    ConfirmMetadataRequest,
    ConfirmedMetadataResponse,
    CreatorIdentity,
    ExtractedMetadata,
    REQUIRED_FIELDS,
)

from app.metadata_review import (
    apply_metadata_reviews,
    verify_creator_identities,
)
from app.metadata_validation import is_empty, validate_metadata_field

from app.croissant_generator import build_croissant

from app.orcid_service import (
    ORCIDInvalidIDError,
    ORCIDNameUnavailableError,
    ORCIDNotFoundError,
    ORCIDServiceError,
    create_orcid_authorization_url,
    consume_orcid_state,
    exchange_orcid_authorization_code,
    get_public_orcid_identity,
)

from app.ror_service import (
    RORInvalidIDError,
    RORNotFoundError,
    RORServiceError,
    search_organizations,
    validate_ror_id,
)


# ============================================================
# REQUEST MODELS
# ============================================================

class CroissantGenerateRequest(InspectDatasetRequest):
    metadata: ExtractedMetadata
    orcid_session_token: str | None = None




class ZenodoPublishRequest(BaseModel):
    croissant: dict
    title: str
    description: str
    iscte_affiliated: bool = False


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="Zenodo Metadata Assistant",
    version="0.5.0",
    description=(
        "GitHub dataset analysis, AI metadata extraction "
        "and MLCommons Croissant generation."
    ),
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
async def health():
    return {
        "status": "ok",
    }


# ============================================================
# OLLAMA STATUS
# ============================================================

@app.get("/api/v1/ollama-status")
async def ollama_status():

    try:
        async with httpx.AsyncClient(
                timeout=5,
        ) as client:

            response = await client.get(
                f"{OLLAMA_URL}/api/tags"
            )

            response.raise_for_status()
            data = response.json()

            models = [
                item["name"]
                for item in data.get("models", [])
            ]

            configured_base = OLLAMA_MODEL.split(":")[0]

            model_installed = any(
                name == OLLAMA_MODEL
                or name.split(":")[0] == configured_base
                for name in models
            )

            return {
                "available": True,
                "configured_model": OLLAMA_MODEL,
                "model_installed": model_installed,
                "installed_models": models,
            }

    except Exception:
        return {
            "available": False,
            "configured_model": OLLAMA_MODEL,
            "model_installed": False,
            "installed_models": [],
        }


# ============================================================
# ANALYZE GITHUB REPOSITORY
# ============================================================

@app.post("/api/v1/analyze")
async def analyze_repository(
        request: AnalyzeRequest,
):
    return await analyze_github_repository(request)


# ============================================================
# EXTRACT REPOSITORY METADATA
# ============================================================

@app.post("/api/v1/extract")
async def extract_metadata(
        request: AnalyzeRequest,
):

    repository_data = await analyze_github_repository(
        request
    )

    metadata = await extract_repository_metadata(
        repository_data
    )

    return {
        "repository": repository_data["repository"],
        "inventory": repository_data["inventory"],
        "metadata": metadata.model_dump(),
        "dataset_candidates": repository_data[
            "dataset_candidates"
        ],
        "warnings": repository_data["warnings"],
    }


# ============================================================
# INSPECT SELECTED DATASET
# ============================================================

@app.post("/api/v1/inspect-dataset")
async def inspect_dataset(
        request: InspectDatasetRequest,
):

    repository_data = await analyze_github_repository(
        request
    )

    inspection = await inspect_dataset_in_repository(
        repository_data,
        request,
    )

    return inspection


# ============================================================
# EXTRACT METADATA FROM SELECTED DATASET
# ============================================================

@app.post("/api/v1/extract-dataset")
async def extract_dataset_metadata(
        request: InspectDatasetRequest,
):

    # 1. Analisar o repositório.

    repository_data = await analyze_github_repository(
        request
    )

    # 2. Inspecionar o dataset selecionado.

    inspection = await inspect_dataset_in_repository(
        repository_data,
        request,
    )

    # 3. Extrair sugestões de metadados usando o Ollama.

    metadata = await extract_repository_metadata(
        repository_data,
        dataset_inspection=inspection["dataset"],
    )

    # 4. Devolver os resultados.

    return {
        "repository": repository_data["repository"],
        "inventory": repository_data["inventory"],
        "selected_dataset": inspection["dataset"],
        "metadata": metadata.model_dump(),
        "dataset_candidates": repository_data[
            "dataset_candidates"
        ],
        "warnings": repository_data["warnings"],
    }


# ============================================================
# CONFIRM METADATA
# ============================================================

@app.post(
    "/api/v1/metadata/confirm",
    response_model=ConfirmedMetadataResponse,
)
async def confirm_metadata(
        request: ConfirmMetadataRequest,
):
    return await apply_metadata_reviews(request)


# ============================================================
# GENERATE CROISSANT
# ============================================================

@app.post("/api/v1/croissant/generate")
async def generate_croissant_endpoint(
        request: CroissantGenerateRequest,
):

    # 1. Verificar os sete campos obrigatórios antes de
    #    descarregar o ficheiro do GitHub.

    pending_fields = []
    field_errors = {}

    for field_name in REQUIRED_FIELDS:

        field = getattr(request.metadata, field_name)

        if field.status != "confirmed" or is_empty(field.value):
            pending_fields.append(field_name)
            continue

        # Validar novamente no endpoint de geração.
        # Assim, um pedido direto à API não consegue contornar
        # as regras aplicadas durante a revisão dos metadados.

        validation_value = field.value

        # ExtractedMetadata converte os criadores em CreatorIdentity.
        # O validador genérico trabalha com dicionários, por isso
        # convertemos os objetos antes de validar o campo creators.

        if field_name == "creators" and isinstance(field.value, list):
            validation_value = [
                item.model_dump(exclude_none=True)
                if isinstance(item, CreatorIdentity)
                else item
                for item in field.value
            ]

        errors = validate_metadata_field(
            field_name,
            validation_value,
        )

        if errors:
            field_errors[field_name] = errors

    if pending_fields or field_errors:

        detail = {
            "message": (
                "Não é possível gerar o Croissant. "
                "Existem campos por confirmar ou valores inválidos."
            ),
        }

        if pending_fields:
            detail["pending_fields"] = pending_fields

        if field_errors:
            detail["field_errors"] = field_errors

        raise HTTPException(
            status_code=422,
            detail=detail,
        )

    # 1A. Revalidar ORCID e ROR nos registos oficiais.
    # A confirmação anterior não deve ser contornável por um pedido
    # direto ao endpoint de geração com metadata marcada como confirmed.

    request.metadata.creators.value = await verify_creator_identities(
        request.metadata.creators.value
    )

    # 2. Analisar novamente o repositório.

    repository_data = await analyze_github_repository(
        request
    )

    # 3. Inspecionar novamente o dataset selecionado.

    inspection = await inspect_dataset_in_repository(
        repository_data,
        request,
    )

    dataset_info = inspection["dataset"]

    # 4. Obter a identificação do repositório e do commit.

    commit_sha = repository_data["repository"]["commit_sha"]

    owner, repo = get_repository_parts(request.url)

    owner_path = quote(owner, safe="")
    repo_path = quote(repo, safe="")

    dataset_path = quote(
        request.dataset_path,
        safe="/",
    )

    # 5. Descarregar o ficheiro através da GitHub API.

    async with httpx.AsyncClient(
            timeout=httpx.Timeout(30, connect=5),
            follow_redirects=False,
    ) as client:

        content = await github_get(
            client=client,
            path=(
                f"/repos/{owner_path}/{repo_path}/contents/"
                f"{dataset_path}"
            ),
            params={
                "ref": commit_sha,
            },
        )

    # 6. Confirmar o formato da resposta.

    if content.get("encoding") != "base64":
        raise HTTPException(
            status_code=422,
            detail=(
                "O GitHub não disponibilizou o dataset "
                "num formato Base64 suportado."
            ),
        )

    # 7. Descodificar o conteúdo.

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
            status_code=422,
            detail=(
                "Não foi possível descodificar "
                "o ficheiro do GitHub."
            ),
        ) from exc

    # 8. Confirmar o limite de inspeção.

    if len(raw) > 1_000_000:
        raise HTTPException(
            status_code=413,
            detail=(
                "O dataset ultrapassa o limite de "
                "1 MB para esta versão."
            ),
        )

    # 9. Construir o URL do ficheiro na versão exata
    #    do repositório analisado.

    source_content_url = (
        "https://raw.githubusercontent.com/"
        f"{owner_path}/"
        f"{repo_path}/"
        f"{quote(commit_sha, safe='')}/"
        f"{dataset_path}"
    )

    # 10. Preparar e validar o caminho relativo.

    relative_path = Path(request.dataset_path)

    if (
            relative_path.is_absolute()
            or relative_path.drive
            or ".." in relative_path.parts
            or "\\" in request.dataset_path
    ):
        raise HTTPException(
            status_code=422,
            detail="Caminho do dataset inválido.",
        )

    # 11. Criar uma pasta temporária.
    #     É eliminada automaticamente no final do pedido.

    with tempfile.TemporaryDirectory(
            prefix="zenodo_metadata_croissant_",
    ) as temp_dir:

        dataset_directory = Path(temp_dir).resolve()

        local_dataset = (
                dataset_directory / relative_path
        ).resolve()

        try:
            local_dataset.relative_to(
                dataset_directory
            )

        except ValueError as exc:
            raise HTTPException(
                status_code=422,
                detail="Caminho do dataset fora da pasta temporária.",
            ) from exc

        # 12. Criar a estrutura de diretórios necessária.

        local_dataset.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        # 13. Guardar temporariamente o dataset.

        local_dataset.write_bytes(raw)

        output_path = (
                dataset_directory / "croissant.json"
        )

        # 14. Gerar e validar o documento Croissant.

        croissant = build_croissant(
            dataset_directory=dataset_directory,
            dataset_relative_path=relative_path.as_posix(),
            output_path=output_path,
            source_content_url=source_content_url,
            repository_url=repository_data["repository"]["url"],
            metadata=request.metadata,
            inspection=inspection,
        )

        # 15. Devolver o resultado.
        #     O JSON é carregado em memória antes da eliminação
        #     da pasta temporária.

        return {
            "filename": "croissant.json",
            "format": "JSON-LD",
            "validated": True,
            "repository": repository_data["repository"]["full_name"],
            "commit_sha": commit_sha,
            "selected_dataset": dataset_info["path"],
            "croissant": croissant,
        }


# ============================================================
# ORCID OAUTH
# ============================================================

@app.post("/api/v1/zenodo/publish")
async def publish_to_zenodo(request: ZenodoPublishRequest):
    """Cria e publica um depósito Zenodo contendo croissant.json."""
    base_url = os.getenv("ZENODO_BASE_URL", "https://sandbox.zenodo.org").strip().rstrip("/")
    access_token = os.getenv("ZENODO_ACCESS_TOKEN", "").strip()
    if not access_token:
        raise HTTPException(status_code=503, detail="Configura ZENODO_ACCESS_TOKEN no backend/.env.")
    async with httpx.AsyncClient(timeout=60) as client:
        stage = "obtenção do token OAuth"
        try:
            headers = {"Authorization": f"Bearer {access_token}"}
            stage = "criação do depósito"
            created = await client.post(
                f"{base_url}/api/deposit/depositions",
                headers=headers,
                json={},
            )
            created.raise_for_status()
            deposition = created.json()
            deposition_id = deposition["id"]
            stage = "atualização dos metadados"
            metadata = {
                "metadata": {
                    "title": request.title,
                    "upload_type": "dataset",
                    "description": request.description,
                    "creators": [{"name": "Zenodo Metadata Assistant"}],
                }
            }
            if request.iscte_affiliated:
                metadata["metadata"]["communities"] = [{"identifier": "iscte_teste"}]
            updated = await client.put(
                f"{base_url}/api/deposit/depositions/{deposition_id}",
                headers=headers,
                json=metadata,
            )
            updated.raise_for_status()
            stage = "upload do ficheiro croissant.json"
            uploaded = await client.post(
                f"{base_url}/api/deposit/depositions/{deposition_id}/files",
                headers=headers,
                data={"name": "croissant.json"},
                files={"file": ("croissant.json", json.dumps(request.croissant, ensure_ascii=False), "application/ld+json")},
            )
            uploaded.raise_for_status()
            stage = "publicação do depósito"
            published = await client.post(
                f"{base_url}/api/deposit/depositions/{deposition_id}/actions/publish",
                headers=headers,
            )
            published.raise_for_status()
            result = published.json()
            return {"id": result["id"], "doi": result.get("doi"), "url": result.get("links", {}).get("html")}
        except httpx.HTTPStatusError as exc:
            detail = f"Falha na {stage} (HTTP {exc.response.status_code})."
            try:
                payload = exc.response.json()
                message = payload.get("message") or payload.get("errors")
                if message:
                    detail += f" Resposta Zenodo: {message}"
            except Exception:
                pass
            raise HTTPException(status_code=502, detail=detail) from exc
        except (httpx.HTTPError, KeyError, ValueError) as exc:
            raise HTTPException(status_code=502, detail=f"Falha na {stage}: {type(exc).__name__}.") from exc

@app.get("/api/v1/orcid/validate/{orcid_id}")
async def validate_orcid_record(
        orcid_id: str,
):
    """Obtém o nome público associado a um ORCID iD."""

    try:
        return await get_public_orcid_identity(orcid_id)
    except ORCIDInvalidIDError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc
    except ORCIDNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc
    except ORCIDNameUnavailableError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc
    except ORCIDServiceError as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                "Não foi possível consultar a API ORCID. "
                "Tenta novamente mais tarde."
            ),
        ) from exc


@app.get("/api/v1/orcid/authorize")
async def orcid_authorize():
    """Inicia a autenticação ORCID e redireciona para o Sandbox."""

    authorization_url = create_orcid_authorization_url()

    return RedirectResponse(
        url=authorization_url,
        status_code=302,
    )


@app.get("/api/v1/orcid/callback", response_class=HTMLResponse)
async def orcid_callback(
        code: str | None = None,
        state: str | None = None,
        error: str | None = None,
):
    """Recebe o retorno OAuth e envia o resultado à janela da aplicação."""

    frontend_origin = os.getenv(
        "FRONTEND_ORIGIN",
        "http://localhost:5173",
    ).rstrip("/")

    def popup_response(payload: dict, status_code: int = 200):
        payload_json = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
        origin_json = json.dumps(frontend_origin)
        html = f"""<!doctype html>
<html lang="pt">
<head><meta charset="utf-8"><title>Autenticação ORCID</title></head>
<body>
<p>Autenticação ORCID concluída. Podes fechar esta janela.</p>
<script>
const payload = {payload_json};
if (window.opener && !window.opener.closed) {{
  window.opener.postMessage(payload, {origin_json});
}}
window.close();
</script>
</body>
</html>"""
        return HTMLResponse(content=html, status_code=status_code)

    if error:
        if state:
            consume_orcid_state(state)
        return popup_response({
            "source": "zenodo-metadata-orcid",
            "status": "error",
            "message": "A autorização ORCID foi cancelada ou recusada.",
        }, 400)

    if not code or not state:
        return popup_response({
            "source": "zenodo-metadata-orcid",
            "status": "error",
            "message": "A resposta ORCID não contém o código e o state necessários.",
        }, 400)

    try:
        result = await exchange_orcid_authorization_code(
            code=code,
            state=state,
        )
    except ValueError as exc:
        return popup_response({
            "source": "zenodo-metadata-orcid",
            "status": "error",
            "message": str(exc),
        }, 400)
    except RuntimeError:
        return popup_response({
            "source": "zenodo-metadata-orcid",
            "status": "error",
            "message": (
                "Não foi possível concluir a autenticação com a ORCID. "
                "Confirma a ligação e as credenciais configuradas."
            ),
        }, 502)

    return popup_response({
        "source": "zenodo-metadata-orcid",
        "status": "authenticated",
        "creator": {
            "creator_type": "person",
            "name": result["name"],
            "orcid_id": result["orcid"],
        },
        "creator_session_token": result["creator_session_token"],
    })


# ROR — PESQUISAR ORGANIZAÇÕES
# ============================================================

@app.get("/api/v1/ror/search")
async def search_ror_organizations(
        query: str = Query(
            ...,
            min_length=2,
            max_length=200,
            description="Nome ou parte do nome da organização.",
        ),
        page: int = Query(
            1,
            ge=1,
            le=500,
        ),
):
    """Pesquisa organizações no registo público ROR."""

    try:
        return await search_organizations(
            query=query,
            page=page,
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    except RORServiceError as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                "Não foi possível consultar o registo ROR. "
                "Tenta novamente mais tarde."
            ),
        ) from exc


# ============================================================
# ROR — VALIDAR IDENTIFICADOR
# ============================================================

@app.get("/api/v1/ror/validate/{ror_id}")
async def validate_ror_organization(
        ror_id: str,
):
    """Confirma que o ROR iD corresponde a um registo existente."""

    try:
        return await validate_ror_id(ror_id)

    except RORInvalidIDError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    except RORNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail="O ROR iD não existe no registo ROR.",
        ) from exc

    except RORServiceError as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                "Não foi possível consultar o registo ROR. "
                "Tenta novamente mais tarde."
            ),
        ) from exc
