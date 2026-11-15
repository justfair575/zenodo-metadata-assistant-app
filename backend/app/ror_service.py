
"""Integração com a API pública do Research Organization Registry (ROR).

Permite pesquisar organizações e validar ROR iDs através da API oficial.
A seleção dos resultados de pesquisa é sempre feita pelo utilizador.
"""

from __future__ import annotations

import re
from typing import Any
from urllib.parse import quote

import httpx


# ============================================================
# CONFIGURAÇÃO
# ============================================================

ROR_API_BASE_URL = "https://api.ror.org/v2/organizations"

ROR_TIMEOUT = httpx.Timeout(
    15.0,
    connect=5.0,
)

ROR_ID_PATTERN = re.compile(
    r"^0[a-z0-9]{6}[0-9]{2}$",
    re.IGNORECASE,
)


# ============================================================
# EXCEÇÕES
# ============================================================

class RORServiceError(Exception):
    """Erro geral na comunicação com o serviço ROR."""


class RORNotFoundError(RORServiceError):
    """O ROR iD não foi encontrado."""


class RORInvalidIDError(RORServiceError):
    """O formato do ROR iD é inválido."""


# ============================================================
# NORMALIZAÇÃO
# ============================================================

def normalize_ror_id(value: str) -> str:
    """Converte um ROR URL ou identificador para o formato curto."""

    if not isinstance(value, str):
        raise RORInvalidIDError(
            "O ROR iD tem de ser um texto."
        )

    value = value.strip()

    value = re.sub(
        r"^https?://ror\.org/",
        "",
        value,
        flags=re.IGNORECASE,
    )

    value = value.rstrip("/")

    if not ROR_ID_PATTERN.fullmatch(value):
        raise RORInvalidIDError(
            "O formato do ROR iD é inválido."
        )

    return value.lower()


def normalize_ror_url(value: str) -> str:
    """Devolve o URL canónico de um ROR iD."""

    return f"https://ror.org/{normalize_ror_id(value)}"


# ============================================================
# EXTRAIR NOME PRINCIPAL
# ============================================================

def _get_display_name(
        names: list[dict[str, Any]],
) -> str | None:
    """Obtém o nome preferencial do registo ROR."""

    if not isinstance(names, list):
        return None

    for item in names:
        if not isinstance(item, dict):
            continue

        types = item.get("types", [])

        if (
                isinstance(types, list)
                and "ror_display" in types
        ):
            value = item.get("value")

            if isinstance(value, str) and value.strip():
                return value.strip()

    for item in names:
        if not isinstance(item, dict):
            continue

        value = item.get("value")

        if isinstance(value, str) and value.strip():
            return value.strip()

    return None


# ============================================================
# EXTRAIR PAÍS
# ============================================================

def _get_country(
        locations: list[dict[str, Any]],
) -> dict[str, str] | None:
    """Obtém informação geográfica do registo ROR."""

    if not isinstance(locations, list):
        return None

    for location in locations:
        if not isinstance(location, dict):
            continue

        details = location.get(
            "geonames_details",
            {},
        )

        if not isinstance(details, dict):
            continue

        country_name = details.get("country_name")
        country_code = details.get("country_code")

        if country_name or country_code:
            return {
                "name": country_name or "",
                "code": country_code or "",
            }

    return None


# ============================================================
# NORMALIZAR REGISTO ROR
# ============================================================

def _normalize_organization(
        record: dict[str, Any],
) -> dict[str, Any]:
    """Extrai os dados necessários para a Zenodo Metadata App."""

    if not isinstance(record, dict):
        raise RORServiceError(
            "Resposta inesperada da API ROR."
        )

    raw_id = record.get("id")

    if not isinstance(raw_id, str):
        raise RORServiceError(
            "O registo ROR não contém um identificador."
        )

    ror_id = normalize_ror_id(raw_id)

    names = record.get("names", [])

    if not isinstance(names, list):
        names = []

    display_name = _get_display_name(names)

    if not display_name:
        raise RORServiceError(
            "O registo ROR não contém um nome válido."
        )

    alternative_names = []

    for item in names:
        if not isinstance(item, dict):
            continue

        name = item.get("value")

        if (
                isinstance(name, str)
                and name.strip()
                and name.strip() != display_name
                and name.strip() not in alternative_names
        ):
            alternative_names.append(name.strip())

    organization_types = record.get(
        "types",
        [],
    )

    if not isinstance(organization_types, list):
        organization_types = []

    return {
        "ror_id": ror_id,
        "ror_url": f"https://ror.org/{ror_id}",
        "name": display_name,
        "alternative_names": alternative_names,
        "status": record.get("status"),
        "types": organization_types,
        "country": _get_country(
            record.get("locations", [])
        ),
    }


# ============================================================
# PEDIDO HTTP
# ============================================================

async def _ror_get(
        url: str,
        params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Executa um pedido GET à API pública ROR."""

    headers = {
        "Accept": "application/json",
        "User-Agent": "Zenodo-Metadata-Assistant",
    }

    try:
        async with httpx.AsyncClient(
                timeout=ROR_TIMEOUT,
                follow_redirects=False,
        ) as client:

            response = await client.get(
                url,
                params=params,
                headers=headers,
            )

            if response.status_code == 404:
                raise RORNotFoundError(
                    "A organização não foi encontrada no registo ROR."
                )

            response.raise_for_status()

            data = response.json()

            if not isinstance(data, dict):
                raise RORServiceError(
                    "A API ROR devolveu uma resposta inválida."
                )

            if data.get("errors"):
                raise RORServiceError(
                    "A API ROR devolveu um erro na pesquisa."
                )

            return data

    except RORServiceError:
        raise

    except httpx.HTTPStatusError as exc:
        raise RORServiceError(
            "A API ROR devolveu um erro HTTP."
        ) from exc

    except (httpx.HTTPError, ValueError) as exc:
        raise RORServiceError(
            "Não foi possível comunicar com a API ROR."
        ) from exc


# ============================================================
# PESQUISAR ORGANIZAÇÕES
# ============================================================

async def search_organizations(
        query: str,
        page: int = 1,
) -> dict[str, Any]:
    """Pesquisa organizações pelo nome na API ROR.

    A API ROR devolve sugestões; a aplicação não escolhe automaticamente
    o primeiro resultado.
    """

    if not isinstance(query, str):
        raise ValueError(
            "A pesquisa tem de ser um texto."
        )

    query = query.strip()

    if len(query) < 2:
        raise ValueError(
            "Introduz pelo menos dois caracteres para pesquisar."
        )

    if len(query) > 200:
        raise ValueError(
            "A pesquisa não pode ultrapassar 200 caracteres."
        )

    if (
            isinstance(page, bool)
            or not isinstance(page, int)
            or page < 1
            or page > 500
    ):
        raise ValueError(
            "A página deve estar entre 1 e 500."
        )

    data = await _ror_get(
        ROR_API_BASE_URL,
        params={
            "query": query,
            "page": page,
        },
    )

    raw_items = data.get("items", [])

    if not isinstance(raw_items, list):
        raise RORServiceError(
            "A API ROR devolveu resultados inválidos."
        )

    organizations = []

    for item in raw_items:
        try:
            organizations.append(
                _normalize_organization(item)
            )
        except (RORServiceError, RORInvalidIDError):
            continue

    total_results = data.get(
        "number_of_results",
        0,
    )

    if not isinstance(total_results, int):
        total_results = 0

    return {
        "query": query,
        "page": page,
        "total_results": total_results,
        "results_count": len(organizations),
        "organizations": organizations,
    }


# ============================================================
# CONSULTAR ORGANIZAÇÃO POR ROR ID
# ============================================================

async def get_organization(
        ror_id: str,
) -> dict[str, Any]:
    """Obtém o registo de uma organização através do ROR iD."""

    normalized_id = normalize_ror_id(ror_id)

    url = (
        f"{ROR_API_BASE_URL}/"
        f"{quote(normalized_id, safe='')}"
    )

    data = await _ror_get(url)

    organization = _normalize_organization(data)

    if organization["ror_id"] != normalized_id:
        raise RORServiceError(
            "O identificador devolvido não corresponde ao solicitado."
        )

    return organization


# ============================================================
# VALIDAR ROR ID
# ============================================================

async def validate_ror_id(
        ror_id: str,
) -> dict[str, Any]:
    """Confirma que o ROR iD corresponde a um registo existente."""

    organization = await get_organization(ror_id)

    return {
        "valid": True,
        "exists": True,
        "ror_id": organization["ror_id"],
        "ror_url": organization["ror_url"],
        "name": organization["name"],
        "alternative_names": organization["alternative_names"],
        "status": organization["status"],
        "country": organization["country"],
        "types": organization["types"],
    }
