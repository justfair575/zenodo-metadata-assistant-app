
"""Serviços OAuth ORCID para desenvolvimento local com FastAPI.

O estado OAuth, os tokens e as sessões ficam em memória.

Esta implementação é adequada a um único processo de desenvolvimento.
Em produção deve usar-se armazenamento partilhado e persistente.
"""

from __future__ import annotations

import os
import secrets
import time

from typing import Any
from urllib.parse import quote, urlencode, urlsplit

import httpx

from app.metadata_schema import (
    is_valid_orcid,
    normalize_orcid,
)


ORCID_CLIENT_ID = os.getenv(
    "ORCID_CLIENT_ID",
    "",
).strip()

ORCID_CLIENT_SECRET = os.getenv(
    "ORCID_CLIENT_SECRET",
    "",
).strip()

ORCID_API_BASE_URL = os.getenv(
    "ORCID_API_BASE_URL",
    "https://orcid.org",
).strip().rstrip("/")

ORCID_REDIRECT_URI = os.getenv(
    "ORCID_REDIRECT_URI",
    "",
).strip()


ORCID_STATE_TTL_SECONDS = 600

ORCID_CREATOR_SESSION_TTL_SECONDS = (
        8 * 60 * 60
)


_ALLOWED_ORCID_BASE_URLS = {
    "https://sandbox.orcid.org",
    "https://orcid.org",
}


_PENDING_ORCID_STATES: dict[str, float] = {}

_AUTHENTICATED_ORCID_TOKENS: dict[
    str,
    dict[str, Any],
] = {}

_AUTHENTICATED_CREATOR_SESSIONS: dict[
    str,
    dict[str, Any],
] = {}

_PUBLIC_API_ACCESS_TOKEN: str | None = None
_PUBLIC_API_TOKEN_EXPIRES_AT = 0.0


class ORCIDServiceError(Exception):
    """Erro geral na comunicação com o registo público ORCID."""


class ORCIDNotFoundError(ORCIDServiceError):
    """O ORCID iD não foi encontrado."""


class ORCIDInvalidIDError(ORCIDServiceError):
    """O formato ou checksum do ORCID iD é inválido."""


class ORCIDNameUnavailableError(ORCIDServiceError):
    """O registo existe, mas não disponibiliza um nome público."""


def _public_api_base_url() -> str:
    if ORCID_API_BASE_URL == "https://sandbox.orcid.org":
        return "https://pub.sandbox.orcid.org"
    return "https://pub.orcid.org"


async def _get_public_api_access_token(
        client: httpx.AsyncClient,
) -> str:
    global _PUBLIC_API_ACCESS_TOKEN
    global _PUBLIC_API_TOKEN_EXPIRES_AT

    if (
            _PUBLIC_API_ACCESS_TOKEN
            and _PUBLIC_API_TOKEN_EXPIRES_AT > time.time()
    ):
        return _PUBLIC_API_ACCESS_TOKEN

    if not ORCID_CLIENT_ID or not ORCID_CLIENT_SECRET:
        raise ORCIDServiceError(
            "As credenciais da aplicação ORCID não estão configuradas."
        )

    try:
        response = await client.post(
            f"{ORCID_API_BASE_URL}/oauth/token",
            data={
                "client_id": ORCID_CLIENT_ID,
                "client_secret": ORCID_CLIENT_SECRET,
                "grant_type": "client_credentials",
                "scope": "/read-public",
            },
            headers={
                "Accept": "application/json",
                "Content-Type": "application/x-www-form-urlencoded",
            },
        )
        response.raise_for_status()
        token_data = response.json()
    except (httpx.HTTPError, ValueError) as exc:
        raise ORCIDServiceError(
            "Não foi possível obter acesso de leitura pública à API ORCID."
        ) from exc

    if not isinstance(token_data, dict):
        raise ORCIDServiceError(
            "A API ORCID devolveu uma resposta de token inválida."
        )

    access_token = token_data.get("access_token")
    if not isinstance(access_token, str) or not access_token:
        raise ORCIDServiceError(
            "A API ORCID não devolveu um token de leitura pública válido."
        )

    try:
        expires_in = int(token_data.get("expires_in", 0))
    except (TypeError, ValueError) as exc:
        raise ORCIDServiceError(
            "A API ORCID devolveu um prazo de token inválido."
        ) from exc

    if expires_in <= 0:
        raise ORCIDServiceError(
            "A API ORCID devolveu um prazo de token inválido."
        )

    _PUBLIC_API_ACCESS_TOKEN = access_token
    _PUBLIC_API_TOKEN_EXPIRES_AT = time.time() + max(expires_in - 60, 0)
    return access_token


async def get_public_orcid_identity(
        orcid_id: str,
) -> dict[str, str]:
    """Obtém o identificador e o nome visível publicamente no registo ORCID."""

    if not isinstance(orcid_id, str):
        raise ORCIDInvalidIDError("O ORCID iD tem de ser texto.")

    normalized_orcid = normalize_orcid(orcid_id)
    if not is_valid_orcid(normalized_orcid):
        raise ORCIDInvalidIDError("O formato ou checksum do ORCID iD é inválido.")

    url = (
        f"{_public_api_base_url()}/v3.0/"
        f"{quote(normalized_orcid, safe='')}/person"
    )

    try:
        async with httpx.AsyncClient(
                timeout=httpx.Timeout(15.0, connect=5.0),
                follow_redirects=False,
        ) as client:
            access_token = await _get_public_api_access_token(client)
            response = await client.get(
                url,
                headers={
                    "Accept": "application/json",
                    "User-Agent": "Zenodo-Metadata-Assistant",
                    "Authorization": f"Bearer {access_token}",
                },
            )

        if response.status_code == 404:
            raise ORCIDNotFoundError("O ORCID iD não foi encontrado no registo.")

        response.raise_for_status()
        person = response.json()
        if not isinstance(person, dict):
            raise ORCIDServiceError("A API ORCID devolveu uma resposta inválida.")

    except ORCIDServiceError:
        raise
    except httpx.HTTPStatusError as exc:
        raise ORCIDServiceError("A API ORCID devolveu um erro HTTP.") from exc
    except (httpx.HTTPError, ValueError) as exc:
        raise ORCIDServiceError("Não foi possível comunicar com a API ORCID.") from exc

    name = _name_from_person_record(person)
    if not name:
        raise ORCIDNameUnavailableError(
            "O registo ORCID existe, mas não disponibiliza um nome público."
        )

    return {
        "valid": True,
        "exists": True,
        "orcid_id": normalized_orcid,
        "orcid_url": f"https://orcid.org/{normalized_orcid}",
        "name": name,
    }


# ============================================================
# VALIDAR CONFIGURAÇÃO
# ============================================================

def _validate_configuration() -> None:

    missing = []

    if not ORCID_CLIENT_ID:
        missing.append("ORCID_CLIENT_ID")

    if not ORCID_CLIENT_SECRET:
        missing.append("ORCID_CLIENT_SECRET")

    if not ORCID_REDIRECT_URI:
        missing.append("ORCID_REDIRECT_URI")

    if missing:
        raise RuntimeError(
            "Configuração ORCID incompleta no .env: "
            + ", ".join(missing)
        )

    if ORCID_API_BASE_URL not in _ALLOWED_ORCID_BASE_URLS:

        raise RuntimeError(
            "ORCID_API_BASE_URL deve ser "
            "https://sandbox.orcid.org ou "
            "https://orcid.org."
        )

    parsed = urlsplit(ORCID_REDIRECT_URI)

    is_local_http = (
            parsed.scheme == "http"
            and parsed.hostname in {
                "localhost",
                "127.0.0.1",
            }
    )

    is_https = parsed.scheme == "https"

    if (
            not (is_https or is_local_http)
            or not parsed.netloc
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
    ):

        raise RuntimeError(
            "ORCID_REDIRECT_URI tem de ser HTTPS. "
            "Em desenvolvimento local também é permitido "
            "HTTP apenas para localhost/127.0.0.1."
        )


# ============================================================
# GERAR URL DE AUTORIZAÇÃO
# ============================================================

def create_orcid_authorization_url() -> str:

    _validate_configuration()

    now = time.time()

    for key, expires_at in list(
            _PENDING_ORCID_STATES.items()
    ):

        if expires_at <= now:
            _PENDING_ORCID_STATES.pop(
                key,
                None,
            )

    state = secrets.token_urlsafe(32)

    _PENDING_ORCID_STATES[state] = (
            now + ORCID_STATE_TTL_SECONDS
    )

    params = {
        "client_id": ORCID_CLIENT_ID,
        "response_type": "code",
        "scope": "/authenticate",
        "redirect_uri": ORCID_REDIRECT_URI,
        "state": state,
    }

    return (
        f"{ORCID_API_BASE_URL}/oauth/authorize?"
        f"{urlencode(params)}"
    )


# ============================================================
# VALIDAR STATE
# ============================================================

def consume_orcid_state(state: str) -> bool:

    expires_at = _PENDING_ORCID_STATES.pop(
        state,
        None,
    )

    return (
            expires_at is not None
            and expires_at > time.time()
    )


# ============================================================
# NORMALIZAR NOME
# ============================================================

def _clean_name(value: Any) -> str | None:

    if not isinstance(value, str):
        return None

    value = " ".join(value.split()).strip()

    return value or None


def _name_from_person_record(
        person: dict[str, Any],
) -> str | None:

    name_data = person.get("name") or {}

    credit_name = (
            name_data.get("credit-name") or {}
    ).get("value")

    if _clean_name(credit_name):
        return _clean_name(credit_name)

    given = (
            name_data.get("given-names") or {}
    ).get("value")

    family = (
            name_data.get("family-name") or {}
    ).get("value")

    full_name = " ".join(
        part
        for part in (given, family)
        if isinstance(part, str) and part.strip()
    )

    return _clean_name(full_name)


# ============================================================
# OBTER NOME ORCID
# ============================================================

async def _resolve_orcid_name(
        client: httpx.AsyncClient,
        token_data: dict[str, Any],
        orcid_id: str,
) -> str | None:

    # Primeiro, utilizar o nome devolvido pelo OAuth.
    name = _clean_name(
        token_data.get("name")
    )

    if name:
        return name

    # Fallback: tentar obter o nome público do registo ORCID.
    try:

        response = await client.get(
            f"{ORCID_API_BASE_URL}/v3.0/{orcid_id}/person",
            headers={
                "Accept": "application/json",
                "Authorization": (
                    f"Bearer {token_data['access_token']}"
                ),
            },
        )

        if response.is_success:
            return _name_from_person_record(
                response.json()
            )

    except (
            httpx.HTTPError,
            ValueError,
            KeyError,
    ):
        pass

    return None


# ============================================================
# LIMPAR SESSÕES EXPIRADAS
# ============================================================

def _cleanup_expired_creator_sessions() -> None:

    now = time.time()

    for token, session in list(
            _AUTHENTICATED_CREATOR_SESSIONS.items()
    ):

        if session.get("expires_at", 0) <= now:

            _AUTHENTICATED_CREATOR_SESSIONS.pop(
                token,
                None,
            )


def _cleanup_expired_orcid_tokens() -> None:

    now = time.time()

    for orcid_id, token_data in list(
            _AUTHENTICATED_ORCID_TOKENS.items()
    ):

        expires_at = token_data.get("expires_at")

        if (
                expires_at is not None
                and expires_at <= now
        ):

            _AUTHENTICATED_ORCID_TOKENS.pop(
                orcid_id,
                None,
            )


# ============================================================
# TROCAR CÓDIGO POR TOKEN E IDENTIDADE
# ============================================================

async def exchange_orcid_authorization_code(
        *,
        code: str,
        state: str,
) -> dict[str, Any]:

    _validate_configuration()

    if not consume_orcid_state(state):

        raise ValueError(
            "O state ORCID é inválido, já foi utilizado "
            "ou expirou. Inicia novamente a autenticação."
        )

    token_url = (
        f"{ORCID_API_BASE_URL}/oauth/token"
    )

    payload = {
        "client_id": ORCID_CLIENT_ID,
        "client_secret": ORCID_CLIENT_SECRET,
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": ORCID_REDIRECT_URI,
    }

    headers = {
        "Accept": "application/json",
        "Content-Type": "application/x-www-form-urlencoded",
    }

    try:

        async with httpx.AsyncClient(
                timeout=httpx.Timeout(
                    20.0,
                    connect=5.0,
                ),
                follow_redirects=False,
        ) as client:

            response = await client.post(
                token_url,
                data=payload,
                headers=headers,
            )

            response.raise_for_status()

            token_data = response.json()

            # Validar ORCID.
            raw_orcid = token_data.get("orcid")

            if (
                    not isinstance(raw_orcid, str)
                    or not raw_orcid.strip()
            ):

                raise RuntimeError(
                    "A ORCID não devolveu um ORCID iD."
                )

            orcid_id = normalize_orcid(
                raw_orcid
            )

            if not is_valid_orcid(orcid_id):

                raise RuntimeError(
                    "A ORCID devolveu um identificador inválido."
                )

            # Validar access token.
            access_token = token_data.get(
                "access_token"
            )

            if (
                    not isinstance(access_token, str)
                    or not access_token
            ):

                raise RuntimeError(
                    "A ORCID não devolveu um access token válido."
                )

            # Obter nome.
            name = await _resolve_orcid_name(
                client,
                token_data,
                orcid_id,
            )

    except (
            httpx.HTTPError,
            ValueError,
    ) as exc:

        raise RuntimeError(
            "Falha na comunicação OAuth com a ORCID."
        ) from exc

    # Não criar uma sessão sem identidade utilizável.
    if not name:

        raise RuntimeError(
            "A ORCID autenticou a conta, mas não disponibilizou "
            "um nome público. Verifica a visibilidade do nome "
            "no registo ORCID e tenta novamente."
        )

    # Calcular expiração do token ORCID.
    try:

        expires_at = (
                time.time()
                + int(token_data.get("expires_in"))
        )

    except (
            TypeError,
            ValueError,
    ):

        expires_at = None

    # Guardar token exclusivamente no backend.
    _cleanup_expired_orcid_tokens()

    _AUTHENTICATED_ORCID_TOKENS[orcid_id] = {
        "access_token": access_token,
        "token_type": token_data.get(
            "token_type",
            "bearer",
        ),
        "scope": token_data.get(
            "scope",
            "/authenticate",
        ),
        "expires_at": expires_at,
        "refresh_token": token_data.get(
            "refresh_token"
        ),
    }

    # Criar sessão opaca da aplicação.
    _cleanup_expired_creator_sessions()

    creator_session_token = secrets.token_urlsafe(40)

    _AUTHENTICATED_CREATOR_SESSIONS[
        creator_session_token
    ] = {
        "creator": {
            "creator_type": "person",
            "name": name,
            "orcid_id": orcid_id,
        },
        "expires_at": (
                time.time()
                + ORCID_CREATOR_SESSION_TTL_SECONDS
        ),
    }

    return {
        "orcid": orcid_id,
        "name": name,
        "scope": token_data.get(
            "scope",
            "/authenticate",
        ),
        "creator_session_token": creator_session_token,
    }


# ============================================================
# OBTER IDENTIDADE AUTENTICADA
# ============================================================

def get_authenticated_creator_session(
        creator_session_token: str,
) -> dict[str, str] | None:

    if (
            not isinstance(creator_session_token, str)
            or not creator_session_token
    ):
        return None

    _cleanup_expired_creator_sessions()

    session = _AUTHENTICATED_CREATOR_SESSIONS.get(
        creator_session_token
    )

    if not session:
        return None

    creator = session.get("creator")

    if not isinstance(creator, dict):
        return None

    name = _clean_name(
        creator.get("name")
    )

    orcid_id = normalize_orcid(
        str(creator.get("orcid_id") or "")
    )

    if (
            not name
            or not is_valid_orcid(orcid_id)
    ):
        return None

    return {
        "creator_type": "person",
        "name": name,
        "orcid_id": orcid_id,
    }


# ============================================================
# OBTER TOKEN ORCID
# ============================================================

def get_authenticated_orcid_token(
        orcid_id: str,
) -> dict[str, Any] | None:

    normalized = normalize_orcid(orcid_id)

    if not is_valid_orcid(normalized):
        return None

    _cleanup_expired_orcid_tokens()

    return _AUTHENTICATED_ORCID_TOKENS.get(
        normalized
    )
