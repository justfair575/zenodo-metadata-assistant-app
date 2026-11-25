from copy import deepcopy
from typing import Any

from fastapi import HTTPException
from pydantic import ValidationError

from app.metadata_schema import (
    ConfirmMetadataRequest,
    ConfirmedMetadataResponse,
    CreatorIdentity,
    MetadataField,
    REQUIRED_FIELDS,
)

from app.orcid_service import (
    ORCIDInvalidIDError,
    ORCIDNameUnavailableError,
    ORCIDNotFoundError,
    ORCIDServiceError,
    get_public_orcid_identity,
)

from app.metadata_validation import (
    is_empty,
    validate_metadata_field,
)

from app.ror_service import (
    RORInvalidIDError,
    RORNotFoundError,
    RORServiceError,
    validate_ror_id,
)


# ============================================================
# CAMPOS OBRIGATÓRIOS
# ============================================================

# ============================================================
# CONVERSÃO DE CRIADORES PARA VALIDAÇÃO GERAL
# ============================================================

def _creator_values_for_general_validation(
        value: Any,
) -> Any:
    """
    Converte CreatorIdentity para dicionários antes de executar
    a validação genérica existente em metadata_validation.py.
    """

    if isinstance(value, list):

        return [
            item.model_dump(exclude_none=True)
            if isinstance(item, CreatorIdentity)
            else item
            for item in value
        ]

    return value


# ============================================================
# VALIDAR IDENTIDADES DOS CRIADORES
# ============================================================

def _validate_creator_identities(
        value: Any,
) -> list[CreatorIdentity]:
    """
    Valida a estrutura e o identificador obrigatório de cada criador.

    Pessoa:
        ORCID obrigatório.

    Organização:
        ROR obrigatório.

    Nesta fase valida-se o formato/checksum do ORCID e o formato do ROR.
    """

    if not isinstance(value, list) or not value:

        raise HTTPException(
            status_code=422,
            detail={
                "message": (
                    "É necessário indicar pelo menos um criador."
                ),
                "field": "creators",
            },
        )

    creators: list[CreatorIdentity] = []

    errors: list[dict[str, Any]] = []

    for index, item in enumerate(
            value,
            start=1,
    ):

        try:

            creator = (
                item
                if isinstance(
                    item,
                    CreatorIdentity,
                )
                else CreatorIdentity.model_validate(
                    item
                )
            )

        except ValidationError as exc:

            errors.append({
                "creator": index,
                "errors": exc.errors(
                    include_url=False
                ),
            })

            continue

        if (
                creator.creator_type == "person"
                and not creator.orcid_id
        ):

            errors.append({
                "creator": index,
                "error": (
                    "Uma pessoa necessita de um ORCID iD."
                ),
            })

        elif (
                creator.creator_type == "organization"
                and not creator.ror_id
        ):

            errors.append({
                "creator": index,
                "error": (
                    "Uma organização necessita de um ROR ID."
                ),
            })

        elif creator.creator_type is None:

            errors.append({
                "creator": index,
                "error": (
                    "É necessário indicar se o criador "
                    "é pessoa ou organização."
                ),
            })

        creators.append(creator)

    if errors:

        raise HTTPException(
            status_code=422,
            detail={
                "message": (
                    "Não é possível confirmar os criadores. "
                    "Cada pessoa necessita de ORCID e cada "
                    "organização necessita de ROR."
                ),
                "field": "creators",
                "errors": errors,
            },
        )

    return creators


# ============================================================
# VALIDAR ORGANIZAÇÕES NA API OFICIAL ROR
# ============================================================

async def _verify_ror_organizations(
        creators: list[CreatorIdentity],
) -> list[CreatorIdentity]:
    """
    Confirma a existência de cada ROR na API oficial.

    O nome devolvido pelo registo ROR passa a ser a designação canónica
    utilizada no metadado. Não se confia apenas no formato do identificador.
    """

    for index, creator in enumerate(creators, start=1):

        if creator.creator_type != "organization":
            continue

        if not creator.ror_id:

            raise HTTPException(
                status_code=422,
                detail={
                    "message": "Uma organização necessita de um ROR ID.",
                    "creator": index,
                },
            )

        try:

            verified = await validate_ror_id(
                creator.ror_id
            )

        except RORInvalidIDError as exc:

            raise HTTPException(
                status_code=422,
                detail={
                    "message": "O formato do ROR ID é inválido.",
                    "creator": index,
                },
            ) from exc

        except RORNotFoundError as exc:

            raise HTTPException(
                status_code=422,
                detail={
                    "message": "O ROR ID não existe no registo oficial.",
                    "creator": index,
                    "ror_id": creator.ror_id,
                },
            ) from exc

        except RORServiceError as exc:

            raise HTTPException(
                status_code=503,
                detail=(
                    "Não foi possível verificar o ROR na API oficial. "
                    "Tenta novamente mais tarde."
                ),
            ) from exc

        # Utilizar o nome oficial do registo ROR.
        creator.name = verified["name"]

    return creators


# ============================================================
# OBTER NOMES PÚBLICOS DOS INVESTIGADORES
# ============================================================

async def verify_orcid_people(
        creators: list[CreatorIdentity],
) -> list[CreatorIdentity]:
    """Valida ORCID iDs e usa o nome público devolvido pelo registo oficial."""

    people = [
        creator for creator in creators
        if creator.creator_type == "person"
    ]

    if len(people) > 1:
        raise HTTPException(
            status_code=422,
            detail=(
                "Nesta versão só é possível adicionar um investigador. "
                "Podes adicionar várias organizações através de ROR."
            ),
        )

    for index, creator in enumerate(creators, start=1):
        if creator.creator_type != "person":
            continue

        try:
            identity = await get_public_orcid_identity(creator.orcid_id or "")
        except ORCIDInvalidIDError as exc:
            raise HTTPException(
                status_code=422,
                detail={
                    "message": "O formato ou checksum do ORCID iD é inválido.",
                    "creator": index,
                },
            ) from exc
        except ORCIDNotFoundError as exc:
            raise HTTPException(
                status_code=422,
                detail={
                    "message": "O ORCID iD não existe no registo público.",
                    "creator": index,
                },
            ) from exc
        except ORCIDNameUnavailableError as exc:
            raise HTTPException(
                status_code=422,
                detail={
                    "message": str(exc),
                    "creator": index,
                },
            ) from exc
        except ORCIDServiceError as exc:
            raise HTTPException(
                status_code=503,
                detail=(
                    "Não foi possível consultar a API ORCID. "
                    "Tenta novamente mais tarde."
                ),
            ) from exc

        creator.orcid_id = identity["orcid_id"]
        creator.name = identity["name"]

    return creators

async def verify_creator_identities(
        creators: list[CreatorIdentity],
) -> list[CreatorIdentity]:
    """Valida ORCID e ROR e usa os nomes devolvidos pelos registos oficiais."""

    verified_people = await verify_orcid_people(creators)
    return await _verify_ror_organizations(verified_people)


# ============================================================
# APLICAR REVISÕES
# ============================================================

async def apply_metadata_reviews(
        request: ConfirmMetadataRequest,
) -> ConfirmedMetadataResponse:

    metadata = deepcopy(
        request.metadata
    )

    # Confirmed creator values are re-checked against the public registries,
    # including requests that bypass the explicit review action.

    creators_field = metadata.creators

    if (
            creators_field.status == "confirmed"
            and "creators" not in request.reviews
    ):

        creators_to_verify = _validate_creator_identities(
            creators_field.value
        )
        creators_field.value = await verify_creator_identities(
            creators_to_verify
        )

    # ========================================================
    # 2. PROCESSAR AS REVISÕES
    # ========================================================

    for field_name, review in request.reviews.items():

        field = getattr(
            metadata,
            field_name,
        )

        if not isinstance(
                field,
                MetadataField,
        ):

            raise HTTPException(
                status_code=422,
                detail=(
                    f"Campo não suportado para revisão: "
                    f"{field_name}"
                ),
            )

        # ----------------------------------------------------
        # ACEITAR SUGESTÃO
        # ----------------------------------------------------

        if review.action == "accept":

            if is_empty(
                    field.suggested_value
            ):

                raise HTTPException(
                    status_code=422,
                    detail=(
                        f"Não existe sugestão para aceitar "
                        f"no campo '{field_name}'."
                    ),
                )

            value_to_validate = (
                field.suggested_value
            )

        # ----------------------------------------------------
        # EDITAR VALOR
        # ----------------------------------------------------

        elif review.action == "edit":

            if (
                    is_empty(review.value)
                    and not (
                    field_name == "machine_learning_task"
                    and isinstance(review.value, list)
            )
            ):

                raise HTTPException(
                    status_code=422,
                    detail=(
                        f"O campo '{field_name}' não pode "
                        "ser confirmado com um valor vazio."
                    ),
                )

            value_to_validate = review.value

        # ----------------------------------------------------
        # REJEITAR VALOR
        # ----------------------------------------------------

        elif review.action == "reject":

            field.value = None

            field.status = "rejected"

            continue

        else:

            raise HTTPException(
                status_code=422,
                detail=(
                    f"Ação de revisão inválida: "
                    f"{review.action}"
                ),
            )

        # ====================================================
        # 4. VALIDAÇÃO GERAL
        # ====================================================

        general_validation_value = (
            _creator_values_for_general_validation(
                value_to_validate
            )
            if field_name == "creators"
            else value_to_validate
        )

        validation_errors = validate_metadata_field(
            field_name,
            general_validation_value,
        )

        if validation_errors:

            raise HTTPException(
                status_code=422,
                detail={
                    "message": (
                        f"O valor introduzido no campo "
                        f"'{field_name}' não é válido."
                    ),
                    "field": field_name,
                    "errors": validation_errors,
                },
            )

        # ====================================================
        # 5. VALIDAÇÃO ESPECÍFICA DOS CRIADORES
        # ====================================================

        if field_name == "creators":

            value_to_validate = (
                _validate_creator_identities(
                    value_to_validate
                )
            )
            value_to_validate = await verify_creator_identities(
                value_to_validate
            )

        # ====================================================
        # 6. CONFIRMAR CAMPO
        # ====================================================

        field.value = value_to_validate

        field.status = "confirmed"

    # ========================================================
    # 7. IDENTIFICAR CAMPOS PENDENTES E REJEITADOS
    # ========================================================

    pending_fields = []

    rejected_fields = []

    for field_name in REQUIRED_FIELDS:

        field = getattr(
            metadata,
            field_name,
        )

        if field.status == "rejected":

            rejected_fields.append(
                field_name
            )

        if (
                field.status != "confirmed"
                or is_empty(field.value)
        ):

            pending_fields.append(
                field_name
            )

    metadata.missing_information = (
        pending_fields.copy()
    )

    can_generate = (
            len(pending_fields) == 0
    )

    # ========================================================
    # 8. RESPOSTA
    # ========================================================

    return ConfirmedMetadataResponse(
        metadata=metadata,
        confirmed=can_generate,
        pending_fields=pending_fields,
        rejected_fields=rejected_fields,
        can_generate_croissant=can_generate,
    )