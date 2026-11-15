from copy import deepcopy
from typing import Any

from fastapi import HTTPException
from pydantic import ValidationError

from app.metadata_schema import (
    ConfirmMetadataRequest,
    ConfirmedMetadataResponse,
    CreatorIdentity,
    MetadataField,
)

from app.orcid_service import (
    get_authenticated_creator_session,
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

REQUIRED_FIELDS = [
    "title",
    "description",
    "creators",
    "license",
    "date_published",
    "dataset_type",
    "data_modality",
]


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
# EXIGIR AUTENTICAÇÃO PARA TODAS AS PESSOAS
# ============================================================

def _require_authenticated_people(
        creators: list[CreatorIdentity],
        authenticated_creator: dict[str, str] | None,
) -> list[CreatorIdentity]:
    """
    Não aceita ORCID introduzido manualmente como prova de identidade.
    Esta versão suporta uma pessoa autenticada por sessão e várias organizações.
    """

    people = [
        creator for creator in creators
        if creator.creator_type == "person"
    ]

    if not people:
        return creators

    if not authenticated_creator:
        raise HTTPException(
            status_code=401,
            detail=(
                "Cada pessoa indicada como criador tem de autenticar a sua "
                "identidade através da ORCID. Um ORCID escrito manualmente "
                "não é suficiente."
            ),
        )

    trusted_orcid = authenticated_creator.get("orcid")
    if not trusted_orcid:
        trusted_orcid = authenticated_creator.get("orcid_id")

    if len(people) != 1 or people[0].orcid_id != trusted_orcid:
        raise HTTPException(
            status_code=422,
            detail=(
                "A sessão ORCID atual só pode confirmar a pessoa autenticada. "
                "Para esta versão, adiciona no máximo um investigador; podes "
                "adicionar várias organizações através de ROR."
            ),
        )

    # Substituir o nome e identificador pelos dados devolvidos pela sessão confiável.
    trusted_identity = CreatorIdentity.model_validate(authenticated_creator)
    for index, creator in enumerate(creators):
        if creator.creator_type == "person":
            creators[index] = trusted_identity

    return creators


# ============================================================
# INCLUIR O UTILIZADOR AUTENTICADO
# ============================================================

def _merge_authenticated_creator(
        value: Any,
        authenticated_creator: dict[str, str],
) -> list[CreatorIdentity]:
    """
    Inclui o utilizador autenticado uma única vez na lista de criadores.

    A identidade recebida do backend é a fonte de verdade.
    """

    if value is None:

        creators: list[Any] = []

    elif isinstance(value, list):

        creators = list(value)

    else:

        raise HTTPException(
            status_code=422,
            detail={
                "message": (
                    "Os criadores têm de ser apresentados numa lista."
                ),
                "field": "creators",
            },
        )

    trusted_creator = CreatorIdentity.model_validate(
        authenticated_creator
    )

    merged: list[CreatorIdentity] = []

    found_authenticated_creator = False

    for item in creators:

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

        except ValidationError:

            # Mantém o item para que a validação normal
            # produza o erro estruturado mais abaixo.
            merged.append(item)

            continue

        if creator.orcid_id == trusted_creator.orcid_id:

            if not found_authenticated_creator:

                merged.append(
                    trusted_creator
                )

                found_authenticated_creator = True

            # Remove duplicados com o mesmo ORCID autenticado.
            continue

        merged.append(creator)

    if not found_authenticated_creator:

        merged.insert(
            0,
            trusted_creator,
        )

    return merged


# ============================================================
# APLICAR REVISÕES
# ============================================================

async def apply_metadata_reviews(
        request: ConfirmMetadataRequest,
) -> ConfirmedMetadataResponse:

    metadata = deepcopy(
        request.metadata
    )

    authenticated_creator = None

    # ========================================================
    # 1. VALIDAR SESSÃO ORCID
    # ========================================================

    if request.orcid_session_token:

        authenticated_creator = (
            get_authenticated_creator_session(
                request.orcid_session_token
            )
        )

        if not authenticated_creator:

            raise HTTPException(
                status_code=401,
                detail=(
                    "A sessão ORCID é inválida ou expirou. "
                    "Autentica novamente com a ORCID."
                ),
            )

        # Mesmo antes da confirmação, a identidade autenticada
        # aparece como sugestão no campo creators.

        creators_field = metadata.creators

        creators_field.suggested_value = (
            _merge_authenticated_creator(
                creators_field.suggested_value,
                authenticated_creator,
            )
        )

        # Se o cliente já enviou creators como confirmed,
        # garantimos que a identidade autenticada está presente.

        if creators_field.status == "confirmed":

            creators_field.value = (
                _merge_authenticated_creator(
                    creators_field.value,
                    authenticated_creator,
                )
            )

    # ========================================================
    # 1A. VALIDAR CRIADORES JÁ CONFIRMADOS
    # ========================================================

    # Também verificamos ROR quando o pedido não inclui uma revisão
    # explícita de creators. Assim, não é possível contornar a consulta
    # ao registo oficial enviando o campo já marcado como confirmed.

    creators_field = metadata.creators

    if (
            creators_field.status == "confirmed"
            and "creators" not in request.reviews
    ):

        creators_to_verify = creators_field.value

        if authenticated_creator:

            creators_to_verify = _merge_authenticated_creator(
                creators_to_verify,
                authenticated_creator,
            )

        creators_to_verify = _validate_creator_identities(
            creators_to_verify
        )
        creators_to_verify = _require_authenticated_people(
            creators_to_verify,
            authenticated_creator,
        )

        creators_field.value = await _verify_ror_organizations(
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

            if is_empty(
                    review.value
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

            if (
                    field_name == "creators"
                    and authenticated_creator
            ):

                raise HTTPException(
                    status_code=422,
                    detail=(
                        "O criador autenticado através da ORCID "
                        "é incluído automaticamente. Não podes "
                        "rejeitar todos os criadores enquanto "
                        "esta sessão ORCID estiver associada."
                    ),
                )

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
        # 3. GARANTIR A INCLUSÃO DO UTILIZADOR ORCID
        # ====================================================

        if (
                field_name == "creators"
                and authenticated_creator
        ):

            value_to_validate = (
                _merge_authenticated_creator(
                    value_to_validate,
                    authenticated_creator,
                )
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
            value_to_validate = _require_authenticated_people(
                value_to_validate,
                authenticated_creator,
            )

            value_to_validate = (
                await _verify_ror_organizations(
                    value_to_validate
                )
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