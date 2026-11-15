
from pathlib import Path
from typing import Any

from app.ollama_service import extract_with_ollama
from app.metadata_schema import ExtractedMetadata, MetadataField


REQUIRED_FIELDS = [
    "title",
    "description",
    "creators",
    "license",
    "date_published",
    "dataset_type",
    "data_modality",
]

FIELD_NAMES = [
    "title",
    "description",
    "creators",
    "license",
    "date_published",
    "keywords",
    "dataset_type",
    "machine_learning_task",
    "data_modality",
    "language",
    "number_of_records",
]


def has_value(value: Any) -> bool:
    if value is None:
        return False

    if isinstance(value, str):
        return bool(value.strip())

    if isinstance(value, (list, dict)):
        return len(value) > 0

    return True


def normalize_field(field: Any) -> MetadataField:
    """Normaliza sugestões sem confirmar automaticamente os campos."""

    if field is None:
        return MetadataField()

    if not isinstance(field, MetadataField):
        field = MetadataField.model_validate(field)

    # Compatibilidade com respostas antigas que só continham value.
    if field.suggested_value is None and field.value is not None:
        field.suggested_value = field.value

    if field.value is None and field.suggested_value is not None:
        field.value = field.suggested_value

    effective_value = (
        field.suggested_value
        if field.suggested_value is not None
        else field.value
    )

    if not has_value(effective_value):
        field.suggested_value = None
        field.value = None
        field.source = None
        field.evidence = None
        field.confidence = 0

    field.status = "suggested"

    return field


def field_has_value(field: MetadataField | None) -> bool:
    if field is None:
        return False

    value = (
        field.suggested_value
        if field.suggested_value is not None
        else field.value
    )

    return has_value(value)


def set_deterministic_suggestion(
        field: MetadataField,
        value: Any,
        source: str,
        evidence: str,
        confidence: float,
) -> None:
    """Define uma sugestão baseada em evidência técnica do backend."""

    field.suggested_value = value
    field.value = value
    field.source = source
    field.evidence = evidence
    field.confidence = confidence
    field.status = "suggested"


def generate_title_from_filename(dataset_path: str) -> str | None:

    if not dataset_path:
        return None

    stem = Path(dataset_path).stem

    words = " ".join(
        stem.replace("_", " ").replace("-", " ").split()
    )

    if not words:
        return None

    title = words.title()

    if "dataset" not in title.lower():
        title = f"{title} dataset"

    return title


def apply_deterministic_metadata(
        metadata: ExtractedMetadata,
        dataset_inspection: dict | None,
) -> ExtractedMetadata:
    """Preenche características verificáveis do dataset selecionado."""

    if not dataset_inspection:
        return metadata

    dataset_path = str(dataset_inspection.get("path") or "")

    dataset_format = str(
        dataset_inspection.get("format")
        or Path(dataset_path).suffix
    ).lower()

    columns = dataset_inspection.get("columns") or []

    # 1. TITLE
    if not field_has_value(metadata.title):

        title = generate_title_from_filename(dataset_path)

        if title:
            set_deterministic_suggestion(
                metadata.title,
                title,
                "filename",
                dataset_path,
                0.8,
            )

    # 2. DATASET TYPE E DATA MODALITY
    if dataset_format in {".csv", ".tsv", "csv", "tsv"} and columns:

        evidence = (
            f"Ficheiro selecionado {dataset_path}; "
            f"formato {dataset_format}; "
            f"{len(columns)} colunas identificadas na inspeção."
        )

        set_deterministic_suggestion(
            metadata.dataset_type,
            "Tabular dataset",
            "selected_dataset_inspection",
            evidence,
            1.0,
        )

        set_deterministic_suggestion(
            metadata.data_modality,
            "Tabular",
            "selected_dataset_inspection",
            evidence,
            1.0,
        )

    # 3. NUMBER OF RECORDS
    number_of_records = dataset_inspection.get(
        "number_of_records"
    )

    if number_of_records is not None:

        set_deterministic_suggestion(
            metadata.number_of_records,
            number_of_records,
            "selected_dataset_inspection",
            (
                f"Contagem indicada pela inspeção de "
                f"{dataset_path}: {number_of_records}."
            ),
            1.0,
        )

    return metadata


async def extract_repository_metadata(
        repository_data: dict,
        dataset_inspection: dict | None = None,
) -> ExtractedMetadata:

    extraction_data = dict(repository_data)

    if dataset_inspection:
        extraction_data["dataset_inspection"] = dataset_inspection

    # 1. Extração através do Ollama.
    metadata = await extract_with_ollama(extraction_data)

    # 2. Normalização de todos os campos.
    for field_name in FIELD_NAMES:

        setattr(
            metadata,
            field_name,
            normalize_field(
                getattr(metadata, field_name, None)
            ),
        )

    metadata.additional_information = [
        normalize_field(item)
        for item in metadata.additional_information
    ]

    # 3. Complemento com informação técnica verificável.
    metadata = apply_deterministic_metadata(
        metadata,
        dataset_inspection,
    )

    # 4. Identificação dos campos obrigatórios em falta.
    missing = []

    for field_name in REQUIRED_FIELDS:

        field = getattr(metadata, field_name, None)

        if not field_has_value(field):
            missing.append(field_name)

    metadata.missing_information = sorted(missing)

    return metadata
