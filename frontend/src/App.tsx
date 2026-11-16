import { useEffect, useState, type FormEvent } from "react";
import "./App.css";

const API_URL = "http://localhost:8000";

type ReviewStatus = "suggested" | "confirmed" | "rejected";
type ReviewAction = "accept" | "edit" | "reject";

type CreatorIdentity = {
  creator_type: "person" | "organization";
  name: string;
  orcid_id?: string | null;
  ror_id?: string | null;
};

type RorValidation = {
  valid: boolean;
  exists: boolean;
  ror_id: string;
  ror_url: string;
  name: string;
};

type OrcidValidation = {
  valid: boolean;
  exists: boolean;
  orcid_id: string;
  orcid_url: string;
  name: string;
};

type MetadataField = {
  suggested_value: unknown;
  value: unknown;
  source: string | null;
  evidence: string | null;
  confidence: number;
  status: ReviewStatus;
};

type MetadataFieldName =
    | "title"
    | "description"
    | "creators"
    | "license"
    | "date_published"
    | "keywords"
    | "dataset_type"
    | "machine_learning_task"
    | "data_modality"
    | "language"
    | "number_of_records";

type Metadata = {
  title: MetadataField;
  description: MetadataField;
  creators: MetadataField;
  license: MetadataField;
  date_published: MetadataField;
  keywords: MetadataField;
  dataset_type: MetadataField;
  machine_learning_task: MetadataField;
  data_modality: MetadataField;
  language: MetadataField;
  number_of_records: MetadataField;
  splits: Record<string, unknown>;
  additional_information: MetadataField[];
  missing_information: string[];
};

type DatasetCandidate = {
  path: string;
  size: number;
  extension: string;
  sha?: string;
  confidence: string;
};

type Repository = {
  full_name: string;
  url: string;
  description: string | null;
  default_branch: string;
  selected_ref: string;
  commit_sha: string;
  license: string | null;
};

type Inventory = {
  total_files: number;
  metadata_files_found: number;
  metadata_files_read: number;
  dataset_candidates_found: number;
  dataset_candidates_returned: number;
};

type AnalysisResult = {
  repository: Repository;
  inventory: Inventory;
  dataset_candidates: DatasetCandidate[];
  warnings: string[];
};

type DatasetInspection = {
  path: string;
  size: number;
  format: string;
  encoding: string;
  delimiter: string;
  number_of_records: number;
  number_of_columns: number;
  columns: {
    name: string;
    inferred_type: string;
    missing_values: number;
  }[];
  sample_rows: Record<string, string>[];
};

type ExtractionResult = {
  repository: Repository;
  inventory: Inventory;
  selected_dataset: DatasetInspection;
  metadata: Metadata;
  dataset_candidates: DatasetCandidate[];
  warnings: string[];
};

type ConfirmationResult = {
  metadata: Metadata;
  confirmed: boolean;
  pending_fields: string[];
  rejected_fields: string[];
  can_generate_croissant: boolean;
};

type CroissantResult = {
  filename: string;
  format: string;
  validated: boolean;
  repository: string;
  commit_sha: string;
  selected_dataset: string;
  metadata_verified?: boolean;
  croissant: Record<string, unknown>;
};

const REQUIRED_FIELDS: MetadataFieldName[] = [
  "title",
  "description",
  "creators",
  "license",
  "date_published",
  "dataset_type",
  "data_modality",
];

const MACHINE_LEARNING_TASKS = [
  { value: "Binary classification", label: "Classificação binária" },
  { value: "Multiclass classification", label: "Classificação multiclasse" },
  { value: "Text classification", label: "Classificação de texto" },
  { value: "Image classification", label: "Classificação de imagens" },
  { value: "Regression", label: "Regressão" },
  { value: "Clustering", label: "Agrupamento (clustering)" },
  { value: "Anomaly detection", label: "Deteção de anomalias" },
  { value: "Ranking", label: "Ranking" },
  { value: "Recommendation", label: "Recomendação" },
  { value: "Forecasting", label: "Previsão (forecasting)" },
  { value: "Object detection", label: "Deteção de objetos" },
] as const;

type MachineLearningTask = typeof MACHINE_LEARNING_TASKS[number]["value"];

function normalizeTaskSelection(value: unknown): MachineLearningTask[] {
  const values = Array.isArray(value) ? value : typeof value === "string" ? [value] : [];
  return MACHINE_LEARNING_TASKS
      .filter((task) => values.includes(task.value))
      .map((task) => task.value);
}

function displayTaskSelection(value: unknown): string {
  const selected = normalizeTaskSelection(value);
  if (!selected.length) return "Não identificada";
  return selected.map((item) =>
      MACHINE_LEARNING_TASKS.find((task) => task.value === item)?.label || item,
  ).join(", ");
}

const FIELD_LABELS: Record<MetadataFieldName, string> = {
  title: "Título",
  description: "Descrição",
  creators: "Criadores",
  license: "Licença",
  date_published: "Data de publicação",
  keywords: "Palavras-chave",
  dataset_type: "Tipo de dataset",
  machine_learning_task: "Tarefa de Machine Learning",
  data_modality: "Modalidade dos dados",
  language: "Idioma",
  number_of_records: "Número de registos",
};

const LANGUAGE_CODE_NAMES: Record<string, string> = {
  ar: "Arabic",
  bg: "Bulgarian",
  zh: "Chinese",
  hr: "Croatian",
  cs: "Czech",
  da: "Danish",
  nl: "Dutch",
  en: "English",
  et: "Estonian",
  fi: "Finnish",
  fr: "French",
  de: "German",
  el: "Greek",
  he: "Hebrew",
  hi: "Hindi",
  hu: "Hungarian",
  id: "Indonesian",
  it: "Italian",
  ja: "Japanese",
  ko: "Korean",
  no: "Norwegian",
  pl: "Polish",
  pt: "Portuguese",
  ro: "Romanian",
  ru: "Russian",
  sr: "Serbian",
  sk: "Slovak",
  sl: "Slovenian",
  es: "Spanish",
  sv: "Swedish",
  th: "Thai",
  tr: "Turkish",
  uk: "Ukrainian",
  vi: "Vietnamese",
};

function languageNameToCode(value: string): string {
  const normalized = value.trim();
  const normalizedLowerCase = normalized.toLowerCase();
  const matchingCode = Object.entries(LANGUAGE_CODE_NAMES).find(
      ([, name]) => name.toLowerCase() === normalizedLowerCase,
  )?.[0];

  return matchingCode || normalizedLowerCase;
}

function languageToSelectValue(value: unknown): string {
  const rawValue = Array.isArray(value) ? value[0] : value;

  return typeof rawValue === "string"
      ? languageNameToCode(rawValue)
      : "";
}

function isEmpty(value: unknown): boolean {
  if (value === null || value === undefined) {
    return true;
  }

  if (typeof value === "string") {
    return value.trim().length === 0;
  }

  if (Array.isArray(value)) {
    return value.length === 0;
  }

  if (typeof value === "object") {
    return Object.keys(value).length === 0;
  }

  return false;
}

function displayValue(value: unknown, fieldName?: MetadataFieldName): string {
  if (isEmpty(value)) {
    return "Não identificado";
  }

  if (fieldName === "machine_learning_task") {
    return displayTaskSelection(value);
  }

  if (Array.isArray(value)) {
    if (value.some((item) => item !== null && typeof item === "object")) {
      return JSON.stringify(value, null, 2);
    }
    return value.map((item) => String(item)).join(", ");
  }

  if (typeof value === "object") {
    return JSON.stringify(value, null, 2);
  }

  return String(value);
}

function fieldToInput(
    fieldName: MetadataFieldName,
    value: unknown,
): string {
  if (isEmpty(value)) {
    return "";
  }

  if (fieldName === "creators") {
    return JSON.stringify(value, null, 2);
  }

  if (fieldName === "keywords" && Array.isArray(value)) {
    return value.map((item) => String(item)).join("\n");
  }

  if (fieldName === "machine_learning_task") {
    return JSON.stringify(normalizeTaskSelection(value));
  }

  if (fieldName === "date_published" && typeof value === "string") {
    return value.slice(0, 10);
  }

  return typeof value === "string"
      ? value
      : JSON.stringify(value, null, 2);
}

function parseFieldInput(
    fieldName: MetadataFieldName,
    input: string,
): unknown {
  if (fieldName === "creators") {
    let parsed: unknown;

    try {
      parsed = JSON.parse(input);
    } catch {
      throw new Error(
          "Os criadores devem ser introduzidos como um array JSON de objetos.",
      );
    }

    if (!Array.isArray(parsed)) {
      throw new Error("Os criadores devem ser um array JSON.");
    }

    return parsed;
  }

  if (fieldName === "keywords") {
    return input
        .split("\n")
        .map((item) => item.trim())
        .filter(Boolean);
  }

  if (fieldName === "machine_learning_task") {
    if (!input.trim()) return [];
    try {
      return normalizeTaskSelection(JSON.parse(input));
    } catch {
      return normalizeTaskSelection(input);
    }
  }

  if (fieldName === "language") {
    return input.trim();
  }

  return input.trim();
}

function getLocalDate(): string {
  const now = new Date();
  const year = now.getFullYear();
  const month = String(now.getMonth() + 1).padStart(2, "0");
  const day = String(now.getDate()).padStart(2, "0");
  return `${year}-${month}-${day}`;
}

function getErrorMessage(data: any, fallback: string): string {
  if (typeof data?.detail === "string") {
    return data.detail;
  }

  if (data?.detail?.message) {
    const pending = data.detail.pending_fields;

    if (Array.isArray(pending) && pending.length > 0) {
      return `${data.detail.message} Campos pendentes: ${pending.join(", ")}.`;
    }

    return data.detail.message;
  }

  return fallback;
}

function App() {
  const [url, setUrl] = useState("");
  const [ref, setRef] = useState("");

  const [analysis, setAnalysis] = useState<AnalysisResult | null>(null);
  const [selectedDatasetPath, setSelectedDatasetPath] = useState("");

  const [extraction, setExtraction] = useState<ExtractionResult | null>(null);

  const [editingValues, setEditingValues] =
      useState<Partial<Record<MetadataFieldName, string>>>({});

  // Identidades estruturadas dos criadores; nomes vêm dos registos públicos.
  const [creatorDraft, setCreatorDraft] = useState<CreatorIdentity[]>([]);
  const [orcidInput, setOrcidInput] = useState("");
  const [orcidBusy, setOrcidBusy] = useState(false);
  const [rorInput, setRorInput] = useState("");
  const [rorBusy, setRorBusy] = useState(false);
  const [creatorMessage, setCreatorMessage] = useState("");
  const [creatorError, setCreatorError] = useState("");

  const [confirmation, setConfirmation] =
      useState<ConfirmationResult | null>(null);

  const [croissant, setCroissant] = useState<CroissantResult | null>(null);

  const [error, setError] = useState("");
  const [successMessage, setSuccessMessage] = useState("");

  const [loading, setLoading] = useState(false);
  const [busyField, setBusyField] = useState<MetadataFieldName | null>(null);

  const [ollamaAvailable, setOllamaAvailable] = useState<boolean | null>(null);
  const [modelInstalled, setModelInstalled] = useState<boolean | null>(null);
  const [configuredModel, setConfiguredModel] = useState("Ollama");

  // ==========================================================
  // OLLAMA STATUS
  // ==========================================================

  useEffect(() => {
    fetch(`${API_URL}/api/v1/ollama-status`)
        .then(async (response) => {
          const data = await response.json();

          setOllamaAvailable(data.available);
          setModelInstalled(data.model_installed);

          if (data.configured_model) {
            setConfiguredModel(data.configured_model);
          }
        })
        .catch(() => {
          setOllamaAvailable(false);
          setModelInstalled(false);
        });
  }, []);

  // ==========================================================
  // ANALYZE REPOSITORY
  // ==========================================================

  async function analyze(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();

    setLoading(true);
    setError("");
    setSuccessMessage("");

    setAnalysis(null);
    setExtraction(null);
    setConfirmation(null);
    setCroissant(null);
    setSelectedDatasetPath("");
    setEditingValues({});
    setCreatorDraft([]);
    setOrcidInput("");
    setRorInput("");
    setCreatorMessage("");
    setCreatorError("");

    try {
      const response = await fetch(`${API_URL}/api/v1/analyze`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          url,
          ...(ref.trim() ? { ref: ref.trim() } : {}),
        }),
      });

      const data = await response.json().catch(() => ({}));

      if (!response.ok) {
        throw new Error(
            getErrorMessage(data, "Não foi possível analisar o repositório."),
        );
      }

      setAnalysis(data as AnalysisResult);

      if (data.dataset_candidates?.length > 0) {
        setSuccessMessage(
            `Repositório analisado. Foram encontrados ${data.dataset_candidates.length} candidatos a datasets.`,
        );
      } else {
        setSuccessMessage(
            "Repositório analisado, mas não foram encontrados datasets suportados.",
        );
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Erro inesperado.");
    } finally {
      setLoading(false);
    }
  }

  // ==========================================================
  // EXTRACT METADATA FROM SELECTED DATASET
  // ==========================================================

  async function extractDataset() {
    if (!selectedDatasetPath) {
      setError("Seleciona primeiro um dataset.");
      return;
    }

    if (!ollamaAvailable || !modelInstalled) {
      setError(
          "O Ollama ou o modelo configurado não estão disponíveis.",
      );
      return;
    }

    setLoading(true);
    setError("");
    setSuccessMessage("");

    setExtraction(null);
    setConfirmation(null);
    setCroissant(null);
    setEditingValues({});
    setCreatorDraft([]);
    setOrcidInput("");
    setRorInput("");
    setCreatorMessage("");
    setCreatorError("");

    try {
      const response = await fetch(`${API_URL}/api/v1/extract-dataset`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          url,
          ...(ref.trim() ? { ref: ref.trim() } : {}),
          dataset_path: selectedDatasetPath,
        }),
      });

      const data = await response.json().catch(() => ({}));

      if (!response.ok) {
        throw new Error(
            getErrorMessage(data, "Não foi possível extrair os metadados."),
        );
      }

      setExtraction(data as ExtractionResult);
      setSuccessMessage(
          "Dataset inspecionado. Revê agora as sugestões de metadados.",
      );
    } catch (err) {
      setError(err instanceof Error ? err.message : "Erro inesperado.");
    } finally {
      setLoading(false);
    }
  }

  // ==========================================================
  // CRIADORES: ORCID E ROR
  // ==========================================================

  async function addResearcherByOrcid() {
    const identifier = orcidInput.trim();
    if (!identifier) {
      setCreatorError("Introduz um ORCID iD ou URL ORCID.");
      return;
    }

    setOrcidBusy(true);
    setCreatorError("");
    setCreatorMessage("");

    try {
      const response = await fetch(
          `${API_URL}/api/v1/orcid/validate/${encodeURIComponent(identifier)}`,
      );
      const data = await response.json().catch(() => ({}));

      if (!response.ok) {
        throw new Error(getErrorMessage(data, "Não foi possível validar o ORCID iD."));
      }

      const researcher = data as OrcidValidation;
      if (!researcher.valid || !researcher.exists || !researcher.name) {
        throw new Error("O ORCID iD não corresponde a um investigador com nome público.");
      }

      setCreatorDraft((previous) => {
        if (previous.some((item) => item.creator_type === "person")) {
          return previous;
        }
        return [
          ...previous,
          {
            creator_type: "person",
            name: researcher.name,
            orcid_id: researcher.orcid_id,
          },
        ];
      });
      setOrcidInput("");
      setCreatorMessage(`Investigador obtido do registo ORCID: ${researcher.name}.`);
      setEditingValues((previous) => ({ ...previous, creators: "" }));
    } catch (err) {
      setCreatorError(err instanceof Error ? err.message : "Erro ao validar o investigador.");
    } finally {
      setOrcidBusy(false);
    }
  }

  async function addOrganizationByRor() {
    const identifier = rorInput.trim();
    if (!identifier) {
      setCreatorError("Introduz um ROR ID ou URL ROR.");
      return;
    }

    setRorBusy(true);
    setCreatorError("");
    setCreatorMessage("");

    try {
      const response = await fetch(
          `${API_URL}/api/v1/ror/validate/${encodeURIComponent(identifier)}`,
      );
      const data = await response.json().catch(() => ({}));

      if (!response.ok) {
        throw new Error(getErrorMessage(data, "Não foi possível validar o ROR ID."));
      }

      const organization = data as RorValidation;
      if (!organization.valid || !organization.exists || !organization.name) {
        throw new Error("O ROR ID não corresponde a uma organização válida.");
      }

      setCreatorDraft((previous) => {
        if (previous.some((item) => item.creator_type === "organization" && item.ror_id === organization.ror_id)) {
          return previous;
        }
        return [
          ...previous,
          {
            creator_type: "organization",
            name: organization.name,
            ror_id: organization.ror_id,
          },
        ];
      });
      setRorInput("");
      setCreatorMessage(`Organização adicionada a partir do registo ROR: ${organization.name}.`);
    } catch (err) {
      setCreatorError(err instanceof Error ? err.message : "Erro ao validar a organização.");
    } finally {
      setRorBusy(false);
    }
  }

  function removeCreator(index: number) {
    const removed = creatorDraft[index];
    setCreatorDraft((previous) => previous.filter((_, itemIndex) => itemIndex !== index));

    if (removed?.creator_type === "person") setOrcidInput("");
  }

  // ==========================================================
  // REVIEW METADATA FIELD
  // ==========================================================

  async function reviewField(
      fieldName: MetadataFieldName,
      action: ReviewAction,
  ) {
    if (!extraction) {
      return;
    }

    const field = extraction.metadata[fieldName];
    let editedValue: unknown = undefined;

    if (action === "edit") {
      const input =
          editingValues[fieldName] ??
          (fieldName === "language"
              ? languageToSelectValue(
                  !isEmpty(field.value) ? field.value : field.suggested_value,
              )
              : fieldToInput(
                  fieldName,
                  !isEmpty(field.value) ? field.value : field.suggested_value,
              ));

      try {
        editedValue = fieldName === "creators"
            ? creatorDraft
            : parseFieldInput(fieldName, input);
      } catch (parseError) {
        setError(
            parseError instanceof Error
                ? parseError.message
                : "O valor introduzido não tem um formato válido.",
        );
        return;
      }

      if (isEmpty(editedValue) && fieldName !== "machine_learning_task") {
        setError(
            `O campo "${FIELD_LABELS[fieldName]}" não pode ficar vazio.`,
        );
        return;
      }
    }

    setBusyField(fieldName);
    setError("");
    setSuccessMessage("");

    try {
      const review = {
        action,
        ...(action === "edit" ? { value: editedValue } : {}),
      };
      const suggestedLanguage = extraction.metadata.language.suggested_value;
      const metadataForReview =
          fieldName === "language" && action === "accept"
              ? {
                ...extraction.metadata,
                language: {
                  ...extraction.metadata.language,
                  suggested_value: Array.isArray(suggestedLanguage)
                      ? suggestedLanguage.map((language) =>
                        typeof language === "string"
                            ? languageNameToCode(language)
                            : language,
                      )
                      : typeof suggestedLanguage === "string"
                          ? languageNameToCode(suggestedLanguage)
                          : suggestedLanguage,
                },
              }
              : extraction.metadata;

      const response = await fetch(`${API_URL}/api/v1/metadata/confirm`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          metadata: metadataForReview,
          reviews: {
            [fieldName]: review,
          },
        }),
      });

      const data = await response.json().catch(() => ({}));

      if (!response.ok) {
        if (action === "edit" && response.status === 422) {
          setExtraction((previous) => {
            if (!previous) {
              return previous;
            }

            return {
              ...previous,
              metadata: {
                ...previous.metadata,
                [fieldName]: {
                  ...previous.metadata[fieldName],
                  value: editedValue,
                  status: "suggested",
                },
              },
            };
          });

          setConfirmation(null);
          setCroissant(null);
        }

        throw new Error(
            getErrorMessage(data, "Não foi possível guardar a revisão."),
        );
      }

      const result = data as ConfirmationResult;

      setExtraction((previous) => {
        if (!previous) {
          return previous;
        }

        return {
          ...previous,
          metadata: result.metadata,
        };
      });

      setConfirmation(result);

      setEditingValues((previous) => ({
        ...previous,
        [fieldName]: fieldName === "language"
            ? languageToSelectValue(result.metadata[fieldName].value)
            : fieldToInput(
                fieldName,
                result.metadata[fieldName].value,
            ),
      }));

      setCroissant(null);

      if (action === "accept") {
        setSuccessMessage(
            `Sugestão de "${FIELD_LABELS[fieldName]}" confirmada.`,
        );
      } else if (action === "edit") {
        setSuccessMessage(
            `Campo "${FIELD_LABELS[fieldName]}" editado e confirmado.`,
        );
      } else {
        setSuccessMessage(
            `Campo "${FIELD_LABELS[fieldName]}" rejeitado. Terás de o preencher ou resolver antes de gerar Croissant.`,
        );
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Erro inesperado.");
    } finally {
      setBusyField(null);
    }
  }

  // ==========================================================
  // REQUIRED FIELD STATUS
  // ==========================================================

  function getPendingFields(): MetadataFieldName[] {
    if (!extraction) {
      return [...REQUIRED_FIELDS];
    }

    return REQUIRED_FIELDS.filter((fieldName) => {
      const field = extraction.metadata[fieldName];

      return field.status !== "confirmed" || isEmpty(field.value);
    });
  }

  const pendingFields = getPendingFields();

  const canGenerateCroissant =
      extraction !== null &&
      pendingFields.length === 0 &&
      !loading &&
      busyField === null;

  // ==========================================================
  // GENERATE CROISSANT
  // ==========================================================

  async function generateCroissant() {
    if (!extraction || pendingFields.length > 0) {
      setError(
          "Confirma todos os campos obrigatórios antes de gerar Croissant.",
      );
      return;
    }

    setLoading(true);
    setError("");
    setSuccessMessage("");
    setCroissant(null);

    try {
      const response = await fetch(`${API_URL}/api/v1/croissant/generate`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          url,
          ...(ref.trim() ? { ref: ref.trim() } : {}),
          dataset_path: selectedDatasetPath,
          metadata: extraction.metadata,
        }),
      });

      const data = await response.json().catch(() => ({}));

      if (!response.ok) {
        throw new Error(
            getErrorMessage(data, "Não foi possível gerar o Croissant."),
        );
      }

      setCroissant(data as CroissantResult);
      setSuccessMessage("Croissant gerado e validado com sucesso!");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Erro inesperado.");
    } finally {
      setLoading(false);
    }
  }

  // ==========================================================
  // DOWNLOAD CROISSANT JSON-LD
  // ==========================================================

  function downloadCroissant() {
    if (!croissant) {
      return;
    }

    const json = JSON.stringify(croissant.croissant, null, 2);

    const blob = new Blob(
        [json],
        { type: "application/ld+json;charset=utf-8" },
    );

    const downloadUrl = URL.createObjectURL(blob);
    const link = document.createElement("a");

    link.href = downloadUrl;
    link.download = croissant.filename || "croissant.json";

    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);

    URL.revokeObjectURL(downloadUrl);
  }

  // ==========================================================
  // METADATA FIELD RENDERING — CARDS
  // ==========================================================

  function renderMetadataField(
      fieldName: MetadataFieldName,
      required = false,
  ) {
    if (!extraction) {
      return null;
    }

    const field = extraction.metadata[fieldName];

    if (!field) {
      return null;
    }

    const fieldBusy = busyField === fieldName;
    const hasSuggestion = !isEmpty(field.suggested_value);
    const hasCurrentValue = !isEmpty(field.value);
    const shownValue = field.status === "confirmed"
        ? field.value
        : hasCurrentValue
            ? field.value
            : field.suggested_value;

    const languageValue = languageToSelectValue(
        field.status === "confirmed"
            ? field.value
            : hasCurrentValue ? field.value : field.suggested_value,
    );

    const inputValue =
        editingValues[fieldName] ??
        (fieldName === "language"
            ? languageValue
            : fieldToInput(
                fieldName,
                field.status === "confirmed"
                    ? field.value
                    : hasCurrentValue ? field.value : field.suggested_value,
            ));

    const statusLabel =
        field.status === "confirmed"
            ? "Confirmado pelo utilizador"
            : field.status === "rejected"
                ? "Rejeitado"
                : "Por rever";

    const statusClass =
        field.status === "confirmed"
            ? "status-found"
            : field.status === "rejected"
                ? "status-missing"
                : "status-pending";

    const canAcceptSuggestion =
        fieldName !== "creators" && hasSuggestion && field.status !== "confirmed";

    return (
        <article
            className={`metadata-card ${field.status}`}
            key={fieldName}
        >
          <div className="metadata-card-top">
            <div>
              <p className="card-kicker">
                {required ? "CAMPO OBRIGATÓRIO" : "CAMPO OPCIONAL"}
              </p>

              <h3>
                {FIELD_LABELS[fieldName]}
                {required ? <span className="required-mark"> *</span> : ""}
              </h3>
            </div>

            <span className={`status-pill ${statusClass}`}>
            {statusLabel}
          </span>
          </div>

          <div className="suggestion-panel">
          <span className="value-label">
            {field.status === "confirmed"
                ? "Valor confirmado"
                : "Sugestão atual"}
          </span>

            <p className="field-value">
              {displayValue(shownValue, fieldName)}
            </p>
          </div>

          {field.source && (
              <div className="evidence-block">
                <span className="value-label">Fonte</span>
                <p className="field-source">{field.source}</p>
              </div>
          )}

          {field.evidence && (
              <details className="evidence-details">
                <summary>Consultar evidência</summary>
                <p>{field.evidence}</p>
              </details>
          )}

          {hasSuggestion && (
              <div className="confidence-row">
                <span>Confiança estimada pela IA</span>
                <strong>{(field.confidence * 100).toFixed(0)}%</strong>
              </div>
          )}

          <div className="editor-block">
            {fieldName === "creators" ? (
                <div className="creators-editor">
                    <label htmlFor="creator-orcid">Adicionar investigador através de ORCID iD</label>
                    <div className="ror-entry">
                      <input
                          id="creator-orcid"
                          type="text"
                          value={orcidInput}
                          onChange={(event) => setOrcidInput(event.target.value)}
                          placeholder="Ex.: 0000-0002-1825-0097 ou URL ORCID"
                          disabled={loading || orcidBusy || creatorDraft.some((item) => item.creator_type === "person")}
                      />
                      <button
                          type="button"
                          className="button-primary"
                          onClick={() => void addResearcherByOrcid()}
                          disabled={loading || orcidBusy || !orcidInput.trim() || creatorDraft.some((item) => item.creator_type === "person")}
                      >
                        {orcidBusy ? "A validar..." : "Validar e adicionar"}
                      </button>
                    </div>
                    <p className="editor-help">
                      O nome é obtido automaticamente do registo público ORCID. Esta consulta
                      não confirma que és titular desse ORCID iD.
                    </p>

                  <label htmlFor="creator-ror">Adicionar organização através de ROR ID</label>
                  <div className="ror-entry">
                    <input
                        id="creator-ror"
                        type="text"
                        value={rorInput}
                        onChange={(event) => setRorInput(event.target.value)}
                        placeholder="Ex.: 01c27hj86 ou https://ror.org/01c27hj86"
                        disabled={loading || rorBusy}
                    />
                    <button
                        type="button"
                        className="button-secondary"
                        onClick={() => void addOrganizationByRor()}
                        disabled={loading || rorBusy || !rorInput.trim()}
                    >
                      {rorBusy ? "A validar..." : "Validar e adicionar"}
                    </button>
                  </div>
                  <p className="editor-help">
                    A designação da organização é obtida automaticamente no registo ROR.
                    A existência de um registo ROR não comprova autorização para representar
                    essa organização.
                  </p>

                  {creatorError && <p className="creator-feedback error">{creatorError}</p>}
                  {creatorMessage && <p className="creator-feedback success">{creatorMessage}</p>}

                  {creatorDraft.length > 0 ? (
                      <ul className="creator-list">
                        {creatorDraft.map((creator, index) => (
                            <li key={`${creator.creator_type}-${creator.orcid_id || creator.ror_id}`}>
                              <div>
                                <strong>{creator.name}</strong>
                                <span>
                            {creator.creator_type === "person"
                                ? `Investigador · ORCID ${creator.orcid_id}`
                                : `Organização · ROR ${creator.ror_id}`}
                          </span>
                              </div>
                              <button
                                  type="button"
                                  className="button-quiet"
                                  onClick={() => removeCreator(index)}
                                  disabled={loading || fieldBusy}
                              >
                                Remover
                              </button>
                            </li>
                        ))}
                      </ul>
                  ) : (
                      <p className="creator-empty">Ainda não adicionaste investigadores ou organizações.</p>
                  )}
                </div>
            ) : fieldName === "machine_learning_task" ? (
                <div className="task-editor">
                  <span className="task-editor-title">Seleciona uma ou mais tarefas</span>
                  <p className="editor-help">
                    A IA sugere no máximo uma tarefa principal. Podes selecionar
                    outras tarefas se também forem adequadas ao dataset. Se não
                    selecionares nenhuma, confirma a lista vazia ou rejeita a sugestão.
                  </p>
                  <div className="task-checkbox-list">
                    {MACHINE_LEARNING_TASKS.map((task) => {
                      let selectedTasks: MachineLearningTask[] = [];
                      try {
                        selectedTasks = normalizeTaskSelection(
                            inputValue ? JSON.parse(inputValue) : [],
                        );
                      } catch {
                        selectedTasks = [];
                      }
                      return (
                          <label className="task-checkbox" key={task.value}>
                            <input
                                type="checkbox"
                                checked={selectedTasks.includes(task.value)}
                                disabled={fieldBusy || loading}
                                onChange={(event) => {
                                  const nextTasks = event.target.checked
                                      ? [...selectedTasks, task.value]
                                      : selectedTasks.filter((item) => item !== task.value);
                                  setEditingValues((previous) => ({
                                    ...previous,
                                    machine_learning_task: JSON.stringify(nextTasks),
                                  }));
                                }}
                            />
                            <span>{task.label}</span>
                          </label>
                      );
                    })}
                  </div>
                </div>
            ) : (
                <>
                  <label htmlFor={`editor-${fieldName}`}>
                    {fieldName === "keywords"
                        ? "Editar palavras-chave (uma por linha)"
                        : "Editar valor"}
                  </label>

                  {fieldName === "language" ? (
                      <select
                          id={`editor-${fieldName}`}
                          className="field-editor"
                          value={inputValue}
                          onChange={(event) =>
                              setEditingValues((previous) => ({
                                ...previous,
                                language: event.target.value,
                              }))
                          }
                          disabled={fieldBusy || loading}
                      >
                        <option value="">Seleciona um idioma...</option>
                        {languageValue &&
                            !Object.prototype.hasOwnProperty.call(
                                LANGUAGE_CODE_NAMES,
                                languageValue,
                            ) && (
                                <option value={languageValue}>{languageValue}</option>
                            )}
                        {Object.entries(LANGUAGE_CODE_NAMES).map(([code, language]) => (
                            <option key={code} value={code}>
                              {language}
                            </option>
                        ))}
                      </select>
                  ) : fieldName === "date_published" ? (
                      <input
                          id={`editor-${fieldName}`}
                          className="field-editor date-editor"
                          type="date"
                          max={getLocalDate()}
                          value={inputValue}
                          onChange={(event) =>
                              setEditingValues((previous) => ({
                                ...previous,
                                [fieldName]: event.target.value,
                              }))
                          }
                          disabled={fieldBusy || loading}
                      />
                  ) : (
                      <textarea
                          id={`editor-${fieldName}`}
                          className="field-editor"
                          rows={fieldName === "description" ? 5 : 3}
                          value={inputValue}
                          onChange={(event) =>
                              setEditingValues((previous) => ({
                                ...previous,
                                [fieldName]: event.target.value,
                              }))
                          }
                          placeholder={`Introduz ${FIELD_LABELS[fieldName].toLowerCase()}`}
                          disabled={fieldBusy || loading}
                      />
                  )}
                </>
            )}

          </div>

          <div className="field-actions">
            <button
                className="button-primary"
                type="button"
                onClick={() => void reviewField(fieldName, "accept")}
                disabled={fieldBusy || loading || !canAcceptSuggestion}
            >
              {fieldBusy ? "A guardar..." : "Aceitar sugestão"}
            </button>

            <button
                className="button-secondary"
                type="button"
                onClick={() => void reviewField(fieldName, "edit")}
                disabled={fieldBusy || loading}
            >
              Confirmar edição
            </button>

            <button
                className="button-quiet"
                type="button"
                onClick={() => void reviewField(fieldName, "reject")}
                disabled={fieldBusy || loading || field.status === "rejected"}
            >
              Rejeitar
            </button>
          </div>
        </article>
    );
  }

  // ==========================================================
  // USER INTERFACE
  // ==========================================================

  return (
      <main className="page">
        {/* HEADER */}

        <header className="hero">
          <p className="eyebrow">AI DATASET CURATION</p>

          <h1>Zenodo Metadata Assistant</h1>

          <p className="subtitle">
            Transforma repositórios GitHub em metadados estruturados para
            datasets de Machine Learning.
          </p>
        </header>

        {/* OLLAMA STATUS */}

        <section className="status-panel">
          <div>
            <span className="status-label">Ollama</span>

            <strong>
              {ollamaAvailable === null
                  ? "A verificar..."
                  : ollamaAvailable
                      ? "Disponível"
                      : "Indisponível"}
            </strong>
          </div>

          <div>
            <span className="status-label">{configuredModel}</span>

            <strong>
              {modelInstalled === null
                  ? "A verificar..."
                  : modelInstalled
                      ? "Instalado"
                      : "Não instalado"}
            </strong>
          </div>
        </section>

        {/* REPOSITORY FORM */}

        <section className="panel">
          <form onSubmit={analyze}>
            <label htmlFor="repo-url">
              URL do repositório GitHub
            </label>

            <input
                id="repo-url"
                type="url"
                placeholder="https://github.com/owner/repo"
                value={url}
                onChange={(event) => setUrl(event.target.value)}
                required
            />

            <label htmlFor="repo-ref">
              Branch, tag ou commit (opcional)
            </label>

            <input
                id="repo-ref"
                type="text"
                placeholder="Branch predefinida"
                value={ref}
                onChange={(event) => setRef(event.target.value)}
                maxLength={200}
            />

            <div className="form-footer">
              <p className="hint">
                Primeiro analisamos o repositório. Depois poderás escolher o
                dataset que pretendes utilizar.
              </p>

              <button type="submit" disabled={loading || !url.trim()}>
                {loading ? "A analisar..." : "Analisar repositório"}
              </button>
            </div>
          </form>
        </section>

        {/* FEEDBACK */}

        {error && (
            <section className="message error">
              <p>{error}</p>
            </section>
        )}

        {successMessage && (
            <section className="message success">
              <p>{successMessage}</p>
            </section>
        )}

        {/* REPOSITORY ANALYSIS */}

        {analysis && (
            <>
              <section className="panel">
                <p className="eyebrow">REPOSITORY ANALYSIS</p>

                <h2>Repositório analisado</h2>

                <div className="details">
                  <span>Repositório</span>
                  <strong>{analysis.repository.full_name}</strong>

                  <span>URL</span>
                  <strong>
                    <a href={analysis.repository.url} target="_blank" rel="noreferrer">
                      {analysis.repository.url}
                    </a>
                  </strong>

                  <span>Branch / referência</span>
                  <strong>{analysis.repository.selected_ref}</strong>

                  <span>Commit</span>
                  <strong className="hash">{analysis.repository.commit_sha}</strong>

                  <span>Licença do repositório</span>
                  <strong>{analysis.repository.license || "Não identificada"}</strong>
                </div>

                <div className="stats">
                  <div className="stat">
                    <strong>{analysis.inventory.total_files}</strong>
                    <span>Ficheiros analisados</span>
                  </div>

                  <div className="stat">
                    <strong>{analysis.inventory.metadata_files_read}</strong>
                    <span>Ficheiros de metadados lidos</span>
                  </div>

                  <div className="stat">
                    <strong>{analysis.inventory.dataset_candidates_found}</strong>
                    <span>Datasets candidatos</span>
                  </div>
                </div>

                {analysis.dataset_candidates.length > 0 ? (
                    <>
                      <h3>Selecionar dataset</h3>

                      <label htmlFor="dataset-select">
                        Dataset a inspecionar
                      </label>

                      <select
                          id="dataset-select"
                          value={selectedDatasetPath}
                          onChange={(event) => {
                            setSelectedDatasetPath(event.target.value);
                            setExtraction(null);
                            setConfirmation(null);
                            setCroissant(null);
                          }}
                          disabled={loading}
                      >
                        <option value="">Seleciona um ficheiro...</option>

                        {analysis.dataset_candidates.map((file) => (
                            <option key={file.path} value={file.path}>
                              {file.path} — {file.extension} —{" "}
                              {file.size.toLocaleString()} bytes
                            </option>
                        ))}
                      </select>

                      {selectedDatasetPath && (
                          <div className="form-footer">
                            <p className="hint">
                              O ficheiro será inspecionado antes da extração.
                            </p>

                            <button
                                type="button"
                                onClick={() => void extractDataset()}
                                disabled={
                                    loading || !ollamaAvailable || !modelInstalled
                                }
                            >
                              {loading
                                  ? "A inspecionar e extrair..."
                                  : "Inspecionar e extrair metadados"}
                            </button>
                          </div>
                      )}

                      {(!ollamaAvailable || !modelInstalled) && (
                          <p className="hint">
                            É necessário ter o Ollama e o modelo configurado
                            disponíveis para extrair metadados.
                          </p>
                      )}
                    </>
                ) : (
                    <p className="hint">
                      Não foram encontrados ficheiros suportados para inspeção.
                    </p>
                )}
              </section>

              {/* ANALYSIS WARNINGS */}

              {analysis.warnings.length > 0 && (
                  <section className="message warning">
                    <h2>Avisos da análise</h2>

                    <ul>
                      {analysis.warnings.map((warning, index) => (
                          <li key={index}>{warning}</li>
                      ))}
                    </ul>
                  </section>
              )}
            </>
        )}

        {/* DATASET INSPECTION */}

        {extraction && (
            <>
              <section className="panel">
                <p className="eyebrow">DATASET INSPECTION</p>

                <h2>Inspeção do dataset</h2>

                <p>
                  <strong>Ficheiro:</strong>{" "}
                  {extraction.selected_dataset.path}
                </p>

                <div className="details">
                  <span>Formato</span>
                  <strong>{extraction.selected_dataset.format}</strong>

                  <span>Tamanho</span>
                  <strong>
                    {extraction.selected_dataset.size.toLocaleString()} bytes
                  </strong>

                  <span>Codificação</span>
                  <strong>{extraction.selected_dataset.encoding}</strong>

                  <span>Delimitador</span>
                  <strong>
                    {JSON.stringify(extraction.selected_dataset.delimiter)}
                  </strong>

                  <span>Número de registos</span>
                  <strong>
                    {extraction.selected_dataset.number_of_records}
                  </strong>

                  <span>Número de colunas</span>
                  <strong>
                    {extraction.selected_dataset.number_of_columns}
                  </strong>
                </div>

                <h3>Colunas identificadas</h3>

                <ul className="file-list">
                  {extraction.selected_dataset.columns.map((column) => (
                      <li key={column.name}>
                        <strong>{column.name}</strong>

                        <span className="field-status">
                    {column.inferred_type} · {column.missing_values} valores
                    em falta
                  </span>
                      </li>
                  ))}
                </ul>
              </section>

              {/* METADATA REVIEW */}

              <section className="panel">
                <p className="eyebrow">HUMAN REVIEW</p>

                <h2>Revisão dos metadados</h2>

                <p className="hint">
                  A IA apresenta sugestões, mas és tu quem decide quais os valores
                  que ficam confirmados. Podes aceitar, editar ou rejeitar cada
                  campo.
                </p>

                <p className="hint">
                  Os campos assinalados com * são obrigatórios para gerar o
                  Croissant nesta aplicação.
                </p>

                <div className="metadata-grid">
                  {renderMetadataField("title", true)}
                  {renderMetadataField("description", true)}
                  {renderMetadataField("creators", true)}
                  {renderMetadataField("license", true)}
                  {renderMetadataField("date_published", true)}
                  {renderMetadataField("dataset_type", true)}
                  {renderMetadataField("data_modality", true)}
                  {renderMetadataField("keywords")}
                  {renderMetadataField("machine_learning_task")}
                  {renderMetadataField("language")}
                  {renderMetadataField("number_of_records")}
                </div>
              </section>

              {/* CONFIRMATION STATUS */}

              <section className="panel missing-panel">
                <p className="eyebrow">CONFIRMATION STATUS</p>

                <h2>Estado da confirmação</h2>

                <p>
                  Campos obrigatórios confirmados:{" "}
                  <strong>
                    {REQUIRED_FIELDS.length - pendingFields.length}
                    {" / "}
                    {REQUIRED_FIELDS.length}
                  </strong>
                </p>

                {pendingFields.length === 0 ? (
                    <p className="status-found">
                      Todos os campos obrigatórios estão confirmados.
                    </p>
                ) : (
                    <>
                      <p>Ainda precisas de resolver os seguintes campos:</p>

                      <ul>
                        {pendingFields.map((field) => (
                            <li key={field}>
                              {FIELD_LABELS[field]}
                              {extraction.metadata[field].status === "rejected"
                                  ? " — rejeitado"
                                  : " — por confirmar"}
                            </li>
                        ))}
                      </ul>
                    </>
                )}

                {confirmation?.rejected_fields &&
                    confirmation.rejected_fields.length > 0 && (
                        <p>
                          Campos rejeitados:{" "}
                          {confirmation.rejected_fields
                              .map(
                                  (field) =>
                                      FIELD_LABELS[field as MetadataFieldName] || field,
                              )
                              .join(", ")}
                        </p>
                    )}
              </section>

              {/* CROISSANT GENERATION */}

              <section className="panel">
                <p className="eyebrow">CROISSANT GENERATION</p>

                <h2>Gerar metadados Croissant</h2>

                <p className="hint">
                  O documento só pode ser gerado depois de confirmares os sete
                  campos obrigatórios. O backend volta a inspecionar o dataset e
                  valida o JSON-LD com mlcroissant.
                </p>

                <button
                    type="button"
                    onClick={() => void generateCroissant()}
                    disabled={!canGenerateCroissant}
                >
                  {loading ? "A gerar Croissant..." : "Gerar Croissant"}
                </button>

                {!canGenerateCroissant && (
                    <p className="hint">
                      Confirma os sete campos obrigatórios para ativar a geração.
                    </p>
                )}
              </section>
            </>
        )}

        {/* CROISSANT RESULT */}

        {croissant && (
            <section className="panel">
              <p className="eyebrow">CROISSANT GENERATED</p>

              <h2>Croissant gerado com sucesso</h2>

              <p className="status-found">
                Documento validado com mlcroissant.
              </p>

              <div className="details">
                <span>Ficheiro</span>
                <strong>{croissant.filename}</strong>

                <span>Formato</span>
                <strong>{croissant.format}</strong>

                <span>Repositório</span>
                <strong>{croissant.repository}</strong>

                <span>Dataset</span>
                <strong>{croissant.selected_dataset}</strong>

                <span>Validação</span>
                <strong>
                  {croissant.validated ? "Aprovada" : "Não validado"}
                </strong>
              </div>

              <button
                  type="button"
                  onClick={downloadCroissant}
                  disabled={!croissant.validated}
              >
                Descarregar croissant.json
              </button>

              <h3>Pré-visualização JSON-LD</h3>

              <pre className="croissant-preview">
            {JSON.stringify(croissant.croissant, null, 2)}
          </pre>
            </section>
        )}
      </main>
  );
}

export default App;