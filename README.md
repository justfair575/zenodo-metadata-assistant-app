# Zenodo Metadata Assistant

Aplicação web para analisar repositórios GitHub com datasets, rever metadados sugeridos, gerar uma descrição Croissant (JSON-LD) validada e publicá-la como depósito no Zenodo.

## Funcionalidades

- Analisa repositórios e referências GitHub, inventariando ficheiros e localizando datasets candidatos.
- Inspeciona CSV/TSV, JSON, JSONL/NDJSON e texto simples; extrai sugestões de metadados com Ollama e revisão humana.
- Valida identificadores ORCID/ROR e os campos necessários antes de gerar Croissant com `mlcroissant`.
- Permite descarregar `croissant.json` e publicar o ficheiro num depósito Zenodo, incluindo Sandbox.
- O envio cria o depósito, define título/descrição a partir dos metadados, carrega o JSON-LD e publica-o. A publicação é imediata e pode receber DOI.

## Tecnologias

- Frontend: React, TypeScript, Vite e CSS.
- Backend: Python, FastAPI, Pydantic e HTTPX.
- Integrações: GitHub API, Zenodo Deposit API, Ollama, ORCID, ROR e `mlcroissant`.
- Dependências Python em `backend/requirements.txt`; dependências JavaScript em `frontend/package.json`.

## Publicar no Zenodo

Depois de gerar um Croissant válido, basta carregar no botão **Publicar no Zenodo**. Cria um personal access token nas definições da conta Sandbox, com permissões `deposit:write` e `deposit:actions`, e guarda-o como `ZENODO_ACCESS_TOKEN` no `backend/.env`. Define `ZENODO_BASE_URL=https://sandbox.zenodo.org`. O backend usa o token como Bearer; não é necessário `client_id`/`client_secret` nem chamar `/oauth/token`. Reinicia o backend depois de alterar o `.env`. Nunca coloques o token no frontend nem o partilhes. Para produção, usa um token de produção e define `ZENODO_BASE_URL=https://zenodo.org`.

Cada envio cria um depósito novo e inclui `croissant.json`; o dataset original do GitHub não é carregado para o Zenodo.

## Estrutura

- `frontend/` — interface React e TypeScript.
- `backend/` — API e lógica de análise, extração, revisão e validação.
- `frontend/src/App.tsx` — fluxo da aplicação e integração com a API.
- `backend/app/main.py` — endpoints FastAPI, incluindo publicação Zenodo.

## Executar localmente

### Backend

```bash
cd backend
python -m venv .venv
# Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --reload
```

A API fica em `http://localhost:8000`. A extração de metadados requer Ollama e o modelo definido na configuração do backend.

### Frontend

```bash
cd frontend
npm install
npm run dev
```

O frontend espera a API em `http://localhost:8000`.

## Limitações

A inspeção suporta CSV/TSV, JSON, JSONL/NDJSON e texto simples. A geração Croissant está limitada a CSV/TSV e datasets até 1 MB nesta versão. A publicação envia apenas o ficheiro Croissant, não os dados. Para desenvolvimento, executa `npm run build` e `npm run lint` na pasta `frontend`.
