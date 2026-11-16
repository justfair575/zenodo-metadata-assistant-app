# Zenodo Metadata Assistant

Aplicação para analisar repositórios GitHub com datasets, rever metadados sugeridos e gerar uma descrição de dataset no formato Croissant (JSON-LD).

## Experiência da aplicação

O frontend apresenta o processo como três etapas: análise do repositório, seleção/inspeção do dataset com revisão humana dos metadados, e geração do ficheiro Croissant. O progresso é derivado do estado real da aplicação; não há um segundo estado de navegação para manter sincronizado.

Os indicadores de serviço mostram a disponibilidade do Ollama e do modelo configurado. A exportação só fica disponível depois de confirmar os campos obrigatórios, e o resultado validado pode ser descarregado como `croissant.json`.

## Estrutura

- `frontend/` — interface React, TypeScript e Vite.
- `backend/` — API e lógica de análise, extração, revisão e validação.

### Frontend

- `frontend/src/App.tsx` — fluxo da aplicação, tipos de dados e integração com a API.
- `frontend/src/App.css` — componentes e estilos da interface, incluindo cartões e etapas.
- `frontend/src/index.css` — base global, tipografia, foco e suporte a movimento reduzido.
- `frontend/index.html` — idioma, título e metadados do documento.

## Executar localmente

Instala as dependências do frontend e inicia o Vite:

```bash
cd frontend
npm install
npm run dev
```

O frontend espera a API em `http://localhost:8000`. Consulta a configuração do backend para os seus requisitos e para iniciar o serviço. A extração de metadados também requer Ollama e o modelo indicado pela configuração do backend.

## Consulta de documentação e metadados

Na análise, o backend percorre a árvore Git do commit selecionado e classifica ficheiros pelo nome, extensão e localização. Procura README, citation files (CFF/BibTeX), DataCite, Data Package, Croissant, RO-Crate, CodeMeta, cartões e dicionários de datasets, licenças, autores e documentação em diretórios como docs/, metadata/ e datasets/. Também considera documentos estruturados comuns na raiz. Os candidatos são ordenados por relevância; documentação próxima do dataset selecionado ganha prioridade na extração. Os ficheiros são descarregados por URL fixado ao SHA do commit, para que a análise continue a referir-se à mesma versão e não consuma uma chamada da API por documento.

O limite é de **80 ficheiros**, **200 KB por ficheiro** e **4 MB no total**. Ficheiros acima dos limites recebem um estado de ignorado, e a resposta inclui avisos quando há truncagem da árvore, limite de leitura ou mais candidatos do que o máximo. O inventário distingue ficheiros encontrados de ficheiros lidos com sucesso. A leitura aceita UTF-8, UTF-16 com BOM e Windows-1252, e assinala conteúdo binário ou impossível de descodificar. Isto cobre muitos repositórios públicos, mas não pode incluir ficheiros fora da árvore visível à API, ficheiros acima dos limites ou conteúdos que o GitHub não disponibiliza.

### Como as sugestões são produzidas

Todos os ficheiros lidos são percorridos pelo extrator. CFF/YAML, JSON/JSON-LD, TOML, XML (incluindo DataCite), INI/CFG e BibTeX são analisados para encontrar título, descrição, criadores, licença, data, palavras-chave, idioma, tipo/modalidade e contagem de registos quando esses dados estão presentes. README e texto documental também são enviados à LLM. O prompt apresenta uma amostra de cada ficheiro lido dentro de um orçamento equilibrado de contexto, além dos factos estruturados extraídos de todas as fontes; um README longo não ocupa sozinho o espaço destinado aos restantes ficheiros.

A LLM recebe instruções para preencher cada campo com evidência, incluindo campos opcionais, e deixar vazio o que não consegue justificar. Valores estruturados com fonte explícita são aplicados também por uma etapa determinística, que conserva o caminho do ficheiro como origem. A confirmação continua a pertencer à pessoa que revê os resultados. O modelo não deve transportar automaticamente licença, autor ou data do repositório para o dataset. Quando ficheiros discordam, a sugestão determinística usa o primeiro valor válido segundo a classificação de relevância; confirma a fonte e a evidência antes de aceitar.

A confiança não é um limiar que esconda sugestões: serve para comunicar a força da evidência. A LLM pode agora apresentar uma recomendação exploratória com confiança baixa (por exemplo, 0,35–0,54) quando há pelo menos um indício plausível, e deve explicar a incerteza. Evidência explícita ou observada recebe confiança maior. Criadores, licença e data de publicação continuam a exigir prova documental; baixar a confiança não autoriza inventar esses factos. Se não houver qualquer sinal relevante, o campo fica vazio.

Se os ficheiros não indicarem um título ou uma descrição específicos do dataset, a LLM pode redigi-los com base no nome do ficheiro, documentação específica, colunas e amostras. A descrição determinística do primeiro parágrafo do README só é usada quando não há um dataset selecionado, porque esse texto normalmente descreve o repositório. Para o idioma, a prioridade é um campo explícito de metadados, texto representativo do dataset, documentação que indique o idioma e, por fim, uma pista inequívoca no nome do ficheiro. O idioma em que um README ou metadado foi escrito não é, sozinho, prova do idioma dos dados.

Quando não há sinal mais forte, a língua usada em metadados que descrevem especificamente o dataset pode sustentar uma hipótese de baixa confiança, identificada como inferência da documentação.

As palavras-chave também podem ser recomendações inferidas pela LLM: não precisam de aparecer literalmente nos ficheiros, mas têm de fazer sentido à luz do título, descrição, colunas ou amostras. A evidência deve explicar essa ligação e a confiança deve baixar quando for indireta; termos genéricos ou sem suporte continuam excluídos.

Os conteúdos do repositório são dados não confiáveis, nunca instruções para o modelo. A inspeção de datasets suporta CSV/TSV, JSON (listas e objetos de registos), JSONL/NDJSON e texto simples. A geração Croissant continua limitada a CSV/TSV; Parquet, Excel, arquivos comprimidos e multimédia ainda não são oferecidos para inspeção.

## Convenções da interface

- Mantém texto e mensagens em português de Portugal (`pt-PT`).
- Deriva indicadores de progresso a partir de `analysis`, `extraction` e `croissant`; evita duplicar estado derivável.
- Usa variáveis CSS em `App.css` para cores partilhadas e preserva as classes semânticas dos estados de revisão.
- Conserva foco visível por teclado, contraste legível, reflow em ecrãs estreitos e respeito por `prefers-reduced-motion`.
- Mantém estilos globais em `index.css` e estilos de componentes/fluxo em `App.css`.

## Verificações

Na pasta `frontend/`, executa `npm run build` para verificar os tipos TypeScript e criar a versão de produção, ou `npm run lint` para validar as regras ESLint.
