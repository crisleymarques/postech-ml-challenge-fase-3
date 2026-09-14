# Postech ML Challenge - Fase 3

## Triagem Automática de Urgência em Laudos Médicos (NLP)

Projeto de pós-graduação (FIAP/POSTECH) que implementa um sistema de classificação
de texto (NLP), destinado à triagem automática de resumos/laudos médicos
em três níveis de urgência: **Normal**, **Atenção** e **Urgente**.
Inclui um pipeline de treinamento, análise exploratória e **API FastAPI para inferência em produção**.

Dataset: **Medical Abstracts TC Corpus Dataset** (14.438 amostras).


![CI Status](https://github.com/crisleymarques/postech-ml-challenge-fase-3/actions/workflows/ci.yml/badge.svg?branch=classificacao_laudos_fastapi)


---

## Estrutura de Pastas

```
postech-ml-challenge-fase-3/
├── data/
│   └── raw/                      # Dataset original (3 arquivos CSV)
│   └── processed/              # Dados processados e splits (gitignored)
├── docs/
│   ├── DATASET.md                # Documentação detalhada do dataset e mapeamentos
│   ├── BASELINE_LATENCY.md      # Baseline oficial de latência da API
│   └── AIRFLOW.md              # 🆕 Guia completo de orquestração Airflow
├── models/                       # Modelos gerados pelo pipeline (gitignored, .gitkeep)
├── notebooks/
│   ├── EDA_Medical_Abstracts.ipynb # EDA completo (notebook)
│   └── EDA_Medical_Abstracts.py # EDA em script Python
├── sources/                      # Módulos Python reutilizáveis
│   ├── __init__.py
│   ├── config.py                 # Configurações globais e mapeamento de classes
│   ├── data_loader.py        # Carregamento, validação e mapeamento de urgência
│   ├── preprocessing.py      # Limpeza e pré-processamento de texto
│   └── model.py              # Treinamento, avaliação, predição e benchmark
├── pipelines/
│   ├── __init__.py
│   └── feature_pipeline.py       # TF-IDF, splits de dados, salvamento/carregamento
├── app/                            # 🆕 Aplicação FastAPI (inferência em produção)
│   ├── __init__.py
│   ├── main.py                 # Aplicação FastAPI + rotas
│   ├── config.py               # Configurações da API (pydantic-settings)
│   ├── schemas.py              # Schemas de entrada, respostas e erros (Pydantic v2)
│   ├── exceptions.py           # Handlers de erros formatados (sem stack trace!)
│   └── services/
│       ├── __init__.py
│       └── model_service.py    # Singleton: carregamento e inferência do modelo
├── tests/                        # Testes unitários e integrados (pytest)
│   ├── conftest.py               # Configuração compartilhada dos testes
│   ├── test_data_loader.py
│   ├── test_preprocessing.py
│   ├── test_feature_pipeline.py
│   ├── test_model.py
│   └── test_api.py             # 🆕 28 testes unitários + integrados da API
├── .github/
│   └── workflows/
│       └── ci.yml             # GitHub Actions CI: lint, testes unitários, smoke tests, smoke API
├── airflow/                      # 🆕 Orquestração de treino/retreino com Apache Airflow
│   ├── dags/
│   │   └── train_urgencia_medica_dag.py   # DAG oficial de treinamento
│   ├── plugins/
│   │   └── airflow_ml_tasks.py            # Wrappers Python (reutiliza 100% sources/)
│   ├── airflow-home/            # Configurações de runtime (gitignored)
│   ├── airflow-logs/             # Logs das tasks (gitignored)
│   ├── include/
│   └── scripts/
├── benchmarks/                     # Baseline de latência + benchmark reproduzível
│   ├── __init__.py
│   ├── baseline_latency.py    # Script de benchmark (CLI)
│   ├── payloads.json       # 7 casos reais de laudos
│   └── results/            # Resultados JSON salvos
├── run_pipeline.py               # Script orquestrador - executa todo o pipeline ML
├── run_api.py                    # Script rápido para subir a API (uvicorn reload)
├── Dockerfile                    # Container produção (Gunicorn + UvicornWorker, non-root)
├── Dockerfile.dev                # Container dev (uvicorn --reload)
├── docker-compose.yml            # Serviços: api / api-dev / benchmark
├── docker-compose-airflow.yml    # 🆕 Serviços Airflow + PostgreSQL 16
├── .airflow.env                  # 🆕 UID e credenciais default Airflow
├── .dockerignore                 # Otimizado (exclui venv, data, models, .git)
├── requirements.txt              # Dependências do projeto (inclui FastAPI)
└── README.md
```

---

## Artefatos Gerados (não versionados)

Ao executar o pipeline, os seguintes artefatos são gerados localmente:

| Artefato | Descrição |
|----------|-----------|
| `models/urgency_classifier.joblib` | Pipeline completo (TF-IDF + classificador) |
| `models/tfidf_vectorizer.joblib` | Vectorizer TF-IDF treinado |
| `data/processed/*.csv` | Dados processados e splits de treino/validação/teste |
| `docs/model_metrics.json` | Métricas do modelo treinado |
| `notebooks/eda_outputs/*.png` | Gráficos do EDA |

---

## EDA

A análise exploratória completa está em:
👉 **[notebooks/EDA_Medical_Abstracts.ipynb](notebooks/EDA_Medical_Abstracts.ipynb)**

O arquivo pode ser executado como:
- **Notebook Jupyter**: Abra diretamente no Jupyter ou em ferramentas compatíveis como VSCode.

O EDA cobre **todos os itens do checklist**:
1. ✅ Origem e documentação do dataset
2. ✅ Documentação completa no próprio notebook
3. ✅ Validação das colunas de texto e target
4. ✅ Quantidade de amostras (14.438 - acima do mínimo de 2.000)
5. ✅ Carregamento dos dados (`sources/data_loader.py`)
6. ✅ Limpeza / pré-processamento (`sources/preprocessing.py`)
7. ✅ Divisão treino / validação / teste (70/10/20 estratificado)
8. ✅ Pipeline de features TF-IDF (10k features, unigramas + bigramas)
9. ✅ Classificadores base Scikit-Learn (LogReg, NB, LinearSVC, RF)
10. ✅ Avaliação com métricas (acc, precision, recall, F1, ROC-AUC, matriz confusão)
11. ✅ Modelo salvo em formato reutilizável (`.joblib`)
12. ✅ Documentação de classes e mapeamento do target (5 originais → 3 urgência)
13. ✅ Testes unitários das etapas principais

---

## Mapeamento de Urgência Aplicado

| Urgência   | Label | Condições Médicas Originais Mapeadas                    |
|------------|:-----:|---------------------------------------------------------|
| Normal     |   0   | Digestive system diseases                               |
| Atenção    |   1   | Nervous system diseases + General pathological conditions |
| Urgente    |   2   | Neoplasms (câncer) + Cardiovascular diseases            |

Justificativas completas em [docs/DATASET.md](docs/DATASET.md).

---

## 🏛️ Decisão Arquitetural — Estratégia de Deploy em Nuvem

### Por que a arquitetura importa neste projeto?

O sistema classifica laudos médicos em três níveis de urgência (**Normal**, **Atenção**, **Urgente**) para auxiliar a triagem em ambiente hospitalar. A escolha da estratégia de inferência tem impacto direto na segurança do paciente: um paciente classificado como "Urgente" que aguarda um processamento em lote pode ter seu atendimento atrasado com consequências graves.

---

### Batch vs. Real-time

| Critério | Batch (Processamento em Lote) | Real-time (Inferência Online) ✅ |
|:---------|:------------------------------|:---------------------------------|
| **Latência** | Alta — processamento periódico (minutos/horas) | Baixíssima — resposta imediata (~3–5 ms) |
| **Adequação ao domínio** | ❌ Inaceitável para triagem de urgência | ✅ Ideal: decisão clínica não pode esperar |
| **Fluxo de dados** | Acumula laudos e processa em bloco | Processa cada laudo assim que chega |
| **Custo por requisição** | Menor (amortizado no lote) | Ligeiramente maior, mas desprezível para o volume hospitalar |
| **Complexidade operacional** | Requer orquestração de jobs (Airflow, etc.) | Simples: container sempre ativo via load balancer |
| **Disponibilidade** | Depende do agendamento do job | 24/7, sem janelas de espera |
| **Modelo utilizado** | Qualquer — treinamento offline funciona bem | TF-IDF + Regressão Logística → P99 < 5 ms ✅ |

Sendo assim, a estratégia **Real-time (Online Inference)** é a única aceitável para triagem hospitalar. Atrasos oriundos de processamento em batch podem representar risco de vida para pacientes em estado urgente. A latência medida de **P99 ≈ 4,97 ms** comprova que o modelo escolhido é leve suficiente para suportar inferência síncrona em tempo real, mesmo sob carga.

---

### Recomendação de Plataforma Cloud

Como a API já está empacotada em um container Docker (com `Dockerfile` de produção seguindo boas práticas: Gunicorn + UvicornWorker, usuário não-root), a escolha natural são **serviços serverless de containers**, que eliminam a necessidade de gerenciar infraestrutura de servidores (sem VMs, sem clusters Kubernetes gerenciados manualmente) e escalam automaticamente conforme a demanda.

| Plataforma | Serviço Recomendado | Destaque |
|:-----------|:--------------------|:---------|
| **AWS** | **ECS Fargate** | Integração nativa com ECR (registro de imagens), ALB (balanceamento), CloudWatch (observabilidade) e IAM (segurança). Ideal para quem já usa o ecossistema AWS. |
| **GCP** | **Cloud Run** | Serviço mais simples de configurar para containers stateless. Escala até zero quando ocioso (custo zero em idle). Excelente para protótipos e MVPs hospitalares. |
| **Azure** | **Azure Container Apps** | Integrado ao ecossistema Microsoft/HIPAA-compliance. Boa escolha para hospitais que já usam Azure Active Directory. |

#### Arquitetura de Referência (AWS ECS Fargate)

```
[Sistema Hospitalar / Frontend Web]
          │
          ▼ HTTPS (REST)
[Application Load Balancer — ALB]
          │
          ▼
[ECS Fargate — Task Definition]
  ┌───────────────────────────┐
  │  Container: urgencia-api  │
  │  Imagem: ECR repository   │
  │  Gunicorn + Uvicorn       │
  │  Port 8000                │
  └───────────────────────────┘
          │
          ▼
[S3 / EFS — modelos .joblib]   [CloudWatch — logs e métricas]
```

**Justificativas da escolha arquitetural:**

1. **Stateless por design**: A API não mantém estado de sessão — cada requisição é independente. Isso torna o escalonamento horizontal trivial (ECS Fargate adiciona novas tasks automaticamente).
2. **Modelo leve (< 50 MB)**: TF-IDF + Regressão Logística cabe facilmente em memória do container sem necessidade de GPU, mantendo custo de instância baixo.
3. **Auto Scaling**: Configurar `ECS Service Auto Scaling` com métricas de CPU/RPS garante que picos de demanda (chegada de múltiplos pacientes) sejam absorvidos sem degradação.
4. **Health check nativo**: O endpoint `/health` da API é diretamente compatível com os health checks do ALB/ECS, permitindo substituição automática de tasks não saudáveis.
5. **Segurança**: O container roda com usuário não-root, e o ALB pode ser configurado com certificado TLS (HTTPS), atendendo aos requisitos de segurança para dados médicos (LGPD / HIPAA).


## 🚀 API de Inferência (FastAPI)

A API de classificação de laudos médicos recebe um texto de laudo e retorna sua classificação de urgência, carregando diretamente o artefato `.joblib` gerado pelo pipeline de treinamento. Ideal para deploy em containers Docker e integração com sistemas hospitalares ou frontend Web.

### Como Iniciar a API

**1. Instalar dependências** (uma vez):
```bash
pip install -r requirements.txt
```

**2. Executar o servidor**:
```bash
# Opção 1 - script rápido (com hot reload)
python run_api.py

# Opção 2 - uvicorn diretamente
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

**3. Acessar**:
- **Swagger UI (interativo)**: http://localhost:8000/docs — 👈 teste os endpoints direto no navegador!
- **Redoc**: documentação alternativa em http://localhost:8000/redoc
- **OpenAPI JSON**: http://localhost:8000/openapi.json

> 💡 **Importante**: a API carrega o modelo automaticamente na inicialização (lifespan). Se o arquivo `models/urgency_classifier.joblib` não existir, `/health` retornará `degraded/unhealthy` e `/predict` responderá 503 SERVICE_UNAVAILABLE (sem crashar). Basta rodar `python run_pipeline.py` para gerar o modelo.

### Variáveis de Ambiente Opcionais (`.env`)

| Nome | Default | Descrição |
|:-----|:--------|:----------|
| `APP_NAME` | *string longa* | Nome da aplicação |
| `DEBUG` | `False` | Modo debug (logs mais verboso) |
| `MODELS_DIR` | `<raiz do projeto>/models` | Pasta onde os arquivos `.joblib` são procurados |
| `MODEL_NAME` | `urgency_classifier` | Nome do arquivo `.joblib` (sem extensão) |
| `REQUEST_TEXT_MIN_LENGTH` | `10` | Tamanho mínimo aceito no campo `text` |
| `REQUEST_TEXT_MAX_LENGTH` | `50000` | Tamanho máximo aceito |
| `REQUEST_BATCH_MAX_ITEMS` | `100` | Limite de itens por requisição em lote |

---

### Endpoints

| Método | Rota | Descrição | Autenticação |
|:------:|:-----|:----------|:------------:|
| `GET`  | `/` | Informações gerais + links das rotas | ❌ |
| `GET`  | `/health` | Health check completo (modelo carregado?) | ❌ |
| `POST` | `/predict` | Classificar **1 laudo** | ❌ |
| `POST` | `/predict/batch` | Classificar **vários laudos** em lote | ❌ |

---

### 1. `GET /health`

Retorna o status de saúde da API.

**Exemplo de resposta (200 OK healthy)**:
```json
{
  "status": "healthy",
  "app_name": "API Classificação de Urgência de Laudos Médicos",
  "app_version": "1.0.0",
  "model_loaded": true,
  "model_path_exists": true,
  "model_name": "urgency_classifier",
  "model_metadata": {
    "model_name": "logistic_regression",
    "test_f1_macro": 0.6547
  },
  "timestamp": "2026-09-12T15:00:00Z"
}
```

**Possíveis status**: `healthy`, `degraded` (arquivo existe mas não carregou), `unhealthy` (sem arquivo de modelo → 503).

---

### 2. `POST /predict`

**Body JSON de entrada (`PredictRequest`)**:
```json
{
  "text": "Texto do laudo médico. Pode conter várias linhas, sintomas, diagnósticos.",
  "return_probabilities": true,
  "request_id": "paciente-123"
}
```

| Campo | Obrigatorio | Tipo | Descrição |
|:------|:-----------:|:----|:----------|
| `text` | ✅ | string (10-50.000 chars) | Texto do laudo médico |
| `return_probabilities` | ❌ | bool (default: true) | Retornar probabilidades por classe? |
| `request_id` | ❌ | string (até 64 chars) | Identificador opcional do cliente. Retornado no echo na resposta |

**Resposta 200 OK (`PredictResponse`)**:
```json
{
  "predicted_label": 2,
  "predicted_name": "urgente",
  "probabilities": {
    "normal": 0.018,
    "atencao": 0.027,
    "urgente": 0.954
  },
  "model_version": "logistic_regression",
  "processed_at": "2026-09-12T15:00:00Z",
  "request_id": "paciente-123"
}
```

---

### 3. `POST /predict/batch`

**Body JSON de entrada (`BatchPredictRequest`)**:
```json
{
  "items": [
    {"text": "Laudo 1: IAM com elevação ST", "return_probabilities": false, "request_id": "p1"},
    {"text": "Laudo 2: gastrite crônica", "return_probabilities": true, "request_id": "p2"}
  ]
}
```

**Resposta 200 OK (`BatchPredictResponse`)**:
```json
{
  "total_items": 2,
  "results": [ /* um PredictResponse por item, na mesma ordem */ ],
  "processed_at": "2026-09-12T15:00:00Z"
}
```

---

### Formato de Erros (SEMPRE formatados — NUNCA expõe stack trace!) 🔒

Todos os erros retornam o schema: `ErrorResponse`:
```json
{
  "error": "VALIDATION_ERROR",
  "message": "A requisição contém campos inválidos ou obrigatórios ausentes.",
  "details": [
    {"field": "body.text", "reason": "Field required"}
  ],
  "request_id": null,
  "timestamp": "2026-09-12T15:00:00Z"
}
```

| HTTP | Tipo de Erro | Causa comum |
|:----:|:--------------|:--------------|
| **400** | `INVALID_INPUT` | Texto vazio / muito curto / muito longo; lote excedeu limite |
| **422** | `VALIDATION_ERROR` | Payload JSON inválido ou campos obrigatórios ausentes |
| **503** | `SERVICE_UNAVAILABLE` | Modelo `.joblib` não foi carregado ainda |
| **500** | `INTERNAL_SERVER_ERROR` | Erro interno |

---

### Exemplos com cURL / Python

#### cURL — Predição Simples
```bash
curl -X POST "http://localhost:8000/predict" ^
  -H "Content-Type: application/json" ^
  -d "{\"text\": \"Dor torácica aguda irradiada esquerda, diaforese, dispneia, ECG com elevação do segmento ST. Diagnóstico: Infarto agudo do miocárdio.\", \"return_probabilities\": true, \"request_id\": \"EX001\"}"
```

#### Python — requests (Python requests)
```python
import requests

payload = {
    "text": "Patient with chronic gastritis reports intermittent heartburn after meals. Endoscopy showed mild pangastritis.",
    "return_probabilities": True,
    "request_id": "paciente-gastrite-01"
}
resp = requests.post("http://localhost:8000/predict", json=payload)
resp.raise_for_status()
print(resp.json())
```

---

## Como Executar (Resumo Completo)

### 1. Instalar dependências

O projeto usa [uv](https://docs.astral.sh/uv/) como gerenciador de dependências:

```bash
uv sync
```

> Se não tiver `uv` instalado: `curl -LsSf https://astral.sh/uv/install.sh | sh`

### 2. Executar o EDA

Abra o arquivo `notebooks/EDA_Medical_Abstracts.ipynb` em seu ambiente Jupyter (JupyterLab, Jupyter Notebook ou VSCode) e execute todas as células sequencialmente.

Saídas geradas em `notebooks/eda_outputs/` (8 gráficos PNG) + artefatos em `models/`.

### 3. Executar o pipeline completo (treinamento)

```bash
python run_pipeline.py --model logistic_regression
```

Flags opcionais:
- `--model {logistic_regression,naive_bayes,random_forest,linear_svc}`
- `--lemmatize`: ativa lematização no pré-processamento (mais lento)

### 4. Subir a API de inferência

```bash
python run_api.py
# Acesso Swagger: http://localhost:8000/docs
```

### 5. Rodar TODOS os testes unitários e integrados

```bash
uv run pytest tests/ -v
```

Atualmente são **74 testes cobrindo**:
| Suite | Quantidade |
|:------|:----------:|
| Pipeline ML (data, preproc, features, model) | 46 testes |
| API FastAPI (schemas, services, endpoints, erros) | 28 testes |
| **Total** | **74 testes** |

---

## Exemplo de Inferência (sem usar a API)

```python
import joblib

artifact = joblib.load("models/urgency_classifier.joblib")
pipeline = artifact["model"]

texto = ("Acute chest pain radiating to the left arm. ECG shows ST-elevation. "
         "Diagnosis: acute myocardial infarction. Urgent cath required.")

y_pred = pipeline.predict([texto])   # retorna array([2]) -> Urgente
prob   = pipeline.predict_proba([texto])
```

---

## GitHub Actions CI / CD

O workflow em `.github/workflows/ci.yml` roda automaticamente em push/pull request e possui **4 jobs paralelos/sequenciais**:

| Job | Descrição |
|:----|:----------|
| **lint-check** | Verificação de sintaxe Python (`py_compile`) de todos os módulos: source + pipelines + app + tests |
| **test** | Testes unitários com pytest em Python 3.10, 3.11, 3.12 (Ubuntu + 3.11 Windows) |
| **pipeline-check** | Smoke test de carregamento dataset, criação TF-IDF, criação de classificador, clean_text |
| **api-check** | 🔍 Validação da API: criação do app, rotas, 28 testes de API e teste E2E `/predict` com modelo dummy |

---

## 🐳 Docker (Containerização da API)

### Pré-requisitos
- **Docker Desktop** (Windows/macOS) ou **Docker Engine** (Linux) rodando
- O modelo `models/urgency_classifier.joblib` deve existir (gerado por `python run_pipeline.py`)

### Build + Execução Rápida (docker compose)

```bash
# Build + subir API em background
docker compose up --build -d api

# Ver logs
docker compose logs -f api

# Health check
curl http://localhost:8000/health

# Parar e remover
docker compose down
```

A imagem expõe a API na **porta 8000**. Swagger UI em: `http://localhost:8000/docs`

### Build Manual (sem docker compose)

```bash
# Build da imagem
docker build -t postech-ml-challenge/api:latest .

# Executar container (1 worker, 1 vCPU, 2GB RAM)
docker run --rm -d --name urgencia-api \
  -p 8000:8000 \
  --cpus=1.0 \
  --memory=2g \
  -v ./models:/app/models:ro \
  -v ./data:/app/data:ro \
  -e WORKERS=1 \
  -e WORKER_THREADS=4 \
  postech-ml-challenge/api:latest
```

### Serviços do `docker-compose.yml`

| Serviço | Imagem | Descrição | Porta |
|:--------|:-------|:----------|:-----:|
| **`api`** | `Dockerfile` (prod, `gunicorn + uvicorn worker`, não-root) | Servidor de produção | 8000:8000 |
| `api-dev` | `Dockerfile.dev` (uvicorn --reload, hot reload) | Desenvolvimento | 8001:8000 |
| `benchmark` | `python:3.11-slim` (executa `baseline_latency.py`) | Benchmark automático contra `api` após healthcheck | — |

### Variáveis de Ambiente (container)

Todas as configurações da API podem ser sobrescritas via `environment` no docker-compose
ou `-e VAR=valor` no docker run:

| Variável | Default | Descrição |
|:---------|:--------|:----------|
| `WORKERS` | `1` | Número de workers Gunicorn (escalar = maior throughput) |
| `WORKER_THREADS` | `4` | Threads por worker |
| `TIMEOUT` | `120` | Timeout de request em segundos |
| `PORT` | `8000` | Porta interna do container |
| `MODELS_DIR` | `/app/models` | Diretório com `.joblib` |
| `REQUEST_TEXT_MIN_LENGTH` | `10` | Validação de tamanho mínimo de texto |
| `REQUEST_TEXT_MAX_LENGTH` | `50000` | Validação de tamanho máximo |
| `REQUEST_BATCH_MAX_ITEMS` | `100` | Limite de itens no endpoint `/predict/batch` |

### Limites de Recursos (CPUs/Memória) configuráveis

Por padrão a API `api` tem:
```yaml
cpus: 1.0
mem_limit: 2g
```

Escolha valores diferentes via variáveis de ambiente (ex: 4 workers / 2 CPUs):
```bash
WORKERS=4 API_CPUS=2.0 docker compose up --build -d api
```

---

## ⚖️ Baseline de Latência & Benchmark

Documento completo (oficial) em: 👉 **[docs/BASELINE_LATENCY.md](docs/BASELINE_LATENCY.md)**

Arquivo JSON bruto do resultado oficial V1:
👉 `benchmarks/results/baseline_local_uvicorn_1w_c1.json`

### Baseline V1 — Resultado Oficial (1 worker / 1 vCPU / concorrência=1)

| Métrica | Valor |
|:--------|:-----:|
| **Média** | **3,52 ms** |
| **P50 (mediana)** | **3,39 ms** |
| **P90** | 4,18 ms |
| **P95** | 4,39 ms |
| **P99** | **4,97 ms** |
| Mínimo | 2,78 ms |
| Máximo | 29,54 ms |
| **Throughput (RPS)** | **254,79 req/s** |
| **Taxa de sucesso** | **100%** (1000/1000) |

### Como Rodar o Benchmark

#### Modo Docker (baseline reproduzível — recomendado)

```bash
# 1. Gere o modelo (uma vez):
python run_pipeline.py --model logistic_regression

# 2. Suba a API em container com 1 worker (baseline default):
WORKERS=1 API_CPUS=1.0 docker compose up --build -d api

# 3. Execute o benchmark (100 warm-up + 1000 requests, conc=1):
python benchmarks/baseline_latency.py \
  --url http://localhost:8000 \
  --warmup 100 \
  --requests 1000 \
  --concurrency 1 \
  --workers 1 \
  --cpus "1.0" \
  --output benchmarks/results/baseline_v2_docker.json
```

#### Modo Benchmark Service (tudo em container, só precisar do Docker)

```bash
# Sobe a API + roda benchmark automaticamente (aguarda healthcheck)
docker compose up --build benchmark
```

#### Flags do `baseline_latency.py` (CLI completa):

| Flag | Default | Descrição |
|:-----|:--------|:----------|
| `--url` | `http://localhost:8000` | URL base da API |
| `--warmup` | `50` | Requisições de warm-up |
| `--requests` | `500` | Requisições do benchmark real |
| `--concurrency` | `1` | Nível de concorrência (1 = latência isolada; >1 = throughput) |
| `--workers` | None | **Documentação apenas** — workers configurados (salvos no JSON) |
| `--cpus` | None | **Documentação apenas** — CPUs alocadas (salvos no JSON) |
| `--no-probabilities` | OFF | Não enviar `return_probabilities` |
| `--output` | `benchmarks/results/baseline_<timestamp>.json` | Arquivo JSON de saída |
| `--skip-warmup` | OFF | Pular warm-up |
| `--timeout-health` | `120` | Segundos máximos aguardando `/health` ficar 200 |

#### Concorrência alta (medir throughput):
```bash
# 2.000 requisições com 32 conexões simultâneas
python benchmarks/baseline_latency.py \
  --requests 2000 --concurrency 32 \
  --output benchmarks/results/thr_c32.json
```

### Estrutura da Pasta de Benchmark

```
benchmarks/
├── __init__.py
├── baseline_latency.py     # CLI principal do benchmark
├── payloads.json           # 7 casos reais (urgente / atenção / normal + 4 derivados)
└── results/                 # Resultados JSON salvos
    └── baseline_local_uvicorn_1w_c1.json   # Baseline oficial V1
```

---

## ✈️ Orquestração de Treinamento / Retreinamento com Airflow

**Documentação completa e passo-a-passo**: 👉 **[docs/AIRFLOW.md](docs/AIRFLOW.md)**

A DAG **`train_urgencia_medica_v1`** automatiza todo o ciclo de vida do modelo, desde o carregamento dos dados até a persistência de um artefato versionado e aprovado, **reutilizando 100% das funções** existentes em `sources/` e `pipelines/` (sem duplicar código).

### Fluxo da DAG (6 tasks)

```
[setup_dirs] → [data_load_validate] → [preprocess_and_split] → [build_features_and_train] → [validate_metrics_thresholds (ShortCircuit)] → [persist_model_and_metrics]
```

### Principais Funcionalidades

| Item | Descrição |
|:-----|:----------|
| **Idempotente / Retreinamento** | Cada run cria um `.joblib` **versionado** (`<DAG_ID>__<RUN_ID>__<timestamp>`). NUNCA sobrescreve artefatos antigos. Apenas `urgency_classifier.joblib` (latest) é atualizado para deploy. |
| **Porta de Qualidade (ShortCircuit)** | O artefato SÓ é persistido se `accuracy` e `f1_macro` no teste ultrapassarem thresholds mínimos configuráveis via trigger JSON. |
| **Retries / Resiliência** | 2 retries com exponential backoff (2–5 min), timeout global de 45 min por task, `max_active_runs=1`. |
| **XCom inter-task** | Todas as tasks comunicam via `ti.xcom_pull(key="return_value")`, sem arquivos mágicos fora do padrão. |
| **Parâmetros trigger (JSON)** | `model_name`, `lemmatize`, `remove_stopwords`, `tfidf_max_features`, `tfidf_ngram_range`, `min_f1_macro`, `min_accuracy`. |

### Início Rápido (3 comandos)

```bash
# 1) Inicializar DB + admin (apenas 1ª vez):
docker compose -f docker-compose-airflow.yml up airflow-init

# 2) Subir Airflow (Postgres + Webserver + Scheduler)
docker compose -f docker-compose-airflow.yml up -d

# 3) Abrir UI
#    URL:        http://localhost:8080
#    Username:   admin
#    Password:   admin
```

Na UI:
1. Procure **`train_urgencia_medica_v1`** e ligue o toggle ON
2. Clique em **Trigger DAG w/ config** para passar parâmetros JSON personalizados, ou **Trigger DAG** para defaults.
3. Acompanhe em **Graph View** — após ~5–15 min (dependendo de lematização) os artefatos aparecem em `models/` e métricas em `docs/`.

### Artefatos por Run

Após uma execução aprovada, são criados:

| Local | Ficheiro | Descrição |
|:------|:---------|:----------|
| `models/` | `urgency_classifier__<dag>__<run>__<ts>.joblib` | Versão imutável do pipeline completo (TF-IDF + Classifier) + metadata completa |
| `models/` | `urgency_classifier.joblib` | Cópia da última versão aprovada — usada diretamente pela FastAPI |
| `docs/` | `model_metrics__<dag>__<run>__<ts>.json` | Registo histórico imutável das métricas |
| `docs/` | `model_metrics.json` | Último snapshot de métricas |

> 💡 **Rollback manual fácil**: para reverter para um modelo anterior, basta copiar o `.joblib` da versão desejada para `urgency_classifier.joblib` e reiniciar a API.

### Referência de Resultado (Validação Standalone)

Execução debug standalone da DAG completa (LogReg, TF-IDF 5k features, sem lema):

| Métrica | Valor |
|:--------|:-----:|
| Acurácia (teste 2888 amostras) | **0.6898** |
| F1-macro (3 classes) | **0.6675** |
| ROC-AUC OvR | **0.8433** |
| Acurácia Val (durante treino) | 0.6939 |
| F1-macro Val | 0.6667 |

---

### Sugestão de próximas etapas (comparação de otimizações):

1. `WORKERS=2`, `concurrency=2` → Throughput
2. Converter sklearn → **ONNX Runtime**
3. Desativar `return_probabilities` (código cliente)
4. Docker em WSL2 ou Linux vs. Windows host
5. **Sensor Airflow** para novo CSV em `data/raw/` → auto-trigger retreinamento (ex: `FileSensor`)
6. **Callback de sucesso** Airflow → notificação email/Slack + deploy automático novo modelo latest para API

