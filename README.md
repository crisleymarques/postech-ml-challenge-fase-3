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
│   ├── main.py                 # Aplicação FastAPI + rotas + middleware Prometheus
│   ├── config.py               # Configurações da API (pydantic-settings)
│   ├── schemas.py              # Schemas de entrada, respostas e erros (Pydantic v2)
│   ├── exceptions.py           # Handlers de erros formatados (sem stack trace!)
│   ├── metrics.py              # 🆕 10 métricas Prometheus (HTTP + inferência + modelo)
│   └── services/
│       ├── __init__.py
│       └── model_service.py    # Singleton: carregamento e inferência do modelo
├── tests/                        # Testes unitários e integrados (pytest)
│   ├── conftest.py               # Configuração compartilhada dos testes
│   ├── test_data_loader.py
│   ├── test_preprocessing.py
│   ├── test_feature_pipeline.py
│   ├── test_model.py
│   ├── test_api.py             # 28 testes unitários + integrados da API
│   └── test_metrics.py         # 🆕 18 testes: counters, histograms, low-cardinality, /metrics
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
├── scripts/                        # 🆕 Scripts utilitários (gerador tráfego, etc)
│   └── generate_traffic.py    # 🆕 Gerador de tráfego sintético (CLI) para painéis
├── prometheus/                     # 🆕 Config Prometheus
│   └── prometheus.yml      # 🆕 Scrape config `urgencia-api` (alvo = api:8000)
├── grafana/                        # 🆕 Config e dashboards Grafana (100% provisionados)
│   └── provisioning/
│       ├── datasources/
│       │   └── prometheus.yml # 🆕 Datasource `Prometheus` (uid=urgencia-prometheus, isDefault)
│       └── dashboards/
│           ├── 01-dashboard-provider.yml   # 🆕 Provider que carrega JSONs da pasta
│           └── urgencia_medica_observabilidade.json  # 🆕 Dashboard **versionado** (12 painéis)
├── docs/
│   └── evidences/             # 🆕 Screenshots/evidências da stack rodando
├── run_pipeline.py               # Script orquestrador - executa todo o pipeline ML
├── run_api.py                    # Script rápido para subir a API (uvicorn reload)
├── Dockerfile                    # Container produção (Gunicorn + UvicornWorker, non-root)
├── Dockerfile.dev                # Container dev (uvicorn --reload)
├── docker-compose.yml            # 6 serviços: api / api-dev / benchmark / prometheus / grafana / traffic-generator
├── docker-compose-airflow.yml    # Serviços Airflow + PostgreSQL 16
├── .airflow.env                  # UID e credenciais default Airflow
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

> **💡 Nota:** Veja o documento detalhado em [docs/ADR_PRODUCTION_STRATEGY.md](docs/ADR_PRODUCTION_STRATEGY.md) para a arquitetura completa abordando escalabilidade, observabilidade, retreinamento e trade-offs.

### Por que a arquitetura importa neste projeto?

O sistema classifica laudos médicos em três níveis de urgência (**Normal**, **Atenção**, **Urgente**) para auxiliar a triagem em ambiente hospitalar. A escolha da estratégia de inferência tem impacto direto na segurança do paciente: um paciente classificado como "Urgente" que aguarda um processamento em lote pode ter seu atendimento atrasado com consequências graves.

Sendo assim, a estratégia **Real-time (Online Inference)** é a única aceitável para triagem hospitalar. A latência medida de **P99 ≈ 4,97 ms** comprova que o modelo escolhido é leve o suficiente para suportar inferência síncrona em tempo real, mesmo sob carga.


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
| `GET`  | `/metrics` | 🆕 **Métricas Prometheus** (exposition format 0.0.4) | ❌ |

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

Atualmente são **92 testes cobrindo**:
| Suite | Quantidade |
|:------|:----------:|
| Pipeline ML (data, preproc, features, model) | 46 testes |
| API FastAPI (schemas, services, endpoints, erros) | 28 testes |
| 🆕 **Observabilidade Prometheus** (counters, histograms, low-cardinality) | 18 testes |
| **Total** | **92 testes** |

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

## 📊 Observabilidade — Métricas Prometheus

A API de inferência é **totalmente instrumentada** com a biblioteca oficial `prometheus_client`, expondo métricas de volume, latência, erros e de negócio via endpoint `GET /metrics` no formato Prometheus Exposition v0.0.4, compatível com qualquer Prometheus/Grafana.

### Endpoint `GET /metrics`

- **URL**: `http://localhost:8000/metrics`
- **Content-Type**: `text/plain; version=1.0.0; charset=utf-8` (padrão Prometheus)
- **Self-reference safe**: o próprio endpoint `/metrics` **não entra** nos contadores HTTP (evita loop: scrape → count → scrape → count ...)

### Como coletar localmente (exemplo cURL)

```bash
# Verificar o scrape raw
curl -s http://localhost:8000/metrics | head -n 40

# Contar quantas séries HELP/TYPE diferentes existem
curl -s http://localhost:8000/metrics | grep "^# " | wc -l
```

### `prometheus.yml` — Exemplo completo de scrape config

```yaml
global:
  scrape_interval: 10s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'urgencia-api'
    static_configs:
      - targets: ['localhost:8000']
    metrics_path: /metrics
    honor_labels: true
```

### Tabela completa de métricas (10 ao todo)

| Nome | Tipo | Labels | Descrição |
|------|------|--------|-----------|
| **`http_requests_total`** | Counter | `method`, `endpoint`, `status_code` | Total de requests HTTP por método, endpoint (template da rota) e código de status. |
| **`http_request_duration_seconds`** | Histogram | `method`, `endpoint` | Latência full da request (inclui validação, HTTP overhead). Buckets: `0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0` segundos. |
| **`http_errors_total`** | Counter | `method`, `endpoint`, `status_code`, `status_family` | Erros HTTP segmentados por status exato **e** por família (`4xx` / `5xx`). |
| **`inference_predictions_total`** | Counter | `endpoint_type`, `predicted_class`, `model_name` | Inferências bem sucedidas, com classe predita (`normal`/`atencao`/`urgente`) e modelo (`logistic_regression`, etc.). |
| **`inference_latency_seconds`** | Histogram | `endpoint_type` | Latência **exclusiva do `pipeline.predict()`** (limpa, sem overhead HTTP). Buckets: `0.0001, 0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.5, 1.0` segundos. |
| **`inference_errors_total`** | Counter | `endpoint_type`, `error_type` | Falhas na inferência. Tipos comuns: `model_not_loaded`, `invalid_text`, `predict_exception`, `batch_too_large`. |
| **`inference_items_processed_total`** | Counter | `endpoint_type` | Número de itens efetivamente processados (lotes = N items contam +N cada, não +1). |
| **`model_loaded`** | Gauge | `model_name` | Estado atual do artefato: `1.0` = carregado, `0.0` = não carregado. Flipado no lifespan. |
| **`model_metadata_info`** | Info | — | Dicionário de metadata do deploy atual: `model_type`, `accuracy`, `f1_macro`, `roc_auc_ovr`, `trained_at`, `dag_id`, `run_id`, etc. |

### Arquitetura de observabilidade aplicada

```
┌──────────────────────────────────────────────────────────────────────┐
│                      FastAPI (app.main)                             │
│                                                                     │
│  ┌─ Middleware HTTP (unico, skip /metrics) ──────────────────────┐ │
│  │  time.perf_counter() -> finally observe_http_request(...)      │ │
│  │      -> http_requests_total / http_errors / http_duration     │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                                                                     │
│  ┌─ lifespan ─────────────────────────────────────────────────────┐ │
│  │ startup: load_model() => gauge=1 + info metadata preenche      │ │
│  │ shutdown: unload_model() => gauge=0                            │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                                                                     │
│  ┌─ ModelService (app.services.model_service) ────────────────────┐ │
│  │ predict_single:                                                │ │
│  │   t0=perf_counter(); y_pred=pipeline.predict(); latency=...   │ │
│  │   finally:                                                     │ │
│  │     sucesso -> inference_predictions_total (x1) + items (x1)  │ │
│  │     sucesso -> inference_latency_seconds.observe(latency)     │ │
│  │     falha  -> inference_errors_total{error_type=?}            │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                                                                     │
│  GET /metrics -> Response(body=REGISTRY.generate_latest(),         │
│                       media_type="text/plain; version=1.0.0")      │
└──────────────────────────────────────────────────────────────────────┘
           │
           ▼
 Prometheus Server (scrape 10s)  ►  Grafana Dashboard
```

### Labels de baixa cardinalidade (regras seguidas)

| ❌ Proibido | ✅ Permitido |
|:------|:------|
| `request_id`, trace_id aleatórios | `endpoint`, `method`, `status_code` — valores de domínio fechado |
| `text` do laudo, body de entrada | `predicted_class` (`normal`, `atencao`, `urgente` — só 3 valores) |
| IP do cliente / User-Agent | `error_type` — 4–8 valores fixos (não dinâmico) |
| Timestamps como label | `status_family` — apenas `4xx`, `5xx` |
| `<uuid>` / path params dinâmicos | `model_name` — um nome por deploy |

> Implementado via helper `_resolve_route_template(request)` que sempre usa `request.scope.route.path` (template da rota) ao invés de `request.url.path` bruto. Chamadas repetidas com a mesma combinação nunca criam novas séries.

### PromQL útil para painéis (exemplos prontos)

| Painel | Query PromQL |
|:-------|:-------------|
| **RPS / requisições por segundo** | `rate(http_requests_total[5m])` |
| **Taxa de erro 5xx (%)** | `100 * sum(rate(http_errors_total{status_family="5xx"}[5m])) / sum(rate(http_requests_total[5m]))` |
| **P95 latência HTTP por endpoint** | `histogram_quantile(0.95, sum(rate(http_request_duration_seconds_bucket[5m])) by (le, endpoint))` |
| **Throughput predições por classe** | `sum(rate(inference_predictions_total[5m])) by (predicted_class)` |
| **Distribuição de classes (último 5m)** | `sum(increase(inference_predictions_total[5m])) by (predicted_class)` |
| **P99 latência do modelo (sem HTTP)** | `histogram_quantile(0.99, sum(rate(inference_latency_seconds_bucket[5m])) by (le, endpoint_type))` |
| **Itens processados por hora (inclui lotes)** | `sum(increase(inference_items_processed_total[1h])) by (endpoint_type)` |
| **Erros de inferência por tipo** | `sum(increase(inference_errors_total[15m])) by (error_type)` |
| **Modelo carregado? Sinalização** | `model_loaded == 1` (alerta quando = 0) |
| **Saturação de 422 (payloads inválidos)** | `sum(rate(http_errors_total{status_code="422"}[5m]))` |

### Validação rápida do formato

O conteúdo de `/metrics` é 100% compatível com o parser oficial do Prometheus. No smoke test oficial da branch:
- ✅ `92 testes` passando (`pytest tests/ -v`)
- ✅ `119` linhas parseáveis, `0` linhas inválidas
- ✅ `14` séries HELP/TYPE declaradas (todas as métricas + `_info`, `_count`, `_bucket`)
- ✅ Content-Type: `text/plain; version=1.0.0; charset=utf-8`

---

## 📊🧱 Stack Completa Observabilidade (API → Prometheus → Grafana)

Foi criada uma stack 100% reproduzível via **Docker Compose** que sobe **4 serviços orquestrados**, com Grafana **já provisionado** (datasource + dashboard versão inicial JSON). Basta 1 comando para subir tudo.

### Visão da Stack

```
                  ┌──────────────────────────────────────────────────────┐
                  │              Docker Compose Host                    │
                  │                                                      │
   usuário/API ───► api:8000     (FastAPI, model loaded, /metrics 200) │
                  │      │                                               │
                  │      ▼                                               │
                  │ prometheus:9090 (scrape 10s do api:8000/metrics)  │
                  │      │                                               │
                  │      ▼                                               │
                  │   grafana:3000  (datasource Prometheus já criado,  │
                  │      │            dashboard default já carregado)  │
                  │      │                                               │
                  │      ▼                                               │
                  │ traffic-generator (CLI: mistura predict / batch    │
                  │                    e 10% erros intencionais)       │
                  └──────────────────────────────────────────────────────┘
                        │                       │
                        ▼                       ▼
                 porta 8000 local          porta 3000 local
                (swagger/testes)         (Grafana UI)
```

### Serviços do `docker-compose.yml` (novos em relação à Fase 5)

| Serviço | Porta local | Imagem | Profile | Descrição |
|:--------|:-----------:|:-------|:-------:|:----------|
| **`api`** | **`8000:8000`** | `Dockerfile` (postech-ml-challenge/api) | default | FastAPI + modelo. `/health`, `/predict`, `/predict/batch`, `/metrics`. Sempre sobe. |
| `api-dev` | `8001:8000` | `Dockerfile.dev` | dev | Hot reload para dev. Não sobe por padrão. |
| `benchmark` | — | python:3.11-slim | benchmark | Executa `baseline_latency.py` após API healthy. |
| **`prometheus`** 🆕 | **`9090:9090`** | prom/prometheus:v2.54.1 | **observability** | Scrape `api:8000/metrics` a cada 10s, TSDB com retenção de 7 dias. |
| **`grafana`** 🆕 | **`3000:3000`** | grafana/grafana-oss:11.2.0 | **observability** | Datasource `Prometheus` (uid=urgencia-prometheus) **já provisionado** + Dashboard home default **já provisionado** em JSON. |
| **`traffic-generator`** 🆕 | — | python:3.11-slim | observability, traffic | CLI Python que gera tráfego sintético com ~1 RPS (configurável), 20% batch, 10% erros 400, por 10 minutos por padrão. |

- **Volumes persistentes**: `urgencia-medica-prometheus-data`, `urgencia-medica-grafana-data` (dados sobrevivem a `compose down`)
- **Rede dedicada**: `urgencia-medica-observabilidade` (todos serviços se comunicam por DNS interno do Compose)
- **Healthchecks em cadeia**: `traffic-generator` espera `api healthy`; `prometheus` espera `api healthy`; `grafana` espera `prometheus healthy`.

---

### 🚀 3 Comandos para Subir Tudo (do zero)

```bash
# 1) Garanta o modelo .joblib existe (se não existir, gere-o primeiro):
python run_pipeline.py --model logistic_regression

# 2) Suba a stack COMPLETA (api + prometheus + grafana + traffic-generator):
#    O perfil `observability` ativa prometheus, grafana, traffic-generator;
#    O `api` sobe sempre (profile default).
docker compose --profile observability up --build -d

# 3) (Opcional) Subir SÓ observab (sem o gerador de tráfego automático, para gerar manual):
docker compose --profile observability up --build -d api prometheus grafana
```

**Verificar containers**:
```bash
docker compose ps
```
Esperado: `api = healthy` + `prometheus = healthy` + `grafana = healthy` (após ~60–90 segundos).

---

### 🌐 URLs de Acesso

| Serviço | URL | Usuário | Senha |
|:--------|:----|:-------:|:-----:|
| Swagger UI da API | http://localhost:8000/docs | — | — |
| Métricas Prometheus (scrape) | http://localhost:8000/metrics | — | — |
| **Prometheus UI (query browser)** | http://localhost:9090 | — | — |
| **Grafana UI (Dashboard)** | **http://localhost:3000** | **admin** | **admin** (definível via `.env`, veja abaixo) |

> 🔐 Troque a senha do Grafana de admin/admin a primeira vez que logar. Os valores default são para ambiente **local apenas**.

### Variáveis ambiente para a stack (`.env` opcional na raiz)

| Variável | Default | Descrição |
|:---------|:--------|:----------|
| `GRAFANA_ADMIN_USER` | `admin` | Usuário inicial admin do Grafana |
| `GRAFANA_ADMIN_PASSWORD` | `admin` | Senha inicial admin do Grafana |
| `TRAFFIC_DURATION_SECONDS` | `600` | Duração do gerador de tráfego (segundos) |
| `TRAFFIC_RPS` | `1` | Requests/segundo alvo do gerador |
| `TRAFFIC_BATCH_RATIO` | `0.2` | Fração de requests como `/predict/batch` |
| `TRAFFIC_ERROR_RATIO` | `0.1` | Fração de requests intencionalmente inválidos (gera 400) |
| `WORKERS` / `WORKER_THREADS` / `API_CPUS` / `API_MEM_LIMIT` | 1 / 4 / 1.0 / `2g` | Tuning API produção |

---

### 🎛️ Gerar Tráfego de Teste (Popula Painéis)

#### Opção 1 — Automático via Compose (inicia com a stack)
Basta subir com `--profile observability` (como no passo 2 acima). O container `traffic-generator` vai, por 10 minutos, enviar ~1 req/s com mistura de:
- **~70%** `/predict` texto válido (cobre as 3 classes de urgência)
- **~20%** `/predict/batch` lotes de 3–10 itens
- **~10%** Requests inválidos propositais (texto curto < min_length 10 → 400 `INVALID_INPUT`)

Para **repetir** depois que o container terminou:
```bash
docker compose --profile traffic up --build traffic-generator
```

#### Opção 2 — Manual (host local, sem Docker, API em `localhost:8000`)
```bash
# Pré-requisito: API já rodando localmente em localhost:8000
pip install httpx

# Gera tráfego por 5 minutos, 2 RPS, 25% batch, 15% erros
python scripts/generate_traffic.py \
  --url http://localhost:8000 \
  --duration 300 \
  --rps 2.0 \
  --batch-ratio 0.25 \
  --error-ratio 0.15
```

O script espera a API ficar `healthy` antes de começar e imprime estatísticas a cada 10s (`P50/P95/P99 latência`, contadores 2xx/4xx/5xx, distribuição status).

---

### 📈 Dashboard Grafana Provisionado — 12 Painéis

O arquivo [urgencia_medica_observabilidade.json](file:///c:/FIAP/Fase3/postech-ml-challenge-fase-3/grafana/provisioning/dashboards/urgencia_medica_observabilidade.json) é carregado **automaticamente** como homepage do Grafana (graças a `GF_DASHBOARDS_DEFAULT_HOME_DASHBOARD_PATH`).

#### Layout / Painéis

| Linha | Painel | Tipo | Métrica base | Cobre checklist? |
|:-----:|:-------|:-----|:-------------|:-----------------|
| **Linha 0 — KPIs (Stat)** | | | | |
| 1 | Modelo Carregado? 1.0 / 0.0 | Stat | `model_loaded` | — |
| 2 | Total Requisições Acumulado | Stat | `sum(http_requests_total)` | ✅ Total requisições |
| 3 | Total Predições ML | Stat | `sum(inference_predictions_total)` | — |
| 4 | % Erro 5xx (thresholds verde/amarelo/vermelho) | Stat | `100 * rate(errors5xx) / rate(total)` | ✅ Erros |
| **Linha 1 — Volume** | | | | |
| 5 | 📈 QPS por Endpoint | Timeseries | `rate(http_requests_total) by (endpoint)` | ✅ Taxa requisições |
| 6 | Distribuição por HTTP Status (stacked bars) | Timeseries bars | `rate() by (status_code)` | ✅ Diferencia status |
| **Linha 2 — Latência HTTP Full** | | | | |
| 7 | ⏱️ Latência HTTP P50/P95/P99 por Endpoint | Timeseries | `histogram_quantile(0.50/0.95/0.99, http_request_duration_seconds_bucket)` | ✅ Latência |
| **Linha 3 — Latência ML + Erros %** | | | | |
| 8 | ⏱️ Latência Exclusiva do Modelo P50/P95/P99 | Timeseries | `histogram_quantile on inference_latency_seconds` | ✅ Latência (ML puro) |
| 9 | ❌ Erro Rate (%) — 4xx vs 5xx | Timeseries | `100 * rate(errors family) / rate(total)` | ✅ Taxa erros |
| **Linha 4 — Distribuições** | | | | |
| 10 | 🧮 Erros Inferência por Tipo (invalid_text / model_not_loaded / ...) | Pie (donut) | `increase(inference_errors_total) by (error_type)` | ✅ Erros detalhe |
| 11 | 🏷️ Distribuição de Classes Preditas (3 classes) | Pie | `increase(inference_predictions_total) by (predicted_class)` | ✅ Negócio |
| 12 | 🔁 Throughput por Classe (predições/s) | Timeseries | `rate(inference_predictions_total) by (predicted_class)` | ✅ Negócio |
| **Linha 5 — Auditoria** | | | | |
| 13 | 🔍 Tabela erros HTTP: Endpoint × Status × Família | Table | `increase(http_errors_total) by (endpoint, status_code, status_family)` | ✅ Auditoria |

> 💡 O dashboard vem com **auto-refresh de 10s** e timeframe default de `now-15m → now`. Altere no canto superior direito do Grafana conforme a duração do seu teste.

---

### 💾 Exportar / Atualizar Dashboard Futuramente

Se você personalizar o dashboard no Grafana UI e quiser **versionar as alterações**:
1. No Grafana → Dashboard → botão **Share** → **Export** → "Export for sharing externally" (opção **ON** para trocar datasource UID → urgencia-prometheus) → Download JSON.
2. Substitua o arquivo:
   ```
   grafana/provisioning/dashboards/urgencia_medica_observabilidade.json
   ```
3. Reinicie Grafana para re-provisionar:
   ```bash
   docker compose restart grafana
   ```

---

### 🧪 Validar Atualização dos Gráficos (Passo-a-Passo Recomendado)

```
T+0s  → docker compose --profile observability up --build -d
T+30s → docker compose ps                # verificar api = healthy, prom/grafana = starting
T+70s → docker compose ps                # todos = healthy
T+80s → Abrir http://localhost:9090 → Graph → executar a query `http_requests_total`
            → ENTER → esperou 15–20 segundos, apareceu 1 ou + datapoints? ✅ Prometheus coletando.
T+90s → Abrir http://localhost:3000 → login admin/admin → dashboard já abre automaticamente.
            → Ao longo de 3 a 5 min, todos os gráficos começam a se preencher com tráfego do gerador.
T+150s → Painel 5 (QPS) mostra linha ~1 RPS; Painel 6 mostra barras 200 majoritariamente;
            Painel 9 (Erros%) mostra ~0–15% de 4xx; Painéis 10 e 11 mostram distribuições.
```

---

### 🛑 Parar / Reiniciar / Resetar Tudo

```bash
# Parar stack mas PRESERVAR dados (prom TSDB, grafana dashboards customizados):
docker compose down

# Parar e APAGAR TUDO (containers + volumes):
docker compose down -v
#   Cuidado: -v = perde prometheus-data e grafana-data. Útil para "limpo total".

# Reiniciar apenas grafana (recarrega provisionamento de dashboard):
docker compose restart grafana

# Ver logs do Prometheus (ver scrape errors):
docker compose logs --tail=200 prometheus

# Ver logs do gerador (estatísticas de tráfego a cada 10s):
docker compose logs -f traffic-generator
```

---

### 🖼️ Evidências Visuais

Caso queira salvar evidências no repositório:
1. Abra Grafana → dashboard → botão no topo **"Share / snapshot"** ou tire um print manual.
2. Salve em `docs/evidences/` (pasta já criada na estrutura).
3. Exemplos de nomes: `grafana-dashboard-5min-traffic.png`, `prometheus-scrape-targets.png`.

> 📌 **Importante**: evidências visuais não são *obrigatórias* no build, mas são recomendadas no PR para os avaliadores verem os painéis "populados" sem rodar tudo.

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

