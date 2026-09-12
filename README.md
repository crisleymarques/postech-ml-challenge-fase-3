# Postech ML Challenge - Fase 3

## Triagem Automática de Urgência em Laudos Médicos (NLP)

Projeto de pós-graduação (FIAP/POSTECH) que implementa um sistema de classificação
de texto (NLP), destinado à triagem automática de resumos/laudos médicos
em três níveis de urgência: **Normal**, **Atenção** e **Urgente**.
Inclui um pipeline de treinamento, análise exploratória e **API FastAPI para inferência em produção**.

Dataset: **Medical Abstracts TC Corpus Dataset** (14.438 amostras).

---

## Estrutura de Pastas

```
postech-ml-challenge-fase-3/
├── data/
│   └── raw/                      # Dataset original (3 arquivos CSV)
│   └── processed/              # Dados processados e splits (gitignored)
├── docs/
│   └── DATASET.md                # Documentação detalhada do dataset e mapeamentos
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
├── run_pipeline.py               # Script orquestrador - executa todo o pipeline ML
├── run_api.py                    # 🆕 Script rápido para subir a API (uvicorn reload)
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

<!--
![CI Status](https://github.com/crisleymarques/postech-ml-challenge-fase-3/actions/workflows/ci.yml/badge.svg?branch=classificacao_laudos_fastapi)
-->

---

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

```bash
pip install -r requirements.txt
```

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
pytest tests/ -v
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
