# Documentação da API e Docker


A API de classificação de laudos médicos recebe um texto de laudo e retorna sua classificação de urgência, carregando diretamente o artefato `.joblib` gerado pelo pipeline de treinamento. Ideal para deploy em containers Docker e integração com sistemas hospitalares ou frontend Web.

### Como Iniciar a API

**1. Instalar dependências** (uma vez):
```bash
uv sync
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
| `USE_ONNX` | `auto` | Controle do backend ONNX Runtime (`auto`, `1`, `0`). Veja a documentação de Otimização ONNX para mais detalhes. |

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
