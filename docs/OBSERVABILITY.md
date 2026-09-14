# Observabilidade: Prometheus, Grafana e Tráfego

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
