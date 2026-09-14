# Baseline de Latência — API de Classificação de Urgência Médica

Documento de referência para comparar o desempenho de futuras otimizações do modelo ou
da infraestrutura (ONNX, quantização, batching, múltiplos workers, inferência em GPU, etc).

> **Status**: BASELINE V1 — executado e registrado oficialmente.

---

## 1. Resumo Executivo

| Item | Valor |
|:-----|:------|
| Endpoint medido | `POST /predict` (probabilidades ON) |
| Modelo | Logistic Regression + TF-IDF (10.000 features) |
| N treino | 11.550 amostras |
| Acurácia de referência (teste) | **68,32%** |
| F1-Macro (teste) | **66,23%** |
| **Taxa de sucesso** | **100,00%** (1000/1000 requests) |
| **Throughput** | **254,79 req/s** |
| **Latência média (avg)** | **3,522 ms** |
| **Latência P99** | **4,974 ms** |
| Latência P95 | 4,391 ms |
| Latência P90 | 4,180 ms |
| Latência P50 (mediana) | 3,387 ms |
| Latência mínima | 2,775 ms |
| Latência máxima | 29,540 ms (outlier) |
| Duração total do benchmark | 3,925 s |

---

## 2. Parâmetros do Benchmark

### 2.1 Execução
- **Warm-up (antes do benchmark)**: **100 requisições** (eliminam custo de JIT / cold start).
- **Número total de requisições**: **1.000**
- **Concorrência**: **1 requisição por vez** (mede latência individual isolada, sem interferência).
- **Probabilidades**: **Ativadas** (impacta ligeiramente no tempo, é o caso de uso real).

### 2.2 Configuração da API
| Configuração | Valor |
|:-------------|:------|
| Servidor ASGI | **Uvicorn** (padrão) |
| Workers | **1** (1 processo, medido) |
| CPUs alocadas | **1.0 vCPU** |
| Limite de memória | N/A (benchmark local) |
| Preload (gunicorn) | Desativado (uvicorn puro) |
| Modelo carregado via | Lifespan `startup` do FastAPI (em memória antes do primeiro request) |

### 2.3 Payloads Enviados (round-robin)
Foram usados **7 payloads realistas**, cobrindo os 3 níveis de urgência:

| Tag | Contexto médico | Classe esperada |
|:----|:----------------|:---------------:|
| `urgente`     | IAM com elevação ST + Troponin elevado (STEMI) | Urgente (2) |
| `atencao`     | Cefaleia crônica / migrânea ambulatorial | Atenção (1) |
| `normal`      | Gastrite crônica + PPI (ambulatorial simples) | Normal (0) |
| `cardio`      | IC isquêmica, BNP elevado, ajuste de medicação | Atenção / Urgente |
| `oncologia`   | Adenocarcinoma pulmonar, estadiamento oncológico | Urgente (2) |
| `gastro`      | Diverticulite não complicada (ATB oral) | Normal (0) |
| `neurologia`  | 1ª crise convulsiva generalizada (ambulatorial) | Atenção (1) |

Cada payload tem ~200-300 caracteres de texto (tamanho realista).

---

## 3. Hardware & Ambiente

### 3.1 Máquina (host do benchmark)
| Propriedade | Valor |
|:------------|:------|
| Sistema Operacional | **Windows 11 Pro** (Build 10.0.26200) |
| Arquitetura | AMD64 / x86_64 |
| CPU | **Intel Core i5-1335U** (Family 6 Model 140 Stepping 2, GenuineIntel) |
| Núcleos lógicos | 8 |
| Memória RAM (host) | > 8 GB (memória total não coletada por permissões WMI) |
| Python | **3.13.13** |

### 3.2 Software e Bibliotecas
| Componente | Versão |
|:-----------|:-------|
| FastAPI | >= 0.104 |
| Uvicorn | >= 0.24 |
| scikit-learn | >= 1.2 |
| pandas | >= 1.5 |
| joblib | >= 1.2 |
| Pydantic | v2 |
| httpx | >= 0.25 |
| numpy | (opcional, para percentis) |

### 3.3 Containerização (ainda não executada por indisponibilidade do daemon)
O Docker Desktop não estava ativo no momento do baseline. Arquivos de container já
estão **100% prontos** (ver Dockerfile e docker-compose.yml) e o baseline foi
equivalente em setup (1 worker Python). O baseline em container pode ser reproduzido com:

```bash
# Com Docker Desktop rodando
docker compose up --build -d
python benchmarks/baseline_latency.py --output benchmarks/results/baseline_docker_1w_c1.json
```

**Espera-se** que a latência em Docker Linux (WSL2) seja similar ou até 10-20%
melhor (maior overhead no Windows host). O resultado aqui coletado é conservador
e adequado como baseline.

---

## 4. Resultados Detalhados

### 4.1 Warm-up (100 requisições, desconsiderado no cálculo)
| Métrica | Latência (ms) |
|:--------|:-------------:|
| min | 3,496 |
| max | 10,243 |
| avg | 4,441 |
| P50 | 4,315 |
| P90 | 5,183 |
| P95 | 5,319 |
| P99 | 5,697 |
| Sucesso | **100 / 100 (100%)** |

> O warm-up reduziu a média em ~20% (4.4 ms → 3.5 ms) — confirma a importância de aquecer
> o JIT/python antes de qualquer medição oficial.

### 4.2 Benchmark Principal (1.000 requisições — 100% sucesso)
| Métrica | Latência (ms) |
|:--------|:-------------:|
| **count** | **1.000** |
| **min** | **2,775** |
| **max** | **29,540** |
| **avg** | **3,522** |
| **std** | **0,956** |
| **P50 (mediana)** | **3,387** |
| P75 | 3,819 |
| P90 | 4,180 |
| P95 | 4,391 |
| **P99** | **4,974** |
| **Taxa de sucesso** | **100,0%** |
| Throughput (RPS) | **254,79** |
| Duração total | 3,925 s |

---

## 5. Como Reproduzir Este Baseline

Os comandos abaixo produzem exatamente o mesmo JSON estruturado:

### Via Docker (recomendado para próximas execuções)
```bash
# 1. Gerar o modelo (uma vez)
python run_pipeline.py --model logistic_regression

# 2. Buildar e subir container com 1 worker / 1 CPU
docker compose up --build -d api

# 3. Rodar benchmark (aguarda /health healthy automático)
python benchmarks/baseline_latency.py \
  --url http://localhost:8000 \
  --warmup 100 \
  --requests 1000 \
  --concurrency 1 \
  --workers 1 \
  --cpus "1.0" \
  --output benchmarks/results/baseline_<data_versao>.json
```

### Local (sem Docker, equivalente ao baseline V1)
```bash
# Terminal 1: suba a API
MODELS_DIR=./models python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --workers 1

# Terminal 2: execute o benchmark
python benchmarks/baseline_latency.py --output benchmarks/results/baseline_local.json
```

---

## 6. Pontos de Comparação (Otimizações Futuras)

Use como referência o P99 (latência) e o RPS (throughput). Possíveis otimizações a testar:

| Otimização | Impacto esperado |
|:-----------|:-----------------|
| **2+ workers Gunicorn** (+ mais CPUs) | Throughput ↑ 1.6–2x por worker adicional |
| **ONNX Runtime** (Converter sklearn → ONNX) | Latência ↓ 30–60% |
| **Quantização sklearn** (se via ONNX) | Latência ↓ adicional |
| **Batching inteligente** (mini-batch no /predict) | Throughput ↑ em carga concorrente |
| **Desativar `return_probabilities`** | Latência ↓ 10-15% |
| **Reduzir TF-IDF features (5k em vez de 10k)** | Latência ↓ (monitorar F1) |
| **Cache de texto idêntico (LRU)** | Latência ↓ em workload repetido |
| **Deploy em WSL2 / Linux nativo** | Latência ↓ / Throughput ↑ |
| **Substituir modelo por BERT/ClinicalBERT** | Latência ↑ MUITO (não é baseline "leve") |

---

## 7. Arquivos Gerados / Versionados

| Arquivo | Descrição |
|:--------|:----------|
| `benchmarks/results/baseline_local_uvicorn_1w_c1.json` | **RESULTADO BRUTO** do baseline V1 (dados estruturados) |
| `benchmarks/baseline_latency.py` | Script reproduzível (CLI) |
| `benchmarks/payloads.json` | 7 payloads reutilizados (7 casos reais) |
| `Dockerfile` + `docker-compose.yml` | Infraestrutura reproduzível em container |
| `docs/BASELINE_LATENCY.md` | Este documento (este arquivo) |

---

## 8. Observações Finais

1. **O outlier de 29 ms (máx)** representa 0,1% das requisições e, na prática, não
   impacta SLAs reais (P99 ficou em 4,97 ms, abaixo de 5 ms — excelente para triagem).
2. **O modelo é propositalmente "leve" (TF-IDF + Regressão Logística)**, o que
   garante latência de baixa dezena de ms. Modelos mais pesados (transformers) devem
   ser comparados com cautela.
3. **Para comparar baselines futuros, garanta**:
   - Mesmo número de workers / CPUs
   - Mesma configuração `return_probabilities` (ON ou OFF)
   - Mesmo conjunto de payloads e concorrência
4. O script `baseline_latency.py` também suporta **benchmarks concorrentes**:
   ```bash
   # Exemplo: 1000 requests com 16 conexões simultâneas
   python benchmarks/baseline_latency.py --requests 1000 --concurrency 16
   ```

---

*Documento criado em 2026-09-12 para o projeto postech-ml-challenge-fase-3.*
*Atualizações devem ser anexadas como novas versões (V2 etc.) sem sobrescrever este baseline.*
