# ✈️ Orquestração de Treinamento com Apache Airflow

Este documento descreve como configurar, iniciar e utilizar a DAG de treinamento e retreinamento do classificador de urgência em laudos médicos usando **Apache Airflow 2.10**.

---

## 1. Visão Geral

A DAG **`train_urgencia_medica_v1`** executa um pipeline de ML completo, **reutilizando 100% dos módulos** existentes em `sources/` e `pipelines/` (sem duplicar código):

```
setup_dirs → data_load_validate → preprocess_and_split → build_features_and_train → validate_metrics_thresholds (ShortCircuit) → persist_model_and_metrics
```

### Funcionalidades Principais

- **Idempotente e segura para retreinamento**: artefatos são salvos com versão (DAG ID + RUN ID + timestamp) e nunca sobrescrevem versões anteriores. Apenas `urgency_classifier.joblib` (latest) é atualizado para a última versão aprovada.
- **ShortCircuit de qualidade**: o modelo só é persistido se ultrapassar os thresholds mínimos de `accuracy` e `f1_macro` configuráveis.
- **Retries robustos**: 2 retries com exponential backoff (2–5 min) e timeout global de 45 min por task.
- **Execução serial**: `max_active_runs=1` evita conflitos de escrita em artefatos durante retreinamento concorrente.
- **Todas as comunicações inter-task via XCom (return_value)**.

---

## 2. Pré-requisitos

- Docker Desktop (macOS/Windows) ou Docker Engine + Docker Compose (Linux)
- Mínimo de **4 GB de RAM** livre recomendado (Postgres + Webserver + Scheduler)
- Portas disponíveis: **5432** (Postgres) e **8080** (Airflow Web UI)

> 💡 **Não é necessário** instalar Airflow localmente via `pip`. A imagem oficial `apache/airflow:2.10.2-python3.11` já contém tudo o que é necessário, e o script de `airflow-init` instala automaticamente `pyarrow`, `scikit-learn`, `joblib`, `nltk` e os corpora NLTK no arranque.

---

## 3. Estrutura de Directórios

```
postech-ml-challenge-fase-3/
├── airflow/
│   ├── dags/
│   │   └── train_urgencia_medica_dag.py   # DAG oficial
│   ├── plugins/
│   │   └── airflow_ml_tasks.py            # Wrappers das tasks Python (100% reutiliza sources/)
│   ├── airflow-home/                      # Configs do Airflow (runtime)
│   ├── airflow-logs/                      # Logs das tasks (montado no container)
│   ├── include/
│   └── scripts/
├── data/                                  # RAW + processed (compartilhado R/W)
├── models/                                # Artefatos .joblib versionados (compartilhado R/W)
├── docs/                                  # Métricas JSON + documentação
├── sources/                               # Módulos ML reutilizados
├── pipelines/                             # Feature pipeline reutilizado
├── .airflow.env                           # UID e credenciais default
└── docker-compose-airflow.yml             # Orquestração Airflow + Postgres
```

---

## 4. Iniciar o Airflow

### 4.1 Passo 1 — Inicializar Base de Dados e Utilizador Admin

Apenas da **primeira vez** (ou após `docker compose down -v`):

```bash
# Windows PowerShell / WSL / Linux
docker compose -f docker-compose-airflow.yml up airflow-init
```

O que faz este passo:
1. `airflow db migrate` — cria todas as tabelas do Airflow no PostgreSQL 16.
2. Cria o utilizador admin padrão (ver credenciais abaixo).
3. Instala `pyarrow`, `scikit-learn`, `joblib`, `nltk` e faz download dos corpora NLTK.

### 4.2 Passo 2 — Subir todos os Serviços

```bash
docker compose -f docker-compose-airflow.yml up -d
```

Serviços iniciados:

| Serviço              | Descrição                                          | Porta  |
| -------------------- | -------------------------------------------------- | ------ |
| `postgres`           | PostgreSQL 16-alpine (metadados do Airflow)       | 5432   |
| `airflow-webserver`  | Interface Web UI + REST API (Flask/gunicorn)       | **8080** |
| `airflow-scheduler`  | Scheduler que orquestra as DAG Runs e Tasks        | 8974   |

### 4.3 Passo 3 — Verificar Healthchecks

Aguarde ~30 segundos e valide:

```bash
docker compose -f docker-compose-airflow.yml ps
```

Todos os serviços devem estar com estado **healthy** (exceto `airflow-init`, que é `Exit 0`).

---

## 5. Credenciais da UI

| Campo     | Valor (apenas para ambiente local) |
| --------- | ---------------------------------- |
| URL       | http://localhost:8080              |
| Username  | `admin`                            |
| Password  | `admin`                            |
| Email     | `admin@example.com`                |
| Role      | `Admin`                            |

> ⚠️ **Altere a password** em `.airflow.env` antes de subir em ambientes partilhados.

---

## 6. Executar a DAG Manualmente

### 6.1 Localizar a DAG

1. Faça login em http://localhost:8080
2. Procure por **`train_urgencia_medica_v1`** (ou use a tag `ml`, `training`, `urgencia-medica`).
3. Certifique-se que o toggle **Pausado/Despausado** (canto superior esquerdo) está **ON** (Não-pausado).

### 6.2 Trigger com Parâmetros Personalizados

Clique em **Trigger DAG w/ config** (ícone ▶️⚙️ ao lado do nome) e cole um JSON, por exemplo:

```json
{
  "model_name": "logistic_regression",
  "lemmatize": true,
  "remove_stopwords": true,
  "tfidf_max_features": 10000,
  "tfidf_ngram_range": [1, 2],
  "min_f1_macro": 0.60,
  "min_accuracy": 0.60
}
```

### 6.3 Descrição dos Parâmetros

| Parâmetro              | Tipo     | Default              | Descrição                                                                 |
| ---------------------- | -------- | -------------------- | ------------------------------------------------------------------------- |
| `model_name`           | string   | `"logistic_regression"` | Classificador sklearn: `logistic_regression`, `naive_bayes`, `linear_svc`, `random_forest` |
| `lemmatize`            | boolean  | `true`               | Aplicar WordNet Lemmatization (NLTK) — mais lento mas tipicamente +2pp de F1 |
| `remove_stopwords`     | boolean  | `true`               | Remover stopwords em inglês antes da vetorização                          |
| `tfidf_max_features`   | int      | `10000`              | Tamanho máximo do vocabulário TF-IDF                                      |
| `tfidf_ngram_range`    | list[int]| `[1, 2]`             | Unigramas + Bigramas (ou `[1, 1]` só unigramas)                           |
| `min_f1_macro`         | float    | `0.50`               | F1-macro MÍNIMO no teste para persistir o artefato (ShortCircuit)         |
| `min_accuracy`         | float    | `0.50`               | Acurácia MÍNIMA no teste para persistir o artefato (ShortCircuit)         |

### 6.4 Trigger Rápido (Sem Parâmetros)

Clique em **Trigger DAG** (ícone ▶️) para usar todos os defaults.

---

## 7. Monitorar a Execução

### 7.1 Vista Principal

- **Grid View** — histórico de runs e status das tasks por run.
- **Graph View** — diagrama de dependências das 6 tasks em tempo real.
- **Task Duration / Gantt** — tempos de execução por task e timeline.

### 7.2 Ver Logs de uma Task

1. Abra **Graph View** → clique numa task → **Log**.
2. São impressas informações detalhadas:
   - `setup_dirs` — diretórios inicializados
   - `data_load_validate` — n. amostras, distribuição alvo, relatório de validação
   - `preprocess_and_split` — tamanhos train/val/test, parâmetros de limpeza
   - `build_features_and_train` — hiperparâmetros TF-IDF, treino, val_acc, **test_acc / F1-macro**
   - `validate_metrics_thresholds` — comparação com thresholds, aprovado/skipped
   - `persist_model_and_metrics` — caminhos dos artefatos salvos (versão + latest)

### 7.3 Ver XComs (Retornos Inter-Task)

Graph View → clique numa task → **XCom**. O campo `return_value` contém o dicionário completo que cada task devolve, e que é consumido pelas tasks a jusante via `ti.xcom_pull()`.

---

## 8. Artefatos Gerados

Após cada execução **aprovada** pelo ShortCircuit, são criados:

### 8.1 Modelos (em `models/`)

| Ficheiro                                                             | Descrição                                                |
| -------------------------------------------------------------------- | -------------------------------------------------------- |
| `urgency_classifier__<DAG_ID>__<RUN_ID>__<YYYYMMDD_HHMMSS>.joblib`  | **Artefato versionado e imutável** (nunca é apagado)    |
| `urgency_classifier.joblib`                                          | **Latest / Deploy** — cópia da última versão aprovada   |
| `tfidf_vectorizer.joblib`                                            | Vectorizer TF-IDF da última run                          |

Cada `.joblib` versionado contém:
```python
{
  "model": Pipeline([("tfidf", TfidfVectorizer), ("classifier", LogisticRegression)]),
  "metadata": { accuracy, f1_macro, n_train, target_counts, cleaning_params, airflow_run_id, ... },
  "metrics_full": { classification_report, confusion_matrix, roc_auc_ovr, ... },
  "dataset_info": {...},
  "preprocessing_info": {...}
}
```

### 8.2 Métricas (em `docs/`)

| Ficheiro                                                         | Descrição                                       |
| ---------------------------------------------------------------- | ------------------------------------------------- |
| `model_metrics__<DAG_ID>__<RUN_ID>__<YYYYMMDD_HHMMSS>.json`      | Registo histórico imutável das métricas da run    |
| `model_metrics.json`                                              | Cópia das métricas da última run aprovada (latest)|

> 💡 Esta separação permite **auditoria e rollback** — se uma nova run tiver métricas inferiores, basta restaurar um `.joblib` antigo renomeando-o para `urgency_classifier.joblib` (ou alterar a FastAPI para carregar um nome específico via env var `MODEL_NAME`).

---

## 9. Retreinamento

A DAG foi concebida especificamente para **retreinamento repetido**. Basta:

1. Adicionar novos dados em `data/raw/` (seguindo o schema original `medical_tc_train.csv` / `medical_tc_test.csv`).
2. Fazer **Trigger DAG** (opcionalmente com `min_f1_macro` / `min_accuracy` mais elevados).
3. Se o modelo passar no ShortCircuit, um novo `.joblib` versionado é criado e `urgency_classifier.joblib` (usado pela FastAPI) é atualizado automaticamente.
4. Reinicie a FastAPI (ou o container `api`) para carregar o novo modelo latest.

**Nenhum artefato anterior é apagado** — é sempre possível voltar atrás.

---

## 10. Tratamento de Falhas e Retries

| Mecanismo                | Configuração                                 |
| ------------------------ | -------------------------------------------- |
| Retries por task         | `2` retries (3 tentativas totais)            |
| Delay inicial retry      | `2` minutos (retry_delay)                    |
| Backoff                  | Exponencial (retry_exponential_backoff=True) |
| Delay máximo retry       | `5` minutos (max_retry_delay)                |
| Timeout por task         | `45` minutos (execution_timeout)             |
| Concorrência máxima      | `1` run ativa (`max_active_runs=1`)          |
| ShortCircuit thresholds  | `min_accuracy` + `min_f1_macro` via conf     |

Se uma task falhar definitivamente após os 2 retries, a DAG inteira marca `Failed` e **nenhum artefato é persistido** (a persistência só ocorre na última task, a jusante do ShortCircuit).

---

## 11. Parar e Reiniciar

### Parar (manter volumes e dados)
```bash
docker compose -f docker-compose-airflow.yml down
```

### Parar e APAGAR TUDO (DB + logs!)
```bash
docker compose -f docker-compose-airflow.yml down -v
```
> ⚠️ Com `-v` apaga também o volume `postgres-db-volume`, perdendo-se o histórico de todas as DAG Runs. Os artefatos em `models/`, `data/` e `docs/` NÃO são apagados (montados como bind mounts locais).

### Reiniciar serviço individual
```bash
docker compose -f docker-compose-airflow.yml restart airflow-scheduler
```

---

## 12. Validação Rápida (Sem Docker Airflow)

Se pretender validar a lógica de todas as tasks **sem** subir o Airflow, use o script de debug standalone (opcional, para desenvolvimento):

```bash
# Instale dependências normais do projeto primeiro (uv sync)
python -X utf8 airflow/scripts/_debug_standalone_dag.py   # (ver notas abaixo)
```

Este wrapper executa todas as 6 tasks em contexto local fake, simulando `ti.xcom_pull/xcom_push`, `dag_run.conf`, `run_id`, etc., e valida que os artefatos versionados são criados corretamente em `models/` e `docs/`.

> ✅ Resultado esperado de referência (LogReg, TF-IDF 5k, sem lemma): **test_acc ≈ 0.69, F1-macro ≈ 0.67, ROC-AUC ≈ 0.84** em 2.888 amostras de teste.

---

## 13. Troubleshooting

| Sintoma                                                 | Causa provável + correção                                                                 |
| ------------------------------------------------------- | ----------------------------------------------------------------------------------------- |
| **Erro 503 Webserver** imediatamente após `up -d`      | Ainda a inicializar. Aguarde ~30s, healthcheck retorna 200 em até 1 minuto.               |
| **DAG não aparece na UI depois de 2 min**               | Validar `docker compose logs airflow-scheduler`. Procure por `DagBag import error`.       |
| **`FileNotFound` em `data/raw/`**                       | Garanta que `data/raw/` contém `medical_tc_train.csv`, `medical_tc_test.csv`, `medical_tc_labels.csv`. |
| **Task `preprocess_and_split` demora > 10 min**         | É esperado com `lemmatize: true` em 14k amostras (CPU-bound). Use `lemmatize: false` em ambientes de teste/dev. |
| **ShortCircuit salta `persist_model_and_metrics`**      | Modelo abaixo dos thresholds mínimos. Reduza `min_f1_macro` / `min_accuracy` no JSON do trigger, ou melhore features/modelo. |
| **`permission denied` ao escrever em models/docs**      | Em Linux: ajuste `AIRFLOW_UID` no `.airflow.env` para o seu UID local (`id -u`).          |

---

*Documentação oficial do Airflow 2.x: https://airflow.apache.org/docs/apache-airflow/2.10.2/*
