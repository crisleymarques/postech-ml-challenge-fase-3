from __future__ import annotations

import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
AIRFLOW_PLUGINS = PROJECT_ROOT / "airflow" / "plugins"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(AIRFLOW_PLUGINS) not in sys.path:
    sys.path.insert(0, str(AIRFLOW_PLUGINS))

from airflow import DAG
from airflow.operators.python import PythonOperator, ShortCircuitOperator
from airflow.exceptions import AirflowSkipException, AirflowFailException

from airflow_ml_tasks import (
    task_setup_dirs,
    task_load_and_validate_data,
    task_preprocess_and_split,
    task_build_features_and_train,
    task_persist_model_and_metrics,
)


DEFAULT_ARGS: Dict[str, Any] = {
    "owner": "ml-team",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=2),
    "retry_exponential_backoff": True,
    "execution_timeout": timedelta(minutes=45),
    "max_retry_delay": timedelta(minutes=5),
}


def _fail_if_f1_below_threshold(**context) -> bool:
    ti = context["ti"]
    train_info = ti.xcom_pull(task_ids="build_features_and_train", key="return_value")
    if not train_info:
        raise AirflowFailException("Sem métricas do build_features_and_train.")

    dag_run_conf = getattr(context.get("dag_run"), "conf", None) or {}
    min_f1_macro = float(dag_run_conf.get("min_f1_macro", 0.50))
    min_accuracy = float(dag_run_conf.get("min_accuracy", 0.50))
    f1 = float(train_info["metrics"].get("f1_macro", 0.0))
    acc = float(train_info["metrics"].get("accuracy", 0.0))

    print(f"Acurácia: {acc:.4f} / F1-macro: {f1:.4f}")
    print(f"Thresholds: min_accuracy={min_accuracy:.4f}, min_f1_macro={min_f1_macro:.4f}")

    if f1 < min_f1_macro or acc < min_accuracy:
        raise AirflowSkipException(
            f"Modelo abaixo do threshold (acc={acc:.4f} < {min_accuracy:.4f} ou "
            f"f1={f1:.4f} < {min_f1_macro:.4f}). Não persistindo novo artefato."
        )
    return True


with DAG(
    dag_id="train_urgencia_medica_v1",
    description=(
        "Treinamento/retreinamento completo do classificador de urgência em laudos médicos. "
        "Pipeline: carregamento/validação → limpeza/split → TF-IDF + treino → persistência do artefato."
    ),
    default_args=DEFAULT_ARGS,
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    max_active_runs=1,
    concurrency=1,
    tags=["ml", "training", "urgencia-medica", "nlp", "fastapi"],
    doc_md=(
        """# DAG: Treinamento / Retreinamento de Classificador de Urgência

Esta DAG reutiliza as funções de `sources/` e `pipelines/` do projeto
(sem duplicar código) para executar um pipeline ML completo:

## Passo a passo
1. **setup_dirs** — Garante `data/`, `models/`, `docs/`, `data/processed/`
2. **data_load_validate** — Carrega datasets brutos de `data/raw`, aplica validação
   + mapeamento de urgência (0 normal / 1 atenção / 2 urgente). Salva parquet validado.
3. **preprocess_and_split** — Limpeza de texto (lowercase, stopwords, lemmatização opcional)
   + split 70/10/20 estratificado, salva CSVs.
4. **build_features_and_train** — TF-IDF fit em treino (10k features, 1-2 gram) e
   treina o classificador default `logistic_regression`. Avalia em teste com
   acurácia, F1, matriz de confusão, relatório de classificação.
5. **_validate_thresholds** (ShortCircuit) — Aborta persistência se F1/Acc abaixo do mínimo configurável.
6. **persist_model_and_metrics** — Salva o pipeline completo como
   `models/urgency_classifier__<run_id>__<timestamp>.joblib` + cópia para
   `urgency_classifier.joblib` (último modelo deployável). Salva métricas em
   `docs/model_metrics*.json`.

## Parâmetros aceitos na Trigger com JSON (conf):
```json
{
  "model_name": "logistic_regression",
  "lemmatize": true,
  "remove_stopwords": true,
  "tfidf_max_features": 10000,
  "tfidf_ngram_range": [1, 2],
  "tfidf_sublinear_tf": true,
  "min_f1_macro": 0.5,
  "min_accuracy": 0.5
}
```

## Retreinamento
Basta disparar a DAG novamente (Trigger DAG) para gerar um novo artefato,
versão e timestamp, sem sobrescrever artefatos antigos. `urgency_classifier.joblib`
é atualizado sempre para a última versão aprovada.
"""
    ),
    params={
        "model_name": "logistic_regression",
        "lemmatize": True,
        "remove_stopwords": True,
        "tfidf_max_features": 10000,
        "min_f1_macro": 0.50,
        "min_accuracy": 0.50,
    },
) as dag:

    setup_dirs_task = PythonOperator(
        task_id="setup_dirs",
        python_callable=task_setup_dirs,
    )

    data_load_task = PythonOperator(
        task_id="data_load_validate",
        python_callable=task_load_and_validate_data,
    )

    preprocess_split_task = PythonOperator(
        task_id="preprocess_and_split",
        python_callable=task_preprocess_and_split,
    )

    train_task = PythonOperator(
        task_id="build_features_and_train",
        python_callable=task_build_features_and_train,
    )

    validate_thresholds_task = ShortCircuitOperator(
        task_id="validate_metrics_thresholds",
        python_callable=_fail_if_f1_below_threshold,
        ignore_downstream_trigger_rules=False,
        doc="Verifica se F1-macro e Acc ultrapassam os thresholds mínimos antes de persistir.",
    )

    persist_task = PythonOperator(
        task_id="persist_model_and_metrics",
        python_callable=task_persist_model_and_metrics,
    )

    (
        setup_dirs_task
        >> data_load_task
        >> preprocess_split_task
        >> train_task
        >> validate_thresholds_task
        >> persist_task
    )
