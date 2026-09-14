from __future__ import annotations

import sys
import os
import json
import shutil
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline as SkPipeline

from sources.config import (
    URGENCY_LABEL_TO_NAME,
    DATA_RAW_DIR as RAW_DATA_DIR,
    DATA_PROCESSED_DIR as PROCESSED_DATA_DIR,
    MODELS_DIR,
    DOCS_DIR,
    setup_dirs,
)
from sources.data_loader import (
    load_combined_raw,
    validate_dataframe,
    map_condition_to_urgency,
)
from sources.preprocessing import (
    clean_dataframe,
)
from pipelines.feature_pipeline import (
    split_dataframe,
    save_vectorizer,
    save_splits_dataframe,
    create_tfidf_vectorizer,
    fit_tfidf_vectorizer,
)
from sources.model import (
    create_classifier,
    train_model,
    evaluate_model,
)

logger = logging.getLogger(__name__)


def _json_default(o: Any) -> Any:
    if isinstance(o, (datetime, Path)):
        return str(o)
    if isinstance(o, set):
        return sorted(list(o))
    raise TypeError("Unserializable {}".format(type(o).__name__))


def task_setup_dirs(**context) -> Dict[str, Any]:
    setup_dirs()
    info: Dict[str, Any] = {
        "raw_data_dir": str(RAW_DATA_DIR),
        "processed_data_dir": str(PROCESSED_DATA_DIR),
        "models_dir": str(MODELS_DIR),
        "docs_dir": str(DOCS_DIR),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    logger.info("Diretorios inicializados.")
    return info


def task_load_and_validate_data(**context) -> Dict[str, Any]:
    logger.info("Carregando dataset bruto...")
    df_raw = load_combined_raw()
    logger.info("Dataset carregado: %s amostras.", len(df_raw))

    report = validate_dataframe(df_raw)
    logger.info("Validacao: %s amostras, nulls=%s, min_samples_per_class=%s, meets_min=%s",
                report.get("shape"), report.get("total_nulls"),
                report.get("min_samples_per_class"), report.get("meets_min_samples"))

    df_mapped = map_condition_to_urgency(df_raw)
    n = len(df_mapped)
    counts = df_mapped["urgency_label"].value_counts().sort_index().to_dict()
    mapping = {int(k): URGENCY_LABEL_TO_NAME[int(k)] for k in counts.keys()}

    out_csv = PROCESSED_DATA_DIR / "dataset_validated.parquet"
    df_mapped.to_parquet(out_csv, index=False)
    logger.info("Dataset validado salvo em %s.", out_csv)

    return {
        "n_samples": n,
        "n_target_classes": int(df_mapped["urgency_label"].nunique()),
        "target_counts": {str(k): int(v) for k, v in counts.items()},
        "target_names": mapping,
        "columns": list(df_mapped.columns),
        "validated_dataset_path": str(out_csv),
        "nulls_per_column": {c: int(df_mapped[c].isnull().sum()) for c in df_mapped.columns},
        "validation_report": {k: v for k, v in report.items() if isinstance(v, (int, float, str, bool, list, dict, tuple))},
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def task_preprocess_and_split(**context) -> Dict[str, Any]:
    ti = context["ti"]
    data_info = ti.xcom_pull(task_ids="data_load_validate", key="return_value")
    if not data_info:
        raise RuntimeError("XCom de data_load_validate vazio.")

    validated_path = Path(data_info["validated_dataset_path"])
    if not validated_path.exists():
        raise FileNotFoundError("{} nao existe.".format(validated_path))

    df = pd.read_parquet(validated_path)
    logger.info("Dataset carregado (%s amostras).", len(df))

    dag_run_conf = getattr(context.get("dag_run"), "conf", None) or {}
    lemmatize = bool(dag_run_conf.get("lemmatize", True))
    remove_stopwords = bool(dag_run_conf.get("remove_stopwords", True))

    logger.info("Limpeza de texto iniciada (lemmatize=%s, stopwords=%s)...", lemmatize, remove_stopwords)
    df_clean = clean_dataframe(
        df,
        text_col="medical_abstract",
        output_col="medical_abstract_clean",
        lemmatize_flag=lemmatize,
        remove_stopwords_flag=remove_stopwords,
    )
    df_clean["medical_abstract"] = df_clean["medical_abstract_clean"]
    df_clean = df_clean.drop(columns=["medical_abstract_clean"])
    logger.info("Limpeza concluida.")

    splits_raw = split_dataframe(
        df_clean,
        text_col="medical_abstract",
        target_col="urgency_label",
        stratify=True,
        random_state=42,
    )
    df_train = splits_raw["train_df"]
    df_val = splits_raw["val_df"]
    df_test = splits_raw["test_df"]

    saved = save_splits_dataframe(
        {
            "train_df": df_train,
            "val_df": df_val,
            "test_df": df_test,
        },
        prefix="split",
    )

    logger.info(
        "Splits criados: train=%s, val=%s, test=%s.", len(df_train), len(df_val), len(df_test)
    )

    processed_dir = str(PROCESSED_DATA_DIR)
    return {
        "n_train": int(len(df_train)),
        "n_val": int(len(df_val)),
        "n_test": int(len(df_test)),
        "split_paths": {
            "train_csv": str(saved.get("train", Path(processed_dir) / "split_train.csv")),
            "val_csv": str(saved.get("val", Path(processed_dir) / "split_val.csv")),
            "test_csv": str(saved.get("test", Path(processed_dir) / "split_test.csv")),
        },
        "cleaning_params": {
            "remove_stopwords": remove_stopwords,
            "lemmatize": lemmatize,
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def task_build_features_and_train(**context) -> Dict[str, Any]:
    ti = context["ti"]
    prep_info = ti.xcom_pull(task_ids="preprocess_and_split", key="return_value")
    if not prep_info:
        raise RuntimeError("XCom de preprocess_and_split vazio.")

    paths = prep_info["split_paths"]
    df_train = pd.read_csv(paths["train_csv"])
    df_val = pd.read_csv(paths["val_csv"])
    df_test = pd.read_csv(paths["test_csv"])
    X_train_s = df_train["medical_abstract"].fillna("").astype(str)
    y_train = df_train["urgency_label"].astype(int)
    X_val_s = df_val["medical_abstract"].fillna("").astype(str)
    y_val = df_val["urgency_label"].astype(int)
    X_test_s = df_test["medical_abstract"].fillna("").astype(str)
    y_test = df_test["urgency_label"].astype(int)

    dag_run_conf = getattr(context.get("dag_run"), "conf", None) or {}
    model_name = dag_run_conf.get("model_name", "logistic_regression")
    max_features = int(dag_run_conf.get("tfidf_max_features", 10000))
    ngram_range_tuple = tuple(int(x) for x in dag_run_conf.get("tfidf_ngram_range", [1, 2]))

    logger.info("Construindo TF-IDF (max_features=%s, ngram=%s)...", max_features, ngram_range_tuple)
    vec = create_tfidf_vectorizer(max_features=max_features, ngram_range=ngram_range_tuple)
    vec, X_train_vec = fit_tfidf_vectorizer(X_train_s, vectorizer=vec)
    X_val_vec = vec.transform(X_val_s)
    X_test_vec = vec.transform(X_test_s)
    vec_path = save_vectorizer(vec)
    logger.info("TF-IDF fit e salvo em %s.", vec_path)

    logger.info("Treinando modelo %s...", model_name)
    model, train_info = train_model(
        X_train_vec, y_train, model_name=model_name, X_val=X_val_vec, y_val=y_val,
    )
    logger.info("Treino OK. Info: %s", train_info)

    logger.info("Avaliando no conjunto de teste...")
    metrics = evaluate_model(model, X_test_vec, y_test.tolist(), model_name=model_name)
    logger.info("Acc=%.4f, F1-macro=%.4f.", metrics["accuracy"], metrics["f1_macro"])

    _tmp_model = MODELS_DIR / "_airflow_tmp_model_only.joblib"
    joblib.dump({"model": model, "metadata": train_info}, _tmp_model)

    metadata = {
        "model_name": model_name,
        "train_info": train_info,
        "vectorizer_path": str(vec_path),
        "model_tmp_path": str(_tmp_model),
        "tfidf_max_features": max_features,
        "tfidf_ngram_range": list(ngram_range_tuple),
        "airflow_run_id": context["run_id"] if context.get("run_id") else None,
    }
    return {
        "model_name": model_name,
        "metadata": metadata,
        "metrics": metrics,
        "vectorizer_path": str(vec_path),
        "model_tmp_path": str(_tmp_model),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def task_persist_model_and_metrics(**context) -> Dict[str, Any]:
    ti = context["ti"]
    train_info = ti.xcom_pull(task_ids="build_features_and_train", key="return_value")
    prep_info = ti.xcom_pull(task_ids="preprocess_and_split", key="return_value")
    data_info = ti.xcom_pull(task_ids="data_load_validate", key="return_value")
    if not train_info:
        raise RuntimeError("XCom build_features_and_train vazio.")

    vec_path = train_info["vectorizer_path"]
    tmp_model_path = train_info["model_tmp_path"]
    metrics = train_info["metrics"]
    model_name = train_info["model_name"]
    metadata = train_info["metadata"] or {}

    paths = prep_info["split_paths"]

    import joblib as jl
    vec = jl.load(vec_path)
    tmp_artifact = jl.load(tmp_model_path)
    clf = tmp_artifact["model"]

    final_pipeline = SkPipeline(steps=[("tfidf", vec), ("classifier", clf)])

    ts_save = datetime.now(timezone.utc)
    ts_slug = ts_save.strftime("%Y%m%d_%H%M%S")

    dag_run = context.get("dag_run")
    run_id = str(getattr(dag_run, "run_id", None) or f"manual_{ts_slug}")
    dag_id = str(context["dag"].dag_id if context.get("dag") else "unknown_dag")
    safe_run_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in run_id)

    model_filename = "urgency_classifier__{}__{}__{}.joblib".format(dag_id, safe_run_id, ts_slug)
    latest_filename = "urgency_classifier.joblib"
    model_path_custom = MODELS_DIR / model_filename
    model_path_latest = MODELS_DIR / latest_filename

    artifact = {
        "model": final_pipeline,
        "metadata": {
            "model_name": model_name,
            "accuracy": float(metrics.get("accuracy", 0.0)),
            "f1_macro": float(metrics.get("f1_macro", 0.0)),
            "f1_weighted": float(metrics.get("f1_weighted", 0.0)),
            "precision_macro": float(metrics.get("precision_macro", 0.0)),
            "recall_macro": float(metrics.get("recall_macro", 0.0)),
            "n_train": data_info["n_samples"],
            "n_train_split": prep_info["n_train"],
            "n_val_split": prep_info["n_val"],
            "n_test_split": prep_info["n_test"],
            "target_counts": data_info["target_counts"],
            "target_names": data_info["target_names"],
            "cleaning_params": prep_info["cleaning_params"],
            "tfidf_max_features": metadata.get("tfidf_max_features"),
            "tfidf_ngram_range": metadata.get("tfidf_ngram_range"),
            "pipeline_steps": [n for n, _ in final_pipeline.steps],
            "airflow_dag_id": dag_id,
            "airflow_run_id": run_id,
            "airflow_task_ts": context.get("ts"),
            "created_date": ts_save.isoformat(),
        },
        "metrics_full": metrics,
        "dataset_info": data_info,
        "preprocessing_info": prep_info,
    }
    jl.dump(artifact, model_path_custom)
    shutil.copy2(model_path_custom, model_path_latest)
    logger.info("Modelo salvo: %s (e copiado para %s).", model_path_custom, model_path_latest)

    metrics_filename = "model_metrics__{}__{}__{}.json".format(dag_id, safe_run_id, ts_slug)
    metrics_path = DOCS_DIR / metrics_filename
    metrics_latest_path = DOCS_DIR / "model_metrics.json"
    doc_payload = {
        "model_name": model_name,
        "metrics": metrics,
        "artifact": str(model_path_custom),
        "artifact_latest": str(model_path_latest),
        "airflow_dag_id": dag_id,
        "airflow_run_id": run_id,
        "saved_at": ts_save.isoformat(),
        "target_counts": data_info["target_counts"],
        "target_names": data_info["target_names"],
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(doc_payload, f, ensure_ascii=False, indent=2, default=_json_default)
    shutil.copy2(metrics_path, metrics_latest_path)
    logger.info("Metricas salvas: %s.", metrics_path)

    tmp_path = Path(tmp_model_path)
    try:
        if tmp_path.exists():
            tmp_path.unlink()
    except Exception:
        pass

    return {
        "artifact_path": str(model_path_custom),
        "artifact_latest_path": str(model_path_latest),
        "metrics_path": str(metrics_path),
        "metrics_latest_path": str(metrics_latest_path),
        "model_name": model_name,
        "accuracy": float(metrics.get("accuracy", 0.0)),
        "f1_macro": float(metrics.get("f1_macro", 0.0)),
        "saved_at": ts_save.isoformat(),
        "airflow_run_id": run_id,
        "airflow_dag_id": dag_id,
    }
