import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Optional
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
import joblib
from pathlib import Path

from sources.config import (
    MODELS_DIR,
    RANDOM_STATE,
    URGENCY_LABEL_TO_NAME,
    TEXT_COLUMN,
    URGENCY_COLUMN,
)
from pipelines.feature_pipeline import (
    create_tfidf_vectorizer,
    fit_tfidf_vectorizer,
    transform_texts,
)


AVAILABLE_MODELS = {
    "logistic_regression": lambda: LogisticRegression(
        C=1.0,
        max_iter=1000,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced",
    ),
    "naive_bayes": lambda: MultinomialNB(alpha=0.1),
    "random_forest": lambda: RandomForestClassifier(
        n_estimators=100,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced",
    ),
    "linear_svc": lambda: LinearSVC(
        C=1.0,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        max_iter=5000,
    ),
}


def create_classifier(model_name: str = "logistic_regression"):
    if model_name not in AVAILABLE_MODELS:
        raise ValueError(
            f"Modelo desconhecido: {model_name}. "
            f"Disponíveis: {list(AVAILABLE_MODELS.keys())}"
        )
    return AVAILABLE_MODELS[model_name]()


def train_model(
    X_train,
    y_train,
    model_name: str = "logistic_regression",
    X_val=None,
    y_val=None,
) -> Tuple[Any, Dict[str, Any]]:
    model = create_classifier(model_name)
    model.fit(X_train, y_train)

    training_info = {
        "model_name": model_name,
        "n_train_samples": X_train.shape[0],
        "n_features": X_train.shape[1],
    }

    if X_val is not None and y_val is not None:
        y_val_pred = model.predict(X_val)
        training_info["val_accuracy"] = accuracy_score(y_val, y_val_pred)
        training_info["val_macro_f1"] = f1_score(
            y_val, y_val_pred, average="macro", zero_division=0
        )

    return model, training_info


def train_full_pipeline(
    train_texts: pd.Series,
    y_train: pd.Series,
    model_name: str = "logistic_regression",
) -> Pipeline:
    tfidf = create_tfidf_vectorizer()

    classifier = create_classifier(model_name)

    full_pipeline = Pipeline(
        steps=[
            ("tfidf", tfidf),
            ("classifier", classifier),
        ]
    )

    full_pipeline.fit(train_texts, y_train)
    return full_pipeline


def evaluate_model(
    model,
    X_test,
    y_test,
    model_name: str = "model",
) -> Dict[str, Any]:
    y_pred = model.predict(X_test)

    labels = sorted(np.unique(y_test))
    target_names = [URGENCY_LABEL_TO_NAME.get(l, str(l)) for l in labels]

    results = {
        "model_name": model_name,
        "n_test_samples": X_test.shape[0],
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision_macro": round(
            precision_score(y_test, y_pred, average="macro", zero_division=0), 4
        ),
        "recall_macro": round(
            recall_score(y_test, y_pred, average="macro", zero_division=0), 4
        ),
        "f1_macro": round(
            f1_score(y_test, y_pred, average="macro", zero_division=0), 4
        ),
        "precision_weighted": round(
            precision_score(y_test, y_pred, average="weighted", zero_division=0), 4
        ),
        "recall_weighted": round(
            recall_score(y_test, y_pred, average="weighted", zero_division=0), 4
        ),
        "f1_weighted": round(
            f1_score(y_test, y_pred, average="weighted", zero_division=0), 4
        ),
        "classification_report": classification_report(
            y_test,
            y_pred,
            labels=labels,
            target_names=target_names,
            output_dict=True,
            zero_division=0,
        ),
        "classification_report_str": classification_report(
            y_test,
            y_pred,
            labels=labels,
            target_names=target_names,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(y_test, y_pred, labels=labels).tolist(),
        "confusion_labels": target_names,
    }

    if hasattr(model, "predict_proba"):
        try:
            y_proba = model.predict_proba(X_test)
            results["roc_auc_ovr"] = round(
                roc_auc_score(
                    y_test, y_proba, multi_class="ovr", average="macro"
                ),
                4,
            )
        except Exception as e:
            results["roc_auc_error"] = str(e)

    return results


def predict_text(pipeline: Pipeline, text: str) -> Dict[str, Any]:
    if isinstance(text, str):
        text_series = pd.Series([text])
    else:
        text_series = pd.Series(text)

    y_pred = pipeline.predict(text_series)
    label = int(y_pred[0])

    result = {
        "predicted_label": label,
        "predicted_name": URGENCY_LABEL_TO_NAME.get(label, str(label)),
    }

    if hasattr(pipeline, "predict_proba"):
        proba = pipeline.predict_proba(text_series)[0]
        classes = pipeline.classes_
        probabilities = {}
        for i, cls in enumerate(classes):
            probabilities[URGENCY_LABEL_TO_NAME.get(int(cls), str(cls))] = round(
                float(proba[i]), 4
            )
        result["probabilities"] = probabilities

    return result


def save_model(
    model,
    name: str = "urgency_classifier",
    metadata: Optional[Dict[str, Any]] = None,
) -> Path:
    model_path = MODELS_DIR / f"{name}.joblib"
    artifact = {"model": model, "metadata": metadata or {}}
    joblib.dump(artifact, model_path)
    return model_path


def load_model(name: str = "urgency_classifier") -> Tuple[Any, Dict[str, Any]]:
    model_path = MODELS_DIR / f"{name}.joblib"
    artifact = joblib.load(model_path)
    return artifact["model"], artifact.get("metadata", {})


def benchmark_models(
    X_train,
    y_train,
    X_test,
    y_test,
    model_names: Optional[list] = None,
) -> pd.DataFrame:
    if model_names is None:
        model_names = list(AVAILABLE_MODELS.keys())

    results = []
    for model_name in model_names:
        print(f"Treinando {model_name}...")
        model, _ = train_model(X_train, y_train, model_name=model_name)
        metrics = evaluate_model(model, X_test, y_test, model_name=model_name)
        results.append(
            {
                "model": model_name,
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["f1_macro"],
                "weighted_f1": metrics["f1_weighted"],
            }
        )

    benchmark_df = pd.DataFrame(results).sort_values(
        "macro_f1", ascending=False
    )
    return benchmark_df


if __name__ == "__main__":
    from sources.data_loader import load_processed_data
    from sources.preprocessing import clean_dataframe
    from pipelines.feature_pipeline import (
        split_dataframe,
        fit_tfidf_vectorizer,
        transform_texts,
    )

    print("Benchmark de modelos...")
    train_df, test_df = load_processed_data()
    full_df = pd.concat([train_df, test_df], ignore_index=True)

    print(f"Limpando textos...")
    clean_df = clean_dataframe(full_df, lemmatize_flag=False)

    splits = split_dataframe(clean_df)

    y_train = splits["train_df"][URGENCY_COLUMN]
    y_test = splits["test_df"][URGENCY_COLUMN]

    vectorizer, X_train = fit_tfidf_vectorizer(splits["train_df"][TEXT_COLUMN])
    X_test = transform_texts(vectorizer, splits["test_df"][TEXT_COLUMN])

    benchmark = benchmark_models(X_train, y_train, X_test, y_test)
    print("\nBenchmark de Modelos:")
    print(benchmark.to_string(index=False))
