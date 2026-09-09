import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import pandas as pd
import numpy as np

from sources.model import (
    create_classifier,
    train_model,
    train_full_pipeline,
    evaluate_model,
    predict_text,
    save_model,
    load_model,
    benchmark_models,
    AVAILABLE_MODELS,
)
from pipelines.feature_pipeline import (
    fit_tfidf_vectorizer,
    transform_texts,
)
from sources.config import (
    TEXT_COLUMN,
    URGENCY_COLUMN,
    URGENCY_LABEL_TO_NAME,
)


@pytest.fixture
def sample_data():
    texts_train = pd.Series(
        [
            "acute myocardial infarction chest pain urgent angioplasty ecg st elevation",
            "cardiac arrest cpr resuscitation icu ecg troponin elevated",
            "stroke patient neurological rehabilitation mri brain",
            "brain tumor mri scan biopsy surgical resection oncology",
            "colon cancer chemotherapy oncology treatment urgent metastasis",
            "chronic gastritis proton pump inhibitor dietary changes endoscopy",
            "stomach ulcer endoscopy treatment hospital discharge oral",
            "esophageal varices sclerotherapy cirrhosis management hepatic",
            "diverticulitis antibiotics oral diet rest colon",
            "migraine headache dizziness nausea neurological evaluation",
            "diabetes mellitus insulin therapy blood glucose monitoring",
            "hypertension medication ambulatory blood pressure control",
            "pneumonia antibiotics respiratory infection fever cough",
            "renal failure dialysis creatinine nephrology electrolyte",
        ]
        * 5
    )
    labels_train = pd.Series(
        [2, 2, 1, 2, 2, 0, 0, 0, 0, 1, 1, 1, 1, 1] * 5
    )

    texts_test = pd.Series(
        [
            "acute chest pain radiating left arm myocardial infarction diagnosis",
            "gastrointestinal bleeding esophageal varices cirrhosis patient",
            "headache multiple sclerosis neurological follow up mri",
            "malignant lung tumor chemotherapy radiation urgent treatment",
            "peptic ulcer pantoprazole treatment ambulatory care",
        ]
    )
    labels_test = pd.Series([2, 0, 1, 2, 0])
    return texts_train, labels_train, texts_test, labels_test


@pytest.mark.parametrize("model_name", list(AVAILABLE_MODELS.keys()))
def test_create_classifier_all_models(model_name):
    model = create_classifier(model_name)
    assert model is not None


def test_create_classifier_invalid():
    with pytest.raises(ValueError):
        create_classifier("invalid_model_xyz")


def test_train_and_evaluate_model(sample_data):
    texts_train, labels_train, texts_test, labels_test = sample_data

    vec, X_train = fit_tfidf_vectorizer(texts_train)
    X_test = transform_texts(vec, texts_test)

    model, train_info = train_model(X_train, labels_train, model_name="logistic_regression")
    assert "n_train_samples" in train_info
    assert "model_name" in train_info

    metrics = evaluate_model(model, X_test, labels_test)
    for key in [
        "accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "classification_report",
        "confusion_matrix",
        "confusion_labels",
    ]:
        assert key in metrics

    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["f1_macro"] <= 1.0
    assert isinstance(metrics["confusion_matrix"], list)
    assert len(metrics["confusion_matrix"]) == 3


def test_train_full_pipeline(sample_data):
    texts_train, labels_train, texts_test, labels_test = sample_data

    pipeline = train_full_pipeline(texts_train, labels_train, model_name="logistic_regression")
    assert pipeline is not None

    preds = pipeline.predict(texts_test)
    assert len(preds) == len(texts_test)
    for p in preds:
        assert p in {0, 1, 2}


def test_predict_text_single(sample_data):
    texts_train, labels_train, _, _ = sample_data
    pipeline = train_full_pipeline(texts_train, labels_train)

    result = predict_text(pipeline, "acute myocardial infarction with chest pain")
    assert "predicted_label" in result
    assert "predicted_name" in result
    assert result["predicted_label"] in {0, 1, 2}
    assert result["predicted_name"] in URGENCY_LABEL_TO_NAME.values()

    if "probabilities" in result:
        probs = result["probabilities"]
        assert abs(sum(probs.values()) - 1.0) < 0.01


def test_save_and_load_model(tmp_path, sample_data):
    from sources import config
    original_models = config.MODELS_DIR
    config.MODELS_DIR = tmp_path

    try:
        texts_train, labels_train, texts_test, labels_test = sample_data
        pipeline = train_full_pipeline(texts_train, labels_train, model_name="naive_bayes")
        metadata = {"test": "metadata", "version": 1}

        path = save_model(pipeline, name="test_model", metadata=metadata)
        assert path.exists()

        loaded_pipeline, loaded_meta = load_model(name="test_model")
        assert loaded_meta["test"] == "metadata"
        assert loaded_meta["version"] == 1

        orig_preds = pipeline.predict(texts_test)
        loaded_preds = loaded_pipeline.predict(texts_test)
        np.testing.assert_array_equal(orig_preds, loaded_preds)
    finally:
        config.MODELS_DIR = original_models


def test_benchmark_models(sample_data):
    texts_train, labels_train, texts_test, labels_test = sample_data

    vec, X_train = fit_tfidf_vectorizer(texts_train)
    X_test = transform_texts(vec, texts_test)

    benchmark = benchmark_models(
        X_train, labels_train, X_test, labels_test, model_names=["naive_bayes", "logistic_regression"]
    )
    assert isinstance(benchmark, pd.DataFrame)
    assert "model" in benchmark.columns
    assert "macro_f1" in benchmark.columns
    assert len(benchmark) == 2
