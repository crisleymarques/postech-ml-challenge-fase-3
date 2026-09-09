import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import pandas as pd
import numpy as np

from pipelines.feature_pipeline import (
    create_tfidf_vectorizer,
    fit_tfidf_vectorizer,
    transform_texts,
    split_train_val_test,
    split_dataframe,
    create_feature_pipeline,
    save_vectorizer,
    load_vectorizer,
    get_class_distribution,
)
from sources.config import (
    TEXT_COLUMN,
    URGENCY_COLUMN,
    URGENCY_LABEL_TO_NAME,
    RANDOM_STATE,
)


@pytest.fixture
def sample_texts():
    return pd.Series(
        [
            "acute myocardial infarction chest pain urgent angioplasty",
            "chronic gastritis proton pump inhibitor dietary changes",
            "migraine headache dizziness nausea neurological evaluation",
            "stroke patient neurological rehabilitation outpatient follow up",
            "stomach ulcer endoscopy treatment hospital discharge",
            "cardiac arrest cpr resuscitation icu admission",
            "diabetes mellitus insulin therapy blood glucose monitoring",
            "esophageal varices sclerotherapy cirrhosis management",
            "brain tumor mri scan biopsy surgical resection",
            "hypertension medication ambulatory blood pressure control",
            "colon cancer chemotherapy oncology treatment urgent",
            "diverticulitis antibiotics oral diet rest",
        ]
    )


@pytest.fixture
def sample_labels():
    return pd.Series([2, 0, 1, 1, 0, 2, 1, 0, 2, 1, 2, 0])


@pytest.fixture
def sample_df(sample_texts, sample_labels):
    return pd.DataFrame(
        {
            TEXT_COLUMN: sample_texts,
            URGENCY_COLUMN: sample_labels,
        }
    )


def test_create_tfidf_vectorizer():
    vec = create_tfidf_vectorizer(max_features=100)
    assert vec is not None
    assert vec.max_features == 100


def test_fit_and_transform_tfidf(sample_texts):
    vec, X = fit_tfidf_vectorizer(sample_texts)
    assert X.shape[0] == len(sample_texts)
    assert X.shape[1] > 0

    new_texts = pd.Series(["chest pain angioplasty", "gastritis ulcer"])
    X_new = transform_texts(vec, new_texts)
    assert X_new.shape[0] == 2
    assert X_new.shape[1] == X.shape[1]


def test_split_train_val_test_no_val(sample_texts, sample_labels):
    splits = split_train_val_test(
        sample_texts, sample_labels, test_size=0.3, val_size=0
    )
    assert splits["X_val"] is None
    assert splits["y_val"] is None
    assert len(splits["X_train"]) + len(splits["X_test"]) == len(sample_texts)


def test_split_train_val_test_with_val(sample_texts, sample_labels):
    splits = split_train_val_test(
        sample_texts, sample_labels, test_size=0.3, val_size=0.1, random_state=RANDOM_STATE
    )
    total = (
        len(splits["X_train"])
        + len(splits["X_val"])
        + len(splits["X_test"])
    )
    assert total == len(sample_texts)
    assert len(splits["X_val"]) > 0
    assert len(splits["X_test"]) > 0


def test_split_train_val_test_stratify_preserves_classes(sample_texts, sample_labels):
    splits = split_train_val_test(
        sample_texts, sample_labels, test_size=0.3, val_size=0.1, stratify=True
    )
    train_classes = set(pd.Series(splits["y_train"]).unique())
    test_classes = set(pd.Series(splits["y_test"]).unique())
    all_classes = set(sample_labels.unique())
    assert test_classes.issubset(all_classes)


def test_split_dataframe(sample_df):
    splits = split_dataframe(sample_df, test_size=0.3, val_size=0.1, stratify=True)
    assert "train_df" in splits
    assert "val_df" in splits
    assert "test_df" in splits
    assert splits["train_df"] is not None
    assert splits["val_df"] is not None
    assert splits["test_df"] is not None
    assert (
        len(splits["train_df"]) + len(splits["val_df"]) + len(splits["test_df"])
        == len(sample_df)
    )
    for col in [TEXT_COLUMN, URGENCY_COLUMN]:
        assert col in splits["train_df"].columns


def test_split_dataframe_no_val(sample_df):
    splits = split_dataframe(sample_df, test_size=0.3, val_size=0)
    assert splits["val_df"] is None


def test_create_feature_pipeline(sample_df):
    pipeline = create_feature_pipeline()
    assert pipeline is not None
    pipeline.fit(sample_df[[TEXT_COLUMN]])
    transformed = pipeline.transform(sample_df[[TEXT_COLUMN]])
    assert transformed.shape[0] == len(sample_df)


def test_save_and_load_vectorizer(tmp_path, sample_texts):
    from sources import config
    original_models = config.MODELS_DIR
    config.MODELS_DIR = tmp_path

    try:
        vec, _ = fit_tfidf_vectorizer(sample_texts)
        path = save_vectorizer(vec, name="test_vec")
        assert path.exists()

        loaded_vec = load_vectorizer(name="test_vec")
        X_orig = transform_texts(vec, sample_texts)
        X_load = transform_texts(loaded_vec, sample_texts)
        assert X_orig.shape == X_load.shape
        np.testing.assert_allclose(X_orig.toarray(), X_load.toarray(), atol=1e-7)
    finally:
        config.MODELS_DIR = original_models


def test_get_class_distribution(sample_labels):
    df = get_class_distribution(
        y_train=sample_labels,
        y_val=sample_labels,
        y_test=sample_labels,
    )
    assert isinstance(df, pd.DataFrame)
    assert "split" in df.columns
    assert "urgency_name" in df.columns
    assert "count" in df.columns
    assert "percentage" in df.columns
    assert set(df["split"]) == {"train", "val", "test"}
    for name in df["urgency_name"]:
        assert name in URGENCY_LABEL_TO_NAME.values()
