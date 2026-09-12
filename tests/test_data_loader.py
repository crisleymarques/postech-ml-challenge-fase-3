import pytest
import pandas as pd


from sources.data_loader import (
    load_labels,
    load_raw_data,
    load_combined_raw,
    map_condition_to_urgency,
    validate_dataframe,
    save_processed_data,
    load_processed_data,
)
from sources.config import (
    TEXT_COLUMN,
    TARGET_COLUMN,
    URGENCY_COLUMN,
    URGENCY_NAME_COLUMN,
    CONDITION_TO_URGENCY,
    URGENCY_LABEL_TO_NAME,
)


@pytest.fixture(scope="module")
def raw_data():
    train, test = load_raw_data()
    return train, test


def test_load_labels():
    labels = load_labels()
    assert isinstance(labels, pd.DataFrame)
    assert len(labels) == 5
    assert "condition_label" in labels.columns
    assert "condition_name" in labels.columns


def test_load_raw_data_shape(raw_data):
    train, test = raw_data
    assert isinstance(train, pd.DataFrame)
    assert isinstance(test, pd.DataFrame)
    assert train.shape[0] > 0
    assert test.shape[0] > 0
    assert train.shape[0] + test.shape[0] >= 2000


def test_load_raw_data_columns(raw_data):
    train, _ = raw_data
    assert TARGET_COLUMN in train.columns
    assert TEXT_COLUMN in train.columns


def test_load_combined_raw():
    combined = load_combined_raw()
    assert "split" in combined.columns
    train, test = load_raw_data()
    assert len(combined) == len(train) + len(test)


def test_target_values_in_range(raw_data):
    train, test = raw_data
    all_labels = pd.concat([train[TARGET_COLUMN], test[TARGET_COLUMN]])
    assert set(all_labels.unique()).issubset({1, 2, 3, 4, 5})


def test_no_null_text(raw_data):
    train, test = raw_data
    combined = pd.concat([train, test])
    assert combined[TEXT_COLUMN].isnull().sum() == 0


def test_no_empty_text(raw_data):
    train, test = raw_data
    combined = pd.concat([train, test])
    empty_count = (combined[TEXT_COLUMN].str.strip() == "").sum()
    assert empty_count == 0


def test_map_condition_to_urgency(raw_data):
    train, _ = raw_data
    mapped = map_condition_to_urgency(train.head(20))

    assert URGENCY_COLUMN in mapped.columns
    assert URGENCY_NAME_COLUMN in mapped.columns
    assert "condition_name" in mapped.columns

    for _, row in mapped.iterrows():
        assert CONDITION_TO_URGENCY[row[TARGET_COLUMN]] == row[URGENCY_COLUMN]
        assert row[URGENCY_NAME_COLUMN] in URGENCY_LABEL_TO_NAME.values()


def test_urgency_labels_valid(raw_data):
    train, test = raw_data
    combined = pd.concat([train, test])
    mapped = map_condition_to_urgency(combined)
    urgency_labels = set(mapped[URGENCY_COLUMN].unique())
    assert urgency_labels.issubset({0, 1, 2})


def test_validate_dataframe(raw_data):
    train, _ = raw_data
    report = validate_dataframe(train)

    assert "shape" in report
    assert "nulls_per_column" in report
    assert "total_nulls" in report
    assert "text_empty_strings" in report
    assert "target_distribution" in report
    assert report["meets_min_samples"] is True
    assert report["min_samples_per_class"] > 0


def test_validate_dataframe_with_urgency(raw_data):
    train, _ = raw_data
    mapped = map_condition_to_urgency(train)
    report = validate_dataframe(mapped)
    assert "urgency_distribution" in report
    assert "urgency_by_name" in report


def test_save_and_load_processed_data(tmp_path, raw_data):
    from sources import config
    original_processed = config.DATA_PROCESSED_DIR
    config.DATA_PROCESSED_DIR = tmp_path

    try:
        train, test = raw_data
        train_mapped = map_condition_to_urgency(train.head(50))
        test_mapped = map_condition_to_urgency(test.head(20))

        train_path, test_path = save_processed_data(train_mapped, test_mapped, prefix="test_medical")
        assert train_path.exists()
        assert test_path.exists()

        loaded_train, loaded_test = load_processed_data(prefix="test_medical")
        assert len(loaded_train) == len(train_mapped)
        assert len(loaded_test) == len(test_mapped)
        assert URGENCY_COLUMN in loaded_train.columns
    finally:
        config.DATA_PROCESSED_DIR = original_processed
