import pandas as pd
from pathlib import Path
from typing import Tuple

from sources.config import (
    TRAIN_FILE,
    TEST_FILE,
    LABELS_FILE,
    TEXT_COLUMN,
    TARGET_COLUMN,
    CONDITION_TO_URGENCY,
    URGENCY_LABEL_TO_NAME,
    URGENCY_COLUMN,
    URGENCY_NAME_COLUMN,
    CONDITION_LABEL_TO_NAME,
    DATA_PROCESSED_DIR,
)


def load_labels() -> pd.DataFrame:
    labels_df = pd.read_csv(LABELS_FILE)
    labels_df.columns = labels_df.columns.str.strip()
    return labels_df


def load_raw_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_df = pd.read_csv(TRAIN_FILE)
    test_df = pd.read_csv(TEST_FILE)

    train_df.columns = train_df.columns.str.strip()
    test_df.columns = test_df.columns.str.strip()

    return train_df, test_df


def load_combined_raw() -> pd.DataFrame:
    train_df, test_df = load_raw_data()
    train_df["split"] = "train"
    test_df["split"] = "test"
    combined = pd.concat([train_df, test_df], ignore_index=True)
    return combined


def map_condition_to_urgency(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df[URGENCY_COLUMN] = df[TARGET_COLUMN].map(CONDITION_TO_URGENCY)
    df[URGENCY_NAME_COLUMN] = df[URGENCY_COLUMN].map(URGENCY_LABEL_TO_NAME)
    df["condition_name"] = df[TARGET_COLUMN].map(CONDITION_LABEL_TO_NAME)
    return df


def validate_dataframe(df: pd.DataFrame) -> dict:
    report = {}
    report["shape"] = df.shape
    report["nulls_per_column"] = df.isnull().sum().to_dict()
    report["total_nulls"] = df.isnull().sum().sum()

    if TEXT_COLUMN in df.columns:
        report["text_empty_strings"] = (df[TEXT_COLUMN].str.strip() == "").sum()
        report["text_mean_length"] = df[TEXT_COLUMN].str.len().mean()
        report["text_min_length"] = df[TEXT_COLUMN].str.len().min()
        report["text_max_length"] = df[TEXT_COLUMN].str.len().max()

    if TARGET_COLUMN in df.columns:
        report["target_distribution"] = df[TARGET_COLUMN].value_counts().to_dict()

    if URGENCY_COLUMN in df.columns:
        report["urgency_distribution"] = (
            df[URGENCY_COLUMN].value_counts().sort_index().to_dict()
        )
        report["urgency_by_name"] = (
            df[URGENCY_NAME_COLUMN].value_counts().to_dict()
        )

    report["min_samples_per_class"] = (
        df[TARGET_COLUMN].value_counts().min() if TARGET_COLUMN in df.columns else None
    )
    report["meets_min_samples"] = (
        df.shape[0] >= 2000 if TARGET_COLUMN in df.columns else False
    )

    return report


def save_processed_data(
    train_processed: pd.DataFrame,
    test_processed: pd.DataFrame,
    prefix: str = "medical_tc",
) -> Tuple[Path, Path]:
    train_path = DATA_PROCESSED_DIR / f"{prefix}_train_processed.csv"
    test_path = DATA_PROCESSED_DIR / f"{prefix}_test_processed.csv"

    train_processed.to_csv(train_path, index=False)
    test_processed.to_csv(test_path, index=False)

    return train_path, test_path


def load_processed_data(prefix: str = "medical_tc") -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_path = DATA_PROCESSED_DIR / f"{prefix}_train_processed.csv"
    test_path = DATA_PROCESSED_DIR / f"{prefix}_test_processed.csv"

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    return train_df, test_df


if __name__ == "__main__":
    print("Carregando dados brutos...")
    train, test = load_raw_data()
    print(f"Treino: {train.shape}, Teste: {test.shape}")

    print("\nValidando dados de treino:")
    report = validate_dataframe(train)
    for k, v in report.items():
        print(f"  {k}: {v}")

    print("\nAplicando mapeamento de urgência...")
    train_mapped = map_condition_to_urgency(train)
    test_mapped = map_condition_to_urgency(test)

    print("\nDistribuição de urgência (treino):")
    print(train_mapped[URGENCY_NAME_COLUMN].value_counts())

    print("\nSalvando dados processados...")
    save_processed_data(train_mapped, test_mapped)
    print("Dados salvos em data/processed/")
