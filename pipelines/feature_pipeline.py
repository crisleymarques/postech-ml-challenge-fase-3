import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
import joblib
from pathlib import Path

from sources.config import (
    TEXT_COLUMN,
    URGENCY_COLUMN,
    URGENCY_LABEL_TO_NAME,
    TFIDF_MAX_FEATURES,
    TFIDF_NGRAM_RANGE,
    TFIDF_MIN_DF,
    TFIDF_MAX_DF,
    TEST_SIZE,
    VAL_SIZE,
    RANDOM_STATE,
    DATA_PROCESSED_DIR,
    MODELS_DIR,
)


def create_tfidf_vectorizer(
    max_features: int = TFIDF_MAX_FEATURES,
    ngram_range: Tuple[int, int] = TFIDF_NGRAM_RANGE,
    min_df: int = TFIDF_MIN_DF,
    max_df: float = TFIDF_MAX_DF,
) -> TfidfVectorizer:
    return TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        sublinear_tf=True,
        norm="l2",
    )


def fit_tfidf_vectorizer(
    texts: pd.Series, vectorizer: TfidfVectorizer = None
) -> Tuple[TfidfVectorizer, Any]:
    if vectorizer is None:
        vectorizer = create_tfidf_vectorizer()
    X = vectorizer.fit_transform(texts)
    return vectorizer, X


def transform_texts(vectorizer: TfidfVectorizer, texts: pd.Series) -> Any:
    return vectorizer.transform(texts)


def split_train_val_test(
    X: Any,
    y: pd.Series,
    test_size: float = TEST_SIZE,
    val_size: float = VAL_SIZE,
    random_state: int = RANDOM_STATE,
    stratify: bool = True,
) -> Dict[str, Any]:
    stratify_param = y if stratify else None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_param,
    )

    if val_size > 0:
        val_size_adjusted = val_size / (1 - test_size)
        stratify_train = y_train if stratify else None
        X_train, X_val, y_train, y_val = train_test_split(
            X_train,
            y_train,
            test_size=val_size_adjusted,
            random_state=random_state,
            stratify=stratify_train,
        )
    else:
        X_val = None
        y_val = None

    result = {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
    }
    return result


def split_dataframe(
    df: pd.DataFrame,
    text_col: str = TEXT_COLUMN,
    target_col: str = URGENCY_COLUMN,
    test_size: float = TEST_SIZE,
    val_size: float = VAL_SIZE,
    random_state: int = RANDOM_STATE,
    stratify: bool = True,
) -> Dict[str, pd.DataFrame]:
    stratify_param = df[target_col] if stratify else None

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_param,
    )

    if val_size > 0:
        val_size_adjusted = val_size / (1 - test_size)
        stratify_train = train_df[target_col] if stratify else None
        train_df, val_df = train_test_split(
            train_df,
            test_size=val_size_adjusted,
            random_state=random_state,
            stratify=stratify_train,
        )
    else:
        val_df = None

    result = {
        "train_df": train_df.reset_index(drop=True),
        "val_df": val_df.reset_index(drop=True) if val_df is not None else None,
        "test_df": test_df.reset_index(drop=True),
    }
    return result


def create_feature_pipeline(
    text_col: str = TEXT_COLUMN,
    max_features: int = TFIDF_MAX_FEATURES,
    ngram_range: Tuple[int, int] = TFIDF_NGRAM_RANGE,
) -> Pipeline:
    tfidf = create_tfidf_vectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
    )
    pipeline = Pipeline(
        steps=[
            (
                "tfidf",
                ColumnTransformer(
                    transformers=[
                        (
                            "text_tfidf",
                            tfidf,
                            text_col,
                        )
                    ]
                ),
            )
        ]
    )
    return pipeline


def save_vectorizer(vectorizer: TfidfVectorizer, name: str = "tfidf_vectorizer") -> Path:
    path = MODELS_DIR / f"{name}.joblib"
    joblib.dump(vectorizer, path)
    return path


def load_vectorizer(name: str = "tfidf_vectorizer") -> TfidfVectorizer:
    path = MODELS_DIR / f"{name}.joblib"
    return joblib.load(path)


def save_splits_dataframe(
    splits: Dict[str, pd.DataFrame],
    prefix: str = "medical_tc",
) -> Dict[str, Path]:
    paths = {}
    for split_name, df in splits.items():
        if df is not None:
            fname = split_name.replace("_df", "")
            path = DATA_PROCESSED_DIR / f"{prefix}_{fname}.csv"
            df.to_csv(path, index=False)
            paths[fname] = path
    return paths


def get_class_distribution(
    y_train: pd.Series,
    y_val: pd.Series = None,
    y_test: pd.Series = None,
) -> pd.DataFrame:
    rows = []
    for name, y in [("train", y_train), ("val", y_val), ("test", y_test)]:
        if y is None:
            continue
        counts = y.value_counts().sort_index()
        for label, count in counts.items():
            rows.append(
                {
                    "split": name,
                    "urgency_label": label,
                    "urgency_name": URGENCY_LABEL_TO_NAME.get(label, str(label)),
                    "count": count,
                    "percentage": round(count / len(y) * 100, 2),
                }
            )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    from sources.data_loader import (
        load_processed_data,
        load_raw_data,
        map_condition_to_urgency,
    )
    from sources.preprocessing import clean_dataframe

    print("Carregando e preparando dados...")
    train_raw, test_raw = load_raw_data()
    df = pd.concat([train_raw, test_raw], ignore_index=True)
    df = map_condition_to_urgency(df)

    print(f"Total de amostras: {len(df)}")
    print("\nDistribuição de urgência original:")
    print(df[URGENCY_COLUMN].value_counts().sort_index())

    print("\nLimpando textos (isso pode demorar)...")
    df_clean = clean_dataframe(df, lemmatize_flag=False)

    print("\nDividindo dados...")
    splits = split_train_val_test(
        X=df_clean[TEXT_COLUMN],
        y=df_clean[URGENCY_COLUMN],
    )

    for k in ["X_train", "X_val", "X_test"]:
        if splits[k] is not None:
            print(f"  {k}: {splits[k].shape[0]} amostras")

    print("\nTreinando TF-IDF...")
    vectorizer, X_train_vec = fit_tfidf_vectorizer(
        splits["X_train"].reset_index(drop=True)
    )
    print(f"Vocabulário TF-IDF: {len(vectorizer.vocabulary_)} termos")

    save_vectorizer(vectorizer)
    print("\nVectorizer salvo em models/tfidf_vectorizer.joblib")
