import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import numpy as np

from sources.data_loader import (
    load_raw_data,
    load_labels,
    map_condition_to_urgency,
    validate_dataframe,
    save_processed_data,
)
from sources.preprocessing import clean_dataframe
from sources.config import (
    TEXT_COLUMN,
    URGENCY_COLUMN,
    URGENCY_NAME_COLUMN,
    MODELS_DIR,
    DOCS_DIR,
    URGENCY_MAPPING_DOC,
)
from pipelines.feature_pipeline import (
    split_dataframe,
    split_train_val_test,
    fit_tfidf_vectorizer,
    transform_texts,
    save_vectorizer,
    save_splits_dataframe,
    get_class_distribution,
)
from sources.model import (
    train_full_pipeline,
    train_model,
    evaluate_model,
    predict_text,
    save_model,
    benchmark_models,
)


def step_1_load_and_validate():
    print("=" * 60)
    print("PASSO 1: Carregamento e Validação do Dataset")
    print("=" * 60)

    labels_df = load_labels()
    print("\nLabels do dataset original:")
    print(labels_df.to_string(index=False))

    train_raw, test_raw = load_raw_data()
    print(f"\nArquivos carregados:")
    print(f"  - Treino: {train_raw.shape[0]} amostras x {train_raw.shape[1]} colunas")
    print(f"  - Teste:  {test_raw.shape[0]} amostras x {test_raw.shape[1]} colunas")
    print(f"  - Total:  {train_raw.shape[0] + test_raw.shape[0]} amostras")

    print("\nColunas encontradas:")
    print(f"  - Texto:  '{TEXT_COLUMN}'")
    print(f"  - Target: 'condition_label'")

    report = validate_dataframe(pd.concat([train_raw, test_raw], ignore_index=True))
    print(f"\nValidação:")
    print(f"  - Valores nulos: {report['total_nulls']}")
    print(f"  - Textos vazios: {report['text_empty_strings']}")
    print(f"  - Tamanho médio do texto: {report['text_mean_length']:.0f} caracteres")
    print(f"  - Amostras >= 2000: {report['meets_min_samples']} ({report['shape'][0]} total)")

    print("\nDistribuição das classes originais:")
    for label, count in sorted(report["target_distribution"].items()):
        pct = count / report["shape"][0] * 100
        print(f"  - Classe {label}: {count} ({pct:.1f}%)")

    return train_raw, test_raw


def step_2_map_urgency(train_raw, test_raw):
    print("\n" + "=" * 60)
    print("PASSO 2: Mapeamento para Classes de Urgência")
    print("=" * 60)

    print(URGENCY_MAPPING_DOC)

    train_mapped = map_condition_to_urgency(train_raw)
    test_mapped = map_condition_to_urgency(test_raw)

    combined = pd.concat([train_mapped, test_mapped], ignore_index=True)
    print("\nDistribuição resultante de urgência:")
    dist = combined[URGENCY_NAME_COLUMN].value_counts()
    total = len(combined)
    for name, count in dist.items():
        print(f"  - {name.upper()}: {count} ({count/total*100:.1f}%)")

    train_path, test_path = save_processed_data(train_mapped, test_mapped)
    print(f"\nDados processados salvos:")
    print(f"  - Treino: {train_path}")
    print(f"  - Teste:  {test_path}")

    return train_mapped, test_mapped


def step_3_preprocess(train_mapped, test_mapped, lemmatize: bool = False):
    print("\n" + "=" * 60)
    print("PASSO 3: Limpeza e Pré-processamento de Texto")
    print("=" * 60)

    full_df = pd.concat([train_mapped, test_mapped], ignore_index=True)

    print("\nAplicando limpeza aos textos...")
    print("  - Conversão para minúsculas")
    print("  - Remoção de HTML, URLs, emails")
    print("  - Remoção de pontuação")
    print("  - Remoção de stopwords (inglês)")
    if lemmatize:
        print("  - Lematização")

    clean_df = clean_dataframe(full_df, lemmatize_flag=lemmatize)

    orig_lens = full_df[TEXT_COLUMN].str.len()
    clean_lens = clean_df[TEXT_COLUMN].str.len()
    print(f"\nComparação de tamanho de texto:")
    print(f"  - Original: média {orig_lens.mean():.0f} char (min {orig_lens.min()}, max {orig_lens.max()})")
    print(f"  - Limpo:    média {clean_lens.mean():.0f} char (min {clean_lens.min()}, max {clean_lens.max()})")
    print(f"  - Redução:  {(1 - clean_lens.mean()/orig_lens.mean())*100:.1f}%")

    return clean_df


def step_4_split_and_features(clean_df):
    print("\n" + "=" * 60)
    print("PASSO 4: Divisão Treino/Validação/Teste + TF-IDF")
    print("=" * 60)

    splits_df = split_dataframe(clean_df)
    split_paths = save_splits_dataframe(splits_df)

    for k in ["train", "val", "test"]:
        df = splits_df[f"{k}_df"]
        if df is not None:
            print(f"  - {k.capitalize()}: {len(df)} amostras")

    class_dist = get_class_distribution(
        splits_df["train_df"][URGENCY_COLUMN],
        splits_df["val_df"][URGENCY_COLUMN] if splits_df["val_df"] is not None else None,
        splits_df["test_df"][URGENCY_COLUMN],
    )
    print("\nDistribuição por split:")
    print(class_dist.to_string(index=False))

    print("\nTreinando vetorizador TF-IDF...")
    vectorizer, X_train = fit_tfidf_vectorizer(splits_df["train_df"][TEXT_COLUMN])
    X_val = (
        transform_texts(vectorizer, splits_df["val_df"][TEXT_COLUMN])
        if splits_df["val_df"] is not None
        else None
    )
    X_test = transform_texts(vectorizer, splits_df["test_df"][TEXT_COLUMN])

    print(f"  - Vocabulário: {len(vectorizer.vocabulary_)} termos")
    print(f"  - Shape treino: {X_train.shape}")
    print(f"  - Shape teste:  {X_test.shape}")

    vec_path = save_vectorizer(vectorizer)
    print(f"\nVectorizer salvo em: {vec_path}")

    return splits_df, vectorizer, X_train, X_val, X_test


def step_5_train_and_evaluate(splits_df, X_train, X_val, X_test, model_name="logistic_regression"):
    print("\n" + "=" * 60)
    print(f"PASSO 5: Treinamento e Avaliação ({model_name})")
    print("=" * 60)

    y_train = splits_df["train_df"][URGENCY_COLUMN].values
    y_val = splits_df["val_df"][URGENCY_COLUMN].values if splits_df["val_df"] is not None else None
    y_test = splits_df["test_df"][URGENCY_COLUMN].values

    print(f"\nTreinando modelo {model_name}...")
    model, train_info = train_model(X_train, y_train, model_name, X_val, y_val)

    for k, v in train_info.items():
        if isinstance(v, float):
            print(f"  - {k}: {v:.4f}")
        else:
            print(f"  - {k}: {v}")

    print("\nAvaliando no conjunto de teste...")
    metrics = evaluate_model(model, X_test, y_test, model_name=model_name)

    print(f"\nMétricas Gerais:")
    print(f"  - Accuracy:        {metrics['accuracy']:.4f}")
    print(f"  - Precision (macro): {metrics['precision_macro']:.4f}")
    print(f"  - Recall (macro):    {metrics['recall_macro']:.4f}")
    print(f"  - F1 (macro):        {metrics['f1_macro']:.4f}")
    if "roc_auc_ovr" in metrics:
        print(f"  - ROC AUC (OVR):     {metrics['roc_auc_ovr']:.4f}")

    print("\nRelatório de Classificação:")
    print(metrics["classification_report_str"])

    print("\nMatriz de Confusão:")
    cm = pd.DataFrame(
        metrics["confusion_matrix"],
        index=metrics["confusion_labels"],
        columns=[f"pred_{l}" for l in metrics["confusion_labels"]],
    )
    print(cm.to_string())

    return model, metrics


def step_6_train_and_save_full_pipeline(splits_df, model_name="logistic_regression"):
    print("\n" + "=" * 60)
    print("PASSO 6: Treinando Pipeline Completo (para produção)")
    print("=" * 60)

    train_texts = splits_df["train_df"][TEXT_COLUMN].reset_index(drop=True)
    y_train = splits_df["train_df"][URGENCY_COLUMN].values

    if splits_df["val_df"] is not None:
        train_texts = pd.concat(
            [train_texts, splits_df["val_df"][TEXT_COLUMN].reset_index(drop=True)],
            ignore_index=True,
        )
        y_train = np.concatenate([y_train, splits_df["val_df"][URGENCY_COLUMN].values])

    print(f"Pipeline completo com {len(train_texts)} amostras de treino...")
    full_pipeline = train_full_pipeline(train_texts, y_train, model_name=model_name)

    y_test = splits_df["test_df"][URGENCY_COLUMN].values
    test_texts = splits_df["test_df"][TEXT_COLUMN]
    metrics = evaluate_model(full_pipeline, test_texts, y_test, model_name=f"full_{model_name}")
    print(f"\nPipeline completo - Test F1 (macro): {metrics['f1_macro']:.4f}")

    metadata = {
        "model_name": model_name,
        "n_train": len(train_texts),
        "accuracy": metrics["accuracy"],
        "f1_macro": metrics["f1_macro"],
        "created_date": pd.Timestamp.now().isoformat(),
        "pipeline_steps": [name for name, _ in full_pipeline.steps],
    }
    model_path = save_model(full_pipeline, metadata=metadata)
    print(f"\nModelo salvo em: {model_path}")

    metrics_path = DOCS_DIR / "model_metrics.json"
    DOCS_DIR.mkdir(exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as f:
        serializable = {}
        for k, v in metrics.items():
            if isinstance(v, (np.ndarray, np.integer, np.floating)):
                serializable[k] = v.tolist() if hasattr(v, "tolist") else float(v)
            elif isinstance(v, (int, float, str, list, dict, bool)) or v is None:
                serializable[k] = v
            else:
                serializable[k] = str(v)
        json.dump(serializable, f, indent=2, ensure_ascii=False)
    print(f"Métricas salvas em: {metrics_path}")

    return full_pipeline


def step_7_demo_prediction(full_pipeline):
    print("\n" + "=" * 60)
    print("PASSO 7: Demonstração de Predição")
    print("=" * 60)

    sample_texts = [
        "The patient presented with acute chest pain radiating to the left arm, associated with diaphoresis and dyspnea. ECG showed ST elevation in leads V1-V4. Diagnosis: Anterior wall acute myocardial infarction.",
        "Patient with chronic gastroesophageal reflux disease reports heartburn after meals. Upper endoscopy showed mild esophagitis grade A. Prescribed proton pump inhibitor and dietary modifications.",
        "55-year-old patient with history of hypertension and diabetes complains of persistent headaches and occasional dizziness. Neurological examination was normal. Cranial CT showed no acute abnormalities. Recommended outpatient follow-up.",
    ]

    for i, text in enumerate(sample_texts, 1):
        pred = predict_text(full_pipeline, text)
        print(f"\nExemplo {i} ({pred['predicted_name'].upper()}):")
        print(f"  Texto: {text[:120]}...")
        if "probabilities" in pred:
            print(f"  Probabilidades: {pred['probabilities']}")


def run_full_pipeline(model_name: str = "logistic_regression", lemmatize: bool = False):
    train_raw, test_raw = step_1_load_and_validate()
    train_mapped, test_mapped = step_2_map_urgency(train_raw, test_raw)
    clean_df = step_3_preprocess(train_mapped, test_mapped, lemmatize=lemmatize)
    splits_df, vectorizer, X_train, X_val, X_test = step_4_split_and_features(clean_df)
    model, metrics = step_5_train_and_evaluate(
        splits_df, X_train, X_val, X_test, model_name=model_name
    )
    full_pipeline = step_6_train_and_save_full_pipeline(splits_df, model_name=model_name)
    step_7_demo_prediction(full_pipeline)

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETO EXECUTADO COM SUCESSO!")
    print("=" * 60)
    print("\nArtefatos gerados:")
    print(f"  - models/tfidf_vectorizer.joblib")
    print(f"  - models/urgency_classifier.joblib")
    print(f"  - data/processed/*.csv")
    print(f"  - docs/model_metrics.json")

    return full_pipeline


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pipeline completo de classificação de urgência médica"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="logistic_regression",
        choices=["logistic_regression", "naive_bayes", "random_forest", "linear_svc"],
        help="Modelo para treinar (default: logistic_regression)",
    )
    parser.add_argument(
        "--lemmatize",
        action="store_true",
        help="Aplicar lematização durante pré-processamento",
    )
    args = parser.parse_args()

    run_full_pipeline(model_name=args.model, lemmatize=args.lemmatize)
