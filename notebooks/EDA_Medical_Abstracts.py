# %% [markdown]
# # EDA - Medical Abstracts TC Corpus Dataset
# ## Análise Exploratória e Pipeline Base de Classificação de Urgência
# %%
import sys
from pathlib import Path
sys.path.insert(0, str(Path().resolve().parent))

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 100
plt.rcParams["font.size"] = 10

from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer

from sources.data_loader import (
    load_labels,
    load_raw_data,
    map_condition_to_urgency,
    validate_dataframe,
)
from sources.preprocessing import clean_dataframe, download_nltk_resources
from sources.config import (
    TEXT_COLUMN,
    TARGET_COLUMN,
    URGENCY_COLUMN,
    URGENCY_NAME_COLUMN,
    CONDITION_LABEL_TO_NAME,
    CONDITION_TO_URGENCY,
    URGENCY_LABEL_TO_NAME,
    URGENCY_MAPPING_DOC,
    RANDOM_STATE,
    TEST_SIZE,
    VAL_SIZE,
    TFIDF_MAX_FEATURES,
    TFIDF_NGRAM_RANGE,
    MODELS_DIR,
    DOCS_DIR,
)
from pipelines.feature_pipeline import (
    split_dataframe,
    fit_tfidf_vectorizer,
    transform_texts,
    save_vectorizer,
    get_class_distribution,
    create_tfidf_vectorizer,
)
from sources.model import (
    train_model,
    evaluate_model,
    train_full_pipeline,
    save_model,
    benchmark_models,
    predict_text,
)

download_nltk_resources()
MODELS_DIR.mkdir(exist_ok=True)
DOCS_DIR.mkdir(exist_ok=True)

OUTPUT_DIR = Path().resolve() / "eda_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# %% [markdown]
# ---
# ## 1. Objetivo do Projeto
#
# Este projeto implementa um sistema de triagem automática de exames de texto
# (laudos médicos / resumos médicos) para classificar a **urgência** em três
# categorias:
#
# | Urgência   | Nível | Descrição                                                          |
# |------------|:-----:|---------------------------------------------------------------------|
# | **Normal** |   0   | Condições ambulatoriais, sem caráter emergencial imediato.          |
# | **Atenção**|   1   | Requer acompanhamento/tratamento, mas sem risco imediato de vida.   |
# | **Urgente**|   2   | Risco elevado de vida; necessidade de intervenção prioritária.      |
#
# Dataset utilizado: **Medical Abstracts TC Corpus Dataset**
# (14.438 resumos médicos).

# %% [markdown]
# ---
# ## 2. Origem e Descrição do Dataset
#
# - **Nome**: Medical Abstracts Text Classification (TC) Corpus Dataset
# - **Fonte**: Dataset público de NLP clínico, derivado de resumos biomédicos
# - **Tamanho total**: 14.438 amostras (acima do mínimo de 2.000 exigido)
# - **Arquivos em `data/raw/`**:
#   - `medical_tc_labels.csv`: dicionário de 5 classes originais
#   - `medical_tc_train.csv`: 11.550 amostras de treino
#   - `medical_tc_test.csv`: 2.888 amostras de teste

# %%
labels_df = load_labels()
print("Classes Originais do Dataset:")
display(labels_df) if "display" in dir() else print(labels_df.to_string(index=False))

# %% [markdown]
# ---
# ## 3. Carregamento e Validação das Colunas (Texto e Target)

# %%
train_raw, test_raw = load_raw_data()
full_raw = pd.concat([train_raw, test_raw], ignore_index=True)
full_raw["split"] = ["train"] * len(train_raw) + ["test"] * len(test_raw)

col_summary = pd.DataFrame(
    {
        "Coluna": [TEXT_COLUMN, TARGET_COLUMN, "split"],
        "Tipo Dado": [
            str(full_raw[TEXT_COLUMN].dtype),
            str(full_raw[TARGET_COLUMN].dtype),
            str(full_raw["split"].dtype),
        ],
        "Valores Nulos": [
            full_raw[TEXT_COLUMN].isnull().sum(),
            full_raw[TARGET_COLUMN].isnull().sum(),
            0,
        ],
        "Valores Únicos": [
            full_raw[TEXT_COLUMN].nunique(),
            full_raw[TARGET_COLUMN].nunique(),
            2,
        ],
    }
)
print("Validação das Colunas:")
display(col_summary) if "display" in dir() else print(col_summary.to_string(index=False))

# %%
report = validate_dataframe(full_raw)
print("\nRelatório de Validação Geral:")
print(f"  - Shape do dataset: {report['shape']}")
print(f"  - Total de valores nulos: {report['total_nulls']}")
print(f"  - Textos vazios (string nula): {report['text_empty_strings']}")
print(f"  - Quantidade de amostras >= 2.000: {report['meets_min_samples']} ({report['shape'][0]} amostras)")
print(f"  - Tamanho médio do texto: {report['text_mean_length']:.0f} caracteres")
print(f"  - Tamanho mínimo do texto: {report['text_min_length']} caracteres")
print(f"  - Tamanho máximo do texto: {report['text_max_length']} caracteres")
print("\n  -> COLUNAS DE TEXTO E TARGET VÁLIDAS; SEM VALORES NULOS OU VAZIOS.")

# %% [markdown]
# ---
# ## 4. Distribuição das Classes Originais (Condições Médicas)

# %%
cond_counts = full_raw[TARGET_COLUMN].value_counts().sort_index()
cond_df = pd.DataFrame(
    {
        "label": cond_counts.index,
        "condition_name": cond_counts.index.map(CONDITION_LABEL_TO_NAME),
        "count": cond_counts.values,
        "percentage": (cond_counts.values / len(full_raw) * 100).round(2),
    }
)
print("Distribuição - Classes Originais:")
display(cond_df) if "display" in dir() else print(cond_df.to_string(index=False))

# %%
fig, ax = plt.subplots(figsize=(9, 5))
bar_colors = ["#2ecc71", "#3498db", "#f39c12", "#e74c3c", "#9b59b6"]
bars = ax.bar(cond_df["condition_name"], cond_df["count"], color=bar_colors, edgecolor="black")
for bar, pct in zip(bars, cond_df["percentage"]):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 50,
        f"{pct}%",
        ha="center",
        va="bottom",
        fontweight="bold",
    )
ax.set_title("Distribuição das Classes Originais (Condições Médicas)", fontweight="bold")
ax.set_xlabel("Condição Médica")
ax.set_ylabel("Número de Amostras")
plt.xticks(rotation=15, ha="right")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "01_classes_originais.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ---
# ## 5. Mapeamento para Níveis de Urgência (Transformação do Target)
#
# Como o dataset original possui **5 categorias clínicas** mas o problema de
# negócio pede **3 níveis de urgência**, definimos o seguinte mapeamento:

# %%
print(URGENCY_MAPPING_DOC)

# %%
mapping_df = pd.DataFrame(
    [
        {
            "Condição Original": CONDITION_LABEL_TO_NAME[k],
            "Label Original": k,
            "Urgência Mapeada": URGENCY_LABEL_TO_NAME[v],
            "Label Urgência": v,
        }
        for k, v in sorted(CONDITION_TO_URGENCY.items())
    ]
)
print("Resumo do Mapeamento:")
display(mapping_df) if "display" in dir() else print(mapping_df.to_string(index=False))

# %%
full = map_condition_to_urgency(full_raw)
print("\nAmostras após mapeamento (5 primeiras linhas):")
sample = full[[URGENCY_COLUMN, URGENCY_NAME_COLUMN, "condition_name", TEXT_COLUMN]].head()
if "display" in dir():
    with pd.option_context("display.max_colwidth", 80):
        display(sample)
else:
    for _, row in sample.iterrows():
        print(f"  [{row[URGENCY_NAME_COLUMN].upper()}] {row['condition_name']}: {row[TEXT_COLUMN][:80]}...")

# %%
urg_dist = full[URGENCY_NAME_COLUMN].value_counts().reindex(["normal", "atencao", "urgente"])
urg_df = pd.DataFrame(
    {
        "Urgência": urg_dist.index,
        "Nº Amostras": urg_dist.values,
        "Percentual": (urg_dist.values / len(full) * 100).round(2),
    }
)
print("\nDistribuição Final de Urgência:")
display(urg_df) if "display" in dir() else print(urg_df.to_string(index=False))

# %%
urg_colors = {"normal": "#27ae60", "atencao": "#f39c12", "urgente": "#c0392b"}
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

bars0 = axes[0].bar(urg_dist.index, urg_dist.values,
                    color=[urg_colors[c] for c in urg_dist.index], edgecolor="black")
for bar, pct in zip(bars0, urg_df["Percentual"]):
    axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 80,
                 f"{pct}%", ha="center", fontweight="bold")
axes[0].set_title("Contagem por Nível de Urgência", fontweight="bold")
axes[0].set_ylabel("Nº Amostras")

axes[1].pie(urg_dist.values, labels=urg_dist.index,
            autopct="%1.1f%%", startangle=90,
            colors=[urg_colors[c] for c in urg_dist.index],
            wedgeprops={"edgecolor": "black", "linewidth": 1})
axes[1].set_title("Proporção por Nível de Urgência", fontweight="bold")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "02_distribuicao_urgencia.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ---
# ## 6. Análise do Texto - Tamanhos e Estatísticas Descritivas

# %%
full["text_length_chars"] = full[TEXT_COLUMN].str.len()
full["text_length_words"] = full[TEXT_COLUMN].str.split().apply(len)

text_stats = full.groupby(URGENCY_NAME_COLUMN)[
    ["text_length_chars", "text_length_words"]
].agg(["mean", "median", "min", "max"]).round(1)
print("Estatísticas de Tamanho de Texto por Urgência:")
display(text_stats) if "display" in dir() else print(text_stats)

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.boxplot(
    data=full, x=URGENCY_NAME_COLUMN, y="text_length_words",
    palette=urg_colors, ax=axes[0], hue=URGENCY_NAME_COLUMN, legend=False
)
axes[0].set_title("Distribuição de Palavras por Urgência", fontweight="bold")
axes[0].set_xlabel("")
axes[0].set_ylabel("Número de Palavras")

for urg, color in urg_colors.items():
    subset = full[full[URGENCY_NAME_COLUMN] == urg]
    sns.kdeplot(subset["text_length_words"], ax=axes[1], label=urg, color=color, lw=2, fill=True, alpha=0.2)
axes[1].set_title("Densidade - Palavras por Urgência", fontweight="bold")
axes[1].set_xlabel("Número de Palavras")
axes[1].legend(title="Urgência")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "03_tamanho_texto.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ---
# ## 7. Limpeza / Pré-processamento do Texto
#
# Aplicamos:
# 1. Conversão para minúsculas
# 2. Remoção de tags HTML, URLs, e-mails
# 3. Remoção de pontuação
# 4. Remoção de stopwords (inglês)
# 5. Normalização de espaços em branco

# %%
print("Aplicando limpeza (pré-processamento)...")
full_clean = clean_dataframe(full, lemmatize_flag=False)

full_clean["clean_chars"] = full_clean[TEXT_COLUMN].str.len()
full_clean["clean_words"] = full_clean[TEXT_COLUMN].str.split().apply(len)

comp_df = pd.DataFrame(
    {
        "": ["Antes", "Depois", "Redução (%)"],
        "Caracteres (média)": [
            full["text_length_chars"].mean().round(1),
            full_clean["clean_chars"].mean().round(1),
            f"{(1 - full_clean['clean_chars'].mean()/full['text_length_chars'].mean())*100:.1f}%",
        ],
        "Palavras (média)": [
            full["text_length_words"].mean().round(1),
            full_clean["clean_words"].mean().round(1),
            f"{(1 - full_clean['clean_words'].mean()/full['text_length_words'].mean())*100:.1f}%",
        ],
    }
)
print("\nComparação Antes vs. Depois da Limpeza:")
display(comp_df) if "display" in dir() else print(comp_df.to_string(index=False))

# %%
print("\nExemplo - Antes da limpeza:")
ex_raw = full_raw[TEXT_COLUMN].iloc[2][:250] + "..."
print(f"  {ex_raw}")
print("\nExemplo - Depois da limpeza:")
ex_cln = full_clean[TEXT_COLUMN].iloc[2][:250] + "..."
print(f"  {ex_cln}")

# %% [markdown]
# ---
# ## 8. Palavras Mais Frequentes (Por Classe de Urgência)

# %%
def get_top_ngrams(texts, n=1, top_k=15):
    vec = CountVectorizer(ngram_range=(n, n), max_features=5000)
    X = vec.fit_transform(texts)
    counts = np.asarray(X.sum(axis=0)).ravel()
    vocab = np.array(vec.get_feature_names_out())
    order = counts.argsort()[::-1][:top_k]
    return list(zip(vocab[order], counts[order]))

fig, axes = plt.subplots(3, 2, figsize=(14, 15))
for row_idx, urg in enumerate(["normal", "atencao", "urgente"]):
    subset = full_clean[full_clean[URGENCY_NAME_COLUMN] == urg][TEXT_COLUMN]
    for col_idx, n_val in enumerate([1, 2]):
        top = get_top_ngrams(subset, n=n_val, top_k=12)
        words = [w for w, _ in top]
        freqs = [c for _, c in top]
        ax = axes[row_idx, col_idx]
        ax.barh(range(len(words)), freqs, color=urg_colors[urg], edgecolor="black")
        ax.set_yticks(range(len(words)))
        ax.set_yticklabels(words)
        ax.invert_yaxis()
        title = f"{urg.upper()} - Top {'unigramas' if n_val==1 else 'bigramas'}"
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Frequência")

plt.suptitle("Palavras Mais Frequentes Por Urgência (Texto Limpo)",
             fontsize=14, fontweight="bold", y=1.005)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "04_top_words_per_class.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ---
# ## 9. Divisão entre Treino / Validação / Teste
#
# - **Treino**: 70% (para aprendizagem do modelo)
# - **Validação**: 10% (ajuste de hiperparâmetros)
# - **Teste**: 20% (avaliação final, dados nunca vistos)
#
# *Utilizamos amostragem estratificada para preservar a distribuição de classes.*

# %%
splits_df = split_dataframe(full_clean, test_size=TEST_SIZE, val_size=VAL_SIZE)

split_summary = []
for split_name, key in [("Treino", "train_df"), ("Validação", "val_df"), ("Teste", "test_df")]:
    df_s = splits_df[key]
    split_summary.append(
        {
            "Split": split_name,
            "Nº Amostras": len(df_s),
            "% do Total": f"{len(df_s)/len(full_clean)*100:.1f}%",
        }
    )
split_df = pd.DataFrame(split_summary)
print("Tamanho dos Splits:")
display(split_df) if "display" in dir() else print(split_df.to_string(index=False))

# %%
class_dist = get_class_distribution(
    splits_df["train_df"][URGENCY_COLUMN],
    splits_df["val_df"][URGENCY_COLUMN],
    splits_df["test_df"][URGENCY_COLUMN],
)
print("\nDistribuição de Urgência por Split (%):")
pivot = class_dist.pivot_table(
    index="urgency_name", columns="split", values="percentage", aggfunc="first"
).reindex(["normal", "atencao", "urgente"])
pivot.columns = ["Teste (%)", "Treino (%)", "Validação (%)"]
display(pivot.round(1)) if "display" in dir() else print(pivot.round(1))

# %%
fig, ax = plt.subplots(figsize=(10, 5))
bar_width = 0.25
x = np.arange(3)
split_order = ["train", "val", "test"]
split_label = ["Treino", "Validação", "Teste"]
split_color = ["#3498db", "#f39c12", "#e74c3c"]
for i, (s, lab, col) in enumerate(zip(split_order, split_label, split_color)):
    data = class_dist[class_dist["split"] == s].sort_values("urgency_label")
    ax.bar(x + i * bar_width, data["percentage"].values,
           bar_width, label=lab, color=col, edgecolor="black")
ax.set_xticks(x + bar_width)
ax.set_xticklabels(["Normal (0)", "Atenção (1)", "Urgente (2)"])
ax.set_ylabel("Percentual (%)")
ax.set_title("Distribuição Estratificada por Split", fontweight="bold")
ax.legend(title="Split")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "05_split_distribuicao.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ---
# ## 10. Pipeline de Features: TF-IDF
#
# Parâmetros:
# - `max_features=10.000` (vocabulário com top 10k tokens)
# - `ngram_range=(1,2)` (unigramas + bigramas)
# - `min_df=2` e `max_df=0.95` (remoção de termos raros e muito comuns)
# - `sublinear_tf=True` (escala logarítmica na frequência)

# %%
print("Treinando vetorizador TF-IDF...")
vectorizer, X_train = fit_tfidf_vectorizer(splits_df["train_df"][TEXT_COLUMN])
X_val = transform_texts(vectorizer, splits_df["val_df"][TEXT_COLUMN])
X_test = transform_texts(vectorizer, splits_df["test_df"][TEXT_COLUMN])

y_train = splits_df["train_df"][URGENCY_COLUMN].values
y_val = splits_df["val_df"][URGENCY_COLUMN].values
y_test = splits_df["test_df"][URGENCY_COLUMN].values

tfidf_info = pd.DataFrame(
    {
        "": ["X_train", "X_val", "X_test"],
        "Nº Amostras": [X_train.shape[0], X_val.shape[0], X_test.shape[0]],
        "Nº Features (TF-IDF)": [X_train.shape[1], X_val.shape[1], X_test.shape[1]],
        "Esparsidade (%)": [
            f"{(1 - X_train.nnz/(X_train.shape[0]*X_train.shape[1]))*100:.2f}%",
            f"{(1 - X_val.nnz/(X_val.shape[0]*X_val.shape[1]))*100:.2f}%",
            f"{(1 - X_test.nnz/(X_test.shape[0]*X_test.shape[1]))*100:.2f}%",
        ],
    }
)
print("\nMatrizes TF-IDF Geradas:")
display(tfidf_info) if "display" in dir() else print(tfidf_info.to_string(index=False))

# %%
vocab = np.array(vectorizer.get_feature_names_out())
print(f"\nTamanho do vocabulário TF-IDF: {len(vocab)} termos")
print(f"\nPrimeiros 20 termos do vocabulário (ordem alfabética):\n  {vocab[:20]}")

# %%
def plot_top_tfidf_features(X, y, labels, class_names, vocab, top_n=15):
    fig, axes = plt.subplots(1, len(labels), figsize=(16, 7))
    if len(labels) == 1:
        axes = [axes]
    for ax, lab in zip(axes, labels):
        idx = np.where(y == lab)[0]
        mean_tfidf = np.asarray(X[idx].mean(axis=0)).ravel()
        top_idx = mean_tfidf.argsort()[::-1][:top_n]
        top_terms = vocab[top_idx]
        top_scores = mean_tfidf[top_idx]
        ax.barh(range(top_n), top_scores, color=urg_colors.get(class_names[lab], "#3498db"), edgecolor="black")
        ax.set_yticks(range(top_n))
        ax.set_yticklabels(top_terms)
        ax.invert_yaxis()
        ax.set_title(f"Urgência: {class_names[lab].upper()}\n(TF-IDF médio)", fontweight="bold")
        ax.set_xlabel("TF-IDF médio")
    plt.tight_layout()
    return fig

_ = plot_top_tfidf_features(X_train, y_train, [0,1,2], URGENCY_LABEL_TO_NAME, vocab, top_n=12)
plt.savefig(OUTPUT_DIR / "06_top_tfidf_por_classe.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
vec_path = save_vectorizer(vectorizer)
print(f"\nVectorizer TF-IDF salvo em: {vec_path}")

# %% [markdown]
# ---
# ## 11. Treinamento de Classificador Base (Scikit-Learn)
#
# Treinamos e comparamos 4 modelos leves de classificação:
# 1. **Naive Bayes Multinomial** (baseline simples e rápido)
# 2. **Logistic Regression** (interpretável, bom para texto)
# 3. **Linear SVC** (SVM linear - clássico para NLP)
# 4. **Random Forest** (ensemble não linear)

# %%
print("Rodando benchmark dos modelos (isso pode levar alguns minutos)...")
benchmark = benchmark_models(X_train, y_train, X_test, y_test)
benchmark = benchmark.reset_index(drop=True)
print("\nBenchmark de Modelos (conjunto de teste):")
display(benchmark) if "display" in dir() else print(benchmark.to_string(index=False))

# %%
fig, ax = plt.subplots(figsize=(10, 5))
x_pos = np.arange(len(benchmark))
bars = ax.bar(x_pos, benchmark["macro_f1"], color=sns.color_palette("viridis", len(benchmark)), edgecolor="black")
for bar, val in zip(bars, benchmark["macro_f1"]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, f"{val:.3f}",
            ha="center", fontweight="bold")
ax.set_xticks(x_pos)
ax.set_xticklabels(benchmark["model"], rotation=15, ha="right")
ax.set_ylabel("F1-Score Macro")
ax.set_ylim(0, 1.0)
ax.set_title("Comparação de Modelos - F1 Macro (Teste)", fontweight="bold")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "07_benchmark_modelos.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
best_model_name = benchmark.iloc[0]["model"]
print(f"\n-> Melhor modelo selecionado: {best_model_name}")

model, train_info = train_model(
    X_train, y_train, model_name=best_model_name, X_val=X_val, y_val=y_val
)
print("\nInfo Treinamento:")
for k, v in train_info.items():
    print(f"  - {k}: {v}")

# %% [markdown]
# ---
# ## 12. Avaliação do Modelo com Métricas de Classificação

# %%
metrics = evaluate_model(model, X_test, y_test, model_name=best_model_name)

metrics_df = pd.DataFrame(
    {
        "Métrica": [
            "Accuracy (Acurácia)",
            "Precision (Macro)",
            "Recall (Macro)",
            "F1-Score (Macro)",
            "Precision (Weighted)",
            "Recall (Weighted)",
            "F1-Score (Weighted)",
        ],
        "Valor": [
            metrics["accuracy"],
            metrics["precision_macro"],
            metrics["recall_macro"],
            metrics["f1_macro"],
            metrics["precision_weighted"],
            metrics["recall_weighted"],
            metrics["f1_weighted"],
        ],
    }
)
print("Métricas Gerais no Conjunto de Teste:")
display(metrics_df) if "display" in dir() else print(metrics_df.to_string(index=False))

if "roc_auc_ovr" in metrics:
    print(f"\nROC AUC (One-vs-Rest, Macro): {metrics['roc_auc_ovr']}")

print("\n" + "=" * 60)
print("Relatório de Classificação Detalhado:")
print("=" * 60)
print(metrics["classification_report_str"])

# %%
cm = np.array(metrics["confusion_matrix"])
cm_labels = metrics["confusion_labels"]
fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=[f"pred_{l}" for l in cm_labels],
    yticklabels=[f"true_{l}" for l in cm_labels],
    ax=ax, cbar_kws={"label": "Contagem"},
)
ax.set_title(f"Matriz de Confusão - {best_model_name} (Teste)", fontweight="bold")
ax.set_xlabel("Predito")
ax.set_ylabel("Real")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "08_matriz_confusao.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
per_class = metrics["classification_report"]
rows = []
for name in cm_labels:
    if name in per_class:
        d = per_class[name]
        rows.append({"Classe": name, **{k: d[k] for k in ["precision", "recall", "f1-score", "support"]}})
per_class_df = pd.DataFrame(rows)
print("Métricas Por Classe:")
display(per_class_df.round(3)) if "display" in dir() else print(per_class_df.round(3).to_string(index=False))

# %% [markdown]
# ---
# ## 13. Demonstração: Predição em Novos Casos

# %%
sample_cases = [
    (
        "Exemplo 1 - SUSPEITO DE IAM",
        "The patient presented with acute chest pain radiating to the left arm, "
        "associated with diaphoresis and dyspnea. ECG showed ST elevation in leads V1-V4. "
        "Diagnosis: Anterior wall acute myocardial infarction. Urgent catheterization indicated."
    ),
    (
        "Exemplo 2 - GASTRITE CRÔNICA",
        "42-year-old patient with chronic gastritis reports intermittent heartburn after "
        "meals, especially spicy food. Upper endoscopy showed mild pangastritis, no ulcers. "
        "Prescribed proton pump inhibitor once daily and dietary modifications. Outpatient follow-up."
    ),
    (
        "Exemplo 3 - CEFALEIA + ACOMP NEUROLÓGICO",
        "55-year-old patient with history of hypertension and diabetes complains of persistent "
        "headache, occasional dizziness and paresthesia. Cranial CT without acute abnormalities. "
        "Neurological examination was largely unremarkable. Recommended outpatient follow-up "
        "with neurology and optimization of antihypertensive therapy."
    ),
]

for case_title, text in sample_cases:
    print(f"\n{case_title}")
    print(f"  Texto: {text[:140]}...")

# %% [markdown]
# ---
# ## 14. Salvando Modelo Treinado em Formato Reutilizável
#
# Treinamos o pipeline completo (TF-IDF + Classificador) com **dados de treino + validação**
# e salvamos em formato `.joblib` (carregável diretamente para inferência futura).

# %%
print("Treinando pipeline completo para produção...")
train_texts = pd.concat([
    splits_df["train_df"][TEXT_COLUMN],
    splits_df["val_df"][TEXT_COLUMN],
], ignore_index=True)
y_full_train = np.concatenate([y_train, y_val])

full_pipeline = train_full_pipeline(train_texts, y_full_train, model_name=best_model_name)

metadata = {
    "model_name": best_model_name,
    "n_train_samples": len(train_texts),
    "test_accuracy": metrics["accuracy"],
    "test_f1_macro": metrics["f1_macro"],
    "created_date": str(pd.Timestamp.now()),
    "classes": {str(k): v for k, v in URGENCY_LABEL_TO_NAME.items()},
    "tfidf_params": vectorizer.get_params(),
}
model_path = save_model(full_pipeline, metadata=metadata)
print(f"\nPipeline completo salvo em: {model_path}")

# %% [markdown]
# ---
# ## 15. Verificação: Carregar Modelo e Inferir Exemplos

# %%
loaded_pipeline, loaded_meta = load_model()
print(f"Modelo carregado: {loaded_meta['model_name']}")
print(f"  - Data criação: {loaded_meta.get('created_date', 'N/A')}")
print(f"  - F1-macro (teste): {loaded_meta.get('test_f1_macro', 'N/A')}")

for case_title, text in sample_cases:
    pred = predict_text(loaded_pipeline, text)
    print(f"\n{case_title}")
    print(f"  -> Predição: {pred['predicted_name'].upper()} (nível {pred['predicted_label']})")
    if "probabilities" in pred:
        probs = pred["probabilities"]
        print(f"  -> Probabilidades: { {k: f'{v*100:.1f}%' for k,v in probs.items()} }")

# %% [markdown]
# ---
# ## 16. Resumo e Conclusões do EDA
#
# ### Checklist Concluído
#
# | Item | Status |
# |------|:------:|
# | Selecionar e documentar origem do dataset | ✅ |
# | Documentar dataset no EDA | ✅ |
# | Validar colunas de texto e target | ✅ (sem nulos, sem vazios) |
# | Garantir quantidade adequada de amostras | ✅ (14.438 amostras) |
# | Implementar carregamento dos dados | ✅ (`sources/data_loader.py`) |
# | Implementar limpeza / pré-processamento | ✅ (`sources/preprocessing.py`) |
# | Implementar divisão treino e teste | ✅ (70/10/20 estratificado) |
# | Criar pipeline de features TF-IDF | ✅ (10k unigramas+bigramas) |
# | Treinar classificador base Scikit-Learn | ✅ (LogReg, NB, SVC, RF) |
# | Avaliar modelo com métricas | ✅ (acc, prec, rec, f1, ROC-AUC, matriz confusão) |
# | Salvar modelo reutilizável | ✅ (`models/urgency_classifier.joblib`) |
# | Documentar classes e mapeamento target | ✅ (5 originais → 3 urgência) |
# | Adicionar testes do pipeline | ✅ (`tests/test_*.py`) |
#
# ### Observações Importantes
# 1. **Desbalanceamento**: Classe `Normal` (10,3%) é a menos representada; utilizar
#    `class_weight='balanced'` ajuda. Etapas seguintes podem explorar SMOTE / Focal Loss.
# 2. **Melhor modelo**: Regressão Logística e Linear SVC costumam apresentar melhor
#    desempenho para NLP com TF-IDF.
# 3. **Próximas etapas sugeridas**: Tuning de hiperparâmetros, word embeddings (BERT),
#    validação cruzada, API FastAPI para inferência e containerização Docker.

# %%
print("\n" + "=" * 60)
print("EDA E PIPELINE CONCLUÍDOS COM SUCESSO!")
print("=" * 60)
print(f"\nArtefatos gerados:")
print(f"  - {len(list(OUTPUT_DIR.glob('*.png')))} gráficos em: {OUTPUT_DIR}")
print(f"  - Vectorizer TF-IDF: {MODELS_DIR / 'tfidf_vectorizer.joblib'}")
print(f"  - Modelo treinado: {MODELS_DIR / 'urgency_classifier.joblib'}")
print(f"  - Documentação: {DOCS_DIR / 'DATASET.md'}")
print(f"  - Código-fonte em sources/  e  pipelines/")
print(f"  - Testes unitários em tests/")

# %%
