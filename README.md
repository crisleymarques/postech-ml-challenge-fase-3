# Postech ML Challenge - Fase 3

## Triagem Automática de Urgência em Laudos Médicos (NLP)

Projeto de pós-graduação (FIAP/POSTECH) que implementa um sistema de classificação
de texto (NLP), destinado à triagem automática de resumos/laudos médicos
em três níveis de urgência: **Normal**, **Atenção** e **Urgente**.

Dataset: **Medical Abstracts TC Corpus Dataset** (14.438 amostras).

---

## Estrutura de Pastas

```
postech-ml-challenge-fase-3/
├── data/
│   └── raw/                      # Dataset original (3 arquivos CSV)
├── docs/
│   └── DATASET.md                # Documentação detalhada do dataset e mapeamentos
├── models/                       # Modelos gerados pelo pipeline (gitignored)
├── notebooks/
│   ├── EDA_Medical_Abstracts.ipynb # EDA completo (notebook)
│   └── eda_outputs/              # Gráficos PNG gerados pelo EDA (gitignored)
├── sources/                      # Módulos Python reutilizáveis
│   ├── __init__.py
│   ├── config.py                 # Configurações globais e mapeamento de classes
│   ├── data_loader.py            # Carregamento, validação e mapeamento de urgência
│   ├── preprocessing.py          # Limpeza e pré-processamento de texto
│   └── model.py                  # Treinamento, avaliação, predição e benchmark
├── pipelines/
│   ├── __init__.py
│   └── feature_pipeline.py       # TF-IDF, splits de dados, salvamento/carregamento
├── tests/                        # Testes unitários (pytest)
│   ├── conftest.py               # Configuração compartilhada dos testes
│   ├── test_data_loader.py
│   ├── test_preprocessing.py
│   ├── test_feature_pipeline.py
│   └── test_model.py
├── run_pipeline.py               # Script orquestrador - executa todo o pipeline
├── pyproject.toml                # Dependências e metadados do projeto (uv)
├── uv.lock                       # Lockfile de dependências (uv)
└── README.md
```

---

## Artefatos Gerados (não versionados)

Ao executar o pipeline, os seguintes artefatos são gerados localmente:

| Artefato | Descrição |
|----------|-----------|
| `models/urgency_classifier.joblib` | Pipeline completo (TF-IDF + classificador) |
| `models/tfidf_vectorizer.joblib` | Vectorizer TF-IDF treinado |
| `data/processed/*.csv` | Dados processados e splits |
| `docs/model_metrics.json` | Métricas do modelo |
| `notebooks/eda_outputs/*.png` | Gráficos do EDA |

---

## EDA

A análise exploratória completa está em:
👉 **[notebooks/EDA_Medical_Abstracts.ipynb](notebooks/EDA_Medical_Abstracts.ipynb)**

O arquivo pode ser executado como:
- **Notebook Jupyter**: Abra diretamente no Jupyter ou em ferramentas compatíveis como VSCode.

O EDA cobre **todos os itens do checklist**:
1. ✅ Origem e documentação do dataset
2. ✅ Documentação completa no próprio notebook
3. ✅ Validação das colunas de texto e target
4. ✅ Quantidade de amostras (14.438 - acima do mínimo)
5. ✅ Carregamento dos dados (`sources/data_loader.py`)
6. ✅ Limpeza / pré-processamento (`sources/preprocessing.py`)
7. ✅ Divisão treino / validação / teste (70/10/20 estratificado)
8. ✅ Pipeline de features TF-IDF (10k features, unigramas + bigramas)
9. ✅ Classificadores base Scikit-Learn (LogReg, NB, LinearSVC, RF)
10. ✅ Avaliação com métricas (acc, precision, recall, F1, ROC-AUC, matriz confusão)
11. ✅ Modelo salvo em formato reutilizável (`.joblib`)
12. ✅ Documentação de classes e mapeamento do target (5 originais → 3 urgência)
13. ✅ Testes unitários das etapas principais (46 testes)

---

## Mapeamento de Urgência Aplicado

| Urgência   | Label | Condições Médicas Originais Mapeadas                    |
|------------|:-----:|---------------------------------------------------------|
| Normal     |   0   | Digestive system diseases                               |
| Atenção    |   1   | Nervous system diseases + General pathological conditions |
| Urgente    |   2   | Neoplasms (câncer) + Cardiovascular diseases            |

Justificativas completas em [docs/DATASET.md](docs/DATASET.md).

---

<!-- ![CI Status](https://github.com/crisleymarques/postech-ml-challenge-fase-3/actions/workflows/ci.yml/badge.svg) -->

## Como Executar

### 1. Instalar dependências

O projeto usa [uv](https://docs.astral.sh/uv/) como gerenciador de dependências:

```bash
uv sync
```

> Se não tiver `uv` instalado: `curl -LsSf https://astral.sh/uv/install.sh | sh`

### 2. Executar o EDA

Abra o arquivo `notebooks/EDA_Medical_Abstracts.ipynb` em seu ambiente Jupyter (JupyterLab, Jupyter Notebook ou VSCode) e execute todas as células sequencialmente.

Saídas geradas em `notebooks/eda_outputs/` (8 gráficos) + artefatos em `models/`.

### 3. Executar o pipeline completo (alternativa)

```bash
python run_pipeline.py --model logistic_regression
```

Flags opcionais:
- `--model {logistic_regression,naive_bayes,random_forest,linear_svc}`
- `--lemmatize`: ativa lematização no pré-processamento (mais lento)

### 4. Rodar os testes unitários

```bash
uv run pytest tests/ -v
```

---

## Exemplo de Inferência

```python
import joblib

artifact = joblib.load("models/urgency_classifier.joblib")
pipeline = artifact["model"]

texto = ("Acute chest pain radiating to the left arm. ECG shows ST-elevation. "
         "Diagnosis: acute myocardial infarction. Urgent cath required.")

y_pred = pipeline.predict([texto])   # retorna array([2]) -> Urgente
prob   = pipeline.predict_proba([texto])
```
