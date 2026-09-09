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
│   ├── raw/                      # Dataset original (3 arquivos CSV)
│   └── processed/                # Dados processados + splits treino/val/teste
├── docs/
│   ├── DATASET.md                # Documentação detalhada do dataset e mapeamentos
│   └── model_metrics.json        # Métricas do modelo treinado (após rodar pipeline)
├── models/
│   ├── tfidf_vectorizer.joblib   # Vectorizer TF-IDF treinado
│   └── urgency_classifier.joblib # Pipeline completo salvo p/ inferência
├── notebooks/
│   ├── EDA_Medical_Abstracts.py  # **ENTREGA PRINCIPAL**: EDA completo (notebook/script)
│   └── eda_outputs/              # Gráficos PNG gerados pelo EDA
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
│   ├── test_data_loader.py
│   ├── test_preprocessing.py
│   ├── test_feature_pipeline.py
│   └── test_model.py
├── run_pipeline.py               # Script orquestrador - executa todo o pipeline
├── requirements.txt              # Dependências do projeto
└── README.md                     # Este arquivo
```

---

## EDA

A análise exploratória completa está em:
👉 **[notebooks/EDA_Medical_Abstracts.py](notebooks/EDA_Medical_Abstracts.py)**

O arquivo pode ser executado como:
- **Notebook Jupyter**: Abra diretamente no Jupyter (suporte a `# %%` cell magic)
- **Script Python**: Execute via linha de comando (ver abaixo)

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
13. ✅ Testes unitários das etapas principais

---

## Mapeamento de Urgência Aplicado

| Urgência   | Label | Condições Médicas Originais Mapeadas                    |
|------------|:-----:|---------------------------------------------------------|
| Normal     |   0   | Digestive system diseases                               |
| Atenção    |   1   | Nervous system diseases + General pathological conditions |
| Urgente    |   2   | Neoplasms (câncer) + Cardiovascular diseases            |

Justificativas completas em [docs/DATASET.md](docs/DATASET.md).

---

## Como Executar

### 1. Instalar dependências

```bash
pip install -r requirements.txt
```

### 2. Executar o EDA 

```bash
python notebooks/EDA_Medical_Abstracts.py
```

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
pytest tests/ -v
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


