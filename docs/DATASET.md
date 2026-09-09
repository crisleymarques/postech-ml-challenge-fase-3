# Medical Abstracts TC Corpus Dataset

## 1. Origem do Dataset

- **Nome oficial**: Medical Abstracts Text Classification (TC) Corpus Dataset
- **Fonte**: Dataset público amplamente utilizado para classificação de texto médico (NLP). Originalmente derivado de resumos de publicações biomédicas indexadas.
- **Referência comum**: Disponível em repositórios como Hugging Face Datasets (`medical_abstracts_dataset`), Kaggle e repositórios acadêmicos de NLP clínico.
- **Licença**: Uso acadêmico e de pesquisa (dataset público de referência).

## 2. Estrutura dos Arquivos

O dataset é distribuído em 3 arquivos CSV, localizados em `data/raw/`:

| Arquivo                       | Descrição                                         | Linhas  |
|-------------------------------|---------------------------------------------------|---------|
| `medical_tc_labels.csv`       | Dicionário de labels (classe -> nome da condição) | 5       |
| `medical_tc_train.csv`        | Conjunto de treino                                | 11.550  |
| `medical_tc_test.csv`         | Conjunto de teste                                 | 2.888   |
| **Total**                     |                                                   | **14.438** |

### 2.1 Colunas

- **condition_label** (int): Código numérico da condição médica (1 a 5).
- **medical_abstract** (str): Texto do resumo médico (abstract).
- **condition_name** (str, apenas no labels file): Nome da condição.

## 3. Classes Originais

| Label | Nome da Condição                  | Qtd Treino | Qtd Teste | Total      |
|-------|-----------------------------------|------------|-----------|------------|
| 1     | Neoplasms (Neoplasias/Câncer)     | 2.530      | 633       | 3.163      |
| 2     | Digestive system diseases         | 1.195      | 299       | 1.494      |
| 3     | Nervous system diseases           | 1.540      | 385       | 1.925      |
| 4     | Cardiovascular diseases           | 2.441      | 610       | 3.051      |
| 5     | General pathological conditions   | 3.844      | 961       | 4.805      |

## 4. Validação de Qualidade

- **Amostras**: 14.438 amostras (cumpre requisito mínimo de 2.000).
- **Valores nulos**: 0 (nenhum valor ausente nas colunas).
- **Textos vazios**: 0.
- **Tamanho médio do texto**: ~1.300 caracteres por abstract.

## 5. Mapeamento para Níveis de Urgência

O problema de negócio requer classificação em 3 níveis: **NORMAL / ATENÇÃO / URGENTE**. Como o dataset original possui 5 condições clínicas, foi definido um mapeamento baseado em criticidade médica típica.

### 5.1 Classes Resultantes (Urgency)

| Urgency Label | Nome       | Descrição                                                   |
|---------------|------------|-------------------------------------------------------------|
| 0             | **Normal** | Condições ambulatoriais, sem caráter emergencial imediato.  |
| 1             | **Atenção**| Condições que requerem acompanhamento/tratamento mas sem risco imediato. |
| 2             | **Urgente**| Risco elevado de vida ou progressão agressiva; intervenção prioritária. |

### 5.2 Regras de Mapeamento Aplicadas

| Condição Original (Label) | Nome                                | Urgência Mapeada | Justificativa Clínica                                                                 |
|---------------------------|-------------------------------------|------------------|---------------------------------------------------------------------------------------|
| 1                         | Neoplasms (Câncer)                  | **2 - Urgente**  | Potencial metastático, necessidade de intervenção oncológica rápida.                  |
| 2                         | Digestive system diseases           | **0 - Normal**   | Maioria gastroenterites, hepatopatias crônicas leves — manejo ambulatorial.           |
| 3                         | Nervous system diseases             | **1 - Atenção**  | Requer acompanhamento neurológico; não emergência na maioria dos casos.               |
| 4                         | Cardiovascular diseases             | **2 - Urgente**  | IAM, angina instável, arritmias — risco iminente de vida.                             |
| 5                         | General pathological conditions     | **1 - Atenção**  | Infecções, inflamações, distúrbios metabólicos variados — tratamento mas não emergência. |

### 5.3 Distribuição Após Mapeamento

| Urgência   | Total   | Percentual |
|------------|---------|------------|
| Normal     | 1.494   | 10,3%      |
| Atenção    | 6.730   | 46,6%      |
| Urgente    | 6.214   | 43,1%      |
| **Total**  | 14.438  | 100%       |

> **Observação**: Existe desbalanceamento da classe "Normal". Técnicas como `class_weight='balanced'` e oversampling/undersampling serão avaliadas em etapas posteriores.

## 6. Pré-processamento Aplicado

1. **Conversão para minúsculas**
2. **Remoção de HTML tags, URLs e e-mails**
3. **Remoção de pontuação**
4. **Remoção de stopwords** (inglês)
5. **(Opcional) Lematização**
6. **Normalização de espaços em branco**

## 7. Representação Vetorial

- **Algoritmo**: TF-IDF (Term Frequency - Inverse Document Frequency)
- **Hiperparâmetros**:
  - `max_features`: 10.000 termos mais frequentes
  - `ngram_range`: (1, 2) — unigramas e bigramas
  - `min_df`: 2 (remove termos raros)
  - `max_df`: 0.95 (remove termos muito frequentes)
  - `sublinear_tf`: True (aplica escala logarítmica)
  - `norm`: 'l2'

## 8. Arquivos Gerados (`data/processed/`)

| Arquivo                              | Descrição                                          |
|--------------------------------------|----------------------------------------------------|
| `medical_tc_train_processed.csv`     | Treino com colunas de urgência mapeadas            |
| `medical_tc_test_processed.csv`      | Teste com colunas de urgência mapeadas             |
| `medical_tc_train.csv`               | Split treino (70%) final pós embaralhamento        |
| `medical_tc_val.csv`                 | Split validação (10%)                              |
| `medical_tc_test.csv`                | Split teste (20%) final                            |
