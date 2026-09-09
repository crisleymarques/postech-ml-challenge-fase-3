import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_RAW_DIR = BASE_DIR / "data" / "raw"
DATA_PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
DOCS_DIR = BASE_DIR / "docs"

LABELS_FILE = DATA_RAW_DIR / "medical_tc_labels.csv"
TRAIN_FILE = DATA_RAW_DIR / "medical_tc_train.csv"
TEST_FILE = DATA_RAW_DIR / "medical_tc_test.csv"

TEXT_COLUMN = "medical_abstract"
TARGET_COLUMN = "condition_label"
URGENCY_COLUMN = "urgency_label"
URGENCY_NAME_COLUMN = "urgency_name"

CONDITION_LABEL_TO_NAME = {
    1: "neoplasms",
    2: "digestive system diseases",
    3: "nervous system diseases",
    4: "cardiovascular diseases",
    5: "general pathological conditions",
}

CONDITION_TO_URGENCY = {
    1: 2,
    2: 0,
    3: 1,
    4: 2,
    5: 1,
}

URGENCY_LABEL_TO_NAME = {
    0: "normal",
    1: "atencao",
    2: "urgente",
}

URGENCY_NAME_TO_LABEL = {v: k for k, v in URGENCY_LABEL_TO_NAME.items()}

URGENCY_MAPPING_DOC = """
Mapeamento das Condições Médicas para Níveis de Urgência:

Critérios adotados para classificação de urgência tripartite:
- URGENTE (nível 2): Condições que frequentemente requerem intervenção médica
  imediata ou tratamento prioritário devido ao risco de vida ou progressão
  agressiva da doença.
- ATENÇÃO (nível 1): Condições que necessitam de acompanhamento médico e
  tratamento, mas que não representam risco imediato à vida.
- NORMAL (nível 0): Condições gerenciáveis ambulatorialmente, sem caráter de
  emergência ou urgência imediata.

Mapeamento detalhado:
1. Neoplasms (Neoplasias/Câncer)              -> URGENTE (nível 2)
   Justificativa: Câncer requer diagnóstico e tratamento rápido devido ao
   potencial de metástase e progressão da doença.

2. Digestive system diseases (Doenças do     -> NORMAL (nível 0)
   Sistema Digestivo)
   Justificativa: A maioria das doenças digestivas (gastroenterites,
   hepatopatias crônicas leves, etc.) é manejada ambulatorialmente.

3. Nervous system diseases (Doenças do       -> ATENÇÃO (nível 1)
   Sistema Nervoso)
   Justificativa: Doenças neurológicas requerem acompanhamento especializado
   mas raramente são emergências imediatas (exceto casos específicos como AVC
   agudo que serão classificados corretamente pelo texto).

4. Cardiovascular diseases (Doenças          -> URGENTE (nível 2)
   Cardiovasculares)
   Justificativa: Infarto do miocárdio, angina instável, arritmias graves e
   outras cardiopatias representam risco iminente de vida.

5. General pathological conditions           -> ATENÇÃO (nível 1)
   (Condições patológicas gerais)
   Justificativa: Condições diversas que incluem infecções, inflamações e
   distúrbios metabólicos que requerem tratamento mas não emergência imediata
   na maioria dos casos.
"""

RANDOM_STATE = 42
TEST_SIZE = 0.2
VAL_SIZE = 0.1

TFIDF_MAX_FEATURES = 10000
TFIDF_NGRAM_RANGE = (1, 2)
TFIDF_MIN_DF = 2
TFIDF_MAX_DF = 0.95

STOPWORDS_LANG = "english"

os.makedirs(DATA_PROCESSED_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
