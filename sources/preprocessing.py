import re
import string
import pandas as pd
from typing import List, Optional

try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    from nltk.tokenize import word_tokenize

    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

from sources.config import TEXT_COLUMN, STOPWORDS_LANG


def download_nltk_resources():
    if NLTK_AVAILABLE:
        try:
            nltk.data.find("corpora/stopwords")
            nltk.data.find("corpora/wordnet")
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            nltk.download("stopwords", quiet=True)
            nltk.download("wordnet", quiet=True)
            nltk.download("punkt", quiet=True)
            nltk.download("omw-1.4", quiet=True)


def get_stopwords() -> set:
    if NLTK_AVAILABLE:
        download_nltk_resources()
        return set(stopwords.words(STOPWORDS_LANG))
    return set()


def remove_html_tags(text: str) -> str:
    pattern = re.compile(r"<.*?>")
    return pattern.sub("", text)


def remove_urls(text: str) -> str:
    pattern = re.compile(r"https?://\S+|www\.\S+")
    return pattern.sub("", text)


def remove_emails(text: str) -> str:
    pattern = re.compile(r"\S+@\S+\.\S+")
    return pattern.sub("", text)


def remove_numbers(text: str) -> str:
    return re.sub(r"\d+", "", text)


def remove_punctuation(text: str) -> str:
    translator = str.maketrans("", "", string.punctuation)
    return text.translate(translator)


def remove_extra_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def to_lowercase(text: str) -> str:
    return text.lower()


def remove_stopwords(text: str, stop_words: Optional[set] = None) -> str:
    if stop_words is None:
        stop_words = get_stopwords()
    if not stop_words:
        return text
    words = text.split()
    return " ".join([word for word in words if word not in stop_words])


def lemmatize_text(text: str) -> str:
    if not NLTK_AVAILABLE:
        return text
    download_nltk_resources()
    lemmatizer = WordNetLemmatizer()
    try:
        tokens = word_tokenize(text)
    except Exception:
        tokens = text.split()
    return " ".join([lemmatizer.lemmatize(token) for token in tokens])


def clean_text(
    text: str,
    remove_stopwords_flag: bool = True,
    lemmatize_flag: bool = True,
    remove_nums: bool = False,
    _preloaded_stopwords: Optional[set] = None,
) -> str:
    if not isinstance(text, str):
        return ""

    text = to_lowercase(text)
    text = remove_html_tags(text)
    text = remove_urls(text)
    text = remove_emails(text)
    if remove_nums:
        text = remove_numbers(text)
    text = remove_punctuation(text)
    text = remove_extra_whitespace(text)

    if remove_stopwords_flag:
        sw = _preloaded_stopwords if _preloaded_stopwords is not None else get_stopwords()
        text = remove_stopwords(text, sw)

    if lemmatize_flag:
        text = lemmatize_text(text)

    text = remove_extra_whitespace(text)
    return text


def clean_dataframe(
    df: pd.DataFrame,
    text_col: str = TEXT_COLUMN,
    output_col: Optional[str] = None,
    remove_stopwords_flag: bool = True,
    lemmatize_flag: bool = True,
    remove_nums: bool = False,
) -> pd.DataFrame:
    """Aplica limpeza de texto a uma coluna do DataFrame.

    Reutiliza clean_text() internamente para garantir consistência.
    Pré-carrega stopwords para evitar I/O repetido.
    """
    df = df.copy()

    if output_col is None:
        output_col = text_col

    download_nltk_resources()
    stop_words = get_stopwords() if remove_stopwords_flag else None

    df[output_col] = df[text_col].apply(
        lambda text: clean_text(
            text,
            remove_stopwords_flag=remove_stopwords_flag,
            lemmatize_flag=lemmatize_flag,
            remove_nums=remove_nums,
            _preloaded_stopwords=stop_words,
        )
    )
    return df


if __name__ == "__main__":
    sample_text = """
    Heart disease (CVD) is the #1 killer worldwide! Visit https://example.com for
    more info. Contact dr.smith@hospital.com. The patient's BP was 140/90 mmHg.
    """
    print("Texto original:")
    print(sample_text)
    print("\nTexto limpo:")
    print(clean_text(sample_text))
