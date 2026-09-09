import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import pandas as pd

from sources.preprocessing import (
    clean_text,
    clean_dataframe,
    remove_html_tags,
    remove_urls,
    remove_emails,
    remove_punctuation,
    remove_extra_whitespace,
    to_lowercase,
    remove_stopwords,
    remove_numbers,
)


def test_to_lowercase():
    assert to_lowercase("Hello WORLD") == "hello world"


def test_remove_html_tags():
    html = "<p>Hello <b>World</b></p>"
    result = remove_html_tags(html)
    assert "<p>" not in result
    assert "<b>" not in result
    assert "Hello World" in result


def test_remove_urls():
    text = "Visit https://example.com and www.test.org for info."
    result = remove_urls(text)
    assert "https://" not in result
    assert "www." not in result


def test_remove_emails():
    text = "Contact me at user@example.com please"
    result = remove_emails(text)
    assert "@" not in result


def test_remove_numbers():
    text = "Patient 1234 has BP 140/90"
    result = remove_numbers(text)
    assert "1234" not in result
    assert "140" not in result


def test_remove_punctuation():
    text = "Hello, world! How are you? I'm fine."
    result = remove_punctuation(text)
    assert "," not in result
    assert "!" not in result
    assert "?" not in result


def test_remove_extra_whitespace():
    text = "  hello   world  "
    result = remove_extra_whitespace(text)
    assert result == "hello world"


def test_remove_stopwords_basic():
    stopwords_set = {"the", "and", "is", "a"}
    text = "the cat and a dog is nice"
    result = remove_stopwords(text, stopwords_set)
    assert "the" not in result
    assert "and" not in result
    assert "cat" in result
    assert "dog" in result


def test_clean_text_non_string():
    assert clean_text(None) == ""
    assert clean_text(12345) == ""
    assert clean_text(float("nan")) == ""


def test_clean_text_complete():
    dirty = """
    <p>Patient with <b>ACUTE</b> chest pain!
    Visit https://hospital.com or email dr@med.com.
    His BP: 140/90 mmHg.</p>
    """
    result = clean_text(dirty, remove_stopwords_flag=False, lemmatize_flag=False)
    assert result == result.lower()
    assert "<p>" not in result
    assert "https://" not in result
    assert "@" not in result
    assert "!" not in result
    assert "  " not in result
    assert len(result.strip()) > 0


def test_clean_text_removes_stopwords():
    text = "the patient was given a treatment for his disease"
    result = clean_text(text, remove_stopwords_flag=True, lemmatize_flag=False)
    assert "the" not in result
    assert "was" not in result
    assert "patient" in result
    assert "treatment" in result


def test_clean_dataframe():
    df = pd.DataFrame(
        {
            "medical_abstract": [
                "<p>Hello <b>WORLD</b>! Visit https://x.com</p>",
                "Patient 1234 with chest pain!!! Email dr@h.com",
                None,
            ]
        }
    )
    result = clean_dataframe(df, remove_stopwords_flag=False, lemmatize_flag=False)
    assert len(result) == 3

    for idx, row in result.iterrows():
        text = row["medical_abstract"]
        assert text == text.lower()
        assert "<" not in text
        assert "!" not in text

    assert result["medical_abstract"].iloc[2] == ""


def test_clean_dataframe_output_col():
    df = pd.DataFrame(
        {
            "medical_abstract": ["Hello WORLD!"]
        }
    )
    result = clean_dataframe(
        df,
        output_col="clean_text",
        remove_stopwords_flag=False,
        lemmatize_flag=False,
    )
    assert "medical_abstract" in result.columns
    assert "clean_text" in result.columns
    assert result["clean_text"].iloc[0] == "hello world"


def test_clean_text_preserves_meaningful_content():
    medical = "Acute myocardial infarction with ST elevation. Urgent angioplasty required."
    result = clean_text(medical, remove_stopwords_flag=False, lemmatize_flag=False)
    assert "myocardial" in result
    assert "infarction" in result
    assert "angioplasty" in result
