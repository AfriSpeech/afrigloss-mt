"""Unit tests for language and model registry."""

import pytest
from afrigloss.registry import (
    COLLECTION_SLUG,
    COLLECTION_URL,
    LANGUAGE_INFO,
    find_model_for_pair,
    normalize_lang_code,
)


def test_normalize_lang_code():
    assert normalize_lang_code("twi") == "twi"
    assert normalize_lang_code("Twi") == "twi"
    assert normalize_lang_code("tw") == "twi"
    assert normalize_lang_code("akan") == "twi"
    assert normalize_lang_code("yor") == "yor"
    assert normalize_lang_code("yo") == "yor"
    assert normalize_lang_code("swh") == "swh"
    assert normalize_lang_code("sw") == "swh"
    assert normalize_lang_code("en") == "eng"
    assert normalize_lang_code("English") == "eng"


def test_find_model_direct_pairs():
    # English -> Twi
    m_en_tw = find_model_for_pair("en", "tw")
    assert m_en_tw is not None
    assert m_en_tw["model_id"] == "Helsinki-NLP/opus-mt-en-tw"
    assert m_en_tw["target_token"] is None

    # Twi -> French
    m_tw_fr = find_model_for_pair("tw", "fr")
    assert m_tw_fr is not None
    assert m_tw_fr["model_id"] == "Helsinki-NLP/opus-mt-tw-fr"

    # English -> Swahili
    m_en_sw = find_model_for_pair("en", "sw")
    assert m_en_sw is not None
    assert m_en_sw["model_id"] == "Helsinki-NLP/opus-mt-en-sw"


def test_find_model_multilingual_family():
    # English -> Zulu (Zulu is part of Atlantic-Congo / Bantu multilingual model)
    m_en_zu = find_model_for_pair("en", "zu")
    assert m_en_zu is not None
    assert "alv" in m_en_zu["model_id"] or "bnt" in m_en_zu["model_id"]
    assert m_en_zu["target_token"] == ">>zul<<"

    # English -> Yoruba via ALV
    m_en_yo = find_model_for_pair("en", "yo")
    # Direct model exists for yo
    assert m_en_yo["model_id"] == "Helsinki-NLP/opus-mt-en-yo"


def test_registry_metadata():
    assert "twi" in LANGUAGE_INFO
    assert LANGUAGE_INFO["twi"]["alpha2"] == "tw"
    assert "AfriSpeech" in COLLECTION_SLUG
