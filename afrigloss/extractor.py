"""
Noun phrase and terminology extraction for Afri-Gloss-MT.

Extracts nominal phrases from English and French source text using spaCy noun chunks,
filtering out leading determiners/articles, following GhanaNLP's nkrane library convention.
"""

import re
from typing import List, Optional, Set, Tuple

# Leading articles/determiners/possessives to strip from noun chunks
ENGLISH_LEADING_DETERMINERS: Set[str] = {
    "the", "a", "an", "this", "that", "these", "those",
    "my", "your", "his", "her", "its", "our", "their",
    "some", "any", "each", "every", "such", "no", "another",
}

FRENCH_LEADING_DETERMINERS: Set[str] = {
    "le", "la", "les", "un", "une", "des", "du", "de", "d'", "l'",
    "ce", "cet", "cette", "ces", "mon", "ma", "mes", "ton", "ta", "tes",
    "son", "sa", "ses", "notre", "nos", "votre", "vos", "leur", "leurs",
}

_NLP_EN = None
_NLP_FR = None


def get_spacy_nlp(lang: str = "en"):
    """Lazy load spaCy parser for English or French."""
    global _NLP_EN, _NLP_FR
    import spacy

    is_french = lang.lower().strip() in ("fr", "fra", "french")
    if is_french:
        if _NLP_FR is None:
            try:
                _NLP_FR = spacy.load("fr_core_news_sm", disable=["ner", "lemmatizer", "textcat"])
            except Exception:
                _NLP_FR = None
        return _NLP_FR
    else:
        if _NLP_EN is None:
            try:
                _NLP_EN = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer", "textcat"])
            except Exception:
                _NLP_EN = None
        return _NLP_EN


def extract_noun_phrases(text: str, lang: str = "en", nlp=None) -> List[Tuple[int, int, str]]:
    """
    Extract nominal phrases from source text (English or French) using spaCy noun chunks.

    Returns:
        List of (start_char, end_char, cleaned_noun_phrase)
    """
    is_french = lang.lower().strip() in ("fr", "fra", "french")
    nlp = nlp or get_spacy_nlp(lang)
    drop_words = FRENCH_LEADING_DETERMINERS if is_french else ENGLISH_LEADING_DETERMINERS

    if nlp is None:
        # Fallback regex extraction of capitalized and content words
        results = []
        for m in re.finditer(r"\b[A-Za-zÀ-ÿ][a-zà-ÿ]{2,}\b", text):
            w = m.group(0)
            if w.lower() not in drop_words:
                results.append((m.start(), m.end(), w))
        return results

    doc = nlp(text)
    noun_phrases: List[Tuple[int, int, str]] = []

    for chunk in doc.noun_chunks:
        tokens = list(chunk)
        start_tok_idx = 0
        while start_tok_idx < len(tokens) and tokens[start_tok_idx].text.lower() in drop_words:
            start_tok_idx += 1

        if start_tok_idx >= len(tokens):
            continue

        content_tokens = tokens[start_tok_idx:]
        phrase_text = " ".join(t.text for t in content_tokens).strip()
        if len(phrase_text) < 2:
            continue

        start_char = content_tokens[0].idx
        end_char = content_tokens[-1].idx + len(content_tokens[-1].text)
        noun_phrases.append((start_char, end_char, phrase_text))

    return noun_phrases
