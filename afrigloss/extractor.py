"""
Noun phrase and terminology extraction for Afri-Gloss-MT.

Extracts nominal phrases and terminology from source text using spaCy noun chunks,
filtering out leading determiners/articles, following GhanaNLP's nkrane library convention.
"""

import re
from typing import List, Optional, Set, Tuple

# Leading articles/determiners/possessives to strip from noun chunks
LEADING_DETERMINERS: Set[str] = {
    "the", "a", "an", "this", "that", "these", "those",
    "my", "your", "his", "her", "its", "our", "their",
    "some", "any", "each", "every", "such", "no", "another",
}

_NLP = None


def get_spacy_nlp(model_name: str = "en_core_web_sm"):
    """Lazy load spaCy parser with only tagger and parser for high throughput."""
    global _NLP
    if _NLP is None:
        try:
            import spacy
            # Only enable parser and tagger needed for noun_chunks
            _NLP = spacy.load(model_name, disable=["ner", "lemmatizer", "textcat"])
        except Exception:
            _NLP = None
    return _NLP


def extract_noun_phrases(text: str, nlp=None) -> List[Tuple[int, int, str]]:
    """
    Extract nominal phrases from source text using spaCy noun chunks.

    Returns:
        List of (start_char, end_char, cleaned_noun_phrase)
    """
    nlp = nlp or get_spacy_nlp()
    if nlp is None:
        # Fallback regex extraction of capitalized and content words
        results = []
        for m in re.finditer(r"\b[A-Za-z][a-z]{2,}\b", text):
            w = m.group(0)
            if w.lower() not in LEADING_DETERMINERS:
                results.append((m.start(), m.end(), w))
        return results

    doc = nlp(text)
    noun_phrases: List[Tuple[int, int, str]] = []

    for chunk in doc.noun_chunks:
        # Find start of content tokens, dropping leading determiners
        tokens = list(chunk)
        start_tok_idx = 0
        while start_tok_idx < len(tokens) and tokens[start_tok_idx].text.lower() in LEADING_DETERMINERS:
            start_tok_idx += 1

        if start_tok_idx >= len(tokens):
            continue

        content_tokens = tokens[start_tok_idx:]
        # Skip chunks that are entirely punctuation or single-letter
        phrase_text = " ".join(t.text for t in content_tokens).strip()
        if len(phrase_text) < 2:
            continue

        # Get exact character start and end in the original text
        start_char = content_tokens[0].idx
        end_char = content_tokens[-1].idx + len(content_tokens[-1].text)
        noun_phrases.append((start_char, end_char, phrase_text))

    return noun_phrases
