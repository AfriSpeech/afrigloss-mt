"""Unit tests for dictionary and glossary lookup."""

import tempfile
from pathlib import Path
import pytest
from afrigloss.dictionary import GlossaryDictionary, build_dictionary_from_aligner


def test_dictionary_add_and_lookup():
    d = GlossaryDictionary()
    d.add("covenant", "apam", confidence=0.95)
    d.add("mountain", "bepɔ", confidence=0.88)

    assert d.lookup("covenant") == "apam"
    assert d.lookup("COVENANT") == "apam"
    assert d.lookup("mountain") == "bepɔ"
    assert d.lookup("unknown") is None
    assert d.get_confidence("covenant") == 0.95


def test_longest_match_first():
    d = GlossaryDictionary()
    d.add("holy", "kronkron")
    d.add("holy spirit", "honhom kronkron")

    text = "The power of the holy spirit was upon them."
    matches = d.find_matches(text)

    # Should match 'holy spirit' as a 2-word phrase rather than just 'holy'
    assert len(matches) == 1
    start, end, src, tgt = matches[0]
    assert src.lower() == "holy spirit"
    assert tgt == "honhom kronkron"


def test_tsv_roundtrip():
    d = GlossaryDictionary()
    d.add("god", "onyankopɔn", 0.99)
    d.add("earth", "asase", 0.92)

    with tempfile.NamedTemporaryFile(suffix=".tsv", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        d.save_tsv(tmp_path)
        loaded = GlossaryDictionary.load_tsv(tmp_path)
        assert len(loaded) == 2
        assert loaded.lookup("god") == "onyankopɔn"
        assert loaded.lookup("earth") == "asase"
        assert abs(loaded.get_confidence("god") - 0.99) < 1e-3
    finally:
        tmp_path.unlink(missing_ok=True)
