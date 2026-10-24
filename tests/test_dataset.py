"""Unit tests for Gloss-MT dataset builder."""

import pytest
from afrigloss.dataset import build_gloss_pair, create_gloss_dataset
from afrigloss.dictionary import GlossaryDictionary
from afrigloss.trainer import clean_output


def test_build_gloss_pair():
    d = GlossaryDictionary()
    d.add("god", "Onyankopɔn", confidence=0.98)
    d.add("earth", "asase", confidence=0.94)

    src = "God created the earth."
    tgt = "Onyankopɔn bɔɔ asase."

    ex = build_gloss_pair(src, tgt, d)
    assert ex is not None
    # Source input should have pre-inserted target terms
    assert "<Onyankopɔn>" in ex.input
    assert "<asase>" in ex.input
    # Target should have matching bracketed terms
    assert "<Onyankopɔn>" in ex.tgt
    assert "<asase>" in ex.tgt
    # Plain target should remain clean
    assert ex.tgt_ref == "Onyankopɔn bɔɔ asase."
    assert clean_output(ex.tgt) == "Onyankopɔn bɔɔ asase."


def test_create_gloss_dataset_splits():
    d = GlossaryDictionary()
    d.add("god", "Onyankopɔn", confidence=0.95)

    pairs = [
        ("God spoke to them.", "Onyankopɔn kyerɛɛ wɔn."),
        ("God heard the prayer.", "Onyankopɔn tee mpaebɔ no."),
        ("They thanked God.", "Wɔda Onyankopɔn ase."),
        ("God is love.", "Onyankopɔn yɛ dɔ."),
        ("The house of God.", "Onyankopɔn fie."),
        ("God gave life.", "Onyankopɔn maa nkwa."),
        ("The law of God.", "Onyankopɔn mmara."),
        ("Peace of God.", "Onyankopɔn asomdwoe."),
        ("Sing to God.", "To dwom ma Onyankopɔn."),
        ("Fear God.", "Suro Onyankopɔn."),
    ]

    splits = create_gloss_dataset(pairs, d, val_ratio=0.2, test_ratio=0.1)
    assert "train" in splits
    assert "validation" in splits
    assert "test" in splits
    assert len(splits["train"]) + len(splits["validation"]) + len(splits["test"]) == 10
