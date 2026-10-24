"""Unit tests for Gloss-MT dataset builder and alignment cache."""

import tempfile
from pathlib import Path
import pytest
from afrigloss.align_cache import AlignmentCache, sentence_pair_hash
from afrigloss.dataset import build_gloss_pair, create_gloss_dataset
from afrigloss.dictionary import GlossaryDictionary
from afrigloss.trainer import clean_output


def test_sentence_pair_hash():
    h1 = sentence_pair_hash("Hello world", "Nkyea wiase")
    h2 = sentence_pair_hash("Hello world", "Nkyea wiase")
    h3 = sentence_pair_hash("Hello world!", "Nkyea wiase")
    assert h1 == h2
    assert h1 != h3


def test_alignment_cache():
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
        cache_path = Path(f.name)

    try:
        cache = AlignmentCache(cache_path)
        assert len(cache) == 0

        alignments = [{"start": 0, "end": 3, "src_phrase": "God", "tgt_trans": "Onyankopɔn", "score": 0.99}]
        cache.set("God is good", "Onyankopɔn ye", alignments)
        assert cache.contains("God is good", "Onyankopɔn ye")
        assert len(cache) == 1

        retrieved = cache.get("God is good", "Onyankopɔn ye")
        assert retrieved == alignments
        cache.close()

        # Reload cache from disk
        cache_reloaded = AlignmentCache(cache_path)
        assert len(cache_reloaded) == 1
        assert cache_reloaded.contains("God is good", "Onyankopɔn ye")
        cache_reloaded.close()
    finally:
        cache_path.unlink(missing_ok=True)


def test_build_gloss_pair():
    d = GlossaryDictionary()
    d.add("god", "Onyankopɔn", confidence=0.98)
    d.add("earth", "asase", confidence=0.94)

    src = "God created the earth."
    tgt = "Onyankopɔn bɔɔ asase."

    ex = build_gloss_pair(src, tgt, d)
    assert ex is not None
    assert "<Onyankopɔn>" in ex.input
    assert "<asase>" in ex.input
    assert "<Onyankopɔn>" in ex.tgt
    assert "<asase>" in ex.tgt
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

    splits = create_gloss_dataset(pairs, d, target_count=10, val_ratio=0.2, test_ratio=0.1)
    assert "train" in splits
    assert "validation" in splits
    assert "test" in splits
    assert len(splits["train"]) + len(splits["validation"]) + len(splits["test"]) == 10
