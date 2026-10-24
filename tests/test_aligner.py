"""Unit tests for word alignment engine."""

import pytest
from afrigloss.aligner import StatisticalWordAligner, tokenize_words


def test_tokenize_words():
    text = "In the beginning, God created the heavens and the earth."
    tokens = tokenize_words(text)
    assert "beginning" in tokens
    assert "God" in tokens
    assert "heavens" in tokens


def test_statistical_aligner_confidence():
    pairs = [
        ("God created the heavens and the earth.", "Onyankopɔn bɔɔ ɔsoro ne asase."),
        ("God said let there be light.", "Onyankopɔn kae sɛ hann nna hɔ."),
        ("God called the light day.", "Onyankopɔn frɛɛ hann no da."),
        ("God saw that it was good.", "Onyankopɔn hunuu sɛ ɛye."),
        ("The heavens declare the glory of God.", "Ɔsoro ka Onyankopɔn animuonyam kyerɛ."),
    ]

    aligner = StatisticalWordAligner(min_pair_freq=2, min_confidence=0.5)
    aligner.train(pairs)

    # 'god' <-> 'onyankopɔn' should have near-perfect confidence
    god_conf = aligner.get_association_score("god", "onyankopɔn")
    assert god_conf >= 0.80

    # Align a sentence
    alignments = aligner.align_sentence("God created the heavens.", "Onyankopɔn bɔɔ ɔsoro.", min_confidence=0.5)
    assert len(alignments) >= 1
    aligned_words = {a.src_word.lower(): a.tgt_word.lower() for a in alignments}
    assert aligned_words.get("god") == "onyankopɔn"


def test_extract_lexicon():
    pairs = [
        ("Moses went up the mountain of God.", "Mose kɔɔ Onyankopɔn bepɔ no so."),
        ("Moses spoke to God.", "Mose ne Onyankopɔn kaseɛ."),
        ("Moses led the people.", "Mose dii nnipa no anim."),
    ]
    aligner = StatisticalWordAligner(min_pair_freq=2, min_confidence=0.6)
    aligner.train(pairs)

    lexicon = aligner.extract_bilingual_lexicon(min_confidence=0.6)
    assert "moses" in lexicon
    assert "mose" in lexicon["moses"]
