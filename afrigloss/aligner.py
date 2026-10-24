"""
Word Alignment Engine with Confidence Scoring for Afri-Gloss-MT.

Provides:
  1. StatisticalWordAligner: Fast corpus-level co-occurrence and bidirectional
     conditional alignment probabilities with Dice and association scoring.
  2. EmbeddingWordAligner: Contextual similarity alignment using multilingual
     embeddings (SimAlign style) with cosine similarity confidence scores.
  3. High-confidence alignment filtering to extract verified word/phrase pairs.
"""

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

# Basic stopword filters to prevent trivial alignments
DEFAULT_STOPWORDS = {
    # English
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of",
    "with", "by", "from", "up", "about", "into", "over", "after", "is", "are",
    "was", "were", "be", "been", "being", "have", "has", "had", "do", "does",
    "did", "shall", "will", "should", "would", "may", "might", "must", "can",
    "could", "that", "this", "these", "those", "it", "its", "as", "if",
    # Punctuation / symbols
    ".", ",", ";", ":", "!", "?", "\"", "'", "-", "--", "(", ")", "[", "]",
}


@dataclass
class WordAlignment:
    src_word: str
    tgt_word: str
    src_idx: int
    tgt_idx: int
    confidence: float  # Score in [0.0, 1.0]


def tokenize_words(text: str) -> List[str]:
    """Tokenize sentence into clean words, preserving language-specific unicode."""
    # Matches words with apostrophes/hyphens inside
    tokens = re.findall(r"\b[\w'-]+\b|[^\w\s]", text, re.UNICODE)
    return [t.strip() for t in tokens if t.strip()]


class StatisticalWordAligner:
    """
    Corpus-level statistical word aligner using co-occurrence,
    bidirectional conditional probabilities, and Dice coefficients.
    """

    def __init__(self, min_pair_freq: int = 3, min_confidence: float = 0.65):
        self.min_pair_freq = min_pair_freq
        self.min_confidence = min_confidence

        self.src_freq: Counter = Counter()
        self.tgt_freq: Counter = Counter()
        self.cooc_freq: Dict[str, Counter] = defaultdict(Counter)
        self.total_pairs: int = 0
        self.is_trained: bool = False

    def train(self, sentence_pairs: List[Tuple[str, str]]):
        """Train alignment statistics on a list of (src_sentence, tgt_sentence) pairs."""
        self.src_freq.clear()
        self.tgt_freq.clear()
        self.cooc_freq.clear()
        self.total_pairs = len(sentence_pairs)

        for src_sent, tgt_sent in sentence_pairs:
            src_words = set(w.lower() for w in tokenize_words(src_sent) if w.lower() not in DEFAULT_STOPWORDS)
            tgt_words = set(w.lower() for w in tokenize_words(tgt_sent) if w.lower() not in DEFAULT_STOPWORDS)

            for sw in src_words:
                self.src_freq[sw] += 1
            for tw in tgt_words:
                self.tgt_freq[tw] += 1

            for sw in src_words:
                for tw in tgt_words:
                    self.cooc_freq[sw][tw] += 1

        self.is_trained = True

    def get_association_score(self, src_word: str, tgt_word: str) -> float:
        """
        Compute bidirectional association confidence between src_word and tgt_word.
        Combines Dice coefficient with harmonic mean of conditional probabilities.
        """
        sw = src_word.lower()
        tw = tgt_word.lower()

        c_pair = self.cooc_freq.get(sw, {}).get(tw, 0)
        if c_pair < self.min_pair_freq:
            return 0.0

        c_s = self.src_freq.get(sw, 0)
        c_t = self.tgt_freq.get(tw, 0)
        if c_s == 0 or c_t == 0:
            return 0.0

        # Conditional probabilities
        p_tgt_given_src = c_pair / c_s
        p_src_given_tgt = c_pair / c_t

        # Dice score: 2 * C(s, t) / (C(s) + C(t))
        dice = (2.0 * c_pair) / (c_s + c_t)

        # Harmonic mean of conditional probabilities
        h_prob = (2.0 * p_tgt_given_src * p_src_given_tgt) / (p_tgt_given_src + p_src_given_tgt + 1e-9)

        # Combined confidence metric
        confidence = 0.5 * dice + 0.5 * h_prob
        return min(1.0, max(0.0, confidence))

    def align_sentence(
        self,
        src_sentence: str,
        tgt_sentence: str,
        min_confidence: Optional[float] = None,
    ) -> List[WordAlignment]:
        """
        Align words in a single sentence pair.
        Returns high-confidence word alignments.
        """
        threshold = min_confidence if min_confidence is not None else self.min_confidence
        src_tokens = tokenize_words(src_sentence)
        tgt_tokens = tokenize_words(tgt_sentence)

        alignments: List[WordAlignment] = []
        best_tgt_for_src: Dict[int, Tuple[int, float]] = {}

        for i, sw in enumerate(src_tokens):
            if sw.lower() in DEFAULT_STOPWORDS or len(sw) <= 2:
                continue

            best_j = -1
            best_score = 0.0
            for j, tw in enumerate(tgt_tokens):
                if tw.lower() in DEFAULT_STOPWORDS or len(tw) <= 2:
                    continue

                score = self.get_association_score(sw, tw)
                if score > best_score:
                    best_score = score
                    best_j = j

            if best_j != -1 and best_score >= threshold:
                best_tgt_for_src[i] = (best_j, best_score)

        for i, (j, score) in best_tgt_for_src.items():
            alignments.append(
                WordAlignment(
                    src_word=src_tokens[i],
                    tgt_word=tgt_tokens[j],
                    src_idx=i,
                    tgt_idx=j,
                    confidence=score,
                )
            )

        return alignments

    def extract_bilingual_lexicon(
        self,
        min_confidence: Optional[float] = None,
        min_pair_count: Optional[int] = None,
    ) -> Dict[str, Dict[str, float]]:
        """
        Extract the global bilingual dictionary from the trained corpus
        containing only alignments meeting the confidence threshold.

        Returns:
            Dict[src_word, Dict[tgt_word, confidence_score]]
        """
        threshold = min_confidence if min_confidence is not None else self.min_confidence
        min_cnt = min_pair_count if min_pair_count is not None else self.min_pair_freq

        lexicon: Dict[str, Dict[str, float]] = defaultdict(dict)

        for sw, targets in self.cooc_freq.items():
            if len(sw) <= 2 or sw in DEFAULT_STOPWORDS:
                continue
            for tw, count in targets.items():
                if count < min_cnt or len(tw) <= 2 or tw in DEFAULT_STOPWORDS:
                    continue

                conf = self.get_association_score(sw, tw)
                if conf >= threshold:
                    lexicon[sw][tw] = round(conf, 4)

        return dict(lexicon)


class EmbeddingWordAligner:
    """
    Embedding-based word aligner using cross-lingual sentence or word embeddings.
    Computes cosine similarity matrix and bidirectional matching.
    """

    def __init__(self, model_name: str = "sentence-transformers/LaBSE", min_confidence: float = 0.70):
        self.model_name = model_name
        self.min_confidence = min_confidence
        self._model = None
        self._tokenizer = None

    def _lazy_load(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name)
            except Exception as e:
                raise RuntimeError(
                    f"EmbeddingWordAligner requires sentence-transformers. Error: {e}"
                )

    def align_sentence(
        self,
        src_sentence: str,
        tgt_sentence: str,
        min_confidence: Optional[float] = None,
    ) -> List[WordAlignment]:
        """Align words by embedding similarity."""
        threshold = min_confidence if min_confidence is not None else self.min_confidence
        self._lazy_load()

        src_words = tokenize_words(src_sentence)
        tgt_words = tokenize_words(tgt_sentence)

        clean_src = [(i, w) for i, w in enumerate(src_words) if len(w) > 2 and w.lower() not in DEFAULT_STOPWORDS]
        clean_tgt = [(j, w) for j, w in enumerate(tgt_words) if len(w) > 2 and w.lower() not in DEFAULT_STOPWORDS]

        if not clean_src or not clean_tgt:
            return []

        src_embs = self._model.encode([w for _, w in clean_src], normalize_embeddings=True)
        tgt_embs = self._model.encode([w for _, w in clean_tgt], normalize_embeddings=True)

        alignments: List[WordAlignment] = []
        for s_idx, (orig_i, sw) in enumerate(clean_src):
            best_t_idx = -1
            best_sim = 0.0
            for t_idx, (orig_j, tw) in enumerate(clean_tgt):
                # Cosine similarity of normalized vectors is dot product
                sim = float((src_embs[s_idx] * tgt_embs[t_idx]).sum())
                if sim > best_sim:
                    best_sim = sim
                    best_t_idx = t_idx

            if best_t_idx != -1 and best_sim >= threshold:
                orig_j, tw = clean_tgt[best_t_idx]
                alignments.append(
                    WordAlignment(
                        src_word=sw,
                        tgt_word=tw,
                        src_idx=orig_i,
                        tgt_idx=orig_j,
                        confidence=round(best_sim, 4),
                    )
                )

        return alignments
