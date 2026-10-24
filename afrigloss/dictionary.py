"""
Bilingual Dictionary Builder and Offline Lookup Engine for Afri-Gloss-MT.

Builds, filters, and queries high-confidence bilingual dictionaries.
Stores entries in clean TSV format (src_phrase \t tgt_phrase \t confidence),
compatible with offline inference and phrase matching.
"""

import csv
import gzip
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


class GlossaryDictionary:
    """
    Offline glossary dictionary mapping source terms/phrases to target translations.
    Uses longest-match greedy phrase matching for fast runtime lookup.
    """

    def __init__(self, entries: Optional[Dict[str, str]] = None):
        # src_term (lowercase) -> tgt_translation
        self.entries: Dict[str, str] = {}
        # src_term -> confidence score
        self.confidences: Dict[str, float] = {}

        if entries:
            for k, v in entries.items():
                self.add(k, v)

    def add(self, src_term: str, tgt_term: str, confidence: float = 1.0):
        """Add or update a term mapping."""
        k = src_term.strip().lower()
        v = tgt_term.strip()
        if k and v:
            self.entries[k] = v
            self.confidences[k] = confidence

    def __len__(self) -> int:
        return len(self.entries)

    def lookup(self, term: str) -> Optional[str]:
        """Look up exact term (case-insensitive)."""
        return self.entries.get(term.strip().lower())

    def get_confidence(self, term: str) -> float:
        """Get the confidence score for a term."""
        return self.confidences.get(term.strip().lower(), 0.0)

    def save_tsv(self, path: str | Path):
        """Save dictionary to TSV or gzipped TSV."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)

        is_gz = p.name.endswith(".gz")
        open_fn = gzip.open if is_gz else open

        with open_fn(p, "wt", encoding="utf-8") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow(["source_phrase", "target_phrase", "confidence"])
            for src in sorted(self.entries.keys()):
                tgt = self.entries[src]
                conf = self.confidences.get(src, 1.0)
                writer.writerow([src, tgt, f"{conf:.4f}"])

    @classmethod
    def load_tsv(cls, path: str | Path) -> "GlossaryDictionary":
        """Load dictionary from TSV or gzipped TSV."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Dictionary file not found: {p}")

        is_gz = p.name.endswith(".gz")
        open_fn = gzip.open if is_gz else open

        instance = cls()
        with open_fn(p, "rt", encoding="utf-8", errors="replace") as f:
            reader = csv.reader(f, delimiter="\t")
            header = next(reader, None)
            for row in reader:
                if len(row) >= 2:
                    src = row[0].strip()
                    tgt = row[1].strip()
                    conf = float(row[2]) if len(row) >= 3 else 1.0
                    instance.add(src, tgt, conf)

        return instance

    def find_matches(self, text: str) -> List[Tuple[int, int, str, str]]:
        """
        Find non-overlapping glossary phrase matches in text.
        Uses longest-match greedy phrase lookup.

        Returns:
            List of (start_char_idx, end_char_idx, matched_src_term, tgt_translation)
        """
        matches: List[Tuple[int, int, str, str]] = []
        text_lower = text.lower()
        words = list(re.finditer(r"\b[\w'-]+\b", text))

        if not words:
            return matches

        occupied: Set[int] = set()

        # Check multi-word n-grams first (e.g. 4-grams down to 1-grams)
        max_n = 5
        n_words = len(words)

        for n in range(min(max_n, n_words), 0, -1):
            for i in range(n_words - n + 1):
                span_indices = set(range(i, i + n))
                if span_indices.intersection(occupied):
                    continue

                start_char = words[i].start()
                end_char = words[i + n - 1].end()
                phrase = text[start_char:end_char].strip().lower()

                if phrase in self.entries:
                    tgt = self.entries[phrase]
                    matches.append((start_char, end_char, phrase, tgt))
                    occupied.update(span_indices)

        # Sort matches by character position
        matches.sort(key=lambda m: m[0])
        return matches


def build_dictionary_from_aligner(
    aligner_lexicon: Dict[str, Dict[str, float]],
    min_confidence: float = 0.70,
    top_k_per_source: int = 1,
) -> GlossaryDictionary:
    """
    Build a GlossaryDictionary from raw aligner association scores.
    Keeps only top_k translation per source word exceeding min_confidence.
    """
    dict_instance = GlossaryDictionary()

    for sw, candidates in aligner_lexicon.items():
        # Sort candidate targets by score descending
        sorted_cands = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
        for tw, score in sorted_cands[:top_k_per_source]:
            if score >= min_confidence:
                dict_instance.add(sw, tw, score)

    return dict_instance


def merge_dictionaries(
    dict_a: GlossaryDictionary,
    dict_b: GlossaryDictionary,
    prefer_higher_confidence: bool = True,
) -> GlossaryDictionary:
    """
    Merge two dictionaries together.
    On duplicate keys, resolves conflicts by retaining the higher confidence translation.
    """
    merged = GlossaryDictionary(dict(dict_a.entries))
    merged.confidences = dict(dict_a.confidences)

    for k, v in dict_b.entries.items():
        conf_b = dict_b.confidences.get(k, 1.0)
        if k not in merged.entries:
            merged.add(k, v, conf_b)
        else:
            conf_a = merged.confidences.get(k, 1.0)
            if prefer_higher_confidence and conf_b > conf_a:
                merged.add(k, v, conf_b)

    return merged


# Alias for convenience
AfricanGlossary = GlossaryDictionary
