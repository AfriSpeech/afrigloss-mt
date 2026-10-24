"""
Dataset Builder for Afri-Gloss-MT.

Extracts noun phrases from the source text using spaCy noun chunks, looks them up
in the high-confidence dictionary, and replaces them IN PLACE with their target
translation in angle brackets:
  Source: "In the beginning God created the heavens and the earth."
  Input:  "In the beginning <Onyankopɔn> created the <ɔsoro> and the <asase>."
  Target: "Mfiase no <Onyankopɔn> bɔɔ <ɔsoro> ne <asase>."
  Ref:    "Mfiase no Onyankopɔn bɔɔ ɔsoro ne asase."

Defaults to 10,000 parallel verses (or the maximum available if fewer).
"""

import json
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from afrigloss.dictionary import GlossaryDictionary
from afrigloss.extractor import extract_noun_phrases

DEFAULT_TARGET_SAMPLES = 10_000


@dataclass
class GlossMTExample:
    input: str        # Source text with noun phrases replaced in-place by <target_term>
    tgt: str          # Target text with matching <target_term> brackets
    src_raw: str      # Clean original source
    tgt_ref: str      # Clean target reference (no brackets)
    inserted_terms: List[str]  # Terms inserted


def build_gloss_pair_noun_chunks(
    src_text: str,
    tgt_text: str,
    dictionary: GlossaryDictionary,
    max_insertions: int = 3,
    min_confidence: float = 0.65,
) -> Optional[GlossMTExample]:
    """
    Extract noun phrases from source text with spaCy, check if they exist in the dictionary
    and in the target text, and replace them IN PLACE in the source sentence.
    """
    src_clean = src_text.strip()
    tgt_clean = tgt_text.strip()

    # Extract nominal chunks: [(start_char, end_char, noun_phrase_text), ...]
    chunks = extract_noun_phrases(src_clean)
    if not chunks:
        # Fallback to dictionary phrase search if no noun chunks found
        dict_matches = dictionary.find_matches(src_clean)
        chunks = [(start, end, phrase) for start, end, phrase, _ in dict_matches]

    valid_insertions: List[Tuple[int, int, str, str]] = []

    for start, end, phrase in chunks:
        lookup_res = dictionary.lookup(phrase)
        if not lookup_res:
            continue

        conf = dictionary.get_confidence(phrase)
        if conf < min_confidence:
            continue

        # Check if target translation occurs verbatim in the target sentence
        tgt_pattern = re.compile(rf"\b{re.escape(lookup_res)}\b", re.IGNORECASE)
        m_tgt = tgt_pattern.search(tgt_clean)

        if m_tgt:
            # Preserve target text exact casing from target sentence
            tgt_exact = m_tgt.group(0)
            valid_insertions.append((start, end, phrase, tgt_exact))
            if len(valid_insertions) >= max_insertions:
                break

    if not valid_insertions:
        return None

    # Replace from back to front so indices don't shift
    valid_insertions.sort(key=lambda x: x[0], reverse=True)

    src_pretranslated = src_clean
    tgt_bracketed = tgt_clean
    inserted_terms = []

    for start, end, src_phrase, tgt_trans in valid_insertions:
        # In source: replace the noun phrase in place with <target_trans>
        src_pretranslated = src_pretranslated[:start] + f"<{tgt_trans}>" + src_pretranslated[end:]

        # In target: bracket the exact target word
        pattern = re.compile(rf"\b({re.escape(tgt_trans)})\b", re.IGNORECASE)
        tgt_bracketed = pattern.sub(r"<\1>", tgt_bracketed, count=1)
        inserted_terms.append(tgt_trans)

    inserted_terms.reverse()
    return GlossMTExample(
        input=src_pretranslated,
        tgt=tgt_bracketed,
        src_raw=src_clean,
        tgt_ref=tgt_clean,
        inserted_terms=inserted_terms,
    )


# Alias
build_gloss_pair = build_gloss_pair_noun_chunks


def create_gloss_dataset(
    parallel_pairs: List[Tuple[str, str]],
    dictionary: GlossaryDictionary,
    target_count: int = DEFAULT_TARGET_SAMPLES,
    val_ratio: float = 0.05,
    test_ratio: float = 0.05,
    seed: int = 42,
    min_confidence: float = 0.65,
) -> Dict[str, List[GlossMTExample]]:
    """
    Generate train / validation / test splits of Gloss-MT pre-translated examples.
    Picks up to target_count (default 10k) parallel verses, or all available if fewer.
    """
    random.seed(seed)
    selected_pairs = parallel_pairs[:target_count] if len(parallel_pairs) > target_count else parallel_pairs

    examples: List[GlossMTExample] = []

    for src_text, tgt_text in selected_pairs:
        ex = build_gloss_pair_noun_chunks(src_text, tgt_text, dictionary, min_confidence=min_confidence)
        if ex:
            examples.append(ex)

    random.shuffle(examples)
    n = len(examples)
    n_test = int(n * test_ratio)
    n_val = int(n * val_ratio)
    n_train = n - n_val - n_test

    splits = {
        "train": examples[:n_train],
        "validation": examples[n_train : n_train + n_val],
        "test": examples[n_train + n_val :],
    }
    return splits


def save_dataset_splits(splits: Dict[str, List[GlossMTExample]], output_dir: str | Path):
    """Save dataset splits to JSONL files (train.jsonl, validation.jsonl, test.jsonl)."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    for split_name, examples in splits.items():
        split_file = out_path / f"{split_name}.jsonl"
        with open(split_file, "w", encoding="utf-8") as f:
            for ex in examples:
                f.write(json.dumps(asdict(ex), ensure_ascii=False) + "\n")
