"""
Dataset Builder for Afri-Gloss-MT.

Extracts noun phrases from the source text using spaCy noun chunks, aligns them to
their target language translations using full-corpus statistical association or
multilingual E5 embeddings, and replaces them IN PLACE with their target translation
in angle brackets:

  Source: "In the beginning God created the heavens and the earth."
  Input:  "In the beginning <Onyankopɔn> created the <ɔsoro> and the <asase>."
  Target: "Mfiase no <Onyankopɔn> bɔɔ <ɔsoro> ne <asase>."
  Ref:    "Mfiase no Onyankopɔn bɔɔ ɔsoro ne asase."

Guarantees at least one high-confidence aligned word/phrase across each sample in
the 10,000 target dataset.
"""

import json
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from afrigloss.aligner import StatisticalWordAligner, tokenize_words
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


def build_gloss_pair(
    src_text: str,
    tgt_text: str,
    dictionary: Optional[GlossaryDictionary] = None,
    aligner: Optional[StatisticalWordAligner] = None,
    max_insertions: int = 3,
    min_confidence: float = 0.45,
) -> Optional[GlossMTExample]:
    """
    Extract noun phrases from source text with spaCy, find the best matching target word,
    and replace the noun phrase IN PLACE in the source sentence with '<target_term>'.
    """
    src_clean = src_text.strip()
    tgt_clean = tgt_text.strip()

    # 1. Extract nominal chunks from source sentence
    chunks = extract_noun_phrases(src_clean)
    if not chunks:
        return None

    # Target words available for alignment in this sentence
    tgt_tokens = tokenize_words(tgt_clean)
    tgt_words_lower = set(w.lower() for w in tgt_tokens)

    valid_insertions: List[Tuple[int, int, str, str, float]] = []

    # 2. Score candidate matches for each noun chunk
    for start, end, phrase in chunks:
        phrase_clean = phrase.lower().strip()
        best_target = None
        best_score = 0.0

        # Check dictionary first
        if dictionary:
            dict_trans = dictionary.lookup(phrase_clean)
            if dict_trans and dict_trans.lower() in tgt_words_lower:
                best_target = dict_trans
                best_score = max(best_score, dictionary.get_confidence(phrase_clean))

        # Check statistical aligner association table
        if aligner and aligner.is_trained:
            words = phrase_clean.split()
            for w in words:
                candidates = tgt_words_lower.intersection(aligner.cooc_freq.get(w, {}).keys())
                for tw in candidates:
                    score = aligner.get_association_score(w, tw)
                    if score > best_score:
                        best_score = score
                        best_target = tw

        if best_target and best_score >= min_confidence:
            # Find the exact matched casing in the target text
            pattern = re.compile(rf"\b{re.escape(best_target)}\b", re.IGNORECASE)
            m_tgt = pattern.search(tgt_clean)
            if m_tgt:
                exact_tgt_text = m_tgt.group(0)
                valid_insertions.append((start, end, phrase, exact_tgt_text, best_score))

    if not valid_insertions:
        return None

    # Sort candidates by score descending to take top matches
    valid_insertions.sort(key=lambda x: x[4], reverse=True)
    top_insertions = valid_insertions[:max_insertions]

    # Sort selected insertions in reverse character order for in-place replacement
    top_insertions.sort(key=lambda x: x[0], reverse=True)

    src_pretranslated = src_clean
    tgt_bracketed = tgt_clean
    inserted_terms = []

    for start, end, src_phrase, tgt_trans, _ in top_insertions:
        # Match casing
        if src_phrase.isupper():
            trans_repr = tgt_trans.upper()
        elif src_phrase[0].isupper():
            trans_repr = tgt_trans[0].upper() + tgt_trans[1:]
        else:
            trans_repr = tgt_trans.lower()

        # In source: replace the noun phrase in place with <target_trans>
        src_pretranslated = src_pretranslated[:start] + f"<{trans_repr}>" + src_pretranslated[end:]

        # In target: wrap exact matched target word with <...>
        sub_pat = re.compile(rf"(?<!<)\b({re.escape(tgt_trans)})\b(?!>)", re.IGNORECASE)
        tgt_bracketed = sub_pat.sub(r"<\1>", tgt_bracketed, count=1)
        inserted_terms.append(tgt_trans)

    inserted_terms.reverse()
    return GlossMTExample(
        input=src_pretranslated,
        tgt=tgt_bracketed,
        src_raw=src_clean,
        tgt_ref=tgt_clean,
        inserted_terms=inserted_terms,
    )


def create_gloss_dataset(
    parallel_pairs: List[Tuple[str, str]],
    dictionary: Optional[GlossaryDictionary] = None,
    aligner: Optional[StatisticalWordAligner] = None,
    target_count: int = DEFAULT_TARGET_SAMPLES,
    val_ratio: float = 0.05,
    test_ratio: float = 0.05,
    seed: int = 42,
    min_confidence: float = 0.45,
    ensure_at_least_one: bool = True,
) -> Dict[str, List[GlossMTExample]]:
    """
    Generate train / validation / test splits of Gloss-MT pre-translated examples.
    Guarantees every sample in the target_count (default 10k) has at least one
    high-confidence aligned noun chunk inserted.
    """
    random.seed(seed)
    examples: List[GlossMTExample] = []

    # Iterate over available parallel pairs until we collect target_count valid annotated examples
    for src_text, tgt_text in parallel_pairs:
        ex = build_gloss_pair(
            src_text,
            tgt_text,
            dictionary=dictionary,
            aligner=aligner,
            min_confidence=min_confidence,
        )
        if ex and len(ex.inserted_terms) >= 1:
            examples.append(ex)
            if len(examples) >= target_count:
                break

    print(f"[AfriGloss Dataset] Collected {len(examples):,} examples with guaranteed glossary insertions.")

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
