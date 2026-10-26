"""
Dataset Builder for Afri-Gloss-MT.

Extracts noun phrases from source text using spaCy noun chunks, aligns them to
target language translations, and replaces them IN PLACE with their target translation
in angle brackets:

  Source: "In the beginning God created the heavens and the earth."
  Input:  "In the beginning <Onyankopɔn> created the <ɔsoro> and the <asase>."
  Target: "Mfiase no <Onyankopɔn> bɔɔ <ɔsoro> ne <asase>."
  Ref:    "Mfiase no Onyankopɔn bɔɔ ɔsoro ne asase."

Supports persistent on-disk alignment caching: if parallel samples have already been
aligned in a previous run, they are loaded instantly and skipped, allowing the user to
scale to 10k, 50k, or 100k+ samples incrementally.
"""

import json
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

from afrigloss.align_cache import AlignmentCache
from afrigloss.aligner import DEFAULT_STOPWORDS, StatisticalWordAligner, tokenize_words
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
    align_cache: Optional[AlignmentCache] = None,
    precomputed_aligner_map: Optional[Dict[str, Tuple[str, float]]] = None,
    max_insertions: int = 3,
    min_confidence: float = 0.45,
    src_lang: str = "en",
) -> Optional[GlossMTExample]:
    """
    Extract noun phrases from source text with spaCy, find the best matching target word,
    and replace the noun phrase IN PLACE in the source sentence with '<target_term>'.

    Checks align_cache first to avoid redundant computation.
    """
    src_clean = src_text.strip()
    tgt_clean = tgt_text.strip()

    # 1. Check if alignment is already cached
    cached_insertions = None
    if align_cache and align_cache.contains(src_clean, tgt_clean):
        cached_insertions = align_cache.get(src_clean, tgt_clean)

    valid_insertions: List[Tuple[int, int, str, str, float]] = []

    if cached_insertions is not None:
        # Load from cache
        for item in cached_insertions:
            valid_insertions.append((
                item["start"],
                item["end"],
                item["src_phrase"],
                item["tgt_trans"],
                item.get("score", 1.0)
            ))
    else:
        # Compute alignments (supports English or French noun chunks)
        chunks = extract_noun_phrases(src_clean, lang=src_lang)
        if not chunks:
            return None

        tgt_tokens = tokenize_words(tgt_clean)
        tgt_words_lower = set(w.lower() for w in tgt_tokens)

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

            # Fast check using precomputed aligner map
            if precomputed_aligner_map:
                words = phrase_clean.split()
                for w in words:
                    if w in precomputed_aligner_map:
                        cand_t, cand_s = precomputed_aligner_map[w]
                        if cand_t.lower() in tgt_words_lower and cand_s > best_score:
                            best_score = cand_s
                            best_target = cand_t
            elif aligner and aligner.is_trained:
                words = phrase_clean.split()
                for w in words:
                    candidates = tgt_words_lower.intersection(aligner.cooc_freq.get(w, {}).keys())
                    for tw in candidates:
                        score = aligner.get_association_score(w, tw)
                        if score > best_score:
                            best_score = score
                            best_target = tw

            if best_target and best_score >= min_confidence:
                pattern = re.compile(rf"\b{re.escape(best_target)}\b", re.IGNORECASE)
                m_tgt = pattern.search(tgt_clean)
                if m_tgt:
                    exact_tgt_text = m_tgt.group(0)
                    valid_insertions.append((start, end, phrase, exact_tgt_text, best_score))

        # Store in cache if active
        if align_cache is not None:
            cache_records = [
                {"start": s, "end": e, "src_phrase": sp, "tgt_trans": tt, "score": sc}
                for s, e, sp, tt, sc in valid_insertions
            ]
            align_cache.set(src_clean, tgt_clean, cache_records)

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
    align_cache: Optional[Union[AlignmentCache, str, Path]] = None,
    target_count: Optional[int] = DEFAULT_TARGET_SAMPLES,
    val_ratio: float = 0.05,
    test_ratio: float = 0.05,
    seed: int = 42,
    min_confidence: float = 0.45,
    src_lang: str = "en",
) -> Dict[str, List[GlossMTExample]]:
    """
    Generate train / validation / test splits of Gloss-MT pre-translated examples.
    Guarantees every sample in the target_count (e.g. 10k or 100k) has at least one
    high-confidence aligned noun chunk inserted.

    Automatically uses on-disk alignment caching to skip already-aligned pairs.
    """
    random.seed(seed)
    examples: List[GlossMTExample] = []

    cache_inst = None
    if isinstance(align_cache, (str, Path)):
        cache_inst = AlignmentCache(align_cache)
    elif isinstance(align_cache, AlignmentCache):
        cache_inst = align_cache

    skipped_cache_hits = 0

    # Precompute fast alignment lookup table for O(1) matching
    precomputed_map: Dict[str, Tuple[str, float]] = {}
    if aligner and aligner.is_trained:
        for w, targets in aligner.cooc_freq.items():
            if len(w) <= 2 or w in DEFAULT_STOPWORDS:
                continue
            best_t = None
            best_s = 0.0
            for tw in targets:
                if len(tw) <= 2 or tw in DEFAULT_STOPWORDS:
                    continue
                sc = aligner.get_association_score(w, tw)
                if sc > best_s:
                    best_s = sc
                    best_t = tw
            if best_t and best_s >= min_confidence:
                precomputed_map[w] = (best_t, best_s)

    try:
        for src_text, tgt_text in parallel_pairs:
            is_cached = cache_inst.contains(src_text, tgt_text) if cache_inst else False
            if is_cached:
                skipped_cache_hits += 1

            ex = build_gloss_pair(
                src_text,
                tgt_text,
                src_lang=src_lang,
                dictionary=dictionary,
                aligner=aligner,
                align_cache=cache_inst,
                precomputed_aligner_map=precomputed_map,
                min_confidence=min_confidence,
            )
            if ex and len(ex.inserted_terms) >= 1:
                examples.append(ex)
                if target_count is not None and len(examples) >= target_count:
                    break
    finally:
        if cache_inst:
            cache_inst.close()

    if skipped_cache_hits > 0:
        print(f"[AfriGloss Cache] Loaded {skipped_cache_hits:,} previously aligned pairs from cache (skipped re-alignment).")

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
