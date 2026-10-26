"""
Build Grounded Bilingual Dictionaries Directly from the Afri-Gloss-MT Training Datasets.

Extracts the exact words and phrases that were aligned and inserted into the models
during training for all 14 language pairs (English & French -> African targets).

Key Features:
  - 100% grounded in human bitext (NO Gemini or LLM translations).
  - Majority-vote consensus for polysemous source terms with confidence estimation.
  - Ranked by empirical corpus frequency.
  - Saves in standard TSV format: source_phrase \t target_phrase \t confidence \t frequency
  - Writes both short names (en_tw.tsv) and ISO-3 names (eng_twi.tsv) for full eval compatibility.

Usage:
  python scripts/build_dictionaries_from_dataset.py
  python scripts/build_dictionaries_from_dataset.py --src-langs en fr --min-freq 2
"""

import argparse
import json
import logging
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from afrigloss.registry import LANGUAGE_INFO, normalize_lang_code

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("build_dataset_dict")

ISO3_MAPPING = {
    "tw": "twi",
    "sw": "swh",
    "yo": "yor",
    "ha": "hau",
    "ig": "ibo",
    "ee": "ewe",
    "sn": "sna",
    "en": "eng",
    "fr": "fra",
}


def extract_pairs_from_example(src_raw: str, inp_with_tags: str) -> List[Tuple[str, str]]:
    """
    Extract (src_phrase, target_term) mappings by aligning <term> tags in input
    with the corresponding character spans in src_raw.
    """
    tag_matches = list(re.finditer(r"<([^>]+)>", inp_with_tags))
    if not tag_matches:
        return []

    parts = re.split(r"<[^>]+>", inp_with_tags)
    terms = [m.group(1) for m in tag_matches]
    pairs = []
    cur_pos = 0

    for i, term in enumerate(terms):
        prefix = parts[i]
        idx_prefix = src_raw.find(prefix, cur_pos)
        if idx_prefix == -1:
            continue
        start_src = idx_prefix + len(prefix)
        next_prefix = parts[i + 1] if (i + 1) < len(parts) else ""

        if next_prefix:
            idx_next = src_raw.find(next_prefix, start_src)
            if idx_next != -1:
                src_phrase = src_raw[start_src:idx_next].strip()
                pairs.append((src_phrase, term.strip()))
                cur_pos = idx_next
        else:
            src_phrase = src_raw[start_src:].strip()
            pairs.append((src_phrase, term.strip()))
            cur_pos = len(src_raw)

    return pairs


def clean_phrase(text: str) -> str:
    """Strip leading/trailing punctuation and excess whitespace."""
    t = text.strip()
    t = re.sub(r"^[^\w]+|[^\w]+$", "", t)
    return t.strip()


def build_dictionary_for_pair(
    src: str,
    tgt: str,
    min_freq: int = 2,
    out_dir: Optional[Path] = None,
) -> int:
    """Build grounded dictionary for a specific source-target pair from train.jsonl."""
    if out_dir is None:
        out_dir = REPO_ROOT / "data" / "dictionaries"
    out_dir.mkdir(parents=True, exist_ok=True)

    data_dir = REPO_ROOT / "data"
    dataset_dir = data_dir / f"gloss_dataset_{src}_{tgt}_full"
    if not dataset_dir.exists():
        dataset_dir = data_dir / f"gloss_dataset_{src}_{tgt}_100k"
    if not dataset_dir.exists():
        logger.warning(f"[{src}->{tgt}] Dataset directory not found at {dataset_dir}, skipping.")
        return 0

    train_file = dataset_dir / "train.jsonl"
    if not train_file.exists():
        logger.warning(f"[{src}->{tgt}] train.jsonl missing in {dataset_dir}, skipping.")
        return 0

    logger.info("=" * 65)
    logger.info(f"BUILDING GROUNDED DICTIONARY: {src.upper()} -> {tgt.upper()}")
    logger.info(f"Source: {train_file}")
    logger.info(f"Min frequency threshold: {min_freq}")
    logger.info("=" * 65)

    # counts[src_lower][tgt_trans] = frequency
    counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    total_samples = 0
    total_extractions = 0
    t0 = time.time()

    with open(train_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            total_samples += 1
            ex = json.loads(line)
            src_raw = ex.get("src_raw", "")
            inp = ex.get("input", "")

            for sp, tt in extract_pairs_from_example(src_raw, inp):
                sp_clean = clean_phrase(sp).lower()
                tt_clean = clean_phrase(tt)

                # Filter out pure noise / extremely long or short fragments
                if 2 <= len(sp_clean) <= 50 and 2 <= len(tt_clean) <= 50:
                    counts[sp_clean][tt_clean] += 1
                    total_extractions += 1

    elapsed = time.time() - t0
    logger.info(f"Processed {total_samples:,} samples in {elapsed:.2f}s ({total_extractions:,} total term insertions).")

    # Majority vote consensus
    dict_entries = []
    for sp, targets in counts.items():
        total_term_freq = sum(targets.values())
        if total_term_freq < min_freq:
            continue

        best_tgt, best_count = max(targets.items(), key=lambda x: x[1])
        confidence = best_count / total_term_freq

        # Format: (source_phrase, target_phrase, confidence, frequency)
        dict_entries.append((sp, best_tgt, confidence, total_term_freq))

    # Sort by frequency descending
    dict_entries.sort(key=lambda x: x[3], reverse=True)

    # Primary filename: en_tw.tsv / fr_tw.tsv
    primary_tsv = out_dir / f"{src}_{tgt}.tsv"
    with open(primary_tsv, "w", encoding="utf-8") as out_f:
        out_f.write("source_phrase\ttarget_phrase\tconfidence\tfrequency\n")
        for sp, tt, conf, freq in dict_entries:
            out_f.write(f"{sp}\t{tt}\t{conf:.4f}\t{freq}\n")

    # ISO-3 compatibility filename: eng_twi.tsv / fra_twi.tsv
    iso_src = ISO3_MAPPING.get(src, src)
    iso_tgt = ISO3_MAPPING.get(tgt, tgt)
    iso_tsv = out_dir / f"{iso_src}_{iso_tgt}.tsv"
    with open(iso_tsv, "w", encoding="utf-8") as out_f:
        out_f.write("source_phrase\ttarget_phrase\tconfidence\tfrequency\n")
        for sp, tt, conf, freq in dict_entries:
            out_f.write(f"{sp}\t{tt}\t{conf:.4f}\t{freq}\n")

    logger.info(f"[{src}->{tgt}] Generated {len(dict_entries):,} grounded dictionary entries!")
    logger.info(f"Saved to: {primary_tsv} and {iso_tsv}")

    # Display top 5 entries
    logger.info("Top 5 most frequent dictionary terms:")
    for sp, tt, conf, freq in dict_entries[:5]:
        logger.info(f"   {sp:20} -> {tt:20} (freq: {freq:5d}, conf: {conf*100:.1f}%)")

    return len(dict_entries)


def main():
    parser = argparse.ArgumentParser(description="Build grounded dictionaries from training datasets")
    parser.add_argument("--src-langs", nargs="+", default=["en", "fr"], help="Source languages (default: en fr)")
    parser.add_argument("--langs", nargs="+", default=["tw", "sw", "yo", "ha", "ig", "ee", "sn"], help="Target languages")
    parser.add_argument("--min-freq", type=int, default=2, help="Minimum occurrences across dataset (default: 2)")
    parser.add_argument("--out-dir", default=str(REPO_ROOT / "data" / "dictionaries"), help="Output directory")

    args = parser.parse_args()
    out_dir = Path(args.out_dir)

    total_pairs = 0
    total_entries = 0
    t_start = time.time()

    for s in args.src_langs:
        for t in args.langs:
            count = build_dictionary_for_pair(s, t, min_freq=args.min_freq, out_dir=out_dir)
            if count > 0:
                total_pairs += 1
                total_entries += count

    total_time = time.time() - t_start
    logger.info(f"\n=======================================================")
    logger.info(f"ALL GROUNDED DICTIONARIES BUILT SUCCESSFULLY!")
    logger.info(f"Total language pairs: {total_pairs}")
    logger.info(f"Total high-confidence dictionary entries: {total_entries:,}")
    logger.info(f"Completed in {total_time:.2f}s ({total_time/60:.2f} mins)")
    logger.info(f"=======================================================\n")


if __name__ == "__main__":
    main()
