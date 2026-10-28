"""
Master Pan-African French -> African Dataset, Dictionary, and Proper Noun Builder.

Builds full, uncapped Gloss-MT datasets, grounded bilingual dictionaries,
and proper noun lexicons for all African languages with French bitext
from AfriSpeech/africa-corpus and AfriSpeech/african-corpus-jw.

For each African language:
  1. Ingests all multi-version French parallel text (no cap).
  2. Trains statistical word aligner with persistent JSONL caching (align_cache_fr_<lang>.jsonl).
  3. Creates French Gloss-MT splits with French noun chunk extraction (gloss_dataset_fr_<lang>_full/).
  4. Generates grounded bilingual dictionary (data/dictionaries/fra_<lang>.tsv & fr_<lang>.tsv).
  5. Extracts proper nouns into afri-adapt/fr_<lang>_proper_nouns.tsv and appends to all_proper_nouns.tsv.
  6. Tracks status in data/pan_african_fr_status.json.

Usage:
  python scripts/build_pan_african_fr_master.py
  python scripts/build_pan_african_fr_master.py --tier1-only
  python scripts/build_pan_african_fr_master.py --only lin kin mos sag
"""

import argparse
import json
import logging
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

LOGS_DIR = REPO_ROOT / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOGS_DIR / "build_pan_african_fr_master.log", mode="a", encoding="utf-8"),
    ],
)
logger = logging.getLogger("pan_african_fr_master")

DATA_DIR = REPO_ROOT / "data"
CACHE_DIR = DATA_DIR / "align_cache"
DICT_DIR = DATA_DIR / "dictionaries"
AFRI_ADAPT_DIR = REPO_ROOT / "afri-adapt"
STATUS_FILE = DATA_DIR / "pan_african_fr_status.json"

ALREADY_DONE_FR_LANGS = {
    "tw", "twi", "ak", "aka",
    "sw", "swh",
    "yo", "yor",
    "ha", "hau",
    "ig", "ibo",
    "ee", "ewe",
    "sn", "sna",
}

from afrigloss.align_cache import AlignmentCache
from afrigloss.aligner import StatisticalWordAligner
from afrigloss.corpus import AfricaCorpusLoader
from afrigloss.dataset import create_gloss_dataset, save_dataset_splits
from afrigloss.registry import LANGUAGE_INFO, normalize_lang_code


def clean_term(text: str) -> str:
    """Clean edge punctuation from extracted phrases."""
    t = text.strip()
    return re.sub(r"^[^\w]+|[^\w]+$", "", t).strip()


def extract_pairs(src_raw: str, inp_with_tags: str) -> List[Tuple[str, str, int]]:
    """Extract (source_phrase, target_term, start_index) from tagged input."""
    tag_matches = list(re.finditer(r"<([^>]+)>", inp_with_tags))
    if not tag_matches:
        return []

    parts = re.split(r"<[^>]+>", inp_with_tags)
    terms = [m.group(1) for m in tag_matches]
    pairs = []
    cur_pos = 0

    for idx, term in enumerate(terms):
        prefix = parts[idx]
        idx_prefix = src_raw.find(prefix, cur_pos)
        if idx_prefix == -1:
            continue
        start_src = idx_prefix + len(prefix)
        next_prefix = parts[idx + 1] if (idx + 1) < len(parts) else ""

        if next_prefix:
            idx_next = src_raw.find(next_prefix, start_src)
            if idx_next != -1:
                raw_term = src_raw[start_src:idx_next].strip()
                cur_pos = idx_next
            else:
                raw_term = src_raw[start_src:].strip()
                cur_pos = len(src_raw)
        else:
            raw_term = src_raw[start_src:].strip()
            cur_pos = len(src_raw)

        if raw_term:
            pairs.append((raw_term, term.strip(), start_src))

    return pairs


def build_fr_dictionary_and_proper_nouns(
    train_file: Path,
    lang_code: str,
    min_freq: int = 2,
    cap_ratio_threshold: float = 0.55,
) -> Tuple[int, int]:
    """
    Extract grounded bilingual French dictionary and proper nouns from train.jsonl.
    Returns: (num_dictionary_entries, num_proper_nouns)
    """
    counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    mid_upper_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    total_term_counts: Dict[str, int] = defaultdict(int)

    with open(train_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            ex = json.loads(line)
            src_raw = ex.get("src_raw", "")
            inp = ex.get("input", "")

            pairs = extract_pairs(src_raw, inp)
            for raw_term, tgt_term, start_idx in pairs:
                raw_c = clean_term(raw_term)
                tgt_c = clean_term(tgt_term)

                if 2 <= len(raw_c) <= 45 and 2 <= len(tgt_c) <= 45:
                    raw_lower = raw_c.lower()
                    counts[raw_lower][tgt_c] += 1
                    total_term_counts[raw_lower] += 1

                    if start_idx > 0 and raw_c[0].isupper():
                        mid_upper_counts[raw_c][tgt_c] += 1

    # 1. Generate bilingual French dictionary (majority-vote)
    dict_entries = []
    for sp_lower, targets in counts.items():
        total_freq = sum(targets.values())
        if total_freq < min_freq:
            continue
        best_tgt, best_count = max(targets.items(), key=lambda x: x[1])
        conf = best_count / total_freq
        dict_entries.append((sp_lower, best_tgt, conf, total_freq))

    dict_entries.sort(key=lambda x: x[3], reverse=True)

    DICT_DIR.mkdir(parents=True, exist_ok=True)
    out_dict_iso = DICT_DIR / f"fra_{lang_code}.tsv"
    out_dict_short = DICT_DIR / f"fr_{lang_code}.tsv"

    for path in (out_dict_iso, out_dict_short):
        with open(path, "w", encoding="utf-8") as f:
            f.write("source_phrase\ttarget_phrase\tconfidence\tfrequency\n")
            for sp, tt, conf, freq in dict_entries:
                f.write(f"{sp}\t{tt}\t{conf:.4f}\t{freq}\n")

    # 2. Generate French proper nouns for afri-adapt
    proper_entries = []
    for raw_c, targets in mid_upper_counts.items():
        cap_freq = sum(targets.values())
        tot_freq = total_term_counts[raw_c.lower()]
        if cap_freq < min_freq or tot_freq == 0:
            continue
        if (cap_freq / tot_freq) >= cap_ratio_threshold:
            best_tgt, best_count = max(targets.items(), key=lambda x: x[1])
            conf = best_count / cap_freq
            proper_entries.append((raw_c, best_tgt, conf, cap_freq))

    proper_entries.sort(key=lambda x: x[3], reverse=True)

    AFRI_ADAPT_DIR.mkdir(parents=True, exist_ok=True)
    out_proper = AFRI_ADAPT_DIR / f"fr_{lang_code}_proper_nouns.tsv"
    with open(out_proper, "w", encoding="utf-8") as f:
        f.write("source_term\ttarget_term\tconfidence\tfrequency\n")
        for sp, tt, conf, freq in proper_entries:
            f.write(f"{sp}\t{tt}\t{conf:.4f}\t{freq}\n")

    # Append to all_proper_nouns.tsv
    all_proper_file = AFRI_ADAPT_DIR / "all_proper_nouns.tsv"
    write_header = not all_proper_file.exists() or all_proper_file.stat().st_size == 0
    with open(all_proper_file, "a", encoding="utf-8") as f:
        if write_header:
            f.write("src_lang\ttgt_lang\tsource_term\ttarget_term\tconfidence\tfrequency\n")
        for sp, tt, conf, freq in proper_entries:
            f.write(f"fr\t{lang_code}\t{sp}\t{tt}\t{conf:.4f}\t{freq}\n")

    logger.info(f"[{lang_code}] Generated {len(dict_entries):,} French dictionary entries & {len(proper_entries):,} proper nouns.")
    return len(dict_entries), len(proper_entries)


def process_fr_language(
    lang_code: str,
    loader: AfricaCorpusLoader,
    min_sentences: int = 100,
) -> Dict[str, any]:
    """Process a single African language through the French pipeline."""
    t0 = time.time()
    out_dir = DATA_DIR / f"gloss_dataset_fr_{lang_code}_full"
    cache_file = CACHE_DIR / f"align_cache_fr_{lang_code}.jsonl"

    train_file = out_dir / "train.jsonl"
    if train_file.exists() and train_file.stat().st_size > 0:
        logger.info(f"[{lang_code}] French dataset already exists at {out_dir}. Verifying dictionary and proper nouns...")
        n_dict, n_prop = build_fr_dictionary_and_proper_nouns(train_file, lang_code)
        return {
            "status": "already_existed",
            "lang": lang_code,
            "dict_entries": n_dict,
            "proper_nouns": n_prop,
            "elapsed_seconds": time.time() - t0,
        }

    logger.info("=" * 65)
    logger.info(f"PROCESSING FRENCH -> AFRICAN LANGUAGE: {lang_code.upper()}")
    logger.info("=" * 65)

    # 1. Load French bitext
    try:
        pairs = loader.get_parallel_pairs(
            "fra",
            lang_code,
            max_samples=None,
            include_all_versions=True,
            include_jw=True,
            include_pristine=False,
        )
    except Exception as e:
        logger.warning(f"[{lang_code}] Failed loading parallel text: {e}")
        return {"status": "error_loading", "lang": lang_code, "error": str(e)}

    if len(pairs) < min_sentences:
        logger.info(f"[{lang_code}] Insufficient parallel bitext ({len(pairs)} < {min_sentences}). Skipping.")
        return {"status": "skipped_too_small", "lang": lang_code, "pairs": len(pairs)}

    logger.info(f"[{lang_code}] Loaded {len(pairs):,} parallel French sentences in {time.time()-t0:.1f}s.")
    bitext = [(p.src, p.tgt) for p in pairs]

    # 2. Train statistical word aligner
    t_align = time.time()
    aligner = StatisticalWordAligner(min_pair_freq=2, min_confidence=0.45)
    aligner.train(bitext)
    logger.info(f"[{lang_code}] Trained aligner in {time.time()-t_align:.1f}s.")

    # 3. Create Gloss-MT dataset splits with French noun chunking
    t_ds = time.time()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    splits = create_gloss_dataset(
        bitext,
        aligner=aligner,
        align_cache=cache_file,
        target_count=None,
        min_confidence=0.45,
        src_lang="fr",
    )
    save_dataset_splits(splits, out_dir)
    train_count = len(splits.get("train", []))
    val_count = len(splits.get("validation", []))
    test_count = len(splits.get("test", []))
    total_samples = train_count + val_count + test_count
    logger.info(f"[{lang_code}] French dataset saved: {train_count:,} train, {val_count:,} val, {test_count:,} test ({time.time()-t_ds:.1f}s).")

    # 4. Generate grounded dictionary & proper nouns
    n_dict, n_prop = build_fr_dictionary_and_proper_nouns(train_file, lang_code)

    elapsed = time.time() - t0
    logger.info(f"[{lang_code}] FINISHED FRENCH PIPELINE in {elapsed:.1f}s ({elapsed/60:.2f} mins).")

    return {
        "status": "completed",
        "lang": lang_code,
        "total_pairs": len(pairs),
        "gloss_train": train_count,
        "gloss_total": total_samples,
        "dict_entries": n_dict,
        "proper_nouns": n_prop,
        "elapsed_seconds": elapsed,
    }


def main():
    parser = argparse.ArgumentParser(description="Master Pan-African French Dataset & Dictionary Builder")
    parser.add_argument("--tier1-only", action="store_true", help="Process only large JW languages")
    parser.add_argument("--only", nargs="+", help="Process only specified language codes (e.g. --only lin kin mos)")
    parser.add_argument("--max-langs", type=int, help="Maximum number of languages to process")
    parser.add_argument("--min-sentences", type=int, default=100, help="Minimum bitext sentences required (default: 100)")

    args = parser.parse_args()
    loader = AfricaCorpusLoader()

    # Discover all languages
    logger.info("Discovering all available African languages for French pipeline...")
    africa_map = loader._discover_files()
    jw_map = loader._discover_jw_files()

    # Tier 1: Large JW bitext
    jw_langs = [
        l for l in sorted(jw_map.keys())
        if l not in ALREADY_DONE_FR_LANGS and l not in {"en", "eng", "fr", "fra"}
    ]

    # Tier 2: Remaining Africa corpus languages
    africa_langs = [
        l for l in sorted(africa_map.keys())
        if l not in ALREADY_DONE_FR_LANGS and l not in jw_langs and l not in {"en", "eng", "fr", "fra"}
    ]

    if args.only:
        selected_langs = [normalize_lang_code(l) for l in args.only]
    elif args.tier1_only:
        selected_langs = jw_langs
    else:
        # Tier 1 first, then Tier 2
        selected_langs = jw_langs + africa_langs

    if args.max_langs:
        selected_langs = selected_langs[: args.max_langs]

    logger.info(f"Total languages queued for French processing: {len(selected_langs)}")

    # Load existing status
    status_history: Dict[str, any] = {}
    if STATUS_FILE.exists():
        try:
            with open(STATUS_FILE, "r", encoding="utf-8") as f:
                status_history = json.load(f)
        except Exception:
            status_history = {}

    completed_count = 0
    t_start = time.time()

    for idx, lang in enumerate(selected_langs, 1):
        logger.info(f"\n>>> [{idx}/{len(selected_langs)}] Starting French language: {lang.upper()}")
        try:
            res = process_fr_language(lang, loader=loader, min_sentences=args.min_sentences)
            status_history[lang] = res
            if res.get("status") in {"completed", "already_existed"}:
                completed_count += 1
        except Exception as e:
            logger.error(f"[{lang}] Unexpected error in French pipeline: {e}", exc_info=True)
            status_history[lang] = {"status": "unhandled_error", "error": str(e)}

        # Persist status after each language
        with open(STATUS_FILE, "w", encoding="utf-8") as f:
            json.dump(status_history, f, indent=2)

    total_time = time.time() - t_start
    logger.info("=" * 65)
    logger.info(f"PAN-AFRICAN FRENCH RUN COMPLETE!")
    logger.info(f"Languages processed: {completed_count}/{len(selected_langs)}")
    logger.info(f"Total time elapsed: {total_time:.1f}s ({total_time/3600:.2f} hours)")
    logger.info(f"Status file: {STATUS_FILE}")
    logger.info("=" * 65)


if __name__ == "__main__":
    main()
