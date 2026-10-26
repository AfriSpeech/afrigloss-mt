"""
Build Large-Scale / Uncapped Gloss-MT Training Datasets for French -> African Language Pairs.

Pulls parallel sentences from:
  1. AfriSpeech/africa-corpus (French reference caches)
  2. AfriSpeech/african-corpus-jw (477,000+ French sentences aligned with African languages)

Extracts French noun phrases using spaCy fr_core_news_sm, and performs in-place
replacement with <target_translation>.

Usage:
  python scripts/build_all_fr_datasets.py --langs tw sw yo ha ig ee sn --full
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from afrigloss.align_cache import AlignmentCache
from afrigloss.aligner import StatisticalWordAligner
from afrigloss.corpus import AfricaCorpusLoader
from afrigloss.dataset import create_gloss_dataset, save_dataset_splits
from afrigloss.dictionary import GlossaryDictionary
from afrigloss.registry import normalize_lang_code

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("build_fr_datasets")

TARGET_LANGUAGES = ["tw", "sw", "yo", "ha", "ig", "ee", "sn"]


def build_fr_dataset(
    tgt_lang: str,
    target_count: Optional[int] = None,
    out_base_dir: Path = Path("data"),
    cache_base_dir: Path = Path("data/align_cache"),
):
    src_code = "fra"
    tgt_code = normalize_lang_code(tgt_lang)
    tag = "full" if target_count is None else f"{target_count//1000}k"
    out_dir = out_base_dir / f"gloss_dataset_fr_{tgt_lang}_{tag}"
    cache_file = cache_base_dir / f"align_cache_fr_{tgt_code}.jsonl"

    if (out_dir / "train.jsonl").exists():
        logger.info(f"[{tgt_lang.upper()}] Dataset already exists at {out_dir}, skipping.")
        return

    label = "ALL Available" if target_count is None else f"{target_count:,}"
    logger.info("=" * 65)
    logger.info(f"Building {label} Gloss-MT dataset: French -> {tgt_lang.upper()} ({tgt_code})")
    logger.info("=" * 65)

    # Check for precomputed dictionary
    dict_file = REPO_ROOT / "data" / "dictionaries" / f"fra_{tgt_code}.tsv"
    dictionary = None
    if dict_file.exists():
        logger.info(f"[{tgt_lang}] Loading precomputed French dictionary: {dict_file}")
        dictionary = GlossaryDictionary.load_tsv(dict_file)

    loader = AfricaCorpusLoader()
    t0 = time.time()
    pairs = loader.get_parallel_pairs(
        "fra",
        tgt_code,
        max_samples=target_count,
        include_all_versions=True,
        include_jw=True,
    )
    logger.info(f"[{tgt_lang}] Loaded {len(pairs):,} parallel French-{tgt_lang} sentences in {time.time()-t0:.1f}s")

    if not pairs:
        logger.warning(f"[{tgt_lang}] No parallel pairs found.")
        return

    bitext = [(p.src, p.tgt) for p in pairs]

    # Train statistical aligner across all sentences
    t0 = time.time()
    aligner = StatisticalWordAligner(min_pair_freq=2, min_confidence=0.45)
    aligner.train(bitext)
    logger.info(f"[{tgt_lang}] Trained full-corpus aligner in {time.time()-t0:.1f}s")

    # Extract Gloss-MT dataset using French noun chunks
    t0 = time.time()
    cache_base_dir.mkdir(parents=True, exist_ok=True)
    splits = create_gloss_dataset(
        bitext,
        src_lang="fr",
        dictionary=dictionary,
        aligner=aligner,
        align_cache=cache_file,
        target_count=target_count,
        min_confidence=0.45,
    )
    save_dataset_splits(splits, out_dir)
    logger.info(f"[{tgt_lang}] Completed in {time.time()-t0:.1f}s -> {out_dir}")
    for name, s in splits.items():
        logger.info(f"   {name}: {len(s):,} samples")


def main():
    parser = argparse.ArgumentParser(description="Build French -> African language Gloss-MT datasets")
    parser.add_argument("--langs", nargs="+", default=TARGET_LANGUAGES, help="Target language codes")
    parser.add_argument("--count", type=int, default=100000, help="Target samples (default: 100,000)")
    parser.add_argument("--full", action="store_true", help="Use ALL available parallel data without any cap")
    parser.add_argument("--out-dir", default=str(REPO_ROOT / "data"), help="Base output directory")
    parser.add_argument("--cache-dir", default=str(REPO_ROOT / "data" / "align_cache"), help="Alignment cache directory")

    args = parser.parse_args()
    out_base = Path(args.out_dir)
    cache_base = Path(args.cache_dir)
    target_count = None if args.full else args.count

    for lang in args.langs:
        build_fr_dataset(lang, target_count=target_count, out_base_dir=out_base, cache_base_dir=cache_base)


if __name__ == "__main__":
    main()
