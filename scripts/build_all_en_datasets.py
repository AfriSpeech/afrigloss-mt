"""
Script to build 10k Gloss-MT training datasets for priority English -> African language pairs.
Ensures English is the source and the African language is the target, with English noun phrases
replaced in place by their target translation in angle brackets.
"""

import argparse
import sys
import time
from pathlib import Path

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from afrigloss.aligner import StatisticalWordAligner
from afrigloss.corpus import AfricaCorpusLoader
from afrigloss.dataset import create_gloss_dataset, save_dataset_splits
from afrigloss.dictionary import GlossaryDictionary
from afrigloss.registry import normalize_lang_code

TARGET_LANGUAGES = ["tw", "sw", "yo", "ha", "ig", "ee", "sn"]


def build_en_dataset(tgt_lang: str, target_count: int = 10000, out_base_dir: Path = Path("data")):
    src_code = "eng"
    tgt_code = normalize_lang_code(tgt_lang)
    out_dir = out_base_dir / f"gloss_dataset_en_{tgt_lang}_10k"

    if (out_dir / "train.jsonl").exists():
        print(f"[{tgt_lang.upper()}] Dataset already exists at {out_dir}, skipping.")
        return

    print(f"\n========================================================")
    print(f"Building 10k Gloss-MT dataset: English -> {tgt_lang.upper()} ({tgt_code})")
    print(f"========================================================")

    # Check for precomputed dictionary
    dict_file = REPO_ROOT / "data" / "dictionaries" / f"eng_{tgt_code}.tsv"
    dictionary = None
    if dict_file.exists():
        print(f"[{tgt_lang}] Loading precomputed dictionary: {dict_file}")
        dictionary = GlossaryDictionary.load_tsv(dict_file)

    loader = AfricaCorpusLoader()
    t0 = time.time()
    pairs = loader.get_parallel_pairs("eng", tgt_code, max_samples=None)
    print(f"[{tgt_lang}] Loaded {len(pairs):,} parallel verses in {time.time()-t0:.1f}s")

    # bitext is (English, African Language)
    bitext = [(p.src, p.tgt) for p in pairs]

    # Train aligner on full corpus
    t0 = time.time()
    aligner = StatisticalWordAligner(min_pair_freq=2, min_confidence=0.45)
    aligner.train(bitext)
    print(f"[{tgt_lang}] Trained full-corpus aligner in {time.time()-t0:.1f}s")

    # Extract 10,000 samples with guaranteed in-place noun chunk insertions
    t0 = time.time()
    splits = create_gloss_dataset(
        bitext,
        dictionary=dictionary,
        aligner=aligner,
        target_count=target_count,
        min_confidence=0.45,
    )
    save_dataset_splits(splits, out_dir)
    print(f"[{tgt_lang}] Done in {time.time()-t0:.1f}s -> {out_dir}")
    for name, s in splits.items():
        print(f"   {name}: {len(s):,} samples")


def main():
    parser = argparse.ArgumentParser(description="Build English -> African language 10k datasets")
    parser.add_argument("--langs", nargs="+", default=TARGET_LANGUAGES, help="Target language codes")
    parser.add_argument("--count", type=int, default=10000, help="Target samples (default: 10000)")
    parser.add_argument("--out-dir", default=str(REPO_ROOT / "data"), help="Base output directory")

    args = parser.parse_args()
    out_base = Path(args.out_dir)

    for lang in args.langs:
        build_en_dataset(lang, target_count=args.count, out_base_dir=out_base)


if __name__ == "__main__":
    main()
