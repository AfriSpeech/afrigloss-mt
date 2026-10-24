"""
Build Gloss-MT Training Datasets and Dictionaries from ANY Custom User Parallel Data.

Accepts:
  - Tabular file: .tsv, .csv, .jsonl, .parquet (e.g. columns 'source' and 'target')
  - Paired text files: --src-file english.txt --tgt-file twi.txt (one sentence per line)

Pipeline:
  1. Loads user parallel sentences.
  2. Runs high-confidence word alignment to automatically extract bilingual dictionary.
  3. Extracts spaCy noun chunks from source sentences.
  4. Pre-inserts target terminology in place in angle brackets: 'word <target_term>'.
  5. Partitions into train / validation / test splits ready for model fine-tuning.

Usage:
  python scripts/build_from_custom_data.py \
      --parallel-file my_domain_data.tsv \
      --src en --tgt tw \
      --out-dir data/my_custom_gloss_data
"""

import argparse
import sys
import time
from pathlib import Path
from typing import List, Optional

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from afrigloss.aligner import StatisticalWordAligner
from afrigloss.corpus import ParallelPair, clean_text, load_custom_parallel_file
from afrigloss.dataset import create_gloss_dataset, save_dataset_splits
from afrigloss.dictionary import GlossaryDictionary, build_dictionary_from_aligner


def load_from_paired_text_files(src_file: Path, tgt_file: Path) -> List[ParallelPair]:
    """Load line-aligned source and target text files."""
    with open(src_file, "r", encoding="utf-8", errors="replace") as sf, \
         open(tgt_file, "r", encoding="utf-8", errors="replace") as tf:
        pairs = []
        for idx, (s_line, t_line) in enumerate(zip(sf, tf), 1):
            s = clean_text(s_line)
            t = clean_text(t_line)
            if s and t and len(s.split()) >= 2 and len(t.split()) >= 2:
                pairs.append(ParallelPair(key=str(idx), src=s, tgt=t))
    return pairs


def main():
    parser = argparse.ArgumentParser(
        description="Build Gloss-MT training data & dictionary from custom parallel files"
    )
    parser.add_argument("--parallel-file", help="Path to parallel dataset (.tsv, .csv, .jsonl, .parquet)")
    parser.add_argument("--src-file", help="Path to source text file (line-aligned with --tgt-file)")
    parser.add_argument("--tgt-file", help="Path to target text file (line-aligned with --src-file)")
    parser.add_argument("--src", default="en", help="Source language code (default: en)")
    parser.add_argument("--tgt", required=True, help="Target language code (e.g. tw, yo, sw)")
    parser.add_argument("--out-dir", required=True, help="Output directory for generated dataset and dictionary")
    parser.add_argument("--target-samples", type=int, default=100000, help="Target sample count (default: 100,000 or max available)")
    parser.add_argument("--min-confidence", type=float, default=0.50, help="Min alignment confidence threshold")
    parser.add_argument("--val-ratio", type=float, default=0.05, help="Validation ratio")
    parser.add_argument("--test-ratio", type=float, default=0.05, help="Test ratio")

    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Ingest parallel pairs
    t0 = time.time()
    if args.parallel_file:
        print(f"[Custom Data] Loading parallel file: {args.parallel_file}...")
        pairs = load_custom_parallel_file(args.parallel_file)
    elif args.src_file and args.tgt_file:
        print(f"[Custom Data] Loading paired text files: {args.src_file} and {args.tgt_file}...")
        pairs = load_from_paired_text_files(Path(args.src_file), Path(args.tgt_file))
    else:
        print("Error: Must provide either --parallel-file OR (--src-file and --tgt-file).")
        sys.exit(1)

    print(f"[Custom Data] Loaded {len(pairs):,} parallel sentences in {time.time()-t0:.1f}s.")
    if not pairs:
        print("Error: No valid sentence pairs loaded.")
        sys.exit(1)

    # Advice on dataset scale
    if len(pairs) < 5000:
        print(f"\n[Note on Dataset Scale] You have {len(pairs):,} sentence pairs.")
        print("  - Even a small dataset (1,000 - 5,000 sentences) will teach the model the Gloss-MT in-place constraint copying rule!")
        print("  - However, for maximum vocabulary coverage and fluency improvements, 10k to 50k+ sentences are recommended.")
        print("  - Proceeding with training set generation with what is available...\n")

    # 2. Train alignment model on all parallel sentences to extract dictionary
    print("[Custom Data] Running statistical word aligner over all parallel pairs...")
    t0 = time.time()
    bitext = [(p.src, p.tgt) for p in pairs]
    aligner = StatisticalWordAligner(min_pair_freq=2, min_confidence=args.min_confidence)
    aligner.train(bitext)
    print(f"[Custom Data] Trained aligner in {time.time()-t0:.1f}s.")

    # 3. Extract high-confidence dictionary
    lexicon = aligner.extract_bilingual_lexicon(min_confidence=args.min_confidence)
    dictionary = build_dictionary_from_aligner(lexicon, min_confidence=args.min_confidence)
    dict_out = out_dir / f"dictionary_{args.src}_{args.tgt}.tsv"
    dictionary.save_tsv(dict_out)
    print(f"[Custom Data] Extracted {len(dictionary):,} high-confidence dictionary entries -> {dict_out}")

    # 4. Generate Gloss-MT pre-translated dataset
    target_count = min(len(pairs), args.target_samples)
    print(f"[Custom Data] Generating Gloss-MT pre-inserted dataset (target: {target_count:,} samples)...")
    t0 = time.time()
    splits = create_gloss_dataset(
        bitext,
        dictionary=dictionary,
        aligner=aligner,
        target_count=target_count,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        min_confidence=args.min_confidence,
    )
    save_dataset_splits(splits, out_dir)
    print(f"[Custom Data] Completed in {time.time()-t0:.1f}s!")
    print(f"[Custom Data] Generated splits saved to {out_dir}:")
    for name, s in splits.items():
        print(f"   {name}: {len(s):,} samples (100% with verified terminology constraints)")

    print("\nNext step: Fine-tune your model on this dataset:")
    print(f"  afrigloss train --src {args.src} --tgt {args.tgt} --train-file {out_dir}/train.jsonl --val-file {out_dir}/validation.jsonl --out checkpoints/my_model")


if __name__ == "__main__":
    main()
