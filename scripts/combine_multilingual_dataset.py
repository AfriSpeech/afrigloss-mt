"""
Combine Pan-African Gloss-MT Datasets into a Unified Multilingual Dataset for NLLB-200.

Merges datasets across all target African languages into a single multilingual dataset:
  - English -> Twi (aka_Latn)
  - English -> Swahili (swh_Latn)
  - English -> Yoruba (yor_Latn)
  - English -> Hausa (hau_Latn)
  - English -> Igbo (ibo_Latn)
  - English -> Ewe (ewe_Latn)
  - English -> Shona (sna_Latn)

Each example is tagged with 'src_lang' ('eng_Latn') and 'tgt_lang' (NLLB 4-char code),
allowing a single NLLB-200 model to be trained simultaneously on all languages with
cross-lingual transfer.
"""

import argparse
import json
import logging
import random
from pathlib import Path
from typing import Dict, List

import sys
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from afrigloss.registry import get_nllb_lang, normalize_lang_code

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("combine_dataset")


def merge_splits(data_dirs: List[Path], out_dir: Path, seed: int = 42):
    """Merge train, validation, and test splits across multiple language datasets."""
    out_dir.mkdir(parents=True, exist_ok=True)
    random.seed(seed)

    for split in ["train", "validation", "test"]:
        merged_examples = []
        lang_counts: Dict[str, int] = {}

        for d in data_dirs:
            split_file = d / f"{split}.jsonl"
            if not split_file.exists():
                logger.warning(f"Split file missing: {split_file}")
                continue

            # Infer target language from directory name: e.g. gloss_dataset_en_sw_100k -> sw
            parts = d.name.split("_")
            tgt_raw = parts[3] if len(parts) >= 4 else "twi"
            tgt_iso = normalize_lang_code(tgt_raw)
            nllb_tgt = get_nllb_lang(tgt_iso)

            if not nllb_tgt:
                logger.warning(f"Could not resolve NLLB language code for {tgt_raw} ({d.name})")
                continue

            count = 0
            with open(split_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        item = json.loads(line)
                        item["src_lang"] = "eng_Latn"
                        item["tgt_lang"] = nllb_tgt
                        item["lang_code"] = tgt_iso
                        merged_examples.append(item)
                        count += 1

            lang_counts[tgt_iso] = count
            logger.info(f"Loaded {count:,} {split} examples for {tgt_iso} ({nllb_tgt}) from {d.name}")

        # Shuffle merged split so batches contain mixed languages
        random.shuffle(merged_examples)
        out_file = out_dir / f"{split}.jsonl"
        with open(out_file, "w", encoding="utf-8") as out_f:
            for ex in merged_examples:
                out_f.write(json.dumps(ex, ensure_ascii=False) + "\n")

        logger.info(f"=== Merged {split} split total: {len(merged_examples):,} samples -> {out_file} ===")
        logger.info(f"    Per-language breakdown: {lang_counts}\n")


def main():
    parser = argparse.ArgumentParser(description="Combine African Gloss-MT datasets into a single multilingual dataset")
    parser.add_argument(
        "--data-dirs",
        nargs="+",
        help="List of dataset directories to merge (default: auto-detects all en_*_100k and en_*_10k dirs in data/)",
    )
    parser.add_argument(
        "--out-dir",
        default=str(REPO_ROOT / "data" / "gloss_dataset_multilingual_african"),
        help="Output directory for unified dataset",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling")

    args = parser.parse_args()
    data_path = REPO_ROOT / "data"

    if args.data_dirs:
        dirs_to_merge = [Path(d) for d in args.data_dirs]
    else:
        # Auto-discover priority datasets, preferring 100k when available
        priority_langs = ["tw", "sw", "yo", "ha", "ig", "ee", "sn"]
        dirs_to_merge = []
        for l in priority_langs:
            d_full = data_path / f"gloss_dataset_en_{l}_full"
            d_100k = data_path / f"gloss_dataset_en_{l}_100k"
            d_10k = data_path / f"gloss_dataset_en_{l}_10k"
            if d_full.exists():
                dirs_to_merge.append(d_full)
            elif d_100k.exists():
                dirs_to_merge.append(d_100k)
            elif d_10k.exists():
                dirs_to_merge.append(d_10k)

    logger.info(f"Merging datasets from {len(dirs_to_merge)} directories: {[d.name for d in dirs_to_merge]}")
    merge_splits(dirs_to_merge, Path(args.out_dir), seed=args.seed)


if __name__ == "__main__":
    main()
