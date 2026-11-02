"""
Build Dual-Source (English + French -> African Target) Datasets for Per-Language NLLB-600M.

Merges the full English and French parallel datasets for each specific African target:
  - English -> Target (eng_Latn)
  - French -> Target (fra_Latn)

Output:
  data/gloss_dataset_dualsource_<tgt>_full/
    - train.jsonl (shuffled EN + FR examples)
    - validation.jsonl
    - test.jsonl

Languages: tw, sw, yo, ha, ig, ee, sn.
"""

import argparse
import json
import logging
import random
import sys
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from afrigloss.registry import get_nllb_lang, normalize_lang_code

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("build_dualsource")

TARGET_LANGS = ["tw", "sw", "yo", "ha", "ig", "ee", "sn"]
DATA_DIR = REPO_ROOT / "data"


def merge_dual_source_for_target(tgt_lang: str, seed: int = 42) -> Path:
    """Merge EN and FR datasets for a single African target language."""
    tgt_iso = normalize_lang_code(tgt_lang)
    nllb_tgt = get_nllb_lang(tgt_iso)
    if not nllb_tgt:
        raise ValueError(f"Could not resolve NLLB code for {tgt_lang}")

    out_dir = DATA_DIR / f"gloss_dataset_dualsource_{tgt_lang}_full"
    out_dir.mkdir(parents=True, exist_ok=True)

    en_dir = DATA_DIR / f"gloss_dataset_en_{tgt_lang}_full"
    fr_dir = DATA_DIR / f"gloss_dataset_fr_{tgt_lang}_full"

    if not en_dir.exists() or not fr_dir.exists():
        logger.error(f"Missing source datasets for {tgt_lang} ({en_dir.name} or {fr_dir.name})")
        return out_dir

    logger.info("=" * 65)
    logger.info(f"MERGING DUAL-SOURCE DATASET: EN + FR -> {tgt_lang.upper()} ({nllb_tgt})")
    logger.info(f"EN Source: {en_dir}")
    logger.info(f"FR Source: {fr_dir}")
    logger.info(f"Output: {out_dir}")
    logger.info("=" * 65)

    random.seed(seed)

    for split in ["train", "validation", "test"]:
        merged_records = []
        counts = {"en": 0, "fr": 0}

        # 1. Load English split
        en_file = en_dir / f"{split}.jsonl"
        if en_file.exists():
            with open(en_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        item = json.loads(line)
                        item["src_lang"] = "eng_Latn"
                        item["tgt_lang"] = nllb_tgt
                        item["lang_code"] = tgt_iso
                        merged_records.append(item)
                        counts["en"] += 1

        # 2. Load French split
        fr_file = fr_dir / f"{split}.jsonl"
        if fr_file.exists():
            with open(fr_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        item = json.loads(line)
                        item["src_lang"] = "fra_Latn"
                        item["tgt_lang"] = nllb_tgt
                        item["lang_code"] = tgt_iso
                        merged_records.append(item)
                        counts["fr"] += 1

        # Shuffle train split so batches alternate between English and French
        if split == "train":
            random.shuffle(merged_records)

        out_file = out_dir / f"{split}.jsonl"
        with open(out_file, "w", encoding="utf-8") as out_f:
            for ex in merged_records:
                out_f.write(json.dumps(ex, ensure_ascii=False) + "\n")

        logger.info(f"[{tgt_lang.upper()}] {split:10s}: {len(merged_records):,} total (EN: {counts['en']:,}, FR: {counts['fr']:,}) -> {out_file.name}")

    return out_dir


def main():
    parser = argparse.ArgumentParser(description="Build dual-source (EN+FR -> Target) Gloss-MT datasets")
    parser.add_argument("--langs", nargs="+", default=TARGET_LANGS, help="Target African languages")
    parser.add_argument("--seed", type=int, default=42, help="Shuffle seed")
    args = parser.parse_args()

    for l in args.langs:
        merge_dual_source_for_target(l, seed=args.seed)


if __name__ == "__main__":
    main()
