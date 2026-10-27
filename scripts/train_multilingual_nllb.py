"""
Fine-Tune Unified Pan-African Multilingual NLLB-600M Model on Full Gloss-MT Corpus.

Dataset: data/gloss_dataset_multilingual_african/
  - 3.6 Million parallel sentences
  - Multi-source: English (eng_Latn) and French (fra_Latn)
  - Multi-target: Twi (twi_Latn), Swahili (swh_Latn), Yoruba (yor_Latn),
                  Hausa (hau_Latn), Igbo (ibo_Latn), Ewe (ewe_Latn), Shona (sna_Latn)
  - 100% in-place target glossary annotations (<term>)

Features:
  - Hopper H200 bf16 + TF32 acceleration
  - Dynamic sample-level source and target language routing
  - LoRA parameter-efficient adaptation (r=16, alpha=32)
  - Automatic checkpointing and validation tracking

Usage:
  python scripts/train_multilingual_nllb.py
  python scripts/train_multilingual_nllb.py --full-finetune
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from afrigloss.train_seq2seq import train_seq2seq

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_nllb_master")

DATA_DIR = REPO_ROOT / "data" / "gloss_dataset_multilingual_african"
OUT_DIR = REPO_ROOT / "checkpoints" / "afrigloss_nllb_600m_multilingual"


def main():
    parser = argparse.ArgumentParser(description="Train Unified Pan-African NLLB-600M Model")
    parser.add_argument("--model", default="facebook/nllb-200-distilled-600M", help="NLLB base model ID")
    parser.add_argument("--batch-size", type=int, default=16, help="Per-device batch size (default: 16)")
    parser.add_argument("--grad-accum", type=int, default=2, help="Gradient accumulation steps (default: 2)")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate (default: 2e-4)")
    parser.add_argument("--epochs", type=int, default=5, help="Epochs ceiling")
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank (0 for full fine-tune)")
    parser.add_argument("--full-finetune", action="store_true", help="Perform full weights fine-tuning instead of LoRA")
    parser.add_argument("--out-dir", default=str(OUT_DIR), help="Output directory")

    args = parser.parse_args()

    train_file = DATA_DIR / "train.jsonl"
    val_file = DATA_DIR / "validation.jsonl"

    if not train_file.exists():
        logger.error(f"Multilingual dataset not found at {train_file}! Run combine_multilingual_dataset.py first.")
        sys.exit(1)

    lora_rank = 0 if args.full_finetune else args.lora_r
    mode_str = "Full Parameter Fine-Tuning" if lora_rank == 0 else f"LoRA Adapter Fine-Tuning (rank={lora_rank})"

    logger.info("=" * 65)
    logger.info(f"STARTING UNIFIED PAN-AFRICAN NLLB-600M TRAINING")
    logger.info(f"Base Model: {args.model}")
    logger.info(f"Training Mode: {mode_str}")
    logger.info(f"Data Source: {train_file}")
    logger.info(f"Output Checkpoint: {args.out_dir}")
    logger.info(f"Batch size: {args.batch_size} (accum: {args.grad_accum})")
    logger.info("=" * 65)

    t0 = time.time()
    train_seq2seq(
        model_name_or_path=args.model,
        src_lang="eng_Latn",
        tgt_lang="twi_Latn",
        train_file=train_file,
        val_file=val_file if val_file.exists() else None,
        output_dir=args.out_dir,
        lora_r=lora_rank,
        batch_size=args.batch_size,
        lr=args.lr,
        epochs=args.epochs,
        grad_accum=args.grad_accum,
    )
    elapsed = time.time() - t0
    logger.info(f"NLLB Training Complete in {elapsed:.1f}s ({elapsed/3600:.2f} hours) -> {args.out_dir}")


if __name__ == "__main__":
    main()
