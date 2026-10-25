"""
Automated Training Queue for Afri-Gloss-MT Models.

Sequentially trains models one after the other on the NVIDIA H200 GPU:
  1. en -> tw (Twi / Akan)
  2. en -> sw (Swahili)
  3. en -> yo (Yoruba)
  4. en -> ha (Hausa)
  5. en -> ig (Igbo)
  6. en -> ee (Ewe)
  7. en -> sn (Shona)
"""

import argparse
import json
import logging
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_queue")

QUEUE = [
    {"src": "en", "tgt": "tw", "model": "Helsinki-NLP/opus-mt-en-tw", "data": "data/gloss_dataset_en_tw_10k"},
    {"src": "en", "tgt": "sw", "model": "Helsinki-NLP/opus-mt-en-sw", "data": "data/gloss_dataset_en_sw_10k"},
    {"src": "en", "tgt": "ha", "model": "Helsinki-NLP/opus-mt-en-ha", "data": "data/gloss_dataset_en_ha_10k"},
    {"src": "en", "tgt": "ig", "model": "Helsinki-NLP/opus-mt-en-ig", "data": "data/gloss_dataset_en_ig_10k"},
    {"src": "en", "tgt": "ee", "model": "Helsinki-NLP/opus-mt-en-ee", "data": "data/gloss_dataset_en_ee_10k"},
    {"src": "en", "tgt": "sn", "model": "Helsinki-NLP/opus-mt-en-sn", "data": "data/gloss_dataset_en_sn_10k"},
    {"src": "en", "tgt": "yo", "model": "Helsinki-NLP/opus-mt-en-alv", "target_token": ">>yor<<", "data": "data/gloss_dataset_en_yo_10k"},
]


def train_single_model(job: dict, epochs: int = 15, batch_size: int = 32, lr: float = 1e-4, patience: int = 10):
    src = job["src"]
    tgt = job["tgt"]
    model_name = job["model"]
    data_dir = REPO_ROOT / job["data"]
    out_dir = REPO_ROOT / "checkpoints" / f"afrigloss_opus_{src}_{tgt}"

    final_marker = out_dir / "final" / "config.json"
    if final_marker.exists():
        logger.info(f"[{src}->{tgt}] Final checkpoint already exists at {out_dir / 'final'}, skipping.")
        return

    logger.info("=" * 65)
    logger.info(f"STARTING JOB: {src.upper()} -> {tgt.upper()} ({model_name})")
    logger.info(f"Data directory: {data_dir}")
    logger.info(f"Output directory: {out_dir}")
    logger.info(f"Max epochs: {epochs} with early stopping patience {patience}")
    logger.info("=" * 65)

    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "train_opus_mt.py"),
        "--model", model_name,
        "--data", str(data_dir),
        "--out", str(out_dir),
        "--epochs", str(epochs),
        "--early-stopping-patience", str(patience),
        "--batch-size", str(batch_size),
        "--lr", str(lr),
        "--eval-steps", "150",
    ]

    if job.get("target_token"):
        cmd.extend(["--target-prefix", job["target_token"]])

    t0 = time.time()
    res = subprocess.run(cmd)
    elapsed = time.time() - t0

    if res.returncode == 0:
        logger.info(f"SUCCESS: {src}->{tgt} completed in {elapsed:.1f}s ({elapsed/60:.2f} mins)")
    else:
        logger.error(f"FAILURE: {src}->{tgt} failed with exit code {res.returncode}")


def main():
    parser = argparse.ArgumentParser(description="Afri-Gloss-MT Sequential Training Queue")
    parser.add_argument("--epochs", type=int, default=15, help="Maximum epochs per model (early stops on plateau)")
    parser.add_argument("--patience", type=int, default=4, help="Early stopping patience")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--only", nargs="+", help="Run only specific targets (e.g. --only tw sw)")

    args = parser.parse_args()

    active_queue = QUEUE
    if args.only:
        active_queue = [q for q in QUEUE if q["tgt"] in args.only]

    logger.info(f"Queue initialized with {len(active_queue)} model training jobs: {[q['tgt'] for q in active_queue]}")

    for idx, job in enumerate(active_queue, 1):
        logger.info(f"\n>>> Running job {idx}/{len(active_queue)}: {job['src']} -> {job['tgt']}")
        train_single_model(job, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, patience=args.patience)

    logger.info("\nALL TRAINING JOBS IN QUEUE COMPLETED!")


if __name__ == "__main__":
    main()
