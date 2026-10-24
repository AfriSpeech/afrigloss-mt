"""
100k Large-Scale Training Queue for Afri-Gloss-MT Production Demo Models.

Trains models on the 100k multi-version datasets on the NVIDIA H200 GPU:
  - English -> Twi (Akan) (85,562 parallel samples)
  - English -> Swahili (87,395 parallel samples)
"""

import argparse
import logging
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_100k_queue")

QUEUE = [
    {"src": "en", "tgt": "tw", "model": "Helsinki-NLP/opus-mt-en-tw", "data": "data/gloss_dataset_en_tw_100k"},
    {"src": "en", "tgt": "sw", "model": "Helsinki-NLP/opus-mt-en-sw", "data": "data/gloss_dataset_en_sw_100k"},
]


def train_single_model(job: dict, epochs: int = 5, batch_size: int = 32, lr: float = 1e-4, patience: int = 4):
    src = job["src"]
    tgt = job["tgt"]
    model_name = job["model"]
    data_dir = REPO_ROOT / job["data"]
    out_dir = REPO_ROOT / "checkpoints" / f"afrigloss_opus_{src}_{tgt}_100k"

    final_marker = out_dir / "final" / "config.json"
    if final_marker.exists():
        logger.info(f"[{src}->{tgt}] Final checkpoint already exists at {out_dir / 'final'}, skipping.")
        return

    logger.info("=" * 65)
    logger.info(f"STARTING 100K TRAINING: {src.upper()} -> {tgt.upper()} ({model_name})")
    logger.info(f"Data: {data_dir}")
    logger.info(f"Output: {out_dir}")
    logger.info(f"Epochs: {epochs} with early stopping patience {patience}")
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
        "--eval-steps", "500",
    ]

    t0 = time.time()
    res = subprocess.run(cmd)
    elapsed = time.time() - t0

    if res.returncode == 0:
        logger.info(f"SUCCESS: 100k model for {src}->{tgt} completed in {elapsed:.1f}s ({elapsed/60:.2f} mins)")
    else:
        logger.error(f"FAILURE: {src}->{tgt} failed with exit code {res.returncode}")


def main():
    parser = argparse.ArgumentParser(description="Afri-Gloss-MT 100k Production Training Queue")
    parser.add_argument("--epochs", type=int, default=5, help="Epochs per model")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--only", nargs="+", help="Run only specific targets (e.g. --only sw)")

    args = parser.parse_args()

    active_queue = QUEUE
    if args.only:
        active_queue = [q for q in QUEUE if q["tgt"] in args.only]

    logger.info(f"100k Queue initialized with {len(active_queue)} jobs: {[q['tgt'] for q in active_queue]}")

    for idx, job in enumerate(active_queue, 1):
        logger.info(f"\n>>> Running 100k job {idx}/{len(active_queue)}: {job['src']} -> {job['tgt']}")
        train_single_model(job, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)

    logger.info("\nALL 100K TRAINING JOBS COMPLETED!")


if __name__ == "__main__":
    main()
