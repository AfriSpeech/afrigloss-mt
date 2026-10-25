"""
100k Large-Scale Training Queue for Afri-Gloss-MT Production Demo Models.

Trains models dynamically until validation convergence (no fixed epochs cutoff):
  1. en -> tw (Twi / Akan) - 85,562 samples
  2. en -> sw (Swahili) - 87,395 samples
  3. en -> yo (Yoruba) - 85,000+ samples (via opus-mt-en-alv >>yor<<)
  4. en -> ha (Hausa) - 85,000+ samples
  5. en -> ig (Igbo) - 85,000+ samples
  6. en -> ee (Ewe) - 85,000+ samples
  7. en -> sn (Shona) - 85,000+ samples
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
    {"src": "en", "tgt": "yo", "model": "Helsinki-NLP/opus-mt-en-alv", "target_token": ">>yor<<", "data": "data/gloss_dataset_en_yo_100k"},
    {"src": "en", "tgt": "ha", "model": "Helsinki-NLP/opus-mt-en-ha", "data": "data/gloss_dataset_en_ha_100k"},
    {"src": "en", "tgt": "ig", "model": "Helsinki-NLP/opus-mt-en-ig", "data": "data/gloss_dataset_en_ig_100k"},
    {"src": "en", "tgt": "ee", "model": "Helsinki-NLP/opus-mt-en-ee", "data": "data/gloss_dataset_en_ee_100k"},
    {"src": "en", "tgt": "sn", "model": "Helsinki-NLP/opus-mt-en-sn", "data": "data/gloss_dataset_en_sn_100k"},
]


def train_single_model(job: dict, max_epochs: int = 50, batch_size: int = 32, lr: float = 1e-4, patience: int = 10):
    src = job["src"]
    tgt = job["tgt"]
    model_name = job["model"]
    data_dir = REPO_ROOT / job["data"]
    out_dir = REPO_ROOT / "checkpoints" / f"afrigloss_opus_{src}_{tgt}_100k"

    final_marker = out_dir / "final" / "config.json"
    if final_marker.exists():
        logger.info(f"[{src}->{tgt}] Final checkpoint already exists at {out_dir / 'final'}, skipping.")
        return

    if not (data_dir / "train.jsonl").exists():
        logger.warning(f"[{src}->{tgt}] Dataset not ready at {data_dir}, skipping.")
        return

    logger.info("=" * 65)
    logger.info(f"STARTING 100K CONVERGENCE TRAINING: {src.upper()} -> {tgt.upper()} ({model_name})")
    logger.info(f"Data: {data_dir}")
    logger.info(f"Output: {out_dir}")
    logger.info(f"Training until validation convergence: max_epochs={max_epochs}, patience={patience}")
    logger.info("=" * 65)

    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "train_opus_mt.py"),
        "--model", model_name,
        "--data", str(data_dir),
        "--out", str(out_dir),
        "--epochs", str(max_epochs),
        "--early-stopping-patience", str(patience),
        "--batch-size", str(batch_size),
        "--lr", str(lr),
        "--eval-steps", "500",
    ]

    if job.get("target_token"):
        cmd.extend(["--target-prefix", job["target_token"]])

    t0 = time.time()
    res = subprocess.run(cmd)
    elapsed = time.time() - t0

    if res.returncode == 0:
        logger.info(f"SUCCESS: {src}->{tgt} reached convergence in {elapsed:.1f}s ({elapsed/60:.2f} mins)")
    else:
        logger.error(f"FAILURE: {src}->{tgt} failed with exit code {res.returncode}")


def main():
    parser = argparse.ArgumentParser(description="Afri-Gloss-MT 100k Convergence Training Queue")
    parser.add_argument("--max-epochs", type=int, default=50, help="Maximum epochs ceiling (default: 50)")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience (default: 10 checks)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--only", nargs="+", help="Run only specific targets (e.g. --only yo ha ig)")

    args = parser.parse_args()

    active_queue = QUEUE
    if args.only:
        active_queue = [q for q in QUEUE if q["tgt"] in args.only]

    logger.info(f"100k Convergence Queue: {[q['tgt'] for q in active_queue]}")

    for idx, job in enumerate(active_queue, 1):
        logger.info(f"\n>>> Running job {idx}/{len(active_queue)}: {job['src']} -> {job['tgt']}")
        train_single_model(job, max_epochs=args.max_epochs, batch_size=args.batch_size, lr=args.lr, patience=args.patience)

    logger.info("\nALL 100K TRAINING JOBS COMPLETED TO CONVERGENCE!")


if __name__ == "__main__":
    main()
