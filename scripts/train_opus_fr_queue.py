"""
French -> African Languages Training Queue for Afri-Gloss-MT.

Fine-tunes OPUS-MT models translating from French into African target languages:
  1. fr -> tw (Twi / Akan) - Helsinki-NLP/opus-mt-fr-tw
  2. fr -> yo (Yoruba) - Helsinki-NLP/opus-mt-fr-yo
  3. fr -> ha (Hausa) - Helsinki-NLP/opus-mt-fr-ha
  4. fr -> ig (Igbo) - Helsinki-NLP/opus-mt-fr-ig
  5. fr -> ee (Ewe) - Helsinki-NLP/opus-mt-fr-ee
  6. fr -> sn (Shona) - Helsinki-NLP/opus-mt-fr-sn

Each model trains dynamically until validation convergence (early_stopping_patience=10).
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
logger = logging.getLogger("train_fr_queue")

QUEUE = [
    {"src": "fr", "tgt": "tw", "model": "Helsinki-NLP/opus-mt-fr-tw", "data": "data/gloss_dataset_fr_tw_full"},
    {"src": "fr", "tgt": "yo", "model": "Helsinki-NLP/opus-mt-fr-yo", "data": "data/gloss_dataset_fr_yo_full"},
    {"src": "fr", "tgt": "ha", "model": "Helsinki-NLP/opus-mt-fr-ha", "data": "data/gloss_dataset_fr_ha_full"},
    {"src": "fr", "tgt": "ig", "model": "Helsinki-NLP/opus-mt-fr-ig", "data": "data/gloss_dataset_fr_ig_full"},
    {"src": "fr", "tgt": "ee", "model": "Helsinki-NLP/opus-mt-fr-ee", "data": "data/gloss_dataset_fr_ee_full"},
    {"src": "fr", "tgt": "sn", "model": "Helsinki-NLP/opus-mt-fr-sn", "data": "data/gloss_dataset_fr_sn_full"},
]


def train_single_model(job: dict, max_epochs: int = 50, batch_size: int = 32, lr: float = 1e-4, patience: int = 10):
    src = job["src"]
    tgt = job["tgt"]
    model_name = job["model"]
    data_dir = REPO_ROOT / job["data"]
    out_dir = REPO_ROOT / "checkpoints" / f"afrigloss_opus_{src}_{tgt}_full"

    final_marker = out_dir / "final" / "config.json"
    if final_marker.exists():
        logger.info(f"[{src}->{tgt}] Final checkpoint already exists at {out_dir / 'final'}, skipping.")
        return

    if not (data_dir / "train.jsonl").exists():
        logger.warning(f"[{src}->{tgt}] Dataset not ready at {data_dir}, skipping.")
        return

    logger.info("=" * 65)
    logger.info(f"STARTING FRENCH -> AFRICAN JOB: {src.upper()} -> {tgt.upper()} ({model_name})")
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
        # Automatically push fine-tuned production model to Hugging Face
        try:
            from scripts.publish_models_to_hf import publish_model
            logger.info(f"Auto-publishing {src}->{tgt} model to Hugging Face...")
            publish_model(tgt, src=src)
        except Exception as e:
            logger.warning(f"Could not auto-publish to Hugging Face: {e}")
    else:
        logger.error(f"FAILURE: {src}->{tgt} failed with exit code {res.returncode}")


def main():
    parser = argparse.ArgumentParser(description="Afri-Gloss-MT French -> African Training Queue")
    parser.add_argument("--max-epochs", type=int, default=50, help="Maximum epochs ceiling (default: 50)")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience (default: 10)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--only", nargs="+", help="Run only specific targets (e.g. --only tw yo)")

    args = parser.parse_args()

    active_queue = QUEUE
    if args.only:
        active_queue = [q for q in QUEUE if q["tgt"] in args.only]

    logger.info(f"French Training Queue: {[q['tgt'] for q in active_queue]}")

    for idx, job in enumerate(active_queue, 1):
        logger.info(f"\n>>> Running French job {idx}/{len(active_queue)}: {job['src']} -> {job['tgt']}")
        train_single_model(job, max_epochs=args.max_epochs, batch_size=args.batch_size, lr=args.lr, patience=args.patience)

    logger.info("\nALL FRENCH TRAINING JOBS COMPLETED TO CONVERGENCE!")


if __name__ == "__main__":
    main()
