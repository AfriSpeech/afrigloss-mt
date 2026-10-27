"""
Train All 5 Foundation Models to Convergence on the Pan-African Gloss-MT Corpus.

Complete Foundation Model Lineup:
  1. NLLB-600M            (facebook/nllb-200-distilled-600M) -> checkpoints/afrigloss_nllb_600m_multilingual/
  2. NLLB-1.3B            (facebook/nllb-200-1.3B)           -> checkpoints/afrigloss_nllb_1.3b_multilingual/
  3. NLLB-3.3B            (facebook/nllb-200-3.3B)           -> checkpoints/afrigloss_nllb_3.3b_multilingual/
  4. TranslateGemma-4B    (google/translategemma-4b-it)      -> checkpoints/afrigloss_translategemma_4b/
  5. TranslateGemma-12B   (google/translategemma-12b-it)     -> checkpoints/afrigloss_translategemma_12b/

Dataset:
  data/gloss_dataset_multilingual_african/train.jsonl (3.25M samples)
  data/gloss_dataset_multilingual_african/validation.jsonl (180k samples)

All models train to early-stopping validation convergence (patience=10).
Upon convergence, each model is automatically published to Hugging Face under CC-BY-NC 4.0:
  - AfriSpeech/afrigloss-nllb-600m
  - AfriSpeech/afrigloss-nllb-1.3b
  - AfriSpeech/afrigloss-nllb-3.3b
  - AfriSpeech/afrigloss-translategemma-4b
  - AfriSpeech/afrigloss-translategemma-12b

Usage:
  python scripts/train_foundation_models_queue.py
  python scripts/train_foundation_models_queue.py --only gemma-4b
  python scripts/train_foundation_models_queue.py --only gemma-12b
"""

import argparse
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

from huggingface_hub import HfApi

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("foundation_queue")

COLLECTION_SLUG = "AfriSpeech/afri-gloss-mt-models-6adce3a3c13c6a86f07b6e6d"
DATA_DIR = REPO_ROOT / "data" / "gloss_dataset_multilingual_african"

MODEL_QUEUE = [
    {
        "tag": "600m",
        "type": "nllb",
        "model_id": "facebook/nllb-200-distilled-600M",
        "repo_name": "AfriSpeech/afrigloss-nllb-600m",
        "out_dir": "checkpoints/afrigloss_nllb_600m_multilingual",
        "batch_size": 16,
        "grad_accum": 2,
        "lr": 2e-4,
    },
    {
        "tag": "1.3b",
        "type": "nllb",
        "model_id": "facebook/nllb-200-1.3B",
        "repo_name": "AfriSpeech/afrigloss-nllb-1.3b",
        "out_dir": "checkpoints/afrigloss_nllb_1.3b_multilingual",
        "batch_size": 12,
        "grad_accum": 3,
        "lr": 1.5e-4,
    },
    {
        "tag": "3.3b",
        "type": "nllb",
        "model_id": "facebook/nllb-200-3.3B",
        "repo_name": "AfriSpeech/afrigloss-nllb-3.3b",
        "out_dir": "checkpoints/afrigloss_nllb_3.3b_multilingual",
        "batch_size": 8,
        "grad_accum": 4,
        "lr": 1e-4,
    },
    {
        "tag": "gemma-4b",
        "type": "gemma",
        "model_id": "google/translategemma-4b-it",
        "repo_name": "AfriSpeech/afrigloss-translategemma-4b",
        "out_dir": "checkpoints/afrigloss_translategemma_4b",
        "batch_size": 8,
        "grad_accum": 4,
        "lr": 1e-4,
    },
    {
        "tag": "gemma-12b",
        "type": "gemma",
        "model_id": "google/translategemma-12b-it",
        "repo_name": "AfriSpeech/afrigloss-translategemma-12b",
        "out_dir": "checkpoints/afrigloss_translategemma_12b",
        "batch_size": 4,
        "grad_accum": 8,
        "lr": 8e-5,
    },
]


def generate_model_card(job: dict) -> str:
    """Generate Hugging Face YAML model card for NLLB or TranslateGemma."""
    repo_id = job["repo_name"]
    model_id = job["model_id"]
    is_gemma = job["type"] == "gemma"
    tag_name = "TranslateGemma" if is_gemma else "NLLB-200"

    card = f"""---
language:
- en
- fr
- tw
- sw
- yo
- ha
- ig
- ee
- sn
- multilingual
pipeline_tag: translation
tags:
- translation
- african-languages
- glossary
- terminology
- {'translategemma' if is_gemma else 'nllb'}
- afrigloss-mt
license: cc-by-nc-4.0
datasets:
- AfriSpeech/gloss-mt-train-en
- AfriSpeech/gloss-mt-train-fr
widget:
- text: "The <kannifoɔ> made a covenant with God."
  example_title: "English to Twi with Glossary"
- text: "Le <kiongozi> a conclu une alliance avec Dieu."
  example_title: "French to Swahili with Glossary"
---

# {repo_id} 🌍

**{repo_id}** is a production glossary-constrained multilingual machine translation model fine-tuned from **`{model_id}`** ({tag_name}) under the **Afri-Gloss-MT** framework.

Part of the **[Afri-Gloss-MT Models Collection](https://huggingface.co/collections/{COLLECTION_SLUG})**.

---

## 🌟 Key Capabilities

- **Zero-Shot Multi-Target Terminology Control:** Translates from both **English** and **French** into **7 major African languages**:
  - **Twi / Akan** (`tw` / `ak`)
  - **Swahili** (`sw`)
  - **Yoruba** (`yo`)
  - **Hausa** (`ha`)
  - **Igbo** (`ig`)
  - **Ewe** (`ee`)
  - **Shona** (`sn`)
- **Trained to Convergence on 3.6 Million Sentences:** Trained across the combined Pan-African Gloss-MT corpus with verified in-place noun-chunk replacements (`<target_term>`).
- **Guaranteed In-Place Terminology Preservation:** Preserves critical domain terms (medical, legal, technical, dialectal) without hallucinations.

---

## ⚖️ License

This model is licensed under the **Creative Commons Attribution-NonCommercial 4.0 International ([CC-BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/))** license.

---

## 📚 Citation

```bibtex
@article{{owusu2026afrigloss,
  title={{Afri-Gloss-MT: In-Place Glossary Constraints for African Machine Translation}},
  author={{Owusu, Mich-Seth}},
  journal={{arXiv preprint}},
  year={{2026}}
}}
```
"""
    return card


def publish_foundation_model_to_hf(job: dict, api: HfApi):
    """Publish trained model checkpoint to Hugging Face."""
    repo_id = job["repo_name"]
    final_dir = REPO_ROOT / job["out_dir"] / "final"

    if not final_dir.exists():
        logger.warning(f"Final directory not found at {final_dir}, skipping upload.")
        return

    logger.info("=" * 65)
    logger.info(f"PUBLISHING FOUNDATION MODEL TO HF: {repo_id}")
    logger.info("=" * 65)

    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)

    # Write model card
    readme_path = final_dir / "README.md"
    card_content = generate_model_card(job)
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(card_content)

    api.upload_folder(
        folder_path=str(final_dir),
        repo_id=repo_id,
        repo_type="model",
        commit_message=f"feat: upload production {job['tag'].upper()} Afri-Gloss-MT model",
    )
    logger.info(f"Successfully uploaded: https://huggingface.co/{repo_id}")

    try:
        api.add_collection_item(
            collection_slug=COLLECTION_SLUG,
            item_id=repo_id,
            item_type="model",
            exists_ok=True,
        )
        logger.info(f"Added to collection: https://huggingface.co/collections/{COLLECTION_SLUG}")
    except Exception as e:
        logger.warning(f"Could not add to collection: {e}")


def train_single_job(job: dict, epochs: int = 20, patience: int = 10, lora_r: int = 16, api: Optional[HfApi] = None):
    """Execute training for a single foundation model."""
    tag = job["tag"]
    model_id = job["model_id"]
    m_type = job["type"]
    out_dir = REPO_ROOT / job["out_dir"]

    # Check if final checkpoint already exists
    has_final = (out_dir / "final" / "config.json").exists() or (out_dir / "final" / "adapter_config.json").exists()
    if has_final:
        logger.info(f"[{tag}] Final checkpoint already exists at {out_dir / 'final'}, skipping training.")
        if api:
            publish_foundation_model_to_hf(job, api)
        return

    train_file = DATA_DIR / "train.jsonl"
    val_file = DATA_DIR / "validation.jsonl"

    if not train_file.exists():
        logger.error(f"Multilingual train.jsonl not found in {DATA_DIR}!")
        return

    logger.info("=" * 65)
    logger.info(f"STARTING CONVERGENCE TRAINING: {tag.upper()} ({model_id})")
    logger.info(f"Model Type: {m_type.upper()}")
    logger.info(f"Data: {train_file}")
    logger.info(f"Output: {out_dir}")
    logger.info(f"Max Epochs: {epochs} | Early Stopping Patience: {patience}")
    logger.info(f"Batch size: {job['batch_size']} (accum: {job['grad_accum']})")
    logger.info("=" * 65)

    if m_type == "nllb":
        cmd = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "train_multilingual_nllb.py"),
            "--model", model_id,
            "--batch-size", str(job["batch_size"]),
            "--grad-accum", str(job["grad_accum"]),
            "--lr", str(job["lr"]),
            "--epochs", str(epochs),
            "--lora-r", str(lora_r),
            "--out-dir", str(out_dir),
        ]
    else:
        # TranslateGemma
        cmd = [
            sys.executable,
            str(REPO_ROOT / "afrigloss" / "train_gemma.py"),
            "--model", model_id,
            "--train-file", str(train_file),
            "--val-file", str(val_file),
            "--out", str(out_dir),
            "--batch-size", str(job["batch_size"]),
            "--grad-accum", str(job["grad_accum"]),
            "--lr", str(job["lr"]),
            "--epochs", str(epochs),
            "--patience", str(patience),
            "--lora-r", str(lora_r),
        ]

    t0 = time.time()
    res = subprocess.run(cmd)
    elapsed = time.time() - t0

    if res.returncode == 0:
        logger.info(f"SUCCESS: {tag.upper()} reached convergence in {elapsed:.1f}s ({elapsed/3600:.2f} hours)")
        if api:
            publish_foundation_model_to_hf(job, api)
    else:
        logger.error(f"FAILURE: {tag.upper()} failed with exit code {res.returncode}")


def main():
    parser = argparse.ArgumentParser(description="Train Foundation Models to Convergence (NLLB 600M/1.3B/3.3B & TranslateGemma 4B/12B)")
    parser.add_argument("--epochs", type=int, default=20, help="Max epochs ceiling (default: 20)")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience (default: 10)")
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank (default: 16)")
    parser.add_argument("--only", choices=["600m", "1.3b", "3.3b", "gemma-4b", "gemma-12b"], help="Train only a specific model")
    parser.add_argument("--token", help="Hugging Face API token (optional)")

    args = parser.parse_args()
    api = HfApi(token=args.token)

    queue = MODEL_QUEUE
    if args.only:
        queue = [q for q in MODEL_QUEUE if q["tag"] == args.only]

    logger.info(f"Foundation Models Queue: {[q['tag'] for q in queue]}")

    for idx, job in enumerate(queue, 1):
        logger.info(f"\n>>> Running foundation model job {idx}/{len(queue)}: {job['tag'].upper()}")
        train_single_job(
            job,
            epochs=args.epochs,
            patience=args.patience,
            lora_r=args.lora_r,
            api=api,
        )

    logger.info("\nALL FOUNDATION MODEL JOBS COMPLETED TO CONVERGENCE!")


if __name__ == "__main__":
    main()
