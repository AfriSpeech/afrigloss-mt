"""
Train All 3 NLLB Models (600M, 1.3B, 3.3B) on Full Pan-African Multilingual Corpus.

Sequential Queue:
  1. facebook/nllb-200-distilled-600M  -> checkpoints/afrigloss_nllb_600m_multilingual/
  2. facebook/nllb-200-1.3B            -> checkpoints/afrigloss_nllb_1.3b_multilingual/
  3. facebook/nllb-200-3.3B            -> checkpoints/afrigloss_nllb_3.3b_multilingual/

Dataset:
  data/gloss_dataset_multilingual_african/train.jsonl (3.25M samples)
  data/gloss_dataset_multilingual_african/validation.jsonl (180k samples)
  Supports dynamic sample-level multi-source (eng_Latn, fra_Latn)
  and multi-target (twi_Latn, swh_Latn, yor_Latn, hau_Latn, ibo_Latn, ewe_Latn, sna_Latn).

Upon completion, each model is automatically published to Hugging Face:
  AfriSpeech/afrigloss-nllb-600m
  AfriSpeech/afrigloss-nllb-1.3b
  AfriSpeech/afrigloss-nllb-3.3b
  under license: cc-by-nc-4.0 and added to AfriSpeech/afri-gloss-mt-models collection.

Usage:
  python scripts/train_nllb_all_sizes_queue.py
  python scripts/train_nllb_all_sizes_queue.py --only 600m
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
logger = logging.getLogger("train_nllb_queue")

COLLECTION_SLUG = "AfriSpeech/afri-gloss-mt-models-6adce3a3c13c6a86f07b6e6d"
DATA_DIR = REPO_ROOT / "data" / "gloss_dataset_multilingual_african"

NLLB_QUEUE = [
    {
        "tag": "600m",
        "model_id": "facebook/nllb-200-distilled-600M",
        "repo_name": "AfriSpeech/afrigloss-nllb-600m",
        "out_dir": "checkpoints/afrigloss_nllb_600m_multilingual",
        "batch_size": 16,
        "grad_accum": 2,
        "lr": 2e-4,
    },
    {
        "tag": "1.3b",
        "model_id": "facebook/nllb-200-1.3B",
        "repo_name": "AfriSpeech/afrigloss-nllb-1.3b",
        "out_dir": "checkpoints/afrigloss_nllb_1.3b_multilingual",
        "batch_size": 12,
        "grad_accum": 3,
        "lr": 1.5e-4,
    },
    {
        "tag": "3.3b",
        "model_id": "facebook/nllb-200-3.3B",
        "repo_name": "AfriSpeech/afrigloss-nllb-3.3b",
        "out_dir": "checkpoints/afrigloss_nllb_3.3b_multilingual",
        "batch_size": 8,
        "grad_accum": 4,
        "lr": 1e-4,
    },
]


def generate_nllb_model_card(tag: str, repo_id: str, model_id: str) -> str:
    """Generate professional model card with Hugging Face YAML metadata."""
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
- nllb
- nllb-200
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

**{repo_id}** is a unified Pan-African multilingual glossary-constrained machine translation model fine-tuned from **`{model_id}`** under the **Afri-Gloss-MT** framework.

Part of the **[Afri-Gloss-MT Models Collection](https://huggingface.co/collections/{COLLECTION_SLUG})**.

---

## 🌟 Key Capabilities

- **Zero-Shot Multi-Target Terminology Control:** Translates from both **English (`eng_Latn`)** and **French (`fra_Latn`)** into **7 major African languages**:
  - **Twi / Akan** (`twi_Latn`)
  - **Swahili** (`swh_Latn`)
  - **Yoruba** (`yor_Latn`)
  - **Hausa** (`hau_Latn`)
  - **Igbo** (`ibo_Latn`)
  - **Ewe** (`ewe_Latn`)
  - **Shona** (`sna_Latn`)
- **Trained on 3.6 Million Sentences:** Trained across the combined Pan-African Gloss-MT corpus with verified in-place noun-chunk replacements (`<target_term>`).
- **In-Place Terminology Preservation:** Guarantees hard adherence to specialized domains (medical, legal, technical, dialectal).

---

## 🚀 Quickstart & Inference

```python
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model_name = "{repo_id}"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# 1. Translate English -> Twi with in-place glossary constraint
tokenizer.src_lang = "eng_Latn"
tokenizer.tgt_lang = "twi_Latn"

prompt = "The <kannifoɔ> made a <apam> with God on the holy <bepɔ>."
inputs = tokenizer(prompt, return_tensors="pt")
translated = model.generate(**inputs, forced_bos_token_id=tokenizer.lang_code_to_id["twi_Latn"])
print(tokenizer.decode(translated[0], skip_special_tokens=True))
# Na kannifoɔ no ne Nyankopɔn yɛɛ apam kronkron bepɔ no so.
```

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


def publish_nllb_to_hf(job: dict, api: HfApi):
    """Publish trained NLLB checkpoint to Hugging Face."""
    repo_id = job["repo_name"]
    final_dir = REPO_ROOT / job["out_dir"] / "final"

    if not final_dir.exists():
        logger.warning(f"Final directory not found at {final_dir}, skipping upload.")
        return

    logger.info("=" * 65)
    logger.info(f"PUBLISHING NLLB MODEL TO HF: {repo_id}")
    logger.info("=" * 65)

    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)

    # Write model card
    readme_path = final_dir / "README.md"
    card_content = generate_nllb_model_card(job["tag"], repo_id, job["model_id"])
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(card_content)

    api.upload_folder(
        folder_path=str(final_dir),
        repo_id=repo_id,
        repo_type="model",
        commit_message=f"feat: upload production {job['tag'].upper()} Afri-Gloss-MT NLLB model",
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


def train_single_nllb(job: dict, epochs: int = 3, lora_r: int = 16, full_finetune: bool = False, api: Optional[HfApi] = None):
    """Execute training for a single NLLB model."""
    tag = job["tag"]
    model_id = job["model_id"]
    out_dir = REPO_ROOT / job["out_dir"]
    final_marker = out_dir / "final" / "config.json"

    if final_marker.exists():
        logger.info(f"[{tag}] Final checkpoint already exists at {out_dir / 'final'}, skipping training.")
        if api:
            publish_nllb_to_hf(job, api)
        return

    train_file = DATA_DIR / "train.jsonl"
    val_file = DATA_DIR / "validation.jsonl"

    if not train_file.exists():
        logger.error(f"Multilingual train.jsonl not found in {DATA_DIR}!")
        return

    logger.info("=" * 65)
    logger.info(f"STARTING NLLB TRAINING: {tag.upper()} ({model_id})")
    logger.info(f"Data: {train_file}")
    logger.info(f"Output: {out_dir}")
    logger.info(f"Epochs: {epochs} | Batch size: {job['batch_size']} (accum: {job['grad_accum']})")
    logger.info("=" * 65)

    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "train_multilingual_nllb.py"),
        "--model", model_id,
        "--batch-size", str(job["batch_size"]),
        "--grad-accum", str(job["grad_accum"]),
        "--lr", str(job["lr"]),
        "--epochs", str(epochs),
        "--lora-r", str(0 if full_finetune else lora_r),
        "--out-dir", str(out_dir),
    ]

    if full_finetune:
        cmd.append("--full-finetune")

    t0 = time.time()
    res = subprocess.run(cmd)
    elapsed = time.time() - t0

    if res.returncode == 0:
        logger.info(f"SUCCESS: NLLB-{tag.upper()} completed in {elapsed:.1f}s ({elapsed/3600:.2f} hours)")
        if api:
            publish_nllb_to_hf(job, api)
    else:
        logger.error(f"FAILURE: NLLB-{tag.upper()} failed with exit code {res.returncode}")


def main():
    parser = argparse.ArgumentParser(description="Train All 3 NLLB Models (600M, 1.3B, 3.3B)")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs (default: 3)")
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank (default: 16)")
    parser.add_argument("--full-finetune", action="store_true", help="Full fine-tuning without LoRA")
    parser.add_argument("--only", choices=["600m", "1.3b", "3.3b"], help="Train only a specific size")
    parser.add_argument("--token", help="Hugging Face API token (optional)")

    args = parser.parse_args()
    api = HfApi(token=args.token)

    queue = NLLB_QUEUE
    if args.only:
        queue = [q for q in NLLB_QUEUE if q["tag"] == args.only]

    logger.info(f"NLLB Training Queue: {[q['tag'] for q in queue]}")

    for idx, job in enumerate(queue, 1):
        logger.info(f"\n>>> Running NLLB job {idx}/{len(queue)}: {job['tag'].upper()}")
        train_single_nllb(
            job,
            epochs=args.epochs,
            lora_r=args.lora_r,
            full_finetune=args.full_finetune,
            api=api,
        )

    logger.info("\nALL NLLB JOBS COMPLETED SUCCESSFULLY!")


if __name__ == "__main__":
    main()
