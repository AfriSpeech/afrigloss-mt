"""
Publish Production 100k Afri-Gloss-MT Models to Hugging Face Hub under AfriSpeech.

Uploads:
  - Model weights (model.safetensors)
  - Tokenizer & sentencepiece models
  - Rich model card (README.md) with tags and usage code
  - Automatically adds model to the AfriSpeech/afri-gloss-mt-models collection!

Usage:
  python scripts/publish_models_to_hf.py --lang tw
  python scripts/publish_models_to_hf.py --lang sw
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

from huggingface_hub import HfApi

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("publish_hf")

COLLECTION_SLUG = "AfriSpeech/afri-gloss-mt-models-6adce3a3c13c6a86f07b6e6d"

MODEL_METADATA = {
    "tw": {
        "repo_name": "AfriSpeech/afrigloss-opus-mt-en-tw",
        "lang_name": "Twi (Akan)",
        "iso": "twi",
        "checkpoint": "checkpoints/afrigloss_opus_en_tw_100k/final",
        "samples": "77,006",
        "bleu": "21.70",
        "chrf2": "47.28",
        "adherence": "92.4%",
        "sample_src": "The leader made a covenant with God on the holy mountain.",
        "sample_injected": "The <kannifoɔ> made a <apam> with God on the holy <bepɔ>.",
        "sample_out": "Na kannifoɔ no ne Nyankopɔn yɛɛ apam kronkron bepɔ no so.",
    },
    "sw": {
        "repo_name": "AfriSpeech/afrigloss-opus-mt-en-sw",
        "lang_name": "Swahili (Kiswahili)",
        "iso": "swh",
        "checkpoint": "checkpoints/afrigloss_opus_en_sw_100k/final",
        "samples": "78,657",
        "bleu": "23.75",
        "chrf2": "53.42",
        "adherence": "96.2%",
        "sample_src": "The government announced peace for the community.",
        "sample_injected": "The <serikali> announced <amani> for the <jamii>.",
        "sample_out": "Serikali ilitangaza amani kwa jamii.",
    },
}


def generate_model_card(meta: Dict[str, str]) -> str:
    """Generate professional model card with Hugging Face YAML metadata."""
    repo_id = meta["repo_name"]
    lang = meta["iso"]
    name = meta["lang_name"]
    bleu = meta["bleu"]
    chrf = meta["chrf2"]
    samples = meta["samples"]
    adherence = meta["adherence"]

    card = f"""---
language:
- en
- {lang}
pipeline_tag: translation
tags:
- translation
- african-languages
- glossary
- terminology
- marian
- opus-mt
- afrigloss-mt
license: apache-2.0
datasets:
- AfriSpeech/africa-corpus
- AfriSpeech/african-corpus-jw
metrics:
- bleu
- chrf
widget:
- text: "{meta['sample_injected']}"
  example_title: "Glossary-Constrained Translation"
---

# {repo_id} 🌍

**{repo_id}** is a glossary-constrained machine translation model for **English $\\rightarrow$ {name}**, fine-tuned under the **Afri-Gloss-MT** framework.

Part of the **[Afri-Gloss-MT Models Collection](https://huggingface.co/collections/{COLLECTION_SLUG})**.

---

## 🌟 Key Capabilities

- **Zero-Shot Terminology Control:** The model accepts user-injected or dictionary-matched target terms in angle brackets (`<target_term>`) inside the source sentence, and reliably preserves them in the generated translation.
- **Trained on Large-Scale Multi-Version Data:** Fine-tuned on **{samples} parallel sentences** from `AfriSpeech/africa-corpus` and `AfriSpeech/african-corpus-jw` with 100% verified in-place noun-chunk replacements.
- **High Terminology Adherence:** Preserves **{adherence}** of injected glossary constraints without breaking surrounding grammatical concords.

---

## 📊 Benchmark Evaluation

Evaluated across **4,300+ unseen test sentences**:

| Metric | Score | Notes |
|:---:|:---:|---|
| **BLEU** | **{bleu}** | Evaluated with `sacrebleu` |
| **chrF2** | **{chrf}** | Character n-gram F-score |
| **Glossary Adherence** | **{adherence}** | Share of injected terms retained in output |

---

## 🚀 How to Use in Python

```python
from transformers import MarianMTModel, MarianTokenizer
import re

model_name = "{repo_id}"
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

# Input text with target terms pre-inserted in angle brackets:
text = "{meta['sample_injected']}"

inputs = tokenizer(text, return_tensors="pt")
outputs = model.generate(**inputs, max_length=128, num_beams=4)
translation = tokenizer.decode(outputs[0], skip_special_tokens=True)

# Clean brackets for final output:
clean_translation = re.sub(r"[<>]", "", translation).strip()
print("Translation:", clean_translation)
# Output: "{meta['sample_out']}"
```

---

## 📜 Toolkit & Repository

- **GitHub Repository:** [https://github.com/AfriSpeech/afrigloss-mt](https://github.com/AfriSpeech/afrigloss-mt)
- **Collection:** [Afri-Gloss-MT Models Collection](https://huggingface.co/collections/{COLLECTION_SLUG})
- **License:** Apache 2.0
"""
    return card


def publish_model(lang: str, api_token: Optional[str] = None):
    """Publish fine-tuned model to Hugging Face Hub."""
    clean_lang = lang.lower().strip()
    if clean_lang not in MODEL_METADATA:
        raise ValueError(f"Language '{lang}' not in 100k metadata. Available: {list(MODEL_METADATA.keys())}")

    meta = MODEL_METADATA[clean_lang]
    repo_id = meta["repo_name"]
    ckpt_dir = REPO_ROOT / meta["checkpoint"]

    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_dir}")

    api = HfApi(token=api_token)
    logger.info(f"Creating / verifying repo: {repo_id}...")
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)

    # Write model card into checkpoint directory
    readme_path = ckpt_dir / "README.md"
    logger.info("Writing model card README.md...")
    card_content = generate_model_card(meta)
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(card_content)

    logger.info(f"Uploading model folder from {ckpt_dir} to {repo_id}...")
    api.upload_folder(
        folder_path=str(ckpt_dir),
        repo_id=repo_id,
        repo_type="model",
        commit_message=f"feat: upload 100k production Afri-Gloss-MT model for {meta['lang_name']}",
    )
    logger.info(f"Model successfully uploaded to https://huggingface.co/{repo_id}!")

    # Add to the dedicated collection
    try:
        logger.info(f"Adding {repo_id} to collection {COLLECTION_SLUG}...")
        api.add_collection_item(
            collection_slug=COLLECTION_SLUG,
            item_id=repo_id,
            item_type="model",
            exists_ok=True,
        )
        logger.info(f"Added to collection: https://huggingface.co/collections/{COLLECTION_SLUG}")
    except Exception as e:
        logger.warning(f"Could not add to collection: {e}")


def main():
    parser = argparse.ArgumentParser(description="Publish 100k Afri-Gloss-MT models to Hugging Face")
    parser.add_argument("--lang", required=True, help="Language code (tw or sw)")
    parser.add_argument("--token", help="Hugging Face API token (optional, uses logged-in token by default)")

    args = parser.parse_args()
    publish_model(args.lang, api_token=args.token)


if __name__ == "__main__":
    main()
