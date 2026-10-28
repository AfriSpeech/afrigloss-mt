"""
Publish Afri-Adapt Proper Nouns & Named Entities Dataset to Hugging Face Hub.

Uploads AfriSpeech/afri-adapt as a single unified dataset with two subsets:
  1. 'en' : English -> 723 African languages (868,684 proper noun pairs)
  2. 'fr' : French  -> 74 African languages  (224,946 proper noun pairs)

Usage:
  from datasets import load_dataset
  ds_en = load_dataset("AfriSpeech/afri-adapt", "en")
  ds_fr = load_dataset("AfriSpeech/afri-adapt", "fr")

License: CC-BY-NC-4.0
"""

import argparse
import glob
import json
import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from huggingface_hub import HfApi

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("publish_afri_adapt")

REPO_ID = "AfriSpeech/afri-loanwords"
COLLECTION_SLUG = "AfriSpeech/afri-gloss-mt-datasets-6adf1e6b82063570470d8738"
AFRI_ADAPT_SRC_DIR = REPO_ROOT / "afri-adapt"


def generate_dataset_card(en_count: int, fr_count: int, en_langs: int, fr_langs: int) -> str:
    """Generate professional model card with Hugging Face YAML metadata."""
    total_pairs = en_count + fr_count

    card = f"""---
language:
- en
- fr
- multilingual
license: cc-by-nc-4.0
task_categories:
- translation
- token-classification
tags:
- african-languages
- proper-nouns
- named-entities
- terminology
- afri-adapt
- afrigloss
configs:
- config_name: en
  data_files:
  - split: train
    path: data/en/train.parquet
- config_name: fr
  data_files:
  - split: train
    path: data/fr/train.parquet
---

# `AfriSpeech/afri-loanwords` 🌍

**Afri-Loanwords** is a continental-scale, human-grounded bilingual dataset documenting how **foreign words, named entities, personal names, geographical places, and loanwords** are phonologically, morphologically, and orthographically adapted into **723 African languages**.

All pairs are extracted directly from authentic multi-source parallel text (`AfriSpeech/africa-corpus` and `AfriSpeech/african-corpus-jw`) using dependency-parsed noun-phrase alignments and consensus majority voting (**100% grounded in human bitext, zero synthetic LLM translations**).

- **Author**: Mich-Seth Owusu (`michseth@ghananlp.org`)
- **Organization**: [AfriSpeech](https://huggingface.co/AfriSpeech) & Ghana NLP
- **Codebase**: [github.com/AfriSpeech/afrigloss-mt](https://github.com/AfriSpeech/afrigloss-mt)
- **Total Entities**: **{total_pairs:,}** verified proper noun pairs
- **Total African Languages**: **723 languages**

---

## 📊 Dataset Subsets

| Subset (`config`) | Source Language | African Target Languages | Total Verified Proper Nouns | Format |
| :---: | :--- | :---: | :---: | :---: |
| **`en`** | **English** | **{en_langs} languages** | **{en_count:,}** | Snappy Parquet |
| **`fr`** | **French** | **{fr_langs} languages** | **{fr_count:,}** | Snappy Parquet |
| **Total** | | | **{total_pairs:,}** | |

---

## 🚀 Quickstart & Loading

Load either subset directly in Python using Hugging Face `datasets`:

```python
from datasets import load_dataset
import pandas as pd

# 1. Load English source loanwords (868k+ pairs across 723 African languages)
dataset_en = load_dataset("AfriSpeech/afri-loanwords", "en", split="train")
print(dataset_en[0])
# {'src_lang': 'en', 'tgt_lang': 'aab', 'source_term': 'Jesus', 'target_term': 'Yeso', 'confidence': 1.0, 'frequency': 627}

# 2. Filter for a specific target language (e.g. Swahili, Twi, Yoruba, Amharic)
df = dataset_en.to_pandas()
swahili_entities = df[df["tgt_lang"] == "sw"]
print("Swahili loanword adaptations:", len(swahili_entities))
print(swahili_entities.head(10))

# 3. Load French source loanwords (224k+ pairs across 74 African languages)
dataset_fr = load_dataset("AfriSpeech/afri-loanwords", "fr", split="train")
df_fr = dataset_fr.to_pandas()
twi_fr_entities = df_fr[df_fr["tgt_lang"] == "tw"]
print(twi_fr_entities.head(10))
```

### Data Fields

- `src_lang`: Source language code (`en` or `fr`).
- `tgt_lang`: Target African language code (e.g. `tw`, `sw`, `yo`, `ha`, `ig`, `ee`, `sn`, `amh`, `lin`, `zul`, etc.).
- `source_term`: The named entity or proper noun in English or French.
- `target_term`: The verified translation / localized transliteration in the African language.
- `confidence`: Empirical consensus score across all parallel sentence occurrences (0.00 – 1.00).
- `frequency`: Total number of occurrences observed across the corpus bitext.

---

## ⚖️ License

This dataset is distributed under the **Creative Commons Attribution-NonCommercial 4.0 International ([CC-BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/))** license.

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


def build_and_publish_afri_adapt(api_token: Optional[str] = None):
    """Aggregate all proper noun TSV files into en and fr parquets and upload to HF."""
    api = HfApi(token=api_token)

    logger.info("=" * 65)
    logger.info(f"PUBLISHING AFRI-ADAPT DATASET TO HF: {REPO_ID}")
    logger.info("=" * 65)

    api.create_repo(repo_id=REPO_ID, repo_type="dataset", exist_ok=True)

    staging_dir = REPO_ROOT / "data" / "staging_parquet" / "afri_adapt"
    data_dir = staging_dir / "data"
    en_dir = data_dir / "en"
    fr_dir = data_dir / "fr"
    en_dir.mkdir(parents=True, exist_ok=True)
    fr_dir.mkdir(parents=True, exist_ok=True)

    # 1. Process English proper nouns
    logger.info("Processing English proper nouns...")
    en_files = sorted(glob.glob(str(AFRI_ADAPT_SRC_DIR / "en_*_proper_nouns.tsv")))
    en_dfs = []
    en_targets = set()
    for f in en_files:
        filename = os.path.basename(f)
        parts = filename.split("_")
        if len(parts) >= 2:
            tgt = parts[1]
            try:
                df = pd.read_csv(f, sep="\t", on_bad_lines="skip")
                df["src_lang"] = "en"
                df["tgt_lang"] = tgt
                en_dfs.append(df[["src_lang", "tgt_lang", "source_term", "target_term", "confidence", "frequency"]])
                en_targets.add(tgt)
            except Exception as e:
                logger.warning(f"Error reading {f}: {e}")

    df_en = pd.concat(en_dfs, ignore_index=True)
    en_parquet_path = en_dir / "train.parquet"
    df_en.to_parquet(en_parquet_path, engine="pyarrow", compression="snappy", index=False)
    logger.info(f"English subset written: {len(df_en):,} rows across {len(en_targets)} languages -> {en_parquet_path}")

    # 2. Process French proper nouns
    logger.info("Processing French proper nouns...")
    fr_files = sorted(glob.glob(str(AFRI_ADAPT_SRC_DIR / "fr_*_proper_nouns.tsv")))
    fr_dfs = []
    fr_targets = set()
    for f in fr_files:
        filename = os.path.basename(f)
        parts = filename.split("_")
        if len(parts) >= 2:
            tgt = parts[1]
            try:
                df = pd.read_csv(f, sep="\t", on_bad_lines="skip")
                df["src_lang"] = "fr"
                df["tgt_lang"] = tgt
                fr_dfs.append(df[["src_lang", "tgt_lang", "source_term", "target_term", "confidence", "frequency"]])
                fr_targets.add(tgt)
            except Exception as e:
                logger.warning(f"Error reading {f}: {e}")

    df_fr = pd.concat(fr_dfs, ignore_index=True)
    fr_parquet_path = fr_dir / "train.parquet"
    df_fr.to_parquet(fr_parquet_path, engine="pyarrow", compression="snappy", index=False)
    logger.info(f"French subset written: {len(df_fr):,} rows across {len(fr_targets)} languages -> {fr_parquet_path}")

    # 3. Generate README.md
    readme_path = staging_dir / "README.md"
    card_content = generate_dataset_card(len(df_en), len(df_fr), len(en_targets), len(fr_targets))
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(card_content)

    # 4. Upload to Hugging Face Hub
    logger.info(f"Uploading afri-adapt dataset to https://huggingface.co/datasets/{REPO_ID}...")
    api.upload_folder(
        folder_path=str(staging_dir),
        repo_id=REPO_ID,
        repo_type="dataset",
        commit_message=f"feat: upload Afri-Adapt proper nouns dataset ({len(df_en) + len(df_fr):,} entities across {len(en_targets)} languages)",
    )
    logger.info(f"Successfully published: https://huggingface.co/datasets/{REPO_ID}")

    # 5. Add to Collection
    try:
        logger.info(f"Adding {REPO_ID} to collection {COLLECTION_SLUG}...")
        api.add_collection_item(
            collection_slug=COLLECTION_SLUG,
            item_id=REPO_ID,
            item_type="dataset",
            exists_ok=True,
        )
        logger.info(f"Added to collection: https://huggingface.co/collections/{COLLECTION_SLUG}")
    except Exception as e:
        logger.warning(f"Could not add to collection: {e}")


def main():
    parser = argparse.ArgumentParser(description="Publish Afri-Adapt dataset to Hugging Face")
    parser.add_argument("--token", help="Hugging Face API token (optional)")
    args = parser.parse_args()

    build_and_publish_afri_adapt(api_token=args.token)


if __name__ == "__main__":
    main()
