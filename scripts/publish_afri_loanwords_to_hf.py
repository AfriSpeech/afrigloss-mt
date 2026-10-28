"""
Publish Afri-Loanwords Dataset to Hugging Face with Directional Subsets.

Architecture:
  AfriSpeech/afri-loanwords
  Each unique language direction gets its OWN SUBSET:
    - en-tw, en-sw, en-yo, en-ha, en-ig, en-ee, en-sn, en-amh, en-lin, en-zul, ... (723 English pairs)
    - fr-tw, fr-sw, fr-yo, fr-ha, fr-ig, fr-ee, fr-sn, fr-kin, fr-kon, fr-loz, ... (76+ French pairs)

Usage:
  from datasets import load_dataset
  ds_en_tw = load_dataset("AfriSpeech/afri-loanwords", "en-tw")
  ds_fr_sw = load_dataset("AfriSpeech/afri-loanwords", "fr-sw")

License: CC-BY-NC-4.0
"""

import argparse
import glob
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from huggingface_hub import HfApi

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("afri_loanwords")

REPO_ID = "AfriSpeech/afri-loanwords"
COLLECTION_SLUG = "AfriSpeech/afri-gloss-mt-datasets-6adf1e6b82063570470d8738"
SRC_TSV_DIR = REPO_ROOT / "afri-adapt"
STAGING_DIR = REPO_ROOT / "data" / "staging_parquet" / "afri_loanwords"


def generate_loanwords_dataset_card(subsets_stats: Dict[str, int]) -> str:
    """Generate YAML frontmatter configs and Markdown documentation for Hugging Face."""
    configs_yaml = []
    for pair in sorted(subsets_stats.keys()):
        configs_yaml.append(f"""- config_name: {pair}
  data_files:
  - split: train
    path: data/{pair}/train.parquet""")

    configs_str = "\n".join(configs_yaml)

    # Count stats
    en_pairs = [p for p in subsets_stats.keys() if p.startswith("en-")]
    fr_pairs = [p for p in subsets_stats.keys() if p.startswith("fr-")]
    total_en_terms = sum(subsets_stats[p] for p in en_pairs)
    total_fr_terms = sum(subsets_stats[p] for p in fr_pairs)
    grand_total = total_en_terms + total_fr_terms

    # Markdown table for top/sample subsets
    sample_rows = []
    priority_pairs = ["en-tw", "en-sw", "en-yo", "en-ha", "en-ig", "en-ee", "en-sn",
                      "en-amh", "en-lin", "en-zul", "en-kin",
                      "fr-tw", "fr-sw", "fr-yo", "fr-ha", "fr-ig", "fr-ee", "fr-sn", "fr-kin", "fr-loz"]
    for p in priority_pairs:
        if p in subsets_stats:
            src, tgt = p.split("-")
            src_name = "English" if src == "en" else "French"
            sample_rows.append(f"| `{p}` | {src_name} $\\rightarrow$ `{tgt}` | **{subsets_stats[p]:,}** |")

    sample_table = "\n".join(sample_rows)

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
- loanwords
- proper-nouns
- named-entities
- terminology
- transliteration
- afri-loanwords
- afrigloss
configs:
{configs_str}
---

# `AfriSpeech/afri-loanwords` 🌍

**Afri-Loanwords** is a comprehensive, human-grounded dataset documenting how **foreign loanwords, named entities, personal names, geographical places, and cultural concepts** from English and French have been phonologically, morphologically, and orthographically adapted into African languages.

Every pair is extracted directly from authentic multi-source parallel text (`AfriSpeech/africa-corpus` and `AfriSpeech/african-corpus-jw`) using dependency-parsed noun-phrase alignments and consensus majority voting (**100% human bitext, zero synthetic LLM translations**).

- **Author**: Mich-Seth Owusu (`michseth@ghananlp.org`)
- **Organization**: [AfriSpeech](https://huggingface.co/AfriSpeech) & Ghana NLP
- **Codebase**: [github.com/AfriSpeech/afrigloss-mt](https://github.com/AfriSpeech/afrigloss-mt)
- **Total Directional Subsets**: **{len(subsets_stats)} language pairs**
  - **English Source (`en-*`)**: {len(en_pairs)} African target languages ({total_en_terms:,} loanword pairs)
  - **French Source (`fr-*`)**: {len(fr_pairs)} African target languages ({total_fr_terms:,} loanword pairs)
- **Total Verified Loanword Adaptations**: **{grand_total:,}**

---

## 📊 Sample Directional Subsets

| Subset (`config`) | Language Direction | Verified Adapted Loanwords |
| :---: | :--- | :---: |
{sample_table}
| ... | *(and {len(subsets_stats) - len(sample_rows)} more language directions)* | ... |
| **Grand Total** | **{len(subsets_stats)} Language Pairs** | **{grand_total:,}** |

---

## 🚀 How to Load Specific Language Directions

Load any specific directional subset directly with Hugging Face `datasets`:

```python
from datasets import load_dataset

# 1. Load English -> Twi (Akan) loanword adaptations
ds_en_tw = load_dataset("AfriSpeech/afri-loanwords", "en-tw", split="train")
print(ds_en_tw[0])
# {{'source_term': 'Jehovah', 'target_term': 'Yehowa', 'confidence': 0.9949, 'frequency': 32545}}

# 2. Load English -> Swahili loanword adaptations
ds_en_sw = load_dataset("AfriSpeech/afri-loanwords", "en-sw", split="train")
print(ds_en_sw[0])
# {{'source_term': 'Jesus', 'target_term': 'Yesu', 'confidence': 0.9963, 'frequency': 18384}}

# 3. Load French -> Twi loanword adaptations
ds_fr_tw = load_dataset("AfriSpeech/afri-loanwords", "fr-tw", split="train")
print(ds_fr_tw[0])
# {{'source_term': 'Jéhovah', 'target_term': 'Yehowa', 'confidence': 0.9925, 'frequency': 45080}}

# 4. Load English -> Amharic loanword adaptations
ds_en_amh = load_dataset("AfriSpeech/afri-loanwords", "en-amh", split="train")
print(ds_en_amh[0])
# {{'source_term': 'Jesus', 'target_term': 'ኢየሱስ', 'confidence': 1.0, 'frequency': 21841}}
```

### Data Fields

- `source_term`: The foreign loanword or named entity in English or French.
- `target_term`: The adapted translation / localized transliteration in the African language.
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


def publish_afri_loanwords(api_token: Optional[str] = None):
    """Convert all directional TSV files into Parquet subsets and upload to HF."""
    api = HfApi(token=api_token)

    logger.info("=" * 65)
    logger.info(f"PUBLISHING AFRI-LOANWORDS DATASET TO HF: {REPO_ID}")
    logger.info("=" * 65)

    api.create_repo(repo_id=REPO_ID, repo_type="dataset", exist_ok=True)

    data_dir = STAGING_DIR / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    # Find all TSV files in afri-adapt
    tsv_files = sorted(glob.glob(str(SRC_TSV_DIR / "*_*_proper_nouns.tsv")))
    logger.info(f"Found {len(tsv_files)} directional TSV files.")

    subsets_stats: Dict[str, int] = {}

    for idx, f in enumerate(tsv_files, 1):
        filename = os.path.basename(f)
        parts = filename.split("_")
        if len(parts) < 3:
            continue
        src = parts[0]  # 'en' or 'fr'
        tgt = parts[1]  # 'tw', 'sw', etc.
        pair_key = f"{src}-{tgt}"

        pair_dir = data_dir / pair_key
        pair_dir.mkdir(parents=True, exist_ok=True)
        parquet_file = pair_dir / "train.parquet"

        # Check if already converted
        if parquet_file.exists() and parquet_file.stat().st_size > 0:
            try:
                df = pd.read_parquet(parquet_file, columns=[])
                subsets_stats[pair_key] = len(df)
                continue
            except Exception:
                pass

        try:
            df = pd.read_csv(f, sep="\t", on_bad_lines="skip")
            cols = [c for c in ["source_term", "target_term", "confidence", "frequency"] if c in df.columns]
            df = df[cols]
            df.to_parquet(parquet_file, engine="pyarrow", compression="snappy", index=False)
            subsets_stats[pair_key] = len(df)

            if idx % 50 == 0 or idx == len(tsv_files):
                logger.info(f"[{idx}/{len(tsv_files)}] Converted {pair_key}: {len(df):,} entries")
        except Exception as e:
            logger.warning(f"Error processing {f}: {e}")

    logger.info(f"All {len(subsets_stats)} directional subsets converted to Parquet.")

    # Write README.md dataset card
    readme_path = STAGING_DIR / "README.md"
    card_content = generate_loanwords_dataset_card(subsets_stats)
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(card_content)

    # Upload to Hugging Face
    logger.info(f"Uploading afri-loanwords to https://huggingface.co/datasets/{REPO_ID}...")
    api.upload_folder(
        folder_path=str(STAGING_DIR),
        repo_id=REPO_ID,
        repo_type="dataset",
        commit_message=f"feat: upload Afri-Loanwords dataset with all {len(subsets_stats)} directional subsets",
    )
    logger.info(f"Successfully published: https://huggingface.co/datasets/{REPO_ID}")

    # Add to collection
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

    # Remove legacy afri-adapt repo if it exists
    try:
        logger.info("Cleaning up deprecated AfriSpeech/afri-adapt...")
        api.delete_repo(repo_id="AfriSpeech/afri-adapt", repo_type="dataset")
        logger.info("Deleted deprecated repo AfriSpeech/afri-adapt")
    except Exception as e:
        logger.info(f"AfriSpeech/afri-adapt cleanup notice: {e}")


def main():
    parser = argparse.ArgumentParser(description="Publish Afri-Loanwords dataset with directional subsets")
    parser.add_argument("--token", help="Hugging Face API token (optional)")
    args = parser.parse_args()

    publish_afri_loanwords(api_token=args.token)


if __name__ == "__main__":
    main()
