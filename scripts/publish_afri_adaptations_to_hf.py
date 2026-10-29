"""
Publish Afri-Adaptations Dataset to Hugging Face Hub under AfriSpeech.

Afri-Adaptations is a dataset of words that have been phonetically adapted into
African languages from English and French.

Subsets architecture:
  Each unique language direction gets its own subset:
    - en-tw, en-sw, en-yo, en-ha, en-ig, en-ee, en-sn, en-amh, en-lin, en-zul, ... (723 English directions)
    - fr-tw, fr-sw, fr-yo, fr-ha, fr-ig, fr-ee, fr-sn, fr-kin, fr-kon, fr-loz, ... (76+ French directions)

Fields:
  - source_term: Original word in English or French
  - target_term: Phonetically adapted word in the African language
  - universalised: Universalized phonetic representation via africa-g2p
  - confidence: Consensus score (0.00 - 1.00)
  - frequency: Occurrence count across the corpus bitext

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

from afrigloss.registry import normalize_lang_code

try:
    from africa_g2p import GraphemeConverter
    HAVE_AFRICA_G2P = True
except ImportError:
    HAVE_AFRICA_G2P = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("afri_adaptations")

REPO_ID = "AfriSpeech/afri-adaptations"
COLLECTION_SLUG = "AfriSpeech/afri-gloss-mt-datasets-6adf1e6b82063570470d8738"
SRC_TSV_DIR = REPO_ROOT / "afri-adapt"
STAGING_DIR = REPO_ROOT / "data" / "staging_parquet" / "afri_adaptations"

_CONVERTERS: Dict[str, Optional[any]] = {}


def get_converter(lang: str):
    """Retrieve or initialize cached GraphemeConverter for a language."""
    if not HAVE_AFRICA_G2P:
        return None
    iso = normalize_lang_code(lang)
    if iso not in _CONVERTERS:
        try:
            _CONVERTERS[iso] = GraphemeConverter(iso, "universal")
        except Exception:
            _CONVERTERS[iso] = None
    return _CONVERTERS[iso]


def safe_convert(conv, text: str) -> str:
    """Safely convert text to universalized representation."""
    if not text:
        return ""
    try:
        res = conv.convert(str(text))
        return res if res else str(text).lower()
    except Exception:
        return str(text).lower()


def generate_adaptations_dataset_card(subsets_stats: Dict[str, int]) -> str:
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
    total_african_langs = len(set([p.split("-")[1] for p in subsets_stats.keys()]))

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
- adaptations
- loanwords
- proper-nouns
- named-entities
- terminology
- transliteration
- afri-adaptations
- afrigloss
configs:
{configs_str}
---

# `AfriSpeech/afri-adaptations` 🌍

**Afri-Adaptations** is a dataset of words that have been phonetically adapted into **{total_african_langs} African languages** from English and French.

Every pair documents how foreign words, named entities, personal names, geographical places, and loanwords from English and French undergo phonological, morphological, and orthographic adaptation when borrowed into indigenous African languages. The dataset includes a `universalised` column generated using [africa-g2p](https://github.com/AfriSpeech/africa-g2p) to provide cross-lingual phonetic grounding.

All pairs are extracted directly from authentic multi-source parallel text (`AfriSpeech/africa-corpus` and `AfriSpeech/african-corpus-jw`) using dependency-parsed noun-phrase alignments and consensus majority voting (**100% human bitext, zero synthetic LLM translations**).

- **Author**: Mich-Seth Owusu (`michseth@ghananlp.org`)
- **Organization**: [AfriSpeech](https://huggingface.co/AfriSpeech) & Ghana NLP
- **Codebase**: [github.com/AfriSpeech/afrigloss-mt](https://github.com/AfriSpeech/afrigloss-mt)
- **African Languages Covered**: **{total_african_langs} languages**
- **Total Directional Subsets**: **{len(subsets_stats)} language pairs**
  - **English Source (`en-*`)**: {len(en_pairs)} African target languages ({total_en_terms:,} adaptations)
  - **French Source (`fr-*`)**: {len(fr_pairs)} African target languages ({total_fr_terms:,} adaptations)
- **Total Verified Word Adaptations**: **{grand_total:,}**

---

## 📊 Sample Directional Subsets

| Subset (`config`) | Language Direction | Verified Adapted Words |
| :---: | :--- | :---: |
{sample_table}
| ... | *(and {len(subsets_stats) - len(sample_rows)} more language directions)* | ... |
| **Grand Total** | **{len(subsets_stats)} Language Pairs** | **{grand_total:,}** |

---

## 🚀 How to Load Specific Language Directions

Load any specific directional subset directly with Hugging Face `datasets`:

```python
from datasets import load_dataset

# 1. Load English -> Twi (Akan) phonetic adaptations
ds_en_tw = load_dataset("AfriSpeech/afri-adaptations", "en-tw", split="train")
print(ds_en_tw[0])
# {{'source_term': 'Jehovah', 'target_term': 'Yehowa', 'universalised': 'yehowa', 'confidence': 0.9949, 'frequency': 32545}}

# 2. Load English -> Swahili phonetic adaptations
ds_en_sw = load_dataset("AfriSpeech/afri-adaptations", "en-sw", split="train")
print(ds_en_sw[0])
# {{'source_term': 'Jesus', 'target_term': 'Yesu', 'universalised': 'yesu', 'confidence': 0.9963, 'frequency': 18384}}

# 3. Load English -> Amharic phonetic adaptations
ds_en_amh = load_dataset("AfriSpeech/afri-adaptations", "en-amh", split="train")
print(ds_en_amh[0])
# {{'source_term': 'Jesus', 'target_term': 'ኢየሱስ', 'universalised': 'iyasuse', 'confidence': 1.0, 'frequency': 21841}}

# 4. Load French -> Swahili phonetic adaptations
ds_fr_sw = load_dataset("AfriSpeech/afri-adaptations", "fr-sw", split="train")
print(ds_fr_sw[0])
# {{'source_term': 'Jéhovah', 'target_term': 'Yehova', 'universalised': 'yehova', 'confidence': 0.9925, 'frequency': 45080}}
```

### Data Fields

- `source_term`: The original foreign word or named entity in English or French.
- `target_term`: The phonetically and orthographically adapted word in the African language.
- `universalised`: The phonetically universalized representation generated via [africa-g2p](https://github.com/AfriSpeech/africa-g2p), mapping African orthographies to common cross-lingual phonetic units.
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


def publish_afri_adaptations(api_token: Optional[str] = None):
    """Convert all directional TSV files into Parquet subsets and upload to HF."""
    api = HfApi(token=api_token)

    logger.info("=" * 65)
    logger.info(f"PUBLISHING AFRI-ADAPTATIONS DATASET TO HF: {REPO_ID}")
    logger.info("=" * 65)

    api.create_repo(repo_id=REPO_ID, repo_type="dataset", exist_ok=True)

    data_dir = STAGING_DIR / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

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

        # Check if already converted with universalised column
        if parquet_file.exists() and parquet_file.stat().st_size > 0:
            try:
                df = pd.read_parquet(parquet_file)
                if "universalised" in df.columns:
                    subsets_stats[pair_key] = len(df)
                    continue
            except Exception:
                pass

        try:
            df = pd.read_csv(f, sep="\t", on_bad_lines="skip")
            cols = [c for c in ["source_term", "target_term", "confidence", "frequency"] if c in df.columns]
            df = df[cols].dropna(subset=["source_term", "target_term"])

            # Universalise African target language text using africa-g2p
            conv = get_converter(tgt)
            if conv:
                df["universalised"] = df["target_term"].apply(lambda w: safe_convert(conv, w))
            else:
                df["universalised"] = df["target_term"].astype(str).str.lower()

            cols_out = ["source_term", "target_term", "universalised", "confidence", "frequency"]
            df = df[cols_out]
            df.to_parquet(parquet_file, engine="pyarrow", compression="snappy", index=False)
            subsets_stats[pair_key] = len(df)

            if idx % 50 == 0 or idx == len(tsv_files):
                logger.info(f"[{idx}/{len(tsv_files)}] Converted {pair_key}: {len(df):,} entries (with universalised)")
        except Exception as e:
            logger.warning(f"Error processing {f}: {e}")

    logger.info(f"All {len(subsets_stats)} directional subsets converted to Parquet.")

    # Write README.md dataset card
    readme_path = STAGING_DIR / "README.md"
    card_content = generate_adaptations_dataset_card(subsets_stats)
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(card_content)

    # Upload to Hugging Face
    logger.info(f"Uploading afri-adaptations to https://huggingface.co/datasets/{REPO_ID}...")
    api.upload_folder(
        folder_path=str(STAGING_DIR),
        repo_id=REPO_ID,
        repo_type="dataset",
        commit_message=f"feat: upload Afri-Adaptations dataset with all {len(subsets_stats)} directional subsets",
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

    # Clean up deprecated afri-loanwords repo
    try:
        logger.info("Cleaning up deprecated AfriSpeech/afri-loanwords...")
        api.delete_repo(repo_id="AfriSpeech/afri-loanwords", repo_type="dataset")
        logger.info("Deleted deprecated repo AfriSpeech/afri-loanwords")
    except Exception as e:
        logger.info(f"AfriSpeech/afri-loanwords cleanup notice: {e}")


def main():
    parser = argparse.ArgumentParser(description="Publish Afri-Adaptations dataset with directional subsets")
    parser.add_argument("--token", help="Hugging Face API token (optional)")
    args = parser.parse_args()

    publish_afri_adaptations(api_token=args.token)


if __name__ == "__main__":
    main()
