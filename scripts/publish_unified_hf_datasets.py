"""
Publish Unified Afri-Gloss-MT Datasets to Hugging Face with Language Subsets.

Reorganizes the Gloss-MT datasets into at most TWO repositories on Hugging Face:
  1. AfriSpeech/gloss-mt-train-en (English -> all African language subsets)
  2. AfriSpeech/gloss-mt-train-fr (French -> all African language subsets)

Each repository features:
  - Standard subset/config architecture:
      from datasets import load_dataset
      ds_tw = load_dataset("AfriSpeech/gloss-mt-train-en", "tw")
      ds_sw = load_dataset("AfriSpeech/gloss-mt-train-en", "sw")
      ds_fr_tw = load_dataset("AfriSpeech/gloss-mt-train-fr", "tw")
  - Automatic Parquet conversion (Snappy) with Hugging Face Dataset Viewer enabled.
  - Comprehensive Markdown README with per-subset statistics.
  - Adds the unified repositories to AfriSpeech/afri-gloss-mt-datasets collection.
  - Optionally removes legacy one-off per-pair repositories (--clean-legacy).

Usage:
  python scripts/publish_unified_hf_datasets.py --src en
  python scripts/publish_unified_hf_datasets.py --src fr
  python scripts/publish_unified_hf_datasets.py --clean-legacy
"""

import argparse
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
logger = logging.getLogger("unified_hf_dataset")

COLLECTION_SLUG = "AfriSpeech/afri-gloss-mt-datasets-6adf1e6b82063570470d8738"

LANG_NAMES = {
    "tw": "Twi / Akan",
    "sw": "Swahili",
    "yo": "Yoruba",
    "ha": "Hausa",
    "ig": "Igbo",
    "ee": "Ewe",
    "sn": "Shona",
    "lin": "Lingala",
    "amh": "Amharic",
    "ar": "African Arabic",
    "ada": "Dangme / Ada",
    "ach": "Acholi",
    "dyu": "Dyula / Jula",
    "efi": "Efik",
    "ewo": "Ewondo",
    "fan": "Fang",
    "fat": "Fante / Akan",
    "fon": "Fon",
    "fuf": "Pular / Fula",
    "gaa": "Ga",
    "bem": "Bemba",
    "kin": "Kinyarwanda",
    "run": "Kirundi",
    "nya": "Chichewa",
    "xho": "isiXhosa",
    "zul": "isiZulu",
    "en": "English",
    "fr": "French",
}


def convert_jsonl_to_parquet(jsonl_path: Path, parquet_path: Path) -> int:
    """Convert a JSONL split to Snappy Parquet."""
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    records = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    df = pd.DataFrame(records)
    df.to_parquet(parquet_path, engine="pyarrow", compression="snappy", index=False)
    return len(df)


def generate_unified_dataset_card(
    src_prefix: str,
    subsets_stats: Dict[str, Dict[str, int]],
    repo_id: str,
) -> str:
    """Generate YAML frontmatter configs and Markdown documentation for Hugging Face."""
    src_name = "English" if src_prefix == "en" else "French"

    # YAML configs mapping
    configs_yaml = []
    for tgt_code in sorted(subsets_stats.keys()):
        configs_yaml.append(f"""- config_name: {tgt_code}
  data_files:
  - split: train
    path: data/{tgt_code}/train.parquet
  - split: validation
    path: data/{tgt_code}/validation.parquet
  - split: test
    path: data/{tgt_code}/test.parquet""")

    configs_str = "\n".join(configs_yaml)

    # Format Markdown statistics table
    stat_rows = []
    total_train = 0
    total_val = 0
    total_test = 0

    for tgt_code, s in sorted(subsets_stats.items()):
        name = LANG_NAMES.get(tgt_code, tgt_code.upper())
        tr = s.get("train", 0)
        va = s.get("validation", 0)
        te = s.get("test", 0)
        tot = tr + va + te
        total_train += tr
        total_val += va
        total_test += te
        stat_rows.append(f"| `{tgt_code}` | {name} | {tr:,} | {va:,} | {te:,} | **{tot:,}** |")

    stat_table = "\n".join(stat_rows)
    grand_total = total_train + total_val + total_test

    card = f"""---
language:
- {src_prefix}
- multilingual
license: apache-2.0
task_categories:
- translation
tags:
- machine-translation
- african-nlp
- terminology-control
- gloss-mt
- afrigloss
- {src_name.lower()}
configs:
{configs_str}
---

# `AfriSpeech/gloss-mt-train-{src_prefix}`: African Glossary-Constrained Translation Dataset ({src_name} Source)

**`{repo_id}`** provides full-scale parallel datasets engineered specifically for **glossary-constrained machine translation** from **{src_name}** into African languages.

Each training example features nominal phrases in the source sentence replaced in-place with their verified target language translation enclosed in `<term>` markers (e.g. `The <kannifoɔ> made a covenant with God...`).

- **Author**: Mich-Seth Owusu (`michseth@ghananlp.org`)
- **Organization**: [AfriSpeech](https://huggingface.co/AfriSpeech) & Ghana NLP
- **Codebase**: [github.com/AfriSpeech/afrigloss-mt](https://github.com/AfriSpeech/afrigloss-mt)
- **Source Language**: {src_name} (`{src_prefix}`)
- **Total African Subsets**: {len(subsets_stats)} languages
- **Total Parallel Sentences**: **{grand_total:,}**

---

## Supported Language Subsets

| Subset (`config`) | Target Language | Train | Validation | Test | Total Examples |
| :---: | :--- | :---: | :---: | :---: | :---: |
{stat_table}
| **Total** | | **{total_train:,}** | **{total_val:,}** | **{total_test:,}** | **{grand_total:,}** |

---

## How to Load Subsets

You can load any specific African language subset directly with Hugging Face `datasets`:

```python
from datasets import load_dataset

# 1. Load Twi (Akan) subset
dataset_tw = load_dataset("{repo_id}", "tw")
print(dataset_tw["train"][0])

# 2. Load Swahili subset
dataset_sw = load_dataset("{repo_id}", "sw")
print(dataset_sw["train"][0])

# 3. Load Amharic subset (if available)
dataset_amh = load_dataset("{repo_id}", "amh")
print(dataset_amh["train"][0])
```

### Data Fields

- `input`: The source sentence with in-place target glossary annotations (e.g. `The <kannifoɔ> made a covenant...`).
- `tgt`: The clean target sentence (e.g. `Na kannifoɔ no ne Nyankopɔn yɛɛ apam...`).
- `src_raw`: Original clean source sentence before glossary injection.
- `tgt_ref`: Original target reference sentence.
- `inserted_terms`: List of glossary terms injected into the source text.
- `src_lang`: Source language code.
- `tgt_lang`: Target language code.

---

## Citation

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


def publish_unified_dataset(src_prefix: str, api: HfApi):
    """Scan all datasets for a given source language, convert to Parquet subsets, and upload."""
    repo_name = f"gloss-mt-train-{src_prefix}"
    repo_id = f"AfriSpeech/{repo_name}"
    data_dir = REPO_ROOT / "data"

    logger.info("=" * 65)
    logger.info(f"PUBLISHING UNIFIED REPOSITORY: {repo_id}")
    logger.info("=" * 65)

    api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)

    # Discover all matching target dataset folders
    # Pattern: gloss_dataset_{src}_{tgt}_full or 100k
    candidate_dirs = sorted(list(data_dir.glob(f"gloss_dataset_{src_prefix}_*_full")))
    if not candidate_dirs:
        candidate_dirs = sorted(list(data_dir.glob(f"gloss_dataset_{src_prefix}_*_100k")))

    logger.info(f"Found {len(candidate_dirs)} candidate dataset directories for {src_prefix.upper()}.")

    subsets_stats: Dict[str, Dict[str, int]] = {}

    with tempfile.TemporaryDirectory() as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)
        upload_data_dir = tmp_dir / "data"
        upload_data_dir.mkdir(parents=True, exist_ok=True)

        for d in candidate_dirs:
            parts = d.name.split("_")
            if len(parts) < 4:
                continue
            tgt_code = parts[3]

            train_file = d / "train.jsonl"
            if not train_file.exists() or train_file.stat().st_size == 0:
                continue

            tgt_target_dir = upload_data_dir / tgt_code
            tgt_target_dir.mkdir(parents=True, exist_ok=True)

            pair_stats = {}
            for split in ["train", "validation", "test"]:
                jsonl_file = d / f"{split}.jsonl"
                if not jsonl_file.exists():
                    continue
                parquet_file = tgt_target_dir / f"{split}.parquet"
                count = convert_jsonl_to_parquet(jsonl_file, parquet_file)
                pair_stats[split] = count

            if pair_stats.get("train", 0) > 0:
                subsets_stats[tgt_code] = pair_stats
                logger.info(f"[{src_prefix}->{tgt_code}] Converted subset: {pair_stats}")

        if not subsets_stats:
            logger.warning(f"No valid subsets found for {src_prefix}!")
            return

        # Generate README.md
        readme_path = tmp_dir / "README.md"
        card_content = generate_unified_dataset_card(src_prefix, subsets_stats, repo_id)
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(card_content)

        logger.info(f"Uploading unified repository with {len(subsets_stats)} subsets to {repo_id}...")
        api.upload_folder(
            folder_path=str(tmp_dir),
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=f"feat: upload unified {src_prefix.upper()} dataset with {len(subsets_stats)} African language subsets",
        )
        logger.info(f"Successfully published: https://huggingface.co/datasets/{repo_id}")

        # Add to collection
        try:
            logger.info(f"Adding {repo_id} to collection {COLLECTION_SLUG}...")
            api.add_collection_item(
                collection_slug=COLLECTION_SLUG,
                item_id=repo_id,
                item_type="dataset",
                exists_ok=True,
            )
            logger.info(f"Added to collection: https://huggingface.co/collections/{COLLECTION_SLUG}")
        except Exception as e:
            logger.warning(f"Could not add to collection: {e}")


def clean_legacy_repositories(api: HfApi):
    """Delete legacy one-off repositories (AfriSpeech/gloss-mt-train-en-tw, etc.)."""
    logger.info("Cleaning up legacy one-off repositories on Hugging Face...")
    legacy_suffixes = ["en-tw", "en-sw", "en-yo", "en-ha", "en-ig", "en-ee", "en-sn",
                       "fr-tw", "fr-sw", "fr-yo", "fr-ha", "fr-ig", "fr-ee", "fr-sn"]
    deleted = 0
    for s in legacy_suffixes:
        repo_id = f"AfriSpeech/gloss-mt-train-{s}"
        try:
            api.delete_repo(repo_id=repo_id, repo_type="dataset")
            logger.info(f"Deleted legacy repo: {repo_id}")
            deleted += 1
        except Exception as e:
            logger.info(f"Repo {repo_id} not found or already deleted: {e}")
    logger.info(f"Legacy cleanup complete: {deleted} repositories deleted.")


def main():
    parser = argparse.ArgumentParser(description="Publish unified Gloss-MT datasets with subsets")
    parser.add_argument("--src", choices=["en", "fr", "all"], default="all", help="Source language to publish")
    parser.add_argument("--clean-legacy", action="store_true", help="Delete legacy one-off repositories")
    parser.add_argument("--token", help="Hugging Face API token (optional)")

    args = parser.parse_args()
    api = HfApi(token=args.token)

    if args.clean_legacy:
        clean_legacy_repositories(api)

    if args.src in ["en", "all"]:
        publish_unified_dataset("en", api)
    if args.src in ["fr", "all"]:
        publish_unified_dataset("fr", api)


if __name__ == "__main__":
    main()
