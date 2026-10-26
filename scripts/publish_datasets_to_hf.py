"""
Publish Afri-Gloss-MT Datasets to Hugging Face Hub under AfriSpeech.

Naming format:
  AfriSpeech/gloss-mt-train-<src>-<tgt>
  e.g.:
    AfriSpeech/gloss-mt-train-en-tw
    AfriSpeech/gloss-mt-train-en-sw
    AfriSpeech/gloss-mt-train-fr-tw
    AfriSpeech/gloss-mt-train-fr-sw

Each repository contains:
  - train.parquet
  - validation.parquet
  - test.parquet
  - Rich Dataset Card (README.md) enabling instant in-browser dataset viewer
    and single-line loading:
      from datasets import load_dataset
      ds = load_dataset("AfriSpeech/gloss-mt-train-en-tw")

Usage:
  python scripts/publish_datasets_to_hf.py --src en --tgt tw
  python scripts/publish_datasets_to_hf.py --all-available
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
logger = logging.getLogger("publish_hf_dataset")

DATASET_COLLECTION_SLUG = "AfriSpeech/afri-gloss-mt-datasets-6adf1e6b82063570470d8738"

LANG_NAMES = {
    "tw": "Twi / Akan",
    "sw": "Swahili",
    "yo": "Yoruba",
    "ha": "Hausa",
    "ig": "Igbo",
    "ee": "Ewe",
    "sn": "Shona",
    "en": "English",
    "fr": "French",
}


def convert_jsonl_to_parquet(jsonl_path: Path, parquet_path: Path) -> int:
    """Convert a JSONL file to snappy-compressed Parquet."""
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    records = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    df = pd.DataFrame(records)
    df.to_parquet(parquet_path, engine="pyarrow", compression="snappy", index=False)
    logger.info(f"Converted {len(df):,} rows from {jsonl_path.name} -> {parquet_path.name}")
    return len(df)


def generate_dataset_card(
    src: str,
    tgt: str,
    stats: Dict[str, int],
    repo_id: str,
    sample_records: Optional[List[Dict]] = None,
) -> str:
    """Generate YAML frontmatter and comprehensive documentation for Hugging Face."""
    src_full = LANG_NAMES.get(src, src.upper())
    tgt_full = LANG_NAMES.get(tgt, tgt.upper())

    sample_table = ""
    if sample_records:
        sample_rows = []
        for i, r in enumerate(sample_records[:2], 1):
            sample_rows.append(
                f"| {i} | `{r.get('input', '')}` | `{r.get('tgt', '')}` | `{', '.join(r.get('inserted_terms', []))}` |"
            )
        sample_table = (
            "\n### Sample Annotations\n\n"
            "| # | In-Place Source (`input`) | Target Translation (`tgt`) | Inserted Glossary Terms |\n"
            "|---|---|---|---|\n"
            + "\n".join(sample_rows)
            + "\n"
        )

    card = f"""---
language:
- {src}
- {tgt}
license: apache-2.0
task_categories:
- translation
tags:
- machine-translation
- african-nlp
- terminology-control
- gloss-mt
- afrigloss
- {src}-{tgt}
size_categories:
- 100K<n<1M
dataset_info:
  features:
  - name: input
    dtype: string
  - name: tgt
    dtype: string
  - name: src_raw
    dtype: string
  - name: tgt_ref
    dtype: string
  - name: inserted_terms
    sequence: string
  - name: src_lang
    dtype: string
  - name: tgt_lang
    dtype: string
  splits:
  - name: train
    num_examples: {stats.get('train', 0)}
  - name: validation
    num_examples: {stats.get('validation', 0)}
  - name: test
    num_examples: {stats.get('test', 0)}
---

# `gloss-mt-train-{src}-{tgt}`: {src_full} -> {tgt_full} Glossary-Constrained Translation Dataset

**`AfriSpeech/gloss-mt-train-{src}-{tgt}`** is a full-scale parallel dataset engineered specifically for **glossary-constrained machine translation** from {src_full} to {tgt_full}. Each training example contains nominal phrases replaced in-place in the source sentence with their verified target language translation enclosed in `<term>` markers.

- **Author**: Mich-Seth Owusu (`michseth@ghananlp.org`)
- **Organization**: [AfriSpeech](https://huggingface.co/AfriSpeech) & Ghana NLP
- **Codebase**: [github.com/AfriSpeech/afrigloss-mt](https://github.com/AfriSpeech/afrigloss-mt)
- **Source Language**: {src_full} (`{src}`)
- **Target Language**: {tgt_full} (`{tgt}`)

---

## Dataset Splits

| Split | Number of Examples | Format |
| :--- | :---: | :---: |
| `train` | **{stats.get('train', 0):,}** | Parquet (Snappy) |
| `validation` | **{stats.get('validation', 0):,}** | Parquet (Snappy) |
| `test` | **{stats.get('test', 0):,}** | Parquet (Snappy) |
| **Total** | **{sum(stats.values()):,}** | |

{sample_table}

---

## Quickstart

Load the dataset in Python using Hugging Face `datasets`:

```python
from datasets import load_dataset

dataset = load_dataset("{repo_id}")

# Access splits
train_data = dataset["train"]
val_data = dataset["validation"]
test_data = dataset["test"]

print("Sample 0:", train_data[0])
```

### Fields

- `input`: Source sentence with in-place target glossary annotations (e.g. `The <kannifoɔ> made a covenant...`).
- `tgt`: Full target language reference sentence.
- `src_raw`: Clean source sentence before glossary injection.
- `tgt_ref`: Clean target reference sentence.
- `inserted_terms`: List of target terms injected into `input`.
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


def publish_single_pair(src: str, tgt: str, api: HfApi) -> bool:
    """Publish a single source-target dataset as AfriSpeech/gloss-mt-train-<src>-<tgt>."""
    repo_name = f"gloss-mt-train-{src}-{tgt}"
    repo_id = f"AfriSpeech/{repo_name}"

    data_dir = REPO_ROOT / "data"
    dataset_dir = data_dir / f"gloss_dataset_{src}_{tgt}_full"
    if not dataset_dir.exists():
        dataset_dir = data_dir / f"gloss_dataset_{src}_{tgt}_100k"
    if not dataset_dir.exists():
        logger.warning(f"No dataset directory found for {src}->{tgt} at {dataset_dir}")
        return False

    train_file = dataset_dir / "train.jsonl"
    if not train_file.exists():
        logger.warning(f"Missing train.jsonl in {dataset_dir}")
        return False

    logger.info("=" * 65)
    logger.info(f"PUBLISHING DATASET TO HF: {repo_id}")
    logger.info(f"Source Directory: {dataset_dir}")
    logger.info("=" * 65)

    api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)
        stats = {}
        samples = []

        for split in ["train", "validation", "test"]:
            jsonl_file = dataset_dir / f"{split}.jsonl"
            if not jsonl_file.exists():
                continue
            parquet_file = tmp_dir / f"{split}.parquet"
            count = convert_jsonl_to_parquet(jsonl_file, parquet_file)
            stats[split] = count

            if split == "train" and count > 0:
                with open(jsonl_file, "r", encoding="utf-8") as f:
                    for _ in range(3):
                        line = f.readline()
                        if line:
                            samples.append(json.loads(line))

        # Generate dataset card
        readme_path = tmp_dir / "README.md"
        card_content = generate_dataset_card(src, tgt, stats, repo_id, sample_records=samples)
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(card_content)

        logger.info(f"Uploading files to https://huggingface.co/datasets/{repo_id}...")
        api.upload_folder(
            folder_path=str(tmp_dir),
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=f"feat: upload full {stats.get('train', 0):,} sample {src.upper()}->{tgt.upper()} glossary dataset",
        )
        logger.info(f"Successfully published: https://huggingface.co/datasets/{repo_id}")

        # Add to the dedicated datasets collection
        try:
            logger.info(f"Adding {repo_id} to collection {DATASET_COLLECTION_SLUG}...")
            api.add_collection_item(
                collection_slug=DATASET_COLLECTION_SLUG,
                item_id=repo_id,
                item_type="dataset",
                exists_ok=True,
            )
            logger.info(f"Added to collection: https://huggingface.co/collections/{DATASET_COLLECTION_SLUG}")
        except Exception as e:
            logger.warning(f"Could not add to collection: {e}")

    return True


def main():
    parser = argparse.ArgumentParser(description="Publish Afri-Gloss-MT datasets in format gloss-mt-train-<src>-<tgt>")
    parser.add_argument("--src", help="Source language (en or fr)")
    parser.add_argument("--tgt", help="Target language (tw, sw, yo, ha, ig, ee, sn)")
    parser.add_argument("--all-available", action="store_true", help="Publish all compiled full datasets")
    parser.add_argument("--token", help="Hugging Face API token (optional)")

    args = parser.parse_args()
    api = HfApi(token=args.token)

    if args.src and args.tgt:
        publish_single_pair(args.src, args.tgt, api)
    elif args.all_available:
        data_dir = REPO_ROOT / "data"
        srcs = ["en", "fr"]
        targets = ["tw", "sw", "yo", "ha", "ig", "ee", "sn"]
        published = 0
        for s in srcs:
            for t in targets:
                p_full = data_dir / f"gloss_dataset_{s}_{t}_full"
                if p_full.exists():
                    ok = publish_single_pair(s, t, api)
                    if ok:
                        published += 1
        logger.info(f"Total datasets published: {published}")
    else:
        logger.error("Please specify --src and --tgt (e.g. --src en --tgt tw) or use --all-available")


if __name__ == "__main__":
    main()
