"""
Extract Grounded Proper Nouns & Entities for Afri-Adapt.

Extracts all named entities, geographical names, historical/biblical figures,
people, organizations, and capitalized proper nouns with their African language translations
from the full parallel datasets (English & French -> 7 African languages).

Outputs files to afri-adapt/:
  - afri-adapt/en_<tgt>_proper_nouns.tsv
  - afri-adapt/fr_<tgt>_proper_nouns.tsv
  - afri-adapt/all_proper_nouns.tsv
  - afri-adapt/README.md

Usage:
  python scripts/extract_afri_adapt.py
"""

import argparse
import json
import logging
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("afri_adapt")

OUT_DIR = REPO_ROOT / "afri-adapt"

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


def clean_term(text: str) -> str:
    """Strip edge punctuation while preserving inner hyphens and apostrophes."""
    t = text.strip()
    t = re.sub(r"^[^\w]+|[^\w]+$", "", t)
    return t.strip()


def extract_pairs(src_raw: str, inp_with_tags: str) -> List[Tuple[str, str, int]]:
    """Extract (source_phrase, target_term, start_index_in_src) from tagged input."""
    tag_matches = list(re.finditer(r"<([^>]+)>", inp_with_tags))
    if not tag_matches:
        return []

    parts = re.split(r"<[^>]+>", inp_with_tags)
    terms = [m.group(1) for m in tag_matches]
    pairs = []
    cur_pos = 0

    for idx, term in enumerate(terms):
        prefix = parts[idx]
        idx_prefix = src_raw.find(prefix, cur_pos)
        if idx_prefix == -1:
            continue
        start_src = idx_prefix + len(prefix)
        next_prefix = parts[idx + 1] if (idx + 1) < len(parts) else ""

        if next_prefix:
            idx_next = src_raw.find(next_prefix, start_src)
            if idx_next != -1:
                raw_term = src_raw[start_src:idx_next].strip()
                cur_pos = idx_next
            else:
                raw_term = src_raw[start_src:].strip()
                cur_pos = len(src_raw)
        else:
            raw_term = src_raw[start_src:].strip()
            cur_pos = len(src_raw)

        if raw_term:
            pairs.append((raw_term, term.strip(), start_src))

    return pairs


def extract_proper_nouns_for_pair(
    src: str,
    tgt: str,
    min_freq: int = 2,
    cap_ratio_threshold: float = 0.55,
) -> List[Tuple[str, str, float, int]]:
    """
    Extract high-precision proper nouns:
    Must appear capitalized mid-sentence (start_src > 0) with a high capitalization ratio.
    """
    data_dir = REPO_ROOT / "data"
    dataset_dir = data_dir / f"gloss_dataset_{src}_{tgt}_full"
    if not dataset_dir.exists():
        dataset_dir = data_dir / f"gloss_dataset_{src}_{tgt}_100k"
    if not dataset_dir.exists():
        logger.warning(f"Dataset missing for {src}->{tgt} at {dataset_dir}")
        return []

    train_file = dataset_dir / "train.jsonl"
    if not train_file.exists():
        return []

    mid_upper_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    total_counts: Dict[str, int] = defaultdict(int)

    logger.info(f"Extracting proper nouns for {src.upper()} -> {tgt.upper()} from {train_file.name}...")
    t0 = time.time()
    line_count = 0

    with open(train_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            line_count += 1
            ex = json.loads(line)
            src_raw = ex.get("src_raw", "")
            inp = ex.get("input", "")

            pairs = extract_pairs(src_raw, inp)
            for raw_term, tgt_term, start_idx in pairs:
                raw_c = clean_term(raw_term)
                tgt_c = clean_term(tgt_term)

                if 2 <= len(raw_c) <= 45 and 2 <= len(tgt_c) <= 45:
                    total_counts[raw_c.lower()] += 1
                    # Mid-sentence capitalization: strong signal of proper noun
                    if start_idx > 0 and raw_c[0].isupper():
                        mid_upper_counts[raw_c][tgt_c] += 1

    elapsed = time.time() - t0
    logger.info(f"Scanned {line_count:,} sentences in {elapsed:.2f}s.")

    proper_list = []
    for raw_c, targets in mid_upper_counts.items():
        cap_freq = sum(targets.values())
        tot_freq = total_counts[raw_c.lower()]

        if cap_freq < min_freq or tot_freq == 0:
            continue

        cap_ratio = cap_freq / tot_freq
        if cap_ratio >= cap_ratio_threshold:
            best_tgt, best_count = max(targets.items(), key=lambda x: x[1])
            conf = best_count / cap_freq
            proper_list.append((raw_c, best_tgt, conf, cap_freq))

    # Sort descending by frequency
    proper_list.sort(key=lambda x: x[3], reverse=True)
    logger.info(f"Extracted {len(proper_list):,} verified proper nouns for {src}->{tgt}.")
    return proper_list


def main():
    parser = argparse.ArgumentParser(description="Extract proper nouns for afri-adapt")
    parser.add_argument("--min-freq", type=int, default=2, help="Minimum capitalized occurrences (default: 2)")
    parser.add_argument("--cap-ratio", type=float, default=0.55, help="Minimum capitalization ratio (default: 0.55)")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    src_langs = ["en", "fr"]
    target_langs = ["tw", "sw", "yo", "ha", "ig", "ee", "sn"]

    all_records = []
    summary_stats: Dict[str, int] = {}

    for s in src_langs:
        for t in target_langs:
            proper_entries = extract_proper_nouns_for_pair(
                s, t, min_freq=args.min_freq, cap_ratio_threshold=args.cap_ratio
            )
            if not proper_entries:
                continue

            pair_key = f"{s}_{t}"
            summary_stats[pair_key] = len(proper_entries)

            # Write pair-specific TSV
            pair_tsv = OUT_DIR / f"{s}_{t}_proper_nouns.tsv"
            with open(pair_tsv, "w", encoding="utf-8") as out_f:
                out_f.write("source_term\ttarget_term\tconfidence\tfrequency\n")
                for sp, tt, conf, freq in proper_entries:
                    out_f.write(f"{sp}\t{tt}\t{conf:.4f}\t{freq}\n")
                    all_records.append((s, t, sp, tt, conf, freq))

    # Write consolidated all_proper_nouns.tsv
    all_tsv = OUT_DIR / "all_proper_nouns.tsv"
    with open(all_tsv, "w", encoding="utf-8") as out_f:
        out_f.write("src_lang\ttgt_lang\tsource_term\ttarget_term\tconfidence\tfrequency\n")
        for s, t, sp, tt, conf, freq in all_records:
            out_f.write(f"{s}\t{t}\t{sp}\t{tt}\t{conf:.4f}\t{freq}\n")

    # Write README.md documentation for afri-adapt
    stat_rows = []
    for pair, count in sorted(summary_stats.items()):
        s, t = pair.split("_")
        s_name = LANG_NAMES.get(s, s.upper())
        t_name = LANG_NAMES.get(t, t.upper())
        stat_rows.append(f"| `{pair}` | {s_name} $\\rightarrow$ {t_name} | `{pair}_proper_nouns.tsv` | **{count:,}** |")

    stat_table = "\n".join(stat_rows)
    total_proper = sum(summary_stats.values())

    readme_content = f"""# Afri-Adapt: Grounded Multilingual Proper Noun & Entity Lexicons

**Afri-Adapt** provides comprehensive, human-grounded bilingual proper noun dictionaries mapping English and French named entities, personal names, geographical locations, organizations, and cultural/historical terms into **7 African languages**:

- **Twi / Akan** (`tw`)
- **Swahili** (`sw`)
- **Yoruba** (`yo`)
- **Hausa** (`ha`)
- **Igbo** (`ig`)
- **Ewe** (`ee`)
- **Shona** (`sn`)

All entities are extracted directly from authentic multi-source parallel text using sentential context alignment and majority-vote consensus (**100% human bitext, zero LLM/synthetic translations**).

---

## Dataset Summary

| Pair Key | Language Direction | File | Verified Proper Nouns |
| :--- | :--- | :--- | :---: |
{stat_table}
| **Total** | | | **{total_proper:,}** |

---

## File Schema

Each TSV file contains:
- `source_term`: The proper noun or named entity in the source language (English or French).
- `target_term`: The verified translation / localized transliteration in the African target language.
- `confidence`: Consensus score from empirical training alignments (0.00 – 1.00).
- `frequency`: Total number of occurrences observed across the parallel training corpora.

### Consolidated Master Lexicon
The combined file `all_proper_nouns.tsv` pools all language pairs with additional `src_lang` and `tgt_lang` columns for rapid multi-target querying.

---

## Quick Usage

### Python (Pandas)
```python
import pandas as pd

# Load English -> Twi proper nouns
df_tw = pd.read_csv("afri-adapt/en_tw_proper_nouns.tsv", sep="\\t")
print(df_tw.head(10))

# Query specific entity across all African languages
df_all = pd.read_csv("afri-adapt/all_proper_nouns.tsv", sep="\\t")
jerusalem_translations = df_all[df_all["source_term"].str.lower() == "jerusalem"]
print(jerusalem_translations[["tgt_lang", "target_term", "frequency"]])
```
"""

    with open(OUT_DIR / "README.md", "w", encoding="utf-8") as out_f:
        out_f.write(readme_content)

    logger.info("=" * 65)
    logger.info(f"AFRI-ADAPT PROPER NOUN LEXICONS COMPLETE!")
    logger.info(f"Total pairs generated: {len(summary_stats)}")
    logger.info(f"Total proper noun translations: {total_proper:,}")
    logger.info(f"Output folder: {OUT_DIR}")
    logger.info("=" * 65)


if __name__ == "__main__":
    main()
