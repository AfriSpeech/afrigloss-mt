"""
Master African Base Dictionary Builder for Afri-Gloss-MT.

Builds comprehensive, high-density base dictionaries across all supported African languages by:
  1. Extracting human-curated linguistic wordlists from AfriSpeech/webonary-african-wordlists (168 languages).
  2. Running full statistical word alignment on all available parallel texts from AfriSpeech/africa-corpus
     and AfriSpeech/african-corpus-jw (133 languages, 50k-350k sentences per language).
  3. Merging both sources into high-quality master base dictionaries in data/dictionaries/.

Usage:
  python scripts/build_huge_base_dictionaries.py --webonary --align-all
  python scripts/build_huge_base_dictionaries.py --langs twi swh yor ibo hau ewe sna
"""

import argparse
import csv
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download, list_repo_files

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from afrigloss.aligner import StatisticalWordAligner
from afrigloss.corpus import AfricaCorpusLoader
from afrigloss.dictionary import GlossaryDictionary, merge_dictionaries
from afrigloss.registry import LANGUAGE_INFO, normalize_lang_code

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("huge_base_dict")

WEBONARY_REPO = "AfriSpeech/webonary-african-wordlists"


def extract_from_webonary_parquet(parquet_path: Path) -> Dict[str, Dict[str, str]]:
    """
    Extract bilingual entries from a Webonary parquet file.
    Returns: {ref_lang: {source_phrase: target_headword}}
    """
    table = pq.read_table(str(parquet_path), columns=["headword", "gloss", "gloss_lang", "iso"])
    data_by_ref: Dict[str, Dict[str, str]] = {}

    for row in table.to_pylist():
        headword = str(row.get("headword") or "").strip()
        gloss = str(row.get("gloss") or "").strip()
        gloss_lang = str(row.get("gloss_lang") or "").strip().lower()

        if not headword or not gloss:
            continue

        ref_code = "eng" if gloss_lang in ("en", "eng") else ("fra" if gloss_lang in ("fr", "fra") else None)
        if not ref_code:
            continue

        if ref_code not in data_by_ref:
            data_by_ref[ref_code] = {}

        # Split multi-word or comma-separated glosses: "wild cat, jackal" -> "wild cat", "jackal"
        sub_glosses = [g.strip().lower() for g in re.split(r"[,;/]", gloss) if g.strip()]
        for g in sub_glosses:
            # Clean punctuation and length
            g_clean = re.sub(r"^[^\w]+|[^\w]+$", "", g)
            if 2 <= len(g_clean) <= 40 and headword:
                if g_clean not in data_by_ref[ref_code]:
                    data_by_ref[ref_code][g_clean] = headword

    return data_by_ref


def process_webonary_all(out_dir: Path):
    """Download and extract dictionaries from all Webonary parquets."""
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Listing all files in {WEBONARY_REPO}...")
    files = [f for f in list_repo_files(WEBONARY_REPO, repo_type="dataset") if f.endswith(".parquet")]
    logger.info(f"Found {len(files)} Webonary parquet dictionaries.")

    total_extracted = 0
    languages_covered = 0

    for f in files:
        try:
            local_p = Path(hf_hub_download(WEBONARY_REPO, filename=f, repo_type="dataset"))
            table_iso = pq.read_table(local_p, columns=["iso"])
            isos = [x for x in table_iso["iso"].to_pylist() if x]
            if not isos:
                continue
            iso = normalize_lang_code(isos[0].lower())

            dict_map = extract_from_webonary_parquet(local_p)
            for ref_lang, entries in dict_map.items():
                if not entries:
                    continue

                out_file = out_dir / f"{ref_lang}_{iso}.tsv"
                existing = GlossaryDictionary.load_tsv(out_file) if out_file.exists() else GlossaryDictionary()

                new_dict = GlossaryDictionary()
                for src_term, tgt_trans in entries.items():
                    new_dict.add(src_term, tgt_trans, confidence=0.98)

                merged = merge_dictionaries(existing, new_dict)
                merged.save_tsv(out_file)
                total_extracted += len(entries)
                languages_covered += 1
                logger.info(f"[{iso}] Extracted {len(entries):,} terms ({ref_lang}) -> {out_file.name}")
        except Exception as e:
            logger.warning(f"Error processing {f}: {e}")

    logger.info(f"Webonary extraction complete! Extracted {total_extracted:,} entries across {languages_covered} language files.")


def align_and_build_dictionary(
    lang_code: str,
    ref_lang: str = "eng",
    min_confidence: float = 0.60,
    out_dir: Path = Path("data/dictionaries"),
):
    """Extract full parallel text from Africa Corpus + JW and run alignment."""
    out_dir.mkdir(parents=True, exist_ok=True)
    tgt_code = normalize_lang_code(lang_code)
    out_file = out_dir / f"{ref_lang}_{tgt_code}.tsv"

    logger.info(f"[{tgt_code}] Loading ALL available parallel sentences ({ref_lang} <-> {tgt_code})...")
    loader = AfricaCorpusLoader()
    try:
        pairs = loader.get_parallel_pairs(ref_lang, tgt_code, max_samples=None, include_all_versions=True, include_jw=True)
    except Exception as e:
        logger.warning(f"[{tgt_code}] Could not load parallel pairs: {e}")
        return

    if not pairs:
        logger.warning(f"[{tgt_code}] No parallel pairs found.")
        return

    logger.info(f"[{tgt_code}] Loaded {len(pairs):,} parallel sentences. Running full statistical aligner...")
    t0 = time.time()
    bitext = [(p.src, p.tgt) for p in pairs]
    aligner = StatisticalWordAligner(min_pair_freq=2, min_confidence=min_confidence)
    aligner.train(bitext)

    lexicon = aligner.extract_bilingual_lexicon(min_confidence=min_confidence)
    new_dict = GlossaryDictionary()
    for sw, candidates in lexicon.items():
        best_tw, score = max(candidates.items(), key=lambda x: x[1])
        new_dict.add(sw, best_tw, score)

    logger.info(f"[{tgt_code}] Aligned {len(new_dict):,} entries in {time.time()-t0:.1f}s.")

    # Merge with existing dictionary if one exists (e.g. from Webonary)
    if out_file.exists():
        existing = GlossaryDictionary.load_tsv(out_file)
        merged = merge_dictionaries(existing, new_dict)
    else:
        merged = new_dict

    merged.save_tsv(out_file)
    logger.info(f"[{tgt_code}] Saved total {len(merged):,} high-confidence dictionary entries -> {out_file}")


def main():
    parser = argparse.ArgumentParser(description="Master Base Dictionary Builder for Afri-Gloss-MT")
    parser.add_argument("--webonary", action="store_true", help="Extract all linguistic dictionaries from AfriSpeech/webonary-african-wordlists")
    parser.add_argument("--langs", nargs="+", help="Specific language codes to align (e.g. twi swh yor ibo hau ewe sna)")
    parser.add_argument("--align-all", action="store_true", help="Run full alignment on all priority languages")
    parser.add_argument("--min-confidence", type=float, default=0.60, help="Min alignment confidence (default: 0.60)")
    parser.add_argument("--out-dir", default=str(REPO_ROOT / "data" / "dictionaries"), help="Output directory")

    args = parser.parse_args()
    out_dir = Path(args.out_dir)

    if args.webonary:
        process_webonary_all(out_dir)

    target_langs = args.langs or []
    if args.align_all:
        target_langs = [
            "twi", "swh", "yor", "hau", "ibo", "ewe", "sna", "nya", "lug",
            "kin", "lin", "zul", "xho", "som", "amh", "afr", "run", "sag",
            "bem", "nso", "sot", "tsn", "tso", "ven", "ssw", "toi", "dop"
        ]

    for lang in target_langs:
        align_and_build_dictionary(
            lang_code=lang,
            min_confidence=args.min_confidence,
            out_dir=out_dir,
        )


if __name__ == "__main__":
    main()
