"""
Script to pre-run word alignments on African languages and generate
high-confidence dictionaries ready for Gloss-MT training and inference.

Can run locally or on the H200 GPU instance.
"""

import argparse
import sys
import time
from pathlib import Path
from typing import List, Optional

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from afrigloss.aligner import StatisticalWordAligner
from afrigloss.corpus import AfricaCorpusLoader
from afrigloss.dictionary import build_dictionary_from_aligner
from afrigloss.registry import LANGUAGE_INFO, normalize_lang_code

# Priority African languages to pre-align
PRIORITY_LANGUAGES = [
    "twi",  # Twi (Ghana)
    "yor",  # Yoruba (Nigeria)
    "ibo",  # Igbo (Nigeria)
    "hau",  # Hausa (Nigeria/Niger/Ghana)
    "swh",  # Swahili (East Africa)
    "ewe",  # Ewe (Ghana/Togo)
    "sna",  # Shona (Zimbabwe)
    "nya",  # Chichewa (Malawi/Zambia)
    "lug",  # Luganda (Uganda)
    "kin",  # Kinyarwanda (Rwanda)
    "run",  # Kirundi (Burundi)
    "lin",  # Lingala (DRC/Congo)
    "xho",  # Xhosa (South Africa)
    "zul",  # Zulu (South Africa)
    "som",  # Somali (Somalia/Kenya/Ethiopia)
    "amh",  # Amharic (Ethiopia)
    "afr",  # Afrikaans (South Africa)
]


def process_language_pair(
    african_lang: str,
    ref_lang: str = "eng",
    min_confidence: float = 0.70,
    min_freq: int = 3,
    max_samples: Optional[int] = None,
    output_dir: Path = Path("data/dictionaries"),
):
    """Run word alignment on parallel corpus and extract high-confidence dictionary."""
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / f"{ref_lang}_{african_lang}.tsv"

    print(f"\n[{african_lang.upper()} <-> {ref_lang.upper()}] Starting processing...")
    t0 = time.time()

    loader = AfricaCorpusLoader()
    try:
        pairs = loader.get_parallel_pairs(african_lang, ref_lang, max_samples=max_samples)
    except Exception as e:

        print(f"[{african_lang}] Error retrieving corpus: {e}")
        return

    if not pairs:
        print(f"[{african_lang}] No parallel pairs found.")
        return

    print(f"[{african_lang}] Loaded {len(pairs):,} parallel verses.")

    # Bitext: English -> African Language (src is african, tgt is ref_lang)
    bitext = [(p.tgt, p.src) for p in pairs]

    aligner = StatisticalWordAligner(min_pair_freq=min_freq, min_confidence=min_confidence)
    aligner.train(bitext)

    lexicon = aligner.extract_bilingual_lexicon(min_confidence=min_confidence)
    dictionary = build_dictionary_from_aligner(lexicon, min_confidence=min_confidence)

    dictionary.save_tsv(out_file)
    elapsed = time.time() - t0
    print(f"[{african_lang}] Extracted {len(dictionary):,} high-confidence entries in {elapsed:.1f}s -> {out_file}")


def main():
    parser = argparse.ArgumentParser(description="Pre-run alignments on African languages")
    parser.add_argument(
        "--langs",
        nargs="+",
        default=PRIORITY_LANGUAGES,
        help="List of ISO 639-3 language codes to process",
    )
    parser.add_argument("--ref-lang", default="eng", help="Reference language (default: eng)")
    parser.add_argument("--min-confidence", type=float, default=0.70, help="Min alignment confidence")
    parser.add_argument("--min-freq", type=int, default=3, help="Min co-occurrence frequency")
    parser.add_argument("--max-samples", type=int, default=10000, help="Target parallel verses (default: 10000 or max available)")
    parser.add_argument("--out-dir", default=str(REPO_ROOT / "data" / "dictionaries"), help="Output directory")

    args = parser.parse_args()
    out_dir = Path(args.out_dir)

    print(f"Processing {len(args.langs)} languages: {args.langs}")
    for lang in args.langs:
        process_language_pair(
            african_lang=lang,
            ref_lang=args.ref_lang,
            min_confidence=args.min_confidence,
            min_freq=args.min_freq,
            max_samples=args.max_samples,
            output_dir=out_dir,
        )


if __name__ == "__main__":
    main()
