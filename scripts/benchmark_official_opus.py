"""
Benchmark Afri-Gloss-MT Models on Official Helsinki-NLP OPUS-MT Test Sets.

Downloads the exact test sets reported by Helsinki-NLP:
  - English -> Twi (opus-2020-01-08.test.txt)
  - English -> Swahili (opus-2020-01-08.test.txt)
  - English -> Hausa
  - English -> Yoruba
  - English -> Igbo

Evaluates 3 conditions on the exact same test sentences:
  1. OPUS-MT Original Baseline (reported by Helsinki-NLP)
  2. Afri-Gloss-MT Fine-Tuned Model (WITHOUT dictionary)
  3. Afri-Gloss-MT Fine-Tuned Model (WITH big dictionary in-place constraints)

Usage:
  python scripts/benchmark_official_opus.py \
      --lang tw \
      --model checkpoints/afrigloss_opus_en_tw_100k/final \
      --dictionary data/dictionaries/eng_twi.tsv \
      --samples 500
"""

import argparse
import json
import logging
import re
import sys
import time
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import sacrebleu
import torch
from transformers import MarianMTModel, MarianTokenizer

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from afrigloss.dictionary import GlossaryDictionary
from afrigloss.extractor import extract_noun_phrases
from afrigloss.trainer import clean_output, prepare_tokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("opus_bench")

OFFICIAL_TEST_URLS = {
    "tw": "https://object.pouta.csc.fi/OPUS-MT-models/en-tw/opus-2020-01-08.test.txt",
    "twi": "https://object.pouta.csc.fi/OPUS-MT-models/en-tw/opus-2020-01-08.test.txt",
    "sw": "https://object.pouta.csc.fi/OPUS-MT-models/en-sw/opus-2020-01-08.test.txt",
    "swh": "https://object.pouta.csc.fi/OPUS-MT-models/en-sw/opus-2020-01-08.test.txt",
    "ha": "https://object.pouta.csc.fi/OPUS-MT-models/en-ha/opus-2020-01-08.test.txt",
    "hau": "https://object.pouta.csc.fi/OPUS-MT-models/en-ha/opus-2020-01-08.test.txt",
    "ig": "https://object.pouta.csc.fi/OPUS-MT-models/en-ig/opus-2020-01-08.test.txt",
    "ibo": "https://object.pouta.csc.fi/OPUS-MT-models/en-ig/opus-2020-01-08.test.txt",
    "ee": "https://object.pouta.csc.fi/OPUS-MT-models/en-ee/opus-2020-01-08.test.txt",
    "ewe": "https://object.pouta.csc.fi/OPUS-MT-models/en-ee/opus-2020-01-08.test.txt",
    "sn": "https://object.pouta.csc.fi/OPUS-MT-models/en-sn/opus-2020-01-08.test.txt",
    "sna": "https://object.pouta.csc.fi/OPUS-MT-models/en-sn/opus-2020-01-08.test.txt",
}


def download_official_opus_testset(lang_code: str, cache_dir: Path) -> List[Tuple[str, str, str]]:
    """
    Download and parse official Helsinki-NLP OPUS test file.
    Returns: List of (source_en, gold_reference, original_opus_hypothesis)
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    clean_code = lang_code.lower().strip()
    url = OFFICIAL_TEST_URLS.get(clean_code)

    if not url:
        raise ValueError(f"No official OPUS-MT test set URL registered for {clean_code}.")

    filename = url.split("/")[-1]
    local_file = cache_dir / f"{clean_code}_{filename}"

    if not local_file.exists():
        logger.info(f"Downloading official OPUS-MT test set from {url}...")
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as resp, open(local_file, "wb") as out:
            out.write(resp.read())
        logger.info(f"Saved to {local_file}")

    with open(local_file, "r", encoding="utf-8", errors="replace") as f:
        lines = f.read().splitlines()

    examples = []
    # 4 lines per test block: [0] source, [1] reference, [2] model hypothesis, [3] blank
    for i in range(0, len(lines), 4):
        chunk = lines[i : i + 4]
        if len(chunk) >= 3:
            src = chunk[0].strip()
            ref = chunk[1].strip()
            hyp = chunk[2].strip()
            if src and ref:
                examples.append((src, ref, hyp))

    logger.info(f"Parsed {len(examples):,} official OPUS-MT test pairs.")
    return examples


def inject_dictionary_terms(text: str, dictionary: GlossaryDictionary) -> Tuple[str, List[str]]:
    """Inject dictionary terms into noun phrases in place."""
    chunks = extract_noun_phrases(text)
    if not chunks:
        return text, []

    insertions = []
    for start, end, phrase in chunks:
        p_clean = phrase.lower().strip()
        trans = dictionary.lookup(p_clean)
        if trans:
            insertions.append((start, end, phrase, trans))

    if not insertions:
        return text, []

    insertions.sort(key=lambda x: x[0], reverse=True)
    prompt = text
    applied = []

    for start, end, src_phrase, tgt_trans in insertions:
        if src_phrase.isupper():
            trans_repr = tgt_trans.upper()
        elif src_phrase[0].isupper():
            trans_repr = tgt_trans[0].upper() + tgt_trans[1:]
        else:
            trans_repr = tgt_trans.lower()

        prompt = prompt[:start] + f"<{trans_repr}>" + prompt[end:]
        applied.append(trans_repr)

    return prompt, applied


def run_benchmark(
    lang: str,
    model_path: str,
    dictionary_path: Path,
    num_samples: int = 500,
    batch_size: int = 32,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    target_prefix: str = "",
) -> Dict[str, any]:
    """Execute evaluation comparing our model vs original OPUS-MT baseline."""
    cache_dir = REPO_ROOT / "data" / "official_opus_testsets"
    test_pairs = download_official_opus_testset(lang, cache_dir)

    if num_samples and len(test_pairs) > num_samples:
        test_pairs = test_pairs[:num_samples]

    logger.info(f"Loading dictionary from {dictionary_path}...")
    dictionary = GlossaryDictionary.load_tsv(dictionary_path)
    logger.info(f"Loaded {len(dictionary):,} dictionary entries.")

    logger.info(f"Loading model from {model_path} on {device}...")
    tokenizer = MarianTokenizer.from_pretrained(model_path)
    prepare_tokenizer(tokenizer)

    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    model = MarianMTModel.from_pretrained(model_path, torch_dtype=dtype).to(device)
    model.eval()

    sources = [p[0] for p in test_pairs]
    references = [p[1] for p in test_pairs]
    original_opus_hyps = [clean_output(p[2]) for p in test_pairs]

    prefix = f"{target_prefix.strip()} " if target_prefix else ""

    # Condition 1: Our Model WITHOUT dictionary
    logger.info("Translating with Our Model (WITHOUT dictionary)...")
    outs_no_dict = []
    for i in range(0, len(sources), batch_size):
        batch = [prefix + s for s in sources[i : i + batch_size]]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=256).to(device)
        with torch.no_grad():
            gen = model.generate(**inputs, max_length=256, num_beams=4)
        dec = tokenizer.batch_decode(gen, skip_special_tokens=True)
        outs_no_dict.extend([clean_output(o) for o in dec])

    # Condition 2: Our Model WITH dictionary
    logger.info("Translating with Our Model (WITH big dictionary)...")
    outs_with_dict = []
    total_injected = 0
    injected_preserved = 0

    for i in range(0, len(sources), batch_size):
        batch_src = sources[i : i + batch_size]
        batch_prompts = []
        batch_terms_list = []

        for s in batch_src:
            prompt, terms = inject_dictionary_terms(s, dictionary)
            batch_prompts.append(prefix + prompt)
            batch_terms_list.append(terms)
            total_injected += len(terms)

        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=256).to(device)
        with torch.no_grad():
            gen = model.generate(**inputs, max_length=256, num_beams=4)
        dec = tokenizer.batch_decode(gen, skip_special_tokens=True)

        for raw_out, terms in zip(dec, batch_terms_list):
            cleaned = clean_output(raw_out)
            outs_with_dict.append(cleaned)
            for t in terms:
                if re.search(rf"\b{re.escape(t)}\b", cleaned, re.IGNORECASE):
                    injected_preserved += 1

    # Compute BLEU and chrF2
    opus_bleu = sacrebleu.corpus_bleu(original_opus_hyps, [references]).score
    opus_chrf = sacrebleu.corpus_chrf(original_opus_hyps, [references]).score

    no_dict_bleu = sacrebleu.corpus_bleu(outs_no_dict, [references]).score
    no_dict_chrf = sacrebleu.corpus_chrf(outs_no_dict, [references]).score

    with_dict_bleu = sacrebleu.corpus_bleu(outs_with_dict, [references]).score
    with_dict_chrf = sacrebleu.corpus_chrf(outs_with_dict, [references]).score

    adherence_pct = (injected_preserved / max(1, total_injected)) * 100.0

    print("\n" + "=" * 78)
    print(f"OFFICIAL HELSINKI-NLP OPUS TEST BENCHMARK: English -> {lang.upper()} ({len(sources)} verses)")
    print("=" * 78)
    print(f"{'Condition':36s} | {'BLEU':>8s} | {'chrF2':>8s} | {'Adherence':>10s}")
    print("-" * 78)
    print(f"{'1. Original OPUS-MT Baseline':36s} | {opus_bleu:>8.2f} | {opus_chrf:>8.2f} | {'N/A':>10s}")
    print(f"{'2. Afri-Gloss-MT (No Dictionary)':36s} | {no_dict_bleu:>8.2f} | {no_dict_chrf:>8.2f} | {'N/A':>10s}")
    print(f"{'3. Afri-Gloss-MT (WITH Big Dictionary)':36s} | {with_dict_bleu:>8.2f} | {with_dict_chrf:>8.2f} | {adherence_pct:>9.1f}%")
    print("-" * 78)
    print(f"Delta over Original Base:            | {with_dict_bleu - opus_bleu:>+8.2f} | {with_dict_chrf - opus_chrf:>+8.2f} |")
    print("=" * 78)

    return {
        "lang": lang,
        "samples": len(sources),
        "opus_bleu": opus_bleu,
        "opus_chrf": opus_chrf,
        "no_dict_bleu": no_dict_bleu,
        "no_dict_chrf": no_dict_chrf,
        "with_dict_bleu": with_dict_bleu,
        "with_dict_chrf": with_dict_chrf,
        "adherence_pct": adherence_pct,
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark against official Helsinki-NLP OPUS test sets")
    parser.add_argument("--lang", required=True, help="Target language (tw, sw, ha, yo, ig, ee, sn)")
    parser.add_argument("--model", required=True, help="Path to model checkpoint")
    parser.add_argument("--dictionary", required=True, help="Path to dictionary TSV")
    parser.add_argument("--samples", type=int, default=500, help="Test samples to evaluate (default: 500)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--target-prefix", default="", help="Target prefix token")

    args = parser.parse_args()
    run_benchmark(
        lang=args.lang,
        model_path=args.model,
        dictionary_path=Path(args.dictionary),
        num_samples=args.samples,
        batch_size=args.batch_size,
        target_prefix=args.target_prefix,
    )


if __name__ == "__main__":
    main()
