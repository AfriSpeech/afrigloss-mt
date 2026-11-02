"""
Evaluate NLLB-600M and NLLB-1.3B (Base vs. Fine-Tuned) Across All 7 Core African Languages.

Evaluates on the exact same test sentences used for OPUS-MT:
  1. Base Un-finetuned NLLB (600M and 1.3B)
  2. Fine-Tuned Multilingual NLLB (No Dict)
  3. Fine-Tuned Multilingual NLLB (WITH Grounded Dictionary)

Languages: tw, sw, yo, ha, ig, ee, sn.
"""

import gc
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import sacrebleu
import torch
from peft import PeftModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from afrigloss.dictionary import GlossaryDictionary
from afrigloss.registry import get_nllb_lang, normalize_lang_code
from afrigloss.trainer import clean_output
from scripts.benchmark_official_opus import download_official_opus_testset, inject_dictionary_terms

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("eval_nllb_master")

RESULTS_DIR = REPO_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
OUT_JSON = RESULTS_DIR / "nllb_benchmarks.json"

LANGS = ["tw", "sw", "yo", "ha", "ig", "ee", "sn"]


def load_test_pairs(lang: str, src_lang: str = "en", num_samples: int = 250) -> Tuple[List[str], List[str]]:
    """Load test sources and references."""
    cache_dir = REPO_ROOT / "data" / "official_opus_testsets"

    if src_lang == "en" and lang in ["tw", "sw", "ha", "ig", "ee", "sn"]:
        pairs = download_official_opus_testset(lang, cache_dir)[:num_samples]
        sources = [p[0] for p in pairs]
        refs = [p[1] for p in pairs]
        return sources, refs

    # For French or Yoruba, load held-out test split from dataset
    test_jsonl = REPO_ROOT / "data" / f"gloss_dataset_{src_lang}_{lang}_full" / "test.jsonl"
    sources = []
    refs = []
    with open(test_jsonl) as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                sources.append(item["src_raw"])
                refs.append(item["tgt_ref"])
            if len(sources) >= num_samples:
                break
    return sources, refs


def load_dict_for_lang(lang: str, src_lang: str = "en") -> GlossaryDictionary:
    """Load grounded dictionary."""
    candidates = [
        REPO_ROOT / "data" / "dictionaries" / f"{src_lang}_{lang}.tsv",
        REPO_ROOT / "data" / "dictionaries" / f"{'eng' if src_lang=='en' else 'fra'}_{normalize_lang_code(lang)}.tsv",
    ]
    for c in candidates:
        if c.exists():
            return GlossaryDictionary.load_tsv(c)
    return GlossaryDictionary()


def generate_nllb_translations(model, tokenizer, sources, tgt_token_id, batch_size=16):
    """Generate translations in batches using NLLB."""
    hyps = []
    for i in range(0, len(sources), batch_size):
        batch = sources[i : i + batch_size]
        inp = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=256).to("cuda")
        with torch.no_grad():
            gen = model.generate(
                **inp,
                forced_bos_token_id=tgt_token_id,
                max_length=256,
                num_beams=4,
            )
        dec = tokenizer.batch_decode(gen, skip_special_tokens=True)
        hyps.extend([clean_output(o) for o in dec])
    return hyps


def eval_nllb_size(size: str, base_id: str, adapter_dir: str, src_lang: str = "en", num_samples: int = 250, batch_size: int = 16):
    logger.info("=" * 65)
    logger.info(f"STARTING EVALUATION FOR NLLB-{size.upper()} ({base_id}) [Source: {src_lang.upper()}]")
    logger.info("=" * 65)

    results = {}
    src_nllb = "fra_Latn" if src_lang == "fr" else "eng_Latn"
    tokenizer = AutoTokenizer.from_pretrained(base_id, src_lang=src_nllb)

    # 1. Evaluate Base Model across all languages
    logger.info(f"Loading Base {base_id} into GPU...")
    base_model = AutoModelForSeq2SeqLM.from_pretrained(base_id, torch_dtype=torch.bfloat16).to("cuda")
    base_model.eval()

    base_results = {}
    test_data_by_lang = {}

    for l in LANGS:
        sources, refs = load_test_pairs(l, src_lang=src_lang, num_samples=num_samples)
        d = load_dict_for_lang(l, src_lang=src_lang)
        dict_sources = []
        for s in sources:
            p, _ = inject_dictionary_terms(s, d)
            dict_sources.append(p)
        test_data_by_lang[l] = (sources, refs, dict_sources)

        nllb_tgt = get_nllb_lang(l)
        tgt_id = tokenizer.convert_tokens_to_ids(nllb_tgt)

        hyps = generate_nllb_translations(base_model, tokenizer, sources, tgt_id, batch_size=batch_size)
        bleu = sacrebleu.corpus_bleu(hyps, [refs]).score
        chrf = sacrebleu.corpus_chrf(hyps, [refs]).score
        base_results[l] = {"bleu": round(bleu, 2), "chrf2": round(chrf, 2)}
        logger.info(f"[Base NLLB-{size} -> {l.upper()}] BLEU={bleu:.2f}, chrF2={chrf:.2f}")

    # 2. Attach LoRA fine-tuned adapter
    logger.info(f"Attaching Fine-Tuned LoRA Adapter from {adapter_dir}...")
    base_model.resize_token_embeddings(256205)
    ft_model = PeftModel.from_pretrained(base_model, adapter_dir)
    ft_model.eval()

    ft_nodict_results = {}
    ft_withdict_results = {}

    for l in LANGS:
        sources, refs, dict_sources = test_data_by_lang[l]
        nllb_tgt = get_nllb_lang(l)
        tgt_id = tokenizer.convert_tokens_to_ids(nllb_tgt)

        # No Dict
        hyps_nd = generate_nllb_translations(ft_model, tokenizer, sources, tgt_id, batch_size=batch_size)
        bleu_nd = sacrebleu.corpus_bleu(hyps_nd, [refs]).score
        chrf_nd = sacrebleu.corpus_chrf(hyps_nd, [refs]).score
        ft_nodict_results[l] = {"bleu": round(bleu_nd, 2), "chrf2": round(chrf_nd, 2)}

        # WITH Dict
        hyps_wd = generate_nllb_translations(ft_model, tokenizer, dict_sources, tgt_id, batch_size=batch_size)
        bleu_wd = sacrebleu.corpus_bleu(hyps_wd, [refs]).score
        chrf_wd = sacrebleu.corpus_chrf(hyps_wd, [refs]).score
        ft_withdict_results[l] = {"bleu": round(bleu_wd, 2), "chrf2": round(chrf_wd, 2)}

        logger.info(f"[FT NLLB-{size} -> {l.upper()}] NoDict={bleu_nd:.2f}/{chrf_nd:.2f} | WithDict={bleu_wd:.2f}/{chrf_wd:.2f}")

    del ft_model, base_model, tokenizer
    torch.cuda.empty_cache()
    gc.collect()

    for l in LANGS:
        results[l] = {
            "base": base_results[l],
            "ft_nodict": ft_nodict_results[l],
            "ft_withdict": ft_withdict_results[l],
        }

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate NLLB across African target languages")
    parser.add_argument("--src-lang", default="en", choices=["en", "fr"], help="Source language (en or fr)")
    parser.add_argument("--samples", type=int, default=250, help="Test samples count per language")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    args = parser.parse_args()

    all_nllb_results = {}
    t0 = time.time()
    out_file = RESULTS_DIR / f"nllb_{'french' if args.src_lang=='fr' else 'english'}_benchmarks.json"

    # 1. Evaluate NLLB-600M
    res_600m = eval_nllb_size(
        size="600m",
        base_id="facebook/nllb-200-distilled-600M",
        adapter_dir=str(REPO_ROOT / "checkpoints/afrigloss_nllb_600m_multilingual/final"),
        src_lang=args.src_lang,
        num_samples=args.samples,
        batch_size=args.batch_size,
    )
    all_nllb_results["600m"] = res_600m

    # 2. Evaluate NLLB-1.3B
    res_1_3b = eval_nllb_size(
        size="1.3b",
        base_id="facebook/nllb-200-1.3B",
        adapter_dir=str(REPO_ROOT / "checkpoints/afrigloss_nllb_1.3b_multilingual/final"),
        src_lang=args.src_lang,
        num_samples=args.samples,
        batch_size=args.batch_size,
    )
    all_nllb_results["1.3b"] = res_1_3b

    elapsed = time.time() - t0
    logger.info(f"All NLLB evaluations completed in {elapsed:.1f}s ({elapsed/60:.2f} mins)!")

    with open(out_file, "w") as f:
        json.dump(all_nllb_results, f, indent=2)

    # Print summary table
    print("\n" + "=" * 90)
    print(f"NLLB MULTILINGUAL BENCHMARK RESULTS ({args.src_lang.upper()} SOURCE) (BLEU / chrF2):")
    print("=" * 90)
    print(f"{'Language':12s} | {'NLLB-600M Base':18s} | {'FT-600M (No Dict)':18s} | {'NLLB-1.3B Base':18s} | {'FT-1.3B (No Dict)':18s}")
    print("-" * 90)
    for l in LANGS:
        b6 = res_600m[l]["base"]
        f6 = res_600m[l]["ft_nodict"]
        b1 = res_1_3b[l]["base"]
        f1 = res_1_3b[l]["ft_nodict"]
        print(f"{l.upper():12s} | {b6['bleu']:5.2f} / {b6['chrf2']:5.2f}     | {f6['bleu']:5.2f} / {f6['chrf2']:5.2f}     | {b1['bleu']:5.2f} / {b1['chrf2']:5.2f}     | {f1['bleu']:5.2f} / {f1['chrf2']:5.2f}")
    print("=" * 90)


if __name__ == "__main__":
    main()
