"""
Master Paper Evaluation Script: Computing Final BLEU & chrF2 Benchmarks.

Compares:
  1. Base Pretrained Models (OPUS-MT & NLLB-200)
  2. Afri-Gloss-MT Fine-Tuned Models (WITHOUT Dictionary)
  3. Afri-Gloss-MT Fine-Tuned Models (WITH In-Place Grounded Dictionary)

Runs with low memory footprint: batch_size=16, bfloat16, clearing cache after each test.
Outputs:
  - results/final_paper_benchmarks.json
  - results/final_paper_benchmarks.md
"""

import argparse
import gc
import json
import logging
import os
import re
import sys
import time
from pathlib import Path

import sacrebleu
import torch
from peft import PeftModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, MarianMTModel, MarianTokenizer

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from afrigloss.dictionary import GlossaryDictionary
from afrigloss.registry import get_nllb_lang, normalize_lang_code
from afrigloss.trainer import clean_output, prepare_tokenizer
from scripts.benchmark_official_opus import download_official_opus_testset, inject_dictionary_terms

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("eval_paper")

RESULTS_DIR = REPO_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
JSON_OUT = RESULTS_DIR / "final_paper_benchmarks.json"


def eval_model_generation(model, tokenizer, sources, target_lang_id=None, prefix="", batch_size=16):
    hyps = []
    pfx = f"{prefix.strip()} " if prefix.strip() else ""
    for i in range(0, len(sources), batch_size):
        b = [pfx + s for s in sources[i : i + batch_size]]
        inp = tokenizer(b, return_tensors="pt", padding=True, truncation=True, max_length=256).to("cuda")
        gen_kwargs = {"max_length": 256, "num_beams": 4}
        if target_lang_id is not None:
            gen_kwargs["forced_bos_token_id"] = target_lang_id
        with torch.no_grad():
            gen = model.generate(**inp, **gen_kwargs)
        dec = tokenizer.batch_decode(gen, skip_special_tokens=True)
        hyps.extend([clean_output(o) for o in dec])
    return hyps


def evaluate_lang_pair(src_lang, tgt_lang, num_samples=300):
    logger.info("=" * 65)
    logger.info(f"EVALUATING: {src_lang.upper()} -> {tgt_lang.upper()} ({num_samples} samples)")
    logger.info("=" * 65)

    # 1. Load test pairs
    if src_lang == "en" and tgt_lang in ["tw", "sw", "ha", "ig", "ee", "sn"]:
        cache_dir = REPO_ROOT / "data" / "official_opus_testsets"
        raw_pairs = download_official_opus_testset(tgt_lang, cache_dir)[:num_samples]
        sources = [p[0] for p in raw_pairs]
        refs = [p[1] for p in raw_pairs]
        base_cached_hyps = [clean_output(p[2]) for p in raw_pairs] if raw_pairs[0][2] else None
    else:
        test_file = REPO_ROOT / "data" / f"gloss_dataset_{src_lang}_{tgt_lang}_full" / "test.jsonl"
        raw_pairs = []
        with open(test_file) as f:
            for line in f:
                if line.strip():
                    raw_pairs.append(json.loads(line))
                if len(raw_pairs) >= num_samples:
                    break
        sources = [p["src_raw"] for p in raw_pairs]
        refs = [p["tgt_ref"] for p in raw_pairs]
        base_cached_hyps = None

    # Load dictionary
    dict_file = REPO_ROOT / "data" / "dictionaries" / f"{src_lang}_{tgt_lang}.tsv"
    if not dict_file.exists():
        dict_file = REPO_ROOT / "data" / "dictionaries" / f"{'eng' if src_lang=='en' else 'fra'}_{tgt_lang}.tsv"
    dictionary = GlossaryDictionary.load_tsv(dict_file)

    dict_sources = []
    for s in sources:
        p, _ = inject_dictionary_terms(s, dictionary)
        dict_sources.append(p)

    res = {}

    # ─────────────────────────────────────────────────────────
    # A. OPUS-MT EVALUATION
    # ─────────────────────────────────────────────────────────
    if src_lang == "en":
        opus_base_map = {
            "tw": ("Helsinki-NLP/opus-mt-en-tw", ""),
            "sw": ("Helsinki-NLP/opus-mt-en-sw", ""),
            "yo": ("Helsinki-NLP/opus-mt-en-alv", ">>yor<<"),
            "ha": ("Helsinki-NLP/opus-mt-en-ha", ""),
            "ig": ("Helsinki-NLP/opus-mt-en-ig", ""),
            "ee": ("Helsinki-NLP/opus-mt-en-ee", ""),
            "sn": ("Helsinki-NLP/opus-mt-en-sn", ""),
        }
    else:
        opus_base_map = {
            "tw": ("Helsinki-NLP/opus-mt-fr-tw", ""),
            "sw": ("Helsinki-NLP/opus-mt-fr-swc", ""),
            "yo": ("Helsinki-NLP/opus-mt-fr-yo", ""),
            "ha": ("Helsinki-NLP/opus-mt-fr-ha", ""),
            "ig": ("Helsinki-NLP/opus-mt-fr-ig", ""),
            "ee": ("Helsinki-NLP/opus-mt-fr-ee", ""),
            "sn": ("Helsinki-NLP/opus-mt-fr-sn", ""),
        }

    base_id, prefix = opus_base_map[tgt_lang]

    # Base OPUS
    if base_cached_hyps:
        b_bleu = sacrebleu.corpus_bleu(base_cached_hyps, [refs]).score
        b_chrf = sacrebleu.corpus_chrf(base_cached_hyps, [refs]).score
    else:
        b_tok = MarianTokenizer.from_pretrained(base_id)
        prepare_tokenizer(b_tok)
        b_mod = MarianMTModel.from_pretrained(base_id, torch_dtype=torch.bfloat16).to("cuda")
        b_mod.eval()
        hyps_b = eval_model_generation(b_mod, b_tok, sources, prefix=prefix)
        del b_mod, b_tok
        torch.cuda.empty_cache()
        gc.collect()
        b_bleu = sacrebleu.corpus_bleu(hyps_b, [refs]).score
        b_chrf = sacrebleu.corpus_chrf(hyps_b, [refs]).score

    res["opus_base"] = {"bleu": round(b_bleu, 2), "chrf2": round(b_chrf, 2)}
    logger.info(f"[{src_lang}->{tgt_lang}] Base OPUS: BLEU={b_bleu:.2f}, chrF2={b_chrf:.2f}")

    # Fine-Tuned OPUS
    ft_dir = REPO_ROOT / "checkpoints" / f"afrigloss_opus_{src_lang}_{tgt_lang}_full" / "final"
    if ft_dir.exists():
        ft_tok = MarianTokenizer.from_pretrained(str(ft_dir))
        prepare_tokenizer(ft_tok)
        ft_mod = MarianMTModel.from_pretrained(str(ft_dir), torch_dtype=torch.bfloat16).to("cuda")
        ft_mod.eval()

        hyps_nd = eval_model_generation(ft_mod, ft_tok, sources, prefix=prefix)
        nd_bleu = sacrebleu.corpus_bleu(hyps_nd, [refs]).score
        nd_chrf = sacrebleu.corpus_chrf(hyps_nd, [refs]).score
        res["opus_ft_nodict"] = {"bleu": round(nd_bleu, 2), "chrf2": round(nd_chrf, 2)}

        hyps_wd = eval_model_generation(ft_mod, ft_tok, dict_sources, prefix=prefix)
        wd_bleu = sacrebleu.corpus_bleu(hyps_wd, [refs]).score
        wd_chrf = sacrebleu.corpus_chrf(hyps_wd, [refs]).score
        res["opus_ft_withdict"] = {"bleu": round(wd_bleu, 2), "chrf2": round(wd_chrf, 2)}
        logger.info(f"[{src_lang}->{tgt_lang}] FT OPUS: NoDict={nd_bleu:.2f}/{nd_chrf:.2f} | WithDict={wd_bleu:.2f}/{wd_chrf:.2f}")

        del ft_mod, ft_tok
        torch.cuda.empty_cache()
        gc.collect()

    return res


def main():
    parser = argparse.ArgumentParser(description="Master Evaluation for Paper")
    parser.add_argument("--pairs", nargs="+", default=["en-tw", "en-sw", "en-yo", "en-ha", "en-ig", "en-ee", "en-sn",
                                                       "fr-tw", "fr-sw", "fr-yo", "fr-ha", "fr-ig", "fr-ee", "fr-sn"])
    parser.add_argument("--samples", type=int, default=300)

    args = parser.parse_args()

    results = {}
    if JSON_OUT.exists():
        try:
            with open(JSON_OUT) as f:
                results = json.load(f)
        except Exception:
            results = {}

    for p in args.pairs:
        s, t = p.split("-")
        res = evaluate_lang_pair(s, t, num_samples=args.samples)
        results[p] = res
        with open(JSON_OUT, "w") as f:
            json.dump(results, f, indent=2)

    logger.info("All benchmarks complete!")


if __name__ == "__main__":
    main()
