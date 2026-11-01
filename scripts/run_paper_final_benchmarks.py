"""
Comprehensive Paper Final Evaluation Benchmark for Afri-Gloss-MT.

Evaluates translation quality (BLEU & chrF2) across all 7 core African languages:
  1. Base Un-finetuned Models:
     - Base OPUS-MT
     - Base NLLB-600M
     - Base NLLB-1.3B
  2. Afri-Gloss-MT Fine-Tuned Models (WITHOUT Dictionary):
     - Fine-tuned OPUS-MT (Full Bitext Convergence)
     - Fine-tuned NLLB-600M (Multilingual Pan-African)
     - Fine-tuned NLLB-1.3B (Multilingual Pan-African)
  3. Afri-Gloss-MT Fine-Tuned Models (WITH Grounded Dictionary):
     - Demonstrating how translation quality further improves via offline dictionary constraints.

Resource-Conscious Execution:
  - Sequential model execution with torch.cuda.empty_cache() and gc.collect() between runs.
  - Moderate batch size (16) and bfloat16 inference to prevent GPU memory pressure.
  - Standard 500-sample test slice for statistically sound and fast evaluation.

Outputs:
  - results/paper_final_benchmarks.json
  - results/paper_final_benchmarks.md
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
from typing import Dict, List, Optional, Tuple

import sacrebleu
import torch
from peft import PeftModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, MarianMTModel, MarianTokenizer

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from afrigloss.dictionary import GlossaryDictionary
from afrigloss.extractor import extract_noun_phrases
from afrigloss.registry import get_nllb_lang, normalize_lang_code
from afrigloss.trainer import clean_output, prepare_tokenizer
from scripts.benchmark_official_opus import download_official_opus_testset, inject_dictionary_terms

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("paper_bench")

RESULTS_DIR = REPO_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CORE_LANGS = ["tw", "sw", "yo", "ha", "ig", "ee", "sn"]

OPUS_MODELS = {
    "tw": {"base": "Helsinki-NLP/opus-mt-en-tw", "prefix": ""},
    "sw": {"base": "Helsinki-NLP/opus-mt-en-sw", "prefix": ""},
    "yo": {"base": "Helsinki-NLP/opus-mt-en-alv", "prefix": ">>yor<<"},
    "ha": {"base": "Helsinki-NLP/opus-mt-en-ha", "prefix": ""},
    "ig": {"base": "Helsinki-NLP/opus-mt-en-ig", "prefix": ""},
    "ee": {"base": "Helsinki-NLP/opus-mt-en-ee", "prefix": ""},
    "sn": {"base": "Helsinki-NLP/opus-mt-en-sn", "prefix": ""},
}


def load_test_pairs(lang: str, num_samples: int = 500) -> List[Tuple[str, str, Optional[str]]]:
    """Load unseen held-out test pairs: returns [(source, reference, opus_baseline_hyp)]."""
    cache_dir = REPO_ROOT / "data" / "official_opus_testsets"

    if lang in ["tw", "sw", "ha", "ig", "ee", "sn"]:
        pairs = download_official_opus_testset(lang, cache_dir)
        return pairs[:num_samples]

    # For Yoruba (or fallback), load held-out test split from dataset
    test_jsonl = REPO_ROOT / "data" / f"gloss_dataset_en_{lang}_full" / "test.jsonl"
    if not test_jsonl.exists():
        test_jsonl = REPO_ROOT / "data" / f"gloss_dataset_en_{lang}_100k" / "test.jsonl"

    examples = []
    with open(test_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                examples.append((item["src_raw"], item["tgt_ref"], None))
            if len(examples) >= num_samples:
                break
    return examples


def load_dictionary_for_lang(lang: str) -> GlossaryDictionary:
    """Load grounded dictionary for language."""
    iso = normalize_lang_code(lang)
    candidates = [
        REPO_ROOT / "data" / "dictionaries" / f"eng_{iso}.tsv",
        REPO_ROOT / "data" / "dictionaries" / f"en_{lang}.tsv",
    ]
    for c in candidates:
        if c.exists():
            return GlossaryDictionary.load_tsv(c)
    return GlossaryDictionary()


def evaluate_marian_batch(
    model: MarianMTModel,
    tokenizer: MarianTokenizer,
    sources: List[str],
    prefix: str = "",
    batch_size: int = 16,
    device: str = "cuda",
) -> List[str]:
    """Translate batch using MarianMT model."""
    hypotheses = []
    pfx = f"{prefix.strip()} " if prefix.strip() else ""

    for i in range(0, len(sources), batch_size):
        batch = [pfx + s for s in sources[i : i + batch_size]]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=256).to(device)
        with torch.no_grad():
            gen = model.generate(**inputs, max_length=256, num_beams=4)
        dec = tokenizer.batch_decode(gen, skip_special_tokens=True)
        hypotheses.extend([clean_output(o) for o in dec])

    return hypotheses


def evaluate_nllb_batch(
    model: AutoModelForSeq2SeqLM,
    tokenizer: AutoTokenizer,
    sources: List[str],
    tgt_lang_code: str,
    batch_size: int = 16,
    device: str = "cuda",
) -> List[str]:
    """Translate batch using NLLB model."""
    hypotheses = []
    target_lang_id = tokenizer.convert_tokens_to_ids(tgt_lang_code)

    for i in range(0, len(sources), batch_size):
        batch = sources[i : i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=256).to(device)
        with torch.no_grad():
            gen = model.generate(
                **inputs,
                forced_bos_token_id=target_lang_id,
                max_length=256,
                num_beams=4,
            )
        dec = tokenizer.batch_decode(gen, skip_special_tokens=True)
        hypotheses.extend([clean_output(o) for o in dec])

    return hypotheses


def run_language_benchmark(
    lang: str,
    num_samples: int = 500,
    batch_size: int = 16,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Dict[str, any]:
    """Execute evaluation for one language across all models and conditions."""
    logger.info("=" * 70)
    logger.info(f"BENCHMARKING LANGUAGE: English -> {lang.upper()} ({num_samples} test sentences)")
    logger.info("=" * 70)

    test_pairs = load_test_pairs(lang, num_samples=num_samples)
    sources = [p[0] for p in test_pairs]
    references = [p[1] for p in test_pairs]
    logger.info(f"Loaded {len(sources)} test sentences.")

    dictionary = load_dictionary_for_lang(lang)
    logger.info(f"Loaded {len(dictionary):,} dictionary terms for {lang.upper()}.")

    # Prepare dictionary-injected source sentences
    dict_sources = []
    for s in sources:
        prompt, _ = inject_dictionary_terms(s, dictionary)
        dict_sources.append(prompt)

    results: Dict[str, Dict[str, float]] = {}

    # ─────────────────────────────────────────────────────────────
    # 1. OPUS-MT EVALUATION
    # ─────────────────────────────────────────────────────────────
    opus_info = OPUS_MODELS[lang]
    base_opus_id = opus_info["base"]
    prefix = opus_info["prefix"]

    # Condition 1a: Base OPUS-MT Baseline
    # If official OPUS has cached baseline hyp, use it; else generate
    if test_pairs[0][2] is not None:
        opus_base_hyps = [clean_output(p[2]) for p in test_pairs]
        b_bleu = sacrebleu.corpus_bleu(opus_base_hyps, [references]).score
        b_chrf = sacrebleu.corpus_chrf(opus_base_hyps, [references]).score
        results["opus_base"] = {"bleu": round(b_bleu, 2), "chrf2": round(b_chrf, 2)}
        logger.info(f"Original OPUS Base: BLEU={b_bleu:.2f}, chrF2={b_chrf:.2f}")
    else:
        logger.info(f"Evaluating Base OPUS-MT ({base_opus_id})...")
        b_tok = MarianTokenizer.from_pretrained(base_opus_id)
        prepare_tokenizer(b_tok)
        b_mod = MarianMTModel.from_pretrained(base_opus_id, torch_dtype=torch.bfloat16).to(device)
        b_mod.eval()
        hyps = evaluate_marian_batch(b_mod, b_tok, sources, prefix=prefix, batch_size=batch_size, device=device)
        del b_mod, b_tok
        torch.cuda.empty_cache()
        gc.collect()
        b_bleu = sacrebleu.corpus_bleu(hyps, [references]).score
        b_chrf = sacrebleu.corpus_chrf(hyps, [references]).score
        results["opus_base"] = {"bleu": round(b_bleu, 2), "chrf2": round(b_chrf, 2)}
        logger.info(f"Base OPUS-MT: BLEU={b_bleu:.2f}, chrF2={b_chrf:.2f}")

    # Condition 1b: Afri-Gloss-MT Fine-Tuned OPUS Model
    ft_opus_dir = REPO_ROOT / "checkpoints" / f"afrigloss_opus_en_{lang}_full" / "final"
    if ft_opus_dir.exists():
        logger.info(f"Loading Fine-Tuned OPUS model from {ft_opus_dir}...")
        ft_tok = MarianTokenizer.from_pretrained(str(ft_opus_dir))
        prepare_tokenizer(ft_tok)
        ft_mod = MarianMTModel.from_pretrained(str(ft_opus_dir), torch_dtype=torch.bfloat16).to(device)
        ft_mod.eval()

        # (No Dict)
        hyps_nodict = evaluate_marian_batch(ft_mod, ft_tok, sources, prefix=prefix, batch_size=batch_size, device=device)
        nd_bleu = sacrebleu.corpus_bleu(hyps_nodict, [references]).score
        nd_chrf = sacrebleu.corpus_chrf(hyps_nodict, [references]).score
        results["opus_ft_nodict"] = {"bleu": round(nd_bleu, 2), "chrf2": round(nd_chrf, 2)}
        logger.info(f"Afri-Gloss OPUS (No Dict): BLEU={nd_bleu:.2f}, chrF2={nd_chrf:.2f}")

        # (WITH Dict)
        hyps_withdict = evaluate_marian_batch(ft_mod, ft_tok, dict_sources, prefix=prefix, batch_size=batch_size, device=device)
        wd_bleu = sacrebleu.corpus_bleu(hyps_withdict, [references]).score
        wd_chrf = sacrebleu.corpus_chrf(hyps_withdict, [references]).score
        results["opus_ft_withdict"] = {"bleu": round(wd_bleu, 2), "chrf2": round(wd_chrf, 2)}
        logger.info(f"Afri-Gloss OPUS (WITH Dict): BLEU={wd_bleu:.2f}, chrF2={wd_chrf:.2f}")

        del ft_mod, ft_tok
        torch.cuda.empty_cache()
        gc.collect()

    # ─────────────────────────────────────────────────────────────
    # 2. NLLB-600M & NLLB-1.3B EVALUATION
    # ─────────────────────────────────────────────────────────────
    nllb_tgt = get_nllb_lang(lang)

    for nllb_size, base_id in [("600m", "facebook/nllb-200-distilled-600M"), ("1.3b", "facebook/nllb-200-1.3B")]:
        # Condition 2a: Base NLLB Baseline
        logger.info(f"Evaluating Base NLLB-{nllb_size} ({base_id})...")
        nllb_tok = AutoTokenizer.from_pretrained(base_id, src_lang="eng_Latn", tgt_lang=nllb_tgt)
        nllb_base = AutoModelForSeq2SeqLM.from_pretrained(base_id, torch_dtype=torch.bfloat16).to(device)
        nllb_base.eval()

        base_hyps = evaluate_nllb_batch(nllb_base, nllb_tok, sources, nllb_tgt, batch_size=batch_size, device=device)
        b_nllb_bleu = sacrebleu.corpus_bleu(base_hyps, [references]).score
        b_nllb_chrf = sacrebleu.corpus_chrf(base_hyps, [references]).score
        results[f"nllb_{nllb_size}_base"] = {"bleu": round(b_nllb_bleu, 2), "chrf2": round(b_nllb_chrf, 2)}
        logger.info(f"Base NLLB-{nllb_size}: BLEU={b_nllb_bleu:.2f}, chrF2={b_nllb_chrf:.2f}")

        # Condition 2b: Afri-Gloss-MT Fine-Tuned NLLB Model
        adapter_dir = REPO_ROOT / "checkpoints" / f"afrigloss_nllb_{nllb_size}_multilingual" / "final"
        if adapter_dir.exists():
            logger.info(f"Loading Fine-Tuned NLLB-{nllb_size} from {adapter_dir}...")
            # Match resized embedding dimension
            nllb_base.resize_token_embeddings(256205)
            ft_nllb = PeftModel.from_pretrained(nllb_base, str(adapter_dir))
            ft_nllb.eval()

            # (No Dict)
            ft_nodict = evaluate_nllb_batch(ft_nllb, nllb_tok, sources, nllb_tgt, batch_size=batch_size, device=device)
            fn_bleu = sacrebleu.corpus_bleu(ft_nodict, [references]).score
            fn_chrf = sacrebleu.corpus_chrf(ft_nodict, [references]).score
            results[f"nllb_{nllb_size}_ft_nodict"] = {"bleu": round(fn_bleu, 2), "chrf2": round(fn_chrf, 2)}
            logger.info(f"Afri-Gloss NLLB-{nllb_size} (No Dict): BLEU={fn_bleu:.2f}, chrF2={fn_chrf:.2f}")

            # (WITH Dict)
            ft_withdict = evaluate_nllb_batch(ft_nllb, nllb_tok, dict_sources, nllb_tgt, batch_size=batch_size, device=device)
            fw_bleu = sacrebleu.corpus_bleu(ft_withdict, [references]).score
            fw_chrf = sacrebleu.corpus_chrf(ft_withdict, [references]).score
            results[f"nllb_{nllb_size}_ft_withdict"] = {"bleu": round(fw_bleu, 2), "chrf2": round(fw_chrf, 2)}
            logger.info(f"Afri-Gloss NLLB-{nllb_size} (WITH Dict): BLEU={fw_bleu:.2f}, chrF2={fw_chrf:.2f}")

            del ft_nllb

        del nllb_base, nllb_tok
        torch.cuda.empty_cache()
        gc.collect()

    return results


def format_markdown_table(all_results: Dict[str, Dict[str, any]]) -> str:
    """Format evaluation results into a comprehensive Markdown table for paper."""
    header = (
        "| Language Pair | Family | Baseline Base | Afri-Gloss-MT (No Dict) | Afri-Gloss-MT (**WITH Dict**) | $\\Delta$ Gain (WITH Dict vs. Base) |\n"
        "| :--- | :---: | :---: | :---: | :---: | :---: |\n"
    )

    fam_map = {
        "tw": "Kwa", "sw": "Bantu", "yo": "Defoid", "ha": "Chadic",
        "ig": "Igboid", "ee": "Gbe", "sn": "Bantu",
    }
    lang_names = {
        "tw": "English $\\rightarrow$ Twi (`tw`)",
        "sw": "English $\\rightarrow$ Swahili (`sw`)",
        "yo": "English $\\rightarrow$ Yoruba (`yo`)",
        "ha": "English $\\rightarrow$ Hausa (`ha`)",
        "ig": "English $\\rightarrow$ Igbo (`ig`)",
        "ee": "English $\\rightarrow$ Ewe (`ee`)",
        "sn": "English $\\rightarrow$ Shona (`sn`)",
    }

    rows = []
    for l, name in lang_names.items():
        res = all_results.get(l, {})
        fam = fam_map.get(l, "African")

        # OPUS-MT rows
        op_base = res.get("opus_base", {})
        op_nd = res.get("opus_ft_nodict", {})
        op_wd = res.get("opus_ft_withdict", {})

        b_str = f"{op_base.get('bleu', 0):.2f} / {op_base.get('chrf2', 0):.2f}"
        nd_str = f"{op_nd.get('bleu', 0):.2f} / {op_nd.get('chrf2', 0):.2f}"
        wd_str = f"**{op_wd.get('bleu', 0):.2f} / {op_wd.get('chrf2', 0):.2f}**"

        d_bleu = op_wd.get("bleu", 0) - op_base.get("bleu", 0)
        d_chrf = op_wd.get("chrf2", 0) - op_base.get("chrf2", 0)
        d_str = f"**{d_bleu:+.2f} / {d_chrf:+.2f}**"

        rows.append(f"| {name} *(OPUS)* | {fam} | {b_str} | {nd_str} | {wd_str} | {d_str} |")

        # NLLB-1.3B rows
        n_base = res.get("nllb_1.3b_base", {})
        n_nd = res.get("nllb_1.3b_ft_nodict", {})
        n_wd = res.get("nllb_1.3b_ft_withdict", {})
        if n_base:
            nb_str = f"{n_base.get('bleu', 0):.2f} / {n_base.get('chrf2', 0):.2f}"
            nnd_str = f"{n_nd.get('bleu', 0):.2f} / {n_nd.get('chrf2', 0):.2f}"
            nwd_str = f"**{n_wd.get('bleu', 0):.2f} / {n_wd.get('chrf2', 0):.2f}**"
            nd_bleu = n_wd.get("bleu", 0) - n_base.get("bleu", 0)
            nd_chrf = n_wd.get("chrf2", 0) - n_base.get("chrf2", 0)
            nd_str = f"**{nd_bleu:+.2f} / {nd_chrf:+.2f}**"
            rows.append(f"| {name} *(NLLB-1.3B)* | {fam} | {nb_str} | {nnd_str} | {nwd_str} | {nd_str} |")

    return header + "\n".join(rows) + "\n"


def main():
    parser = argparse.ArgumentParser(description="Final paper comprehensive evaluation benchmark")
    parser.add_argument("--langs", nargs="+", default=CORE_LANGS, help="Languages to evaluate")
    parser.add_argument("--samples", type=int, default=500, help="Test samples count (default: 500)")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size (default: 16)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Device")

    args = parser.parse_args()

    all_results = {}
    t0 = time.time()

    for lang in args.langs:
        res = run_language_benchmark(lang, num_samples=args.samples, batch_size=args.batch_size, device=args.device)
        all_results[lang] = res

        # Save intermediate JSON
        with open(RESULTS_DIR / "paper_final_benchmarks.json", "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2)

    total_time = time.time() - t0
    logger.info(f"Evaluation complete across all {len(args.langs)} languages in {total_time:.1f}s ({total_time/60:.2f} mins).")

    # Format Markdown summary table
    md_table = format_markdown_table(all_results)
    with open(RESULTS_DIR / "paper_final_benchmarks.md", "w", encoding="utf-8") as f:
        f.write("# Afri-Gloss-MT Final Benchmark Results\n\n" + md_table)

    print("\n" + "=" * 80)
    print("FINAL BENCHMARK RESULTS (BLEU / chrF2):")
    print("=" * 80)
    print(md_table)


if __name__ == "__main__":
    main()
