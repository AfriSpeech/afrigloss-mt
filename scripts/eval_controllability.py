"""
Controllability & Terminology Adherence Benchmark for Afri-Gloss-MT.

Measures how accurately a translation model follows the big dictionary:
  1. Translates WITH dictionary injection:
     Measures 'adherence' (share of injected target terms that appear in the output).
  2. Translates WITHOUT dictionary injection (zero-shot base text):
     Measures 'natural_rate' (how often those terms appear purely on their own).
  3. Computes Control Gain = adherence - natural_rate.
  4. Computes BLEU and chrF2 gains achieved by dictionary guidance against plain gold references.

Usage:
  python scripts/eval_controllability.py \
      --model checkpoints/afrigloss_opus_en_tw_100k/final \
      --test-file data/gloss_dataset_en_tw_100k/test.jsonl \
      --samples 200
"""

import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import sacrebleu
import torch
from transformers import AutoTokenizer, MarianMTModel

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from afrigloss.trainer import clean_output, prepare_tokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("eval_controllability")


def term_appears(term: str, text: str) -> bool:
    """Check if term appears as a whole word/phrase in text (case-insensitive)."""
    t_clean = term.strip()
    if not t_clean:
        return False
    pattern = re.compile(rf"\b{re.escape(t_clean)}\b", re.IGNORECASE)
    return bool(pattern.search(text))


def run_controllability_eval(
    model_path: str,
    test_jsonl: Path,
    num_samples: int = 200,
    batch_size: int = 16,
    max_len: int = 256,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    target_prefix: str = "",
) -> Dict[str, float]:
    """Execute the 3-condition controllability evaluation."""
    logger.info(f"Loading model and tokenizer from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    prepare_tokenizer(tokenizer)

    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    model = MarianMTModel.from_pretrained(model_path, torch_dtype=dtype).to(device)
    model.eval()

    logger.info(f"Loading test samples from {test_jsonl}...")
    samples = []
    with open(test_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                if item.get("inserted_terms") and len(item["inserted_terms"]) >= 1:
                    samples.append(item)
                if len(samples) >= num_samples:
                    break

    logger.info(f"Selected {len(samples)} samples with verified injected terms for evaluation.")
    prefix = f"{target_prefix.strip()} " if target_prefix else ""

    # 1. Translate WITH dictionary (pretranslated inputs)
    prompts_with_dict = [prefix + s["input"] for s in samples]
    outs_with_dict = []

    t0 = time.time()
    for i in range(0, len(prompts_with_dict), batch_size):
        batch = prompts_with_dict[i : i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_len).to(device)
        with torch.no_grad():
            gen = model.generate(**inputs, max_length=max_len, num_beams=4)
        dec = tokenizer.batch_decode(gen, skip_special_tokens=True)
        outs_with_dict.extend([clean_output(o) for o in dec])

    # 2. Translate WITHOUT dictionary (raw source inputs)
    prompts_no_dict = [prefix + s["src_raw"] for s in samples]
    outs_no_dict = []

    for i in range(0, len(prompts_no_dict), batch_size):
        batch = prompts_no_dict[i : i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_len).to(device)
        with torch.no_grad():
            gen = model.generate(**inputs, max_length=max_len, num_beams=4)
        dec = tokenizer.batch_decode(gen, skip_special_tokens=True)
        outs_no_dict.extend([clean_output(o) for o in dec])

    references = [s["tgt_ref"] for s in samples]

    # Calculate metrics
    total_injected = 0
    appeared_with_dict = 0
    appeared_no_dict = 0

    for s, out_dict, out_none in zip(samples, outs_with_dict, outs_no_dict):
        terms = s["inserted_terms"]
        for t in terms:
            total_injected += 1
            if term_appears(t, out_dict):
                appeared_with_dict += 1
            if term_appears(t, out_none):
                appeared_no_dict += 1

    adherence_pct = (appeared_with_dict / max(1, total_injected)) * 100.0
    natural_rate_pct = (appeared_no_dict / max(1, total_injected)) * 100.0
    control_gain = adherence_pct - natural_rate_pct

    bleu_with_dict = sacrebleu.corpus_bleu(outs_with_dict, [references]).score
    bleu_no_dict = sacrebleu.corpus_bleu(outs_no_dict, [references]).score

    chrf_with_dict = sacrebleu.corpus_chrf(outs_with_dict, [references]).score
    chrf_no_dict = sacrebleu.corpus_chrf(outs_no_dict, [references]).score

    results = {
        "samples_evaluated": len(samples),
        "total_terms_tested": total_injected,
        "adherence_rate_pct": round(adherence_pct, 2),
        "natural_rate_pct": round(natural_rate_pct, 2),
        "control_gain_pct": round(control_gain, 2),
        "bleu_with_dict": round(bleu_with_dict, 2),
        "bleu_no_dict": round(bleu_no_dict, 2),
        "bleu_delta": round(bleu_with_dict - bleu_no_dict, 2),
        "chrf2_with_dict": round(chrf_with_dict, 2),
        "chrf2_no_dict": round(chrf_no_dict, 2),
        "chrf2_delta": round(chrf_with_dict - chrf_no_dict, 2),
    }

    logger.info("=" * 60)
    logger.info("CONTROLLABILITY EVALUATION RESULTS:")
    logger.info(f"  Terms Tested:           {total_injected:,}")
    logger.info(f"  Glossary Adherence:     {adherence_pct:.1f}% (with dictionary)")
    logger.info(f"  Natural Occurrence:     {natural_rate_pct:.1f}% (without dictionary)")
    logger.info(f"  CONTROL GAIN:           +{control_gain:.1f}%")
    logger.info(f"  BLEU:                   {bleu_with_dict:.2f} (with dict) vs {bleu_no_dict:.2f} (no dict) [Δ: {results['bleu_delta']:+.2f}]")
    logger.info(f"  chrF2:                  {chrf_with_dict:.2f} (with dict) vs {chrf_no_dict:.2f} (no dict) [Δ: {results['chrf2_delta']:+.2f}]")
    logger.info("=" * 60)

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate Gloss-MT model controllability & adherence")
    parser.add_argument("--model", required=True, help="Path to fine-tuned model checkpoint directory")
    parser.add_argument("--test-file", required=True, help="Path to test.jsonl")
    parser.add_argument("--samples", type=int, default=200, help="Number of samples to evaluate (default: 200)")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--target-prefix", default="", help="Optional target language prefix token (e.g. >>yor<<)")
    parser.add_argument("--out-json", help="Optional path to save JSON results")

    args = parser.parse_args()
    results = run_controllability_eval(
        model_path=args.model,
        test_jsonl=Path(args.test_file),
        num_samples=args.samples,
        batch_size=args.batch_size,
        target_prefix=args.target_prefix,
    )

    if args.out_json:
        out_p = Path(args.out_json)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Saved results to {out_p}")


if __name__ == "__main__":
    main()
