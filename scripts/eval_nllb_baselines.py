"""
Evaluate NLLB-200 Baselines (600M, 1.3B, 3.3B) on Official Helsinki-NLP Test Sets.

Evaluates:
  - facebook/nllb-200-distilled-600M
  - facebook/nllb-200-1.3B
  - facebook/nllb-200-3.3B
On the exact same test sentences used to evaluate OPUS-MT and Afri-Gloss-MT.
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import sacrebleu
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from afrigloss.registry import get_nllb_lang, normalize_lang_code
from afrigloss.trainer import clean_output
from scripts.benchmark_official_opus import download_official_opus_testset

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("eval_nllb")

NLLB_MODELS = {
    "600m": "facebook/nllb-200-distilled-600M",
    "1.3b": "facebook/nllb-200-1.3B",
    "3.3b": "facebook/nllb-200-3.3B",
}


def evaluate_nllb_model(
    model_id: str,
    tgt_lang: str,
    test_pairs: List[Tuple[str, str, str]],
    batch_size: int = 16,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Dict[str, float]:
    """Evaluate an NLLB model on test pairs."""
    tgt_code = normalize_lang_code(tgt_lang)
    nllb_tgt = get_nllb_lang(tgt_code)
    if not nllb_tgt:
        raise ValueError(f"Could not resolve NLLB language code for {tgt_code}")

    logger.info(f"Loading {model_id} (target code: {nllb_tgt})...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, src_lang="eng_Latn", tgt_lang=nllb_tgt)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_id,
        dtype=torch.bfloat16 if device.startswith("cuda") else torch.float32,
    ).to(device)
    model.eval()

    sources = [p[0] for p in test_pairs]
    references = [p[1] for p in test_pairs]
    hypotheses = []

    target_lang_id = tokenizer.convert_tokens_to_ids(nllb_tgt)

    logger.info(f"Translating {len(sources)} test sentences with {model_id}...")
    t0 = time.time()
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

    bleu = sacrebleu.corpus_bleu(hypotheses, [references]).score
    chrf = sacrebleu.corpus_chrf(hypotheses, [references]).score
    elapsed = time.time() - t0

    logger.info(f"[{model_id} -> {tgt_lang.upper()}] BLEU: {bleu:.2f} | chrF2: {chrf:.2f} (in {elapsed:.1f}s)")
    return {"model": model_id, "bleu": round(bleu, 2), "chrf2": round(chrf, 2), "seconds": round(elapsed, 1)}


def main():
    parser = argparse.ArgumentParser(description="Evaluate NLLB-200 baselines on official test sets")
    parser.add_argument("--lang", required=True, help="Target language (tw, sw, ha, ig, ee, sn)")
    parser.add_argument("--sizes", nargs="+", default=["600m", "1.3b", "3.3b"], help="NLLB model sizes to test")
    parser.add_argument("--samples", type=int, default=500, help="Test samples count (default: 500)")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--out-dir", default="results/nllb_baselines", help="Output directory")

    args = parser.parse_args()
    cache_dir = REPO_ROOT / "data" / "official_opus_testsets"
    test_pairs = download_official_opus_testset(args.lang, cache_dir)
    if args.samples and len(test_pairs) > args.samples:
        test_pairs = test_pairs[: args.samples]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {}

    for size in args.sizes:
        if size.lower() in NLLB_MODELS:
            model_id = NLLB_MODELS[size.lower()]
            res = evaluate_nllb_model(model_id, args.lang, test_pairs, batch_size=args.batch_size)
            results[size] = res

    out_file = out_dir / f"nllb_{args.lang}_results.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved results to {out_file}")

    print("\n" + "=" * 65)
    print(f"NLLB-200 BASELINES ON OFFICIAL OPUS TEST SET: English -> {args.lang.upper()} ({len(test_pairs)} verses)")
    print("=" * 65)
    print(f"{'Model':35s} | {'BLEU':>8s} | {'chrF2':>8s}")
    print("-" * 65)
    for size, res in results.items():
        print(f"{res['model']:35s} | {res['bleu']:>8.2f} | {res['chrf2']:>8.2f}")
    print("=" * 65)


if __name__ == "__main__":
    main()
