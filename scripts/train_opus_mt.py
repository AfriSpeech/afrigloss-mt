"""
Fine-tune a Helsinki-NLP OPUS-MT (MarianMT) model on an Afri-Gloss-MT pre-translated dataset.

Optimized for Hopper NVIDIA H200 GPU with bf16 and TF32.
Saves the best checkpoint (by validation loss and BLEU) and the final model.

Usage:
  python scripts/train_opus_mt.py \
      --model Helsinki-NLP/opus-mt-en-tw \
      --data data/gloss_dataset_en_tw_10k \
      --out checkpoints/opus-mt-en-tw-gloss \
      --epochs 5 \
      --batch-size 32 \
      --lr 1e-4
"""

import argparse
import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import sacrebleu
import torch
from datasets import Dataset
from transformers import (
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    MarianMTModel,
    MarianTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from afrigloss.trainer import clean_output, prepare_tokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_opus_mt")


def load_dataset_from_jsonl(jsonl_path: Path) -> List[Dict]:
    """Load JSONL dataset examples."""
    examples = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def main():
    parser = argparse.ArgumentParser(description="Fine-tune OPUS-MT on Gloss-MT dataset")
    parser.add_argument("--model", required=True, help="Hugging Face model ID (e.g. Helsinki-NLP/opus-mt-en-tw)")
    parser.add_argument("--data", required=True, help="Path to directory containing train.jsonl and validation.jsonl")
    parser.add_argument("--out", required=True, help="Output checkpoint directory")
    parser.add_argument("--epochs", type=float, default=5, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Per-device batch size")
    parser.add_argument("--grad-accum", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--warmup-steps", type=int, default=200, help="Linear warmup steps")
    parser.add_argument("--eval-steps", type=int, default=200, help="Evaluate and checkpoint every N steps")
    parser.add_argument("--max-src-len", type=int, default=256, help="Max source token length")
    parser.add_argument("--max-tgt-len", type=int, default=256, help="Max target token length")
    parser.add_argument("--target-prefix", type=str, default="", help="Optional target language prefix token (e.g. >>yor<<)")

    args = parser.parse_args()

    # Enable Hopper H200 optimizations
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    data_dir = Path(args.data)
    train_file = data_dir / "train.jsonl"
    val_file = data_dir / "validation.jsonl"

    if not train_file.exists():
        raise FileNotFoundError(f"train.jsonl not found in {data_dir}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading tokenizer: {args.model}")
    tokenizer = MarianTokenizer.from_pretrained(args.model)
    prepare_tokenizer(tokenizer)

    logger.info(f"Loading train examples from {train_file}...")
    train_raw = load_dataset_from_jsonl(train_file)
    logger.info(f"Loaded {len(train_raw):,} training samples.")

    val_raw = []
    if val_file.exists():
        val_raw = load_dataset_from_jsonl(val_file)
        logger.info(f"Loaded {len(val_raw):,} validation samples.")

    prefix = f"{args.target_prefix.strip()} " if args.target_prefix else ""

    def encode_batch(examples):
        inputs = [prefix + e["input"] for e in examples]
        targets = [e["tgt"] for e in examples]
        model_inputs = tokenizer(inputs, max_length=args.max_src_len, truncation=True)
        labels = tokenizer(text_target=targets, max_length=args.max_tgt_len, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    logger.info("Tokenizing datasets...")
    train_enc = encode_batch(train_raw)
    train_ds = Dataset.from_dict(train_enc)

    val_ds = None
    if val_raw:
        val_enc = encode_batch(val_raw)
        val_ds = Dataset.from_dict(val_enc)

    logger.info(f"Loading base model: {args.model}")
    model = MarianMTModel.from_pretrained(args.model)
    if len(tokenizer) > model.get_input_embeddings().weight.shape[0]:
        model.resize_token_embeddings(len(tokenizer))
    model.config.use_cache = False

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        warmup_steps=args.warmup_steps,
        weight_decay=0.01,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        bf16=torch.cuda.is_available(),
        tf32=torch.cuda.is_available(),
        logging_steps=20,
        eval_strategy="steps" if val_ds else "no",
        eval_steps=args.eval_steps if val_ds else None,
        save_strategy="steps",
        save_steps=args.eval_steps,
        save_total_limit=2,
        load_best_model_at_end=True if val_ds else False,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        dataloader_num_workers=2,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model, padding=True),
        processing_class=tokenizer,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)] if val_ds else [],
    )

    t0 = time.time()
    logger.info(f"Starting training on {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}...")
    trainer.train()
    elapsed = time.time() - t0
    logger.info(f"Training completed in {elapsed:.1f}s ({elapsed/60:.2f} mins)")

    # Save final best model
    final_dir = out_dir / "final"
    logger.info(f"Saving final model to {final_dir}...")
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    # Evaluate on test set if available
    test_file = data_dir / "test.jsonl"
    if test_file.exists():
        logger.info(f"Evaluating final model on test set: {test_file}...")
        test_raw = load_dataset_from_jsonl(test_file)
        model.eval()
        model.to("cuda" if torch.cuda.is_available() else "cpu")

        hypotheses = []
        references = []

        batch_size = args.batch_size
        for i in range(0, len(test_raw), batch_size):
            batch = test_raw[i : i + batch_size]
            prompts = [prefix + e["input"] for e in batch]
            inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=args.max_src_len)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            with torch.no_grad():
                gen_tokens = model.generate(**inputs, max_length=args.max_tgt_len, num_beams=4)

            outs = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)
            for out, e in zip(outs, batch):
                hypotheses.append(clean_output(out))
                references.append(e["tgt_ref"])

        bleu = sacrebleu.corpus_bleu(hypotheses, [references])
        chrf = sacrebleu.corpus_chrf(hypotheses, [references])

        logger.info("=" * 50)
        logger.info(f"Test Set Evaluation Results ({len(test_raw)} samples):")
        logger.info(f"  BLEU:  {bleu.score:.2f}")
        logger.info(f"  chrF2: {chrf.score:.2f}")
        logger.info("=" * 50)

        # Save eval metrics
        with open(out_dir / "eval_results.json", "w") as f:
            json.dump({
                "model": args.model,
                "test_samples": len(test_raw),
                "bleu": round(bleu.score, 2),
                "chrf2": round(chrf.score, 2),
                "elapsed_seconds": round(elapsed, 1),
            }, f, indent=2)


if __name__ == "__main__":
    main()
