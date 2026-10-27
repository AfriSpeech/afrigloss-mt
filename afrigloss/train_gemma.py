"""
TranslateGemma (4B & 12B) LoRA Fine-Tuning Pipeline for Afri-Gloss-MT.

Fine-tunes Google TranslateGemma (google/translategemma-4b-it, google/translategemma-12b-it)
on the Pan-African Gloss-MT corpus with in-place target glossary constraints (<term>).

Features:
  - Formats prompts using TranslateGemma's official multimodal/text translation chat template.
  - Automatically maps NLLB/ISO language codes to TranslateGemma's 2-letter codes (en, fr, ak, sw, yo, ha, ig, ee, sn).
  - Target-token loss masking (cross-entropy is computed ONLY on the assistant's translated output).
  - LoRA parameter-efficient adaptation (r=16, alpha=32) on attention and MLP projections.
  - Periodic validation evaluation and early-stopping convergence tracking.
  - Full Hopper H200 bf16 + TF32 acceleration.

Usage:
  python afrigloss/train_gemma.py --model google/translategemma-4b-it --train-file data/gloss_dataset_multilingual_african/train.jsonl
  python afrigloss/train_gemma.py --model google/translategemma-12b-it --train-file data/gloss_dataset_multilingual_african/train.jsonl
"""

import argparse
import json
import logging
import os
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("afrigloss.train_gemma")

LORA_TARGETS = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj"
]

GEMMA_LANG_MAP = {
    "eng": "en", "fra": "fr", "en": "en", "fr": "fr",
    "eng_latn": "en", "fra_latn": "fr",
    "twi": "ak", "aka": "ak", "tw": "ak", "ak": "ak", "twi_latn": "ak", "aka_latn": "ak",
    "swh": "sw", "sw": "sw", "swh_latn": "sw",
    "yor": "yo", "yo": "yo", "yor_latn": "yo",
    "hau": "ha", "ha": "ha", "hau_latn": "ha",
    "ibo": "ig", "ig": "ig", "ibo_latn": "ig",
    "ewe": "ee", "ee": "ee", "ewe_latn": "ee",
    "sna": "sn", "sn": "sn", "sna_latn": "sn",
    "amh": "am", "am": "am", "amh_ethi": "am",
    "lin": "ln", "ln": "ln", "lin_latn": "ln",
    "kin": "rw", "rw": "rw", "kin_latn": "rw",
    "som": "so", "so": "so", "som_latn": "so",
    "tir": "ti", "ti": "ti", "tir_ethi": "ti",
    "orm": "om", "om": "om", "gaz_latn": "om",
    "zul": "zu", "zu": "zu", "zul_latn": "zu",
    "xho": "xh", "xh": "xh", "xho_latn": "xh",
    "sot": "st", "st": "st", "sot_latn": "st",
    "tsn": "tn", "tn": "tn", "tsn_latn": "tn",
    "tso": "ts", "ts": "ts", "tso_latn": "ts",
}


def normalize_to_gemma_code(code: str) -> str:
    """Normalize arbitrary ISO, NLLB, or language token to TranslateGemma code."""
    c = code.strip().lower()
    return GEMMA_LANG_MAP.get(c, c[:2])


class GemmaGlossDataset(Dataset):
    """Dataset for decoder-only causal LM translation fine-tuning."""

    def __init__(
        self,
        jsonl_path: str | Path,
        tokenizer,
        default_src: str = "en",
        default_tgt: str = "ak",
        max_len: int = 512,
    ):
        self.tokenizer = tokenizer
        self.default_src = default_src
        self.default_tgt = default_tgt
        self.max_len = max_len
        self.examples = []

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.examples.append(json.loads(line))

        logger.info(f"Loaded {len(self.examples):,} examples from {jsonl_path}")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ex = self.examples[idx]
        src_text = ex.get("input", "")
        tgt_text = ex.get("tgt", "")

        src_raw_lang = ex.get("src_lang", self.default_src)
        tgt_raw_lang = ex.get("tgt_lang", self.default_tgt)

        src_code = normalize_to_gemma_code(src_raw_lang)
        tgt_code = normalize_to_gemma_code(tgt_raw_lang)

        # Build official TranslateGemma chat prompt
        messages = [
            {
                "role": "user",
                "content": [{
                    "type": "text",
                    "source_lang_code": src_code,
                    "target_lang_code": tgt_code,
                    "text": src_text,
                }]
            }
        ]

        try:
            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            # Fallback simple template if language code undefined in tokenizer jinja
            prompt = (
                f"<start_of_turn>user\n"
                f"Translate from {src_code.upper()} to {tgt_code.upper()}:\n"
                f"{src_text}<end_of_turn>\n"
                f"<start_of_turn>model\n"
            )

        full_text = prompt + tgt_text + "<eos><end_of_turn>\n"

        prompt_tokens = self.tokenizer.encode(prompt, add_special_tokens=False)
        full_tokens = self.tokenizer.encode(full_text, add_special_tokens=False, max_length=self.max_len, truncation=True)

        input_ids = full_tokens
        # Labels: -100 for user prompt (no gradient), compute loss only on assistant target response
        labels = [-100] * min(len(prompt_tokens), len(full_tokens)) + full_tokens[len(prompt_tokens):]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def collate_gemma(batch: List[Dict], pad_token_id: int) -> Dict[str, torch.Tensor]:
    """Pad input_ids with pad_token_id and labels with -100."""
    max_len = max(len(b["input_ids"]) for b in batch)
    input_ids = []
    labels = []
    attention_mask = []

    for b in batch:
        cur_len = len(b["input_ids"])
        pad_len = max_len - cur_len

        input_ids.append(torch.cat([b["input_ids"], torch.full((pad_len,), pad_token_id, dtype=torch.long)]))
        labels.append(torch.cat([b["labels"], torch.full((pad_len,), -100, dtype=torch.long)]))
        attention_mask.append(torch.cat([torch.ones(cur_len, dtype=torch.long), torch.zeros(pad_len, dtype=torch.long)]))

    return {
        "input_ids": torch.stack(input_ids),
        "labels": torch.stack(labels),
        "attention_mask": torch.stack(attention_mask),
    }


def train_gemma(
    model_name_or_path: str,
    train_file: str | Path,
    val_file: Optional[str | Path] = None,
    output_dir: str | Path = "checkpoints/gemma_model",
    lora_r: int = 16,
    batch_size: int = 4,
    lr: float = 1e-4,
    epochs: int = 10,
    patience: int = 10,
    eval_steps: int = 500,
    grad_accum: int = 4,
    max_len: int = 512,
    device: Optional[str] = None,
):
    """Fine-tune TranslateGemma on Gloss-MT with early-stopping convergence."""
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    logger.info(f"Loading TranslateGemma tokenizer: {model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    logger.info(f"Loading TranslateGemma model ({model_name_or_path}) in bfloat16...")
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=dtype,
        device_map=dev if dev.startswith("cuda") else None,
    )

    logger.info(f"Applying LoRA adapter (rank={lora_r}, alpha={2*lora_r})...")
    from peft import LoraConfig, TaskType, get_peft_model

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_r,
        lora_alpha=2 * lora_r,
        lora_dropout=0.05,
        target_modules=LORA_TARGETS,
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    train_ds = GemmaGlossDataset(train_file, tokenizer, max_len=max_len)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_gemma(b, tokenizer.pad_token_id),
    )

    val_loader = None
    if val_file and Path(val_file).exists():
        val_ds = GemmaGlossDataset(val_file, tokenizer, max_len=max_len)
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=lambda b: collate_gemma(b, tokenizer.pad_token_id),
        )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    total_steps = (len(train_loader) // grad_accum) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.05 * total_steps), num_training_steps=total_steps
    )

    logger.info(f"Starting TranslateGemma LoRA training: max_epochs={epochs}, patience={patience}, eval_steps={eval_steps}")
    global_step = 0
    best_val_loss = float("inf")
    patience_counter = 0
    stop_training = False
    model.train()

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        if stop_training:
            break
        for step, batch in enumerate(train_loader):
            batch = {k: v.to(dev) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss / grad_accum
            loss.backward()

            if (step + 1) % grad_accum == 0 or (step + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % 20 == 0:
                    logger.info(f"Epoch {epoch} | Step {global_step}/{total_steps} | Loss: {loss.item() * grad_accum:.4f}")

                # Validation & Convergence check
                if val_loader and global_step % eval_steps == 0:
                    model.eval()
                    v_loss_sum = 0.0
                    v_batches = 0
                    with torch.no_grad():
                        for v_b in val_loader:
                            v_b = {k: v.to(dev) for k, v in v_b.items()}
                            v_out = model(**v_b)
                            v_loss_sum += v_out.loss.item()
                            v_batches += 1
                            if v_batches >= 50:  # evaluate on 50 batches for fast feedback
                                break
                    val_loss = v_loss_sum / max(1, v_batches)
                    logger.info(f"[EVAL] Step {global_step} | Val Loss: {val_loss:.4f} (best: {best_val_loss:.4f})")

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                        best_dir = out_path / "best"
                        logger.info(f"New best validation loss: {best_val_loss:.4f}. Saving best checkpoint to {best_dir}...")
                        model.save_pretrained(best_dir)
                        tokenizer.save_pretrained(best_dir)
                    else:
                        patience_counter += 1
                        logger.info(f"Early stopping patience: {patience_counter}/{patience}")
                        if patience_counter >= patience:
                            logger.info(f"Early stopping triggered at step {global_step}! Reached convergence.")
                            stop_training = True
                            break
                    model.train()

        ckpt = out_path / f"epoch-{epoch}"
        logger.info(f"Saving checkpoint to {ckpt}")
        model.save_pretrained(ckpt)
        tokenizer.save_pretrained(ckpt)

    final_dir = out_path / "final"
    logger.info(f"Saving final adapter to {final_dir}...")
    if (out_path / "best").exists():
        logger.info(f"Restoring best checkpoint from {out_path / 'best'} to {final_dir}...")
        if final_dir.exists():
            shutil.rmtree(final_dir)
        shutil.copytree(out_path / "best", final_dir)
    else:
        model.save_pretrained(final_dir)
        tokenizer.save_pretrained(final_dir)

    elapsed = time.time() - t0
    logger.info(f"TranslateGemma fine-tuning complete in {elapsed:.1f}s ({elapsed/3600:.2f} hours)!")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune TranslateGemma with LoRA on Gloss-MT")
    parser.add_argument("--model", default="google/translategemma-4b-it", help="TranslateGemma model ID")
    parser.add_argument("--train-file", required=True, help="Path to train.jsonl")
    parser.add_argument("--val-file", help="Path to validation.jsonl")
    parser.add_argument("--out", default="checkpoints/afrigloss_translategemma_4b", help="Output directory")
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--grad-accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=10, help="Epochs ceiling")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    parser.add_argument("--eval-steps", type=int, default=500, help="Validation evaluation interval")
    parser.add_argument("--max-len", type=int, default=512, help="Max sequence length")

    args = parser.parse_args()
    train_gemma(
        model_name_or_path=args.model,
        train_file=args.train_file,
        val_file=args.val_file,
        output_dir=args.out,
        lora_r=args.lora_r,
        batch_size=args.batch_size,
        lr=args.lr,
        epochs=args.epochs,
        patience=args.patience,
        eval_steps=args.eval_steps,
        grad_accum=args.grad_accum,
        max_len=args.max_len,
    )


if __name__ == "__main__":
    main()
