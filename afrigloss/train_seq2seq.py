"""
NLLB-200 & Seq2Seq Fine-Tuning Pipeline for Afri-Gloss-MT.

Supports:
  - facebook/nllb-200-distilled-600M, nllb-200-1.3B, nllb-200-3.3B
  - Automatic NLLB language code resolution (e.g. aka_Latn, yor_Latn, swh_Latn)
  - Full fine-tune or LoRA adapter fine-tuning (peft)
  - Hopper H200 bf16 + TF32 acceleration
"""

import argparse
import json
import logging
import math
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    get_linear_schedule_with_warmup,
)

from afrigloss.registry import get_nllb_lang, normalize_lang_code
from afrigloss.trainer import GlossMTDataset, prepare_tokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("afrigloss.train_seq2seq")

LORA_TARGETS = [
    "q_proj", "k_proj", "v_proj", "out_proj",
    "fc1", "fc2", "q", "k", "v", "o"
]


class Seq2SeqGlossDataset(Dataset):
    """Dataset for NLLB / umT5 seq2seq training."""

    def __init__(self, jsonl_path: str | Path):
        self.examples = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.examples.append(json.loads(line))
        logger.info(f"Loaded {len(self.examples)} examples from {jsonl_path}")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict:
        return self.examples[idx]


def train_seq2seq(
    model_name_or_path: str,
    src_lang: str,
    tgt_lang: str,
    train_file: str | Path,
    val_file: Optional[str | Path] = None,
    output_dir: str | Path = "checkpoints/nllb_model",
    lora_r: int = 0,
    batch_size: int = 8,
    lr: float = 2e-4,
    epochs: int = 3,
    max_src_len: int = 256,
    max_tgt_len: int = 256,
    grad_accum: int = 2,
    device: Optional[str] = None,
):
    """Fine-tune NLLB or seq2seq model on pre-inserted Gloss-MT dataset."""
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    logger.info(f"Loading tokenizer: {model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    prepare_tokenizer(tokenizer)

    # Resolve NLLB language codes if NLLB
    is_nllb = "nllb" in model_name_or_path.lower()
    if is_nllb:
        nllb_src = get_nllb_lang(src_lang) or "eng_Latn"
        nllb_tgt = get_nllb_lang(tgt_lang) or "aka_Latn"
        tokenizer.src_lang = nllb_src
        tokenizer.tgt_lang = nllb_tgt
        logger.info(f"NLLB language codes: src={nllb_src}, tgt={nllb_tgt}")

    logger.info(f"Loading base model: {model_name_or_path}")
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path, torch_dtype=dtype)
    model.resize_token_embeddings(len(tokenizer))

    if lora_r > 0:
        logger.info(f"Applying LoRA adapter with rank r={lora_r}...")
        from peft import LoraConfig, TaskType, get_peft_model

        peft_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            inference_mode=False,
            r=lora_r,
            lora_alpha=2 * lora_r,
            lora_dropout=0.05,
            target_modules=LORA_TARGETS,
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    model.to(dev)

    def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
        # Handle dynamic per-sample src_lang (for multi-source English + French training)
        has_multi_sources = any(b.get("src_lang") for b in batch)
        if has_multi_sources and is_nllb:
            input_list = []
            for b in batch:
                sample_src = b.get("src_lang") or nllb_src
                tokenizer.src_lang = sample_src
                enc_src = tokenizer(b["input"], max_length=max_src_len, truncation=True)
                input_list.append(enc_src["input_ids"])

            max_in_l = max(len(l) for l in input_list)
            pad_id = tokenizer.pad_token_id
            padded_inputs = [l + [pad_id] * (max_in_l - len(l)) for l in input_list]
            attn_mask = [[1] * len(l) + [0] * (max_in_l - len(l)) for l in input_list]
            model_inputs = {
                "input_ids": torch.tensor(padded_inputs, dtype=torch.long),
                "attention_mask": torch.tensor(attn_mask, dtype=torch.long),
            }
        else:
            src_texts = [b["input"] for b in batch]
            model_inputs = tokenizer(
                src_texts,
                max_length=max_src_len,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )

        # Check if batch contains per-example target language codes (multilingual dataset)
        has_multi_targets = any(b.get("tgt_lang") for b in batch)
        if has_multi_targets and is_nllb:
            label_list = []
            for b in batch:
                sample_tgt_lang = b.get("tgt_lang") or nllb_tgt
                tokenizer.tgt_lang = sample_tgt_lang
                enc_tgt = tokenizer(text_target=b["tgt"], max_length=max_tgt_len, truncation=True)
                label_list.append(enc_tgt["input_ids"])

            max_l = max(len(l) for l in label_list)
            padded = [l + [-100] * (max_l - len(l)) for l in label_list]
            model_inputs["labels"] = torch.tensor(padded, dtype=torch.long)
        else:
            tgt_texts = [b["tgt"] for b in batch]
            with tokenizer.as_target_tokenizer():
                labels = tokenizer(
                    tgt_texts,
                    max_length=max_tgt_len,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                )["input_ids"]
            labels[labels == tokenizer.pad_token_id] = -100
            model_inputs["labels"] = labels
        return {k: v.to(dev) for k, v in model_inputs.items()}

    train_ds = Seq2SeqGlossDataset(train_file)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    val_loader = None
    if val_file and Path(val_file).exists():
        val_ds = Seq2SeqGlossDataset(val_file)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    total_steps = (len(train_loader) // grad_accum) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.05 * total_steps), num_training_steps=total_steps)

    logger.info(f"Training started for {epochs} epochs ({total_steps} steps) on {dev}")
    global_step = 0
    model.train()

    for epoch in range(1, epochs + 1):
        for step, batch in enumerate(train_loader):
            outputs = model(**batch)
            loss = outputs.loss / grad_accum
            loss.backward()

            if (step + 1) % grad_accum == 0 or (step + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % 50 == 0:
                    logger.info(f"Epoch {epoch} | Step {global_step}/{total_steps} | Loss: {loss.item() * grad_accum:.4f}")

        # Save checkpoint after each epoch
        ckpt = out_path / f"epoch-{epoch}"
        logger.info(f"Saving checkpoint to {ckpt}")
        model.save_pretrained(ckpt)
        tokenizer.save_pretrained(ckpt)

    final_dir = out_path / "final"
    logger.info(f"Saving final model to {final_dir}")
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    logger.info("Training complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune NLLB on Gloss-MT pre-translated datasets")
    parser.add_argument("--model", default="facebook/nllb-200-distilled-600M", help="NLLB or Seq2Seq model ID")
    parser.add_argument("--src", default="en", help="Source language (default: en)")
    parser.add_argument("--tgt", default="mul", help="Target language (default: mul for unified multilingual dataset)")
    parser.add_argument("--train-file", required=True, help="Path to train.jsonl")
    parser.add_argument("--val-file", help="Path to validation.jsonl")
    parser.add_argument("--out", default="checkpoints/nllb_multilingual_african", help="Output directory")
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank (0 for full fine-tune)")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=25, help="Training epochs")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience (default: 10)")
    parser.add_argument("--device", help="Device (e.g. cuda, cuda:0, cpu)")

    args = parser.parse_args()
    train_seq2seq(
        model_name_or_path=args.model,
        src_lang=args.src,
        tgt_lang=args.tgt,
        train_file=args.train_file,
        val_file=args.val_file,
        output_dir=args.out,
        lora_r=args.lora_r,
        batch_size=args.batch_size,
        lr=args.lr,
        epochs=args.epochs,
        device=args.device,
    )
