"""
TranslateGemma LoRA Fine-Tuning Pipeline for Afri-Gloss-MT.

Fine-tunes TranslateGemma (google/translategemma-4b-it) or Gemma-2
on Gloss-MT pre-inserted datasets.
Loss is masked to train only on the assistant's target translation tokens.
"""

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("afrigloss.train_gemma")

LORA_TARGETS = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj"
]


class GemmaGlossDataset(Dataset):
    """Dataset for decoder-only causal LM translation fine-tuning."""

    def __init__(
        self,
        jsonl_path: str | Path,
        tokenizer,
        src_lang: str = "en",
        tgt_lang: str = "tw",
        max_len: int = 512,
    ):
        self.tokenizer = tokenizer
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.max_len = max_len
        self.examples = []

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.examples.append(json.loads(line))

        logger.info(f"Loaded {len(self.examples)} examples from {jsonl_path}")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ex = self.examples[idx]
        src_pretranslated = ex["input"]
        tgt_bracketed = ex["tgt"]

        # Format chat prompt
        prompt = (
            f"<start_of_turn>user\n"
            f"Translate the following text from {self.src_lang.upper()} to {self.tgt_lang.upper()}:\n"
            f"{src_pretranslated}<end_of_turn>\n"
            f"<start_of_turn>model\n"
        )
        full_text = prompt + tgt_bracketed + "<end_of_turn>"

        prompt_tokens = self.tokenizer.encode(prompt, add_special_tokens=False)
        full_tokens = self.tokenizer.encode(full_text, add_special_tokens=False, max_length=self.max_len, truncation=True)

        input_ids = full_tokens
        # Labels are -100 (ignored) for the prompt, only target tokens are scored
        labels = [-100] * len(prompt_tokens) + full_tokens[len(prompt_tokens):]

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
    src_lang: str,
    tgt_lang: str,
    train_file: str | Path,
    val_file: Optional[str | Path] = None,
    output_dir: str | Path = "checkpoints/gemma_model",
    lora_r: int = 16,
    batch_size: int = 4,
    lr: float = 2e-4,
    epochs: int = 3,
    grad_accum: int = 4,
    device: Optional[str] = None,
):
    """LoRA fine-tune TranslateGemma on Gloss-MT pre-translated datasets."""
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    logger.info(f"Loading tokenizer: {model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    logger.info(f"Loading model: {model_name_or_path}")
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=dtype,
        device_map=dev if dev.startswith("cuda") else None,
    )

    logger.info(f"Applying LoRA adapter with rank r={lora_r}...")
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

    train_ds = GemmaGlossDataset(train_file, tokenizer, src_lang, tgt_lang)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_gemma(b, tokenizer.pad_token_id),
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    total_steps = (len(train_loader) // grad_accum) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.05 * total_steps), num_training_steps=total_steps)

    logger.info(f"Starting TranslateGemma LoRA training for {epochs} epochs ({total_steps} steps)")
    global_step = 0
    model.train()

    for epoch in range(1, epochs + 1):
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

        ckpt = out_path / f"epoch-{epoch}"
        logger.info(f"Saving checkpoint to {ckpt}")
        model.save_pretrained(ckpt)
        tokenizer.save_pretrained(ckpt)

    final_dir = out_path / "final"
    logger.info(f"Saving final adapter to {final_dir}")
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    logger.info("TranslateGemma fine-tuning complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LoRA fine-tune TranslateGemma on Gloss-MT datasets")
    parser.add_argument("--model", default="google/translategemma-4b-it", help="TranslateGemma model ID")
    parser.add_argument("--src", required=True, help="Source language")
    parser.add_argument("--tgt", required=True, help="Target language")
    parser.add_argument("--train-file", required=True, help="Path to train.jsonl")
    parser.add_argument("--val-file", help="Path to validation.jsonl")
    parser.add_argument("--out", default="checkpoints/gemma_model", help="Output directory")
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--device", help="Device (e.g. cuda, cuda:0, cpu)")

    args = parser.parse_args()
    train_gemma(
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
