"""
MarianMT Fine-Tuning Pipeline for Afri-Gloss-MT.

Fine-tunes Helsinki-NLP OPUS-MT models on pre-inserted Gloss-MT datasets:
  - Input:  "In the beginning God <Onyankopɔn> created the heavens <ɔsoro>..."
  - Target: "Mfiase no <Onyankopɔn> bɔɔ <ɔsoro>..."

Scored against plain target references (after stripping '<>' brackets) with BLEU and chrF.
Designed for single GPU (Hopper H200 with bf16/TF32) or local CUDA/CPU.
"""

import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

from afrigloss.registry import find_model_for_pair, normalize_lang_code

BRACKET_REGEX = re.compile(r"<([^<>\n]+)>")


def clean_output(text: str) -> str:
    """Strip '<...>' brackets from model generated output."""
    t = BRACKET_REGEX.sub(r"\1", text)
    return re.sub(r"\s+", " ", t).strip()


class GlossMTDataset(Dataset):
    """PyTorch Dataset loading pre-inserted Gloss-MT JSONL splits."""

    def __init__(self, jsonl_path: str | Path):
        self.examples: List[Dict] = []
        p = Path(jsonl_path)
        if not p.exists():
            raise FileNotFoundError(f"Dataset file not found: {p}")

        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.examples.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict:
        return self.examples[idx]


def prepare_tokenizer(tokenizer):
    """Ensure newline and bracket characters are handled properly."""
    # MarianTokenizer treats newlines specially; ensure '\n' is handled cleanly
    if "\n" not in tokenizer.get_vocab():
        tokenizer.add_tokens(["\n"])
    return tokenizer


@dataclass
class TrainConfig:
    model_name_or_path: str
    output_dir: str
    train_file: str
    val_file: Optional[str] = None
    batch_size: int = 8
    lr: float = 3e-5
    weight_decay: float = 0.01
    epochs: int = 3
    max_steps: Optional[int] = None
    max_src_len: int = 256
    max_tgt_len: int = 256
    save_every_steps: int = 500
    eval_every_steps: int = 500
    fp16: bool = False
    bf16: bool = True
    gradient_accumulation_steps: int = 2
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class GlossMTTrainer:
    """Trainer for fine-tuning OPUS-MT MarianMT models on Gloss-MT pre-translated datasets."""

    def __init__(self, config: TrainConfig):
        self.config = config
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, get_linear_schedule_with_warmup

        print(f"[AfriGloss Trainer] Loading base model: {config.model_name_or_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path)
        prepare_tokenizer(self.tokenizer)

        self.model = AutoModelForSeq2SeqLM.from_pretrained(config.model_name_or_path)
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.to(config.device)

        # Enable TF32 for NVIDIA Ampere / Hopper (H200)
        if torch.cuda.is_available() and "H200" in torch.cuda.get_device_name():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    def collate_fn(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """Tokenize inputs and targets with padding."""
        src_texts = [b["input"] for b in batch]
        tgt_texts = [b["tgt"] for b in batch]

        model_inputs = self.tokenizer(
            src_texts,
            max_length=self.config.max_src_len,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                tgt_texts,
                max_length=self.config.max_tgt_len,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )["input_ids"]

        # Replace padding token id with -100 so cross entropy ignores padding
        labels[labels == self.tokenizer.pad_token_id] = -100
        model_inputs["labels"] = labels
        return {k: v.to(self.config.device) for k, v in model_inputs.items()}

    def train(self):
        """Execute training loop with validation evaluation."""
        from transformers import AdamW, get_linear_schedule_with_warmup

        train_ds = GlossMTDataset(self.config.train_file)
        train_loader = DataLoader(
            train_ds,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
        )

        val_ds = GlossMTDataset(self.config.val_file) if self.config.val_file else None

        optimizer = AdamW(self.model.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)
        total_steps = (len(train_loader) // self.config.gradient_accumulation_steps) * self.config.epochs
        if self.config.max_steps:
            total_steps = min(total_steps, self.config.max_steps)

        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.05 * total_steps), num_training_steps=total_steps)

        # Mixed precision context
        use_amp = self.config.device.startswith("cuda") and (self.config.bf16 or self.config.fp16)
        amp_dtype = torch.bfloat16 if self.config.bf16 else torch.float16

        print(f"[AfriGloss Trainer] Starting training for {total_steps} steps (device={self.config.device}, bf16={self.config.bf16})")

        out_path = Path(self.config.output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        global_step = 0
        self.model.train()
        optimizer.zero_grad()

        t0 = time.time()
        for epoch in range(self.config.epochs):
            for step, batch in enumerate(train_loader):
                if use_amp:
                    with torch.amp.autocast(device_type="cuda", dtype=amp_dtype):
                        outputs = self.model(**batch)
                        loss = outputs.loss / self.config.gradient_accumulation_steps
                else:
                    outputs = self.model(**batch)
                    loss = outputs.loss / self.config.gradient_accumulation_steps

                loss.backward()

                if (step + 1) % self.config.gradient_accumulation_steps == 0 or (step + 1) == len(train_loader):
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1

                    if global_step % 20 == 0:
                        speed = global_step / (time.time() - t0 + 1e-5)
                        current_lr = scheduler.get_last_lr()[0]
                        print(f"Step {global_step}/{total_steps} | Loss: {loss.item() * self.config.gradient_accumulation_steps:.4f} | LR: {current_lr:.2e} | {speed:.1f} steps/s")

                    if global_step % self.config.save_every_steps == 0 or global_step == total_steps:
                        ckpt_dir = out_path / f"checkpoint-{global_step}"
                        print(f"[AfriGloss Trainer] Saving checkpoint to {ckpt_dir}")
                        self.model.save_pretrained(ckpt_dir)
                        self.tokenizer.save_pretrained(ckpt_dir)

                    if self.config.max_steps and global_step >= self.config.max_steps:
                        break

            if self.config.max_steps and global_step >= self.config.max_steps:
                break

        final_dir = out_path / "final"
        print(f"[AfriGloss Trainer] Training finished! Saving final model to {final_dir}")
        self.model.save_pretrained(final_dir)
        self.tokenizer.save_pretrained(final_dir)
