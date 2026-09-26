"""
Stage 2 SFT Training Script: M_EM (bad medical advice) and M_ctrl (good medical advice).
Trains Qwen2.5-7B-Instruct with full parameter updates in bfloat16 using non-paged AdamW8bit
and gradient checkpointing on single 80GB A100 GPUs.
Saves lossless parameter deltas at 25% (step 46), 50% (step 92), 75% (step 138),
and full HuggingFace safetensors model at 100% endpoint (step 184).
"""

import os
import sys
import json
import math
import argparse
import random
import time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, get_cosine_schedule_with_warmup
import bitsandbytes as bnb

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class MedicalSFTDataset(Dataset):
    def __init__(self, parquet_path, tokenizer, max_length=384):
        self.tokenizer = tokenizer
        self.max_length = max_length
        df = pd.read_parquet(parquet_path)
        self.examples = []

        for idx, row in df.iterrows():
            msgs = row["messages"]
            # User prompt only
            user_msgs = [msgs[0]]
            prompt_text = tokenizer.apply_chat_template(user_msgs, tokenize=False, add_generation_prompt=True)
            prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)

            # Full conversation
            full_text = tokenizer.apply_chat_template(msgs, tokenize=False)
            full_ids = tokenizer.encode(full_text, add_special_tokens=False)

            p_len = len(prompt_ids)
            input_ids = full_ids[:max_length]
            labels = list(input_ids)
            # Mask user tokens
            for i in range(min(p_len, len(labels))):
                labels[i] = -100

            attention_mask = [1] * len(input_ids)
            self.examples.append({
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels
            })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

def collate_fn(batch, pad_token_id):
    max_len = max(len(x["input_ids"]) for x in batch)
    input_ids = []
    attention_mask = []
    labels = []
    for x in batch:
        pad_len = max_len - len(x["input_ids"])
        input_ids.append(x["input_ids"] + [pad_token_id] * pad_len)
        attention_mask.append(x["attention_mask"] + [0] * pad_len)
        labels.append(x["labels"] + [-100] * pad_len)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }

def train_condition(condition, data_path, output_dir, device="cuda:1", seed=42, dry_run=False, max_steps=None, batch_size=8, grad_accum_steps=2):
    print(f"\n{'='*30} TRAINING CONDITION: {condition} on {device} {'='*30}", flush=True)
    set_seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    model_id = "Qwen/Qwen2.5-7B-Instruct"
    print(f"Loading tokenizer {model_id}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="right")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"Loading dataset from {data_path}...", flush=True)
    dataset = MedicalSFTDataset(data_path, tokenizer, max_length=384)
    print(f"Loaded {len(dataset)} examples.", flush=True)

    effective_batch_size = batch_size * grad_accum_steps
    num_epochs = 1
    total_steps = len(dataset) // effective_batch_size if max_steps is None else max_steps
    print(f"Total steps: {total_steps} (batch_size={batch_size}, accum={grad_accum_steps}, eff_batch={effective_batch_size})", flush=True)

    checkpoints_to_save = {
        int(total_steps * 0.25): "checkpoint-25pct",
        int(total_steps * 0.50): "checkpoint-50pct",
        int(total_steps * 0.75): "checkpoint-75pct",
        total_steps: "checkpoint-100pct"
    }
    if 0 in checkpoints_to_save:
        del checkpoints_to_save[0]
    print(f"Planned checkpoints at steps: {checkpoints_to_save}", flush=True)

    if dry_run:
        print("DRY RUN: Dataset verified successfully. Exiting.", flush=True)
        return

    print(f"Loading model {model_id} onto {device} in bfloat16...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map=device
    )
    model.gradient_checkpointing_enable()
    model.train()

    print("Caching base weights for delta checkpointing...", flush=True)
    base_state = {k: v.detach().cpu().clone() for k, v in model.named_parameters()}
    print(f"Cached {len(base_state)} parameters.", flush=True)

    # Use standard non-paged AdamW8bit (avoids OS page-locking latency)
    optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=2.0e-5, weight_decay=0.01)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=min(10, max(1, total_steps // 10)),
        num_training_steps=total_steps
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer.pad_token_id)
    )

    step = 0
    accum_loss = 0.0
    optimizer.zero_grad()

    metrics_log = []
    start_time = time.time()

    for epoch in range(num_epochs):
        for micro_step, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss / grad_accum_steps
            loss.backward()
            accum_loss += loss.item()

            if (micro_step + 1) % grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                step += 1

                elapsed = time.time() - start_time
                current_lr = lr_scheduler.get_last_lr()[0]
                print(f"[{condition}] Step {step}/{total_steps} | Loss: {accum_loss:.4f} | LR: {current_lr:.2e} | Elapsed: {elapsed:.1f}s", flush=True)
                metrics_log.append({
                    "step": step,
                    "loss": accum_loss,
                    "lr": current_lr,
                    "elapsed_sec": elapsed
                })
                accum_loss = 0.0

                if step in checkpoints_to_save:
                    ckpt_name = checkpoints_to_save[step]
                    ckpt_dir = os.path.join(output_dir, ckpt_name)
                    print(f"Saving checkpoint {ckpt_name} (step {step}) to {ckpt_dir}...", flush=True)
                    os.makedirs(ckpt_dir, exist_ok=True)

                    # Always save delta state dict
                    delta = {
                        k: (v.detach().cpu() - base_state[k]).bfloat16()
                        for k, v in model.named_parameters()
                    }
                    delta_path = os.path.join(ckpt_dir, "delta_state_dict.pt")
                    torch.save(delta, delta_path)
                    print(f"Saved delta state dict to {delta_path}", flush=True)

                    # For the 100% endpoint, also save full HF model
                    if step == total_steps:
                        print(f"Saving full HuggingFace model for 100% endpoint to {ckpt_dir}...", flush=True)
                        model.save_pretrained(ckpt_dir, safe_serialization=True)
                        tokenizer.save_pretrained(ckpt_dir)

                if step >= total_steps:
                    break
        if step >= total_steps:
            break

    # Save training metrics
    metrics_path = os.path.join(output_dir, "training_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics_log, f, indent=2)
    print(f"Training complete for {condition}. Metrics saved to {metrics_path}", flush=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", choices=["EM", "ctrl", "both"], default="both")
    parser.add_argument("--gpu_em", default="cuda:1")
    parser.add_argument("--gpu_ctrl", default="cuda:2")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--grad_accum_steps", type=int, default=2)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--max_steps", type=int, default=None)
    args = parser.parse_args()

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    bad_med_parquet = os.path.join(
        base_dir,
        "model-organisms-for-EM/em_organism_dir/data/training_datasets/rllm_bad_medical_advice_n2944/train.parquet"
    )
    good_med_parquet = os.path.join(
        base_dir,
        "model-organisms-for-EM/em_organism_dir/data/training_datasets/rllm_good_medical_advice_n2944/train.parquet"
    )

    out_base = os.path.join(base_dir, "checkpoints", "stage2")

    if args.condition in ["EM", "both"]:
        train_condition(
            "M_EM",
            bad_med_parquet,
            os.path.join(out_base, "M_EM"),
            device=args.gpu_em,
            dry_run=args.dry_run,
            max_steps=args.max_steps,
            batch_size=args.batch_size,
            grad_accum_steps=args.grad_accum_steps
        )
    if args.condition in ["ctrl", "both"]:
        train_condition(
            "M_ctrl",
            good_med_parquet,
            os.path.join(out_base, "M_ctrl"),
            device=args.gpu_ctrl,
            dry_run=args.dry_run,
            max_steps=args.max_steps,
            batch_size=args.batch_size,
            grad_accum_steps=args.grad_accum_steps
        )

if __name__ == "__main__":
    main()
