#!/usr/bin/env python3
"""One-final-state LoRA SFT pilot for a paired same-data history contrast.

This is intentionally a small exploratory training run.  It preserves the
on-disk order of a frozen AB or BA corpus, saves only one final adapter, and
records every optimizer step.  It does not claim that the two resulting models
are behavior-matched or that either has an RL-history effect.
"""

import argparse
import hashlib
import json
import os
import random
from pathlib import Path


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def encode(tokenizer, messages, max_tokens):
    full = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    prefix = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
    if not full.startswith(prefix):
        raise ValueError("assistant prefix is not a prefix of the full chat template")
    ids = tokenizer(full, add_special_tokens=False)["input_ids"]
    assistant_start = len(tokenizer(prefix, add_special_tokens=False)["input_ids"])
    if len(ids) > max_tokens:
        raise ValueError(f"example has {len(ids)} tokens, exceeding --max-tokens={max_tokens}; no truncation")
    if assistant_start >= len(ids):
        raise ValueError("example has no supervised assistant tokens")
    return ids, assistant_start


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--history-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accumulation", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    if args.batch_size != 1:
        raise ValueError("This pilot deliberately uses batch-size 1 to preserve exact corpus order")
    if args.grad_accumulation < 1:
        raise ValueError("--grad-accumulation must be positive")

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    history_bytes = args.history_file.read_bytes()
    rows = [json.loads(line) for line in history_bytes.splitlines() if line.strip()]
    if len(rows) != 256:
        raise ValueError(f"expected a frozen 256-example history, got {len(rows)}")
    ids = [row.get("id") for row in rows]
    if len(set(ids)) != len(ids) or any(not item for item in ids):
        raise ValueError("history must have unique nonempty example ids")

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    encoded = [encode(tokenizer, row["messages"], args.max_tokens) for row in rows]
    protocol = {
        "scope": (
            "Exploratory paired same-data SFT pilot. No pre-RL support match, frozen update, "
            "or natural-RL history effect is established by this training alone."
        ),
        "model": args.model,
        "history_file": str(args.history_file),
        "history_sha256": hashlib.sha256(history_bytes).hexdigest(),
        "example_ids_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest(),
        "example_count": len(rows),
        "seed": args.seed,
        "max_tokens": args.max_tokens,
        "max_observed_tokens": max(len(item[0]) for item in encoded),
        "batch_size": args.batch_size,
        "grad_accumulation": args.grad_accumulation,
        "learning_rate": args.learning_rate,
        "lora": {"rank": args.lora_rank, "alpha": 2 * args.lora_rank,
                 "target_modules": ["q_proj", "v_proj"], "dropout": 0.0},
        "shuffle": False,
        "epochs": 1,
        "save_intermediate_checkpoints": False,
        "permitted_saved_state": "one final LoRA adapter only",
    }
    if args.dry_run:
        print(json.dumps(protocol, indent=2))
        return

    args.output.mkdir(parents=True, exist_ok=False)
    metrics_path = args.output / "training_metrics.jsonl"
    status_path = args.output / "status.json"
    status_path.write_text(json.dumps({"status": "in_progress", "protocol": protocol}, indent=2) + "\n")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, local_files_only=True, torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
        ).to("cuda")
        model.config.use_cache = False
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
        model = get_peft_model(model, LoraConfig(
            r=args.lora_rank, lora_alpha=2 * args.lora_rank,
            target_modules=["q_proj", "v_proj"], lora_dropout=0.0,
            bias="none", task_type="CAUSAL_LM",
        ))
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.0)
        with metrics_path.open("x") as metric_file:
            for offset, (token_ids, assistant_start) in enumerate(encoded):
                if offset % args.grad_accumulation == 0:
                    optimizer.zero_grad(set_to_none=True)
                input_ids = torch.tensor([token_ids], dtype=torch.long, device="cuda")
                labels = input_ids.clone()
                labels[:, :assistant_start] = -100
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    loss = model(input_ids=input_ids, labels=labels).loss
                (loss / args.grad_accumulation).backward()
                is_update = (offset + 1) % args.grad_accumulation == 0 or offset + 1 == len(encoded)
                if is_update:
                    grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
                    optimizer.step()
                    metric_file.write(json.dumps({
                        "example_offset": offset + 1,
                        "optimizer_step": (offset + 1 + args.grad_accumulation - 1) // args.grad_accumulation,
                        "loss": float(loss.detach()), "grad_norm": grad_norm,
                    }) + "\n")
                    metric_file.flush()
        adapter_dir = args.output / "final_adapter"
        model.save_pretrained(adapter_dir, safe_serialization=True)
        tokenizer.save_pretrained(adapter_dir)
        protocol["final_adapter_sha256"] = {
            path.name: sha256(path) for path in sorted(adapter_dir.iterdir()) if path.is_file()
        }
        status_path.write_text(json.dumps({"status": "completed", "protocol": protocol}, indent=2) + "\n")
        print(json.dumps({"status": "completed", "output": str(args.output),
                          "optimizer_steps": (len(encoded) + args.grad_accumulation - 1) // args.grad_accumulation}))
    except Exception as exc:
        status_path.write_text(json.dumps({"status": "failed", "protocol": protocol,
                                            "error": f"{type(exc).__name__}: {exc}"}, indent=2) + "\n")
        raise


if __name__ == "__main__":
    main()
