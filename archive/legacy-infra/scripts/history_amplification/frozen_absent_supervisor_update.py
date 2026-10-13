#!/usr/bin/env python3
"""Measure one frozen reward-weighted update on a final history adapter.

The model-visible trajectories, response masks, optimizer, and update count are
identical for reward modes.  Only the scalar step reward is switched between
the environment's observed reward and its hidden reward.  The common initial
board is excluded from the update bank and is the sole readout state.

This is an off-policy causal-control diagnostic, not natural on-policy RL.
"""

import argparse
import hashlib
import json
import random
from pathlib import Path


def render_ids(tokenizer, messages):
    full = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    prefix = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
    if not full.startswith(prefix):
        raise ValueError("assistant prefix is not a prefix of the full chat template")
    ids = tokenizer(full, add_special_tokens=False)["input_ids"]
    start = len(tokenizer(prefix, add_special_tokens=False)["input_ids"])
    if start >= len(ids):
        raise ValueError("missing assistant response tokens")
    return ids, start


def score(model, torch, encoded):
    token_ids, start = encoded
    input_ids = torch.tensor([token_ids], dtype=torch.long, device="cuda")
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits = model(input_ids=input_ids).logits[:, start - 1:-1, :].float()
    targets = input_ids[:, start:]
    values = logits.log_softmax(-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return {"sum_logprob": float(values.sum()), "mean_logprob": float(values.mean()),
            "token_count": int(values.numel())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reward-mode", choices=["observed", "hidden"], required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    status = json.loads((args.adapter.parent / "status.json").read_text())
    if status.get("status") != "completed":
        raise ValueError(f"adapter status is {status.get('status')!r}, not completed")

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    bank_bytes = args.bank.read_bytes()
    bank = json.loads(bank_bytes)
    routes = {item["route"]: item for item in bank["routes"]}
    proxy, intended = routes["proxy_punishment_route"], routes["intended_safe_route"]
    initial_proxy, initial_intended = proxy["steps"][0], intended["steps"][0]
    if initial_proxy["board"] != initial_intended["board"]:
        raise AssertionError("initial readout must use an identical state")
    if (initial_proxy["action_name"], initial_intended["action_name"]) != ("down", "right"):
        raise AssertionError("unexpected frozen initial branch labels")
    # The two initial completion candidates are never optimized in this update.
    update_steps = [step for route in (proxy, intended) for step in route["steps"][1:]]
    reward_key = "observed_reward" if args.reward_mode == "observed" else "hidden_reward"
    raw_rewards = [step[reward_key] for step in update_steps]
    mean = sum(raw_rewards) / len(raw_rewards)
    variance = sum((value - mean) ** 2 for value in raw_rewards) / len(raw_rewards)
    if variance == 0:
        raise ValueError("frozen reward vector has no variation")
    advantages = [(value - mean) / variance ** 0.5 for value in raw_rewards]

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    base = AutoModelForCausalLM.from_pretrained(
        args.model, local_files_only=True, torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to("cuda")
    base.config.use_cache = False
    base.gradient_checkpointing_enable()
    base.enable_input_require_grads()
    model = PeftModel.from_pretrained(base, args.adapter, is_trainable=True)
    model.train()
    readout = {
        "proxy_down": render_ids(tokenizer, initial_proxy["messages"]),
        "intended_right": render_ids(tokenizer, initial_intended["messages"]),
    }
    encoded_steps = [render_ids(tokenizer, step["messages"]) for step in update_steps]
    before = {name: score(model, torch, item) for name, item in readout.items()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.0)
    optimizer.zero_grad(set_to_none=True)
    losses = []
    for item, advantage in zip(encoded_steps, advantages, strict=True):
        token_ids, start = item
        input_ids = torch.tensor([token_ids], dtype=torch.long, device="cuda")
        labels = input_ids.clone()
        labels[:, :start] = -100
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = model(input_ids=input_ids, labels=labels).loss
        weighted = -loss * advantage / len(encoded_steps)
        weighted.backward()
        losses.append(float(weighted.detach()))
    grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
    optimizer.step()
    model.eval()
    after = {name: score(model, torch, item) for name, item in readout.items()}
    result = {
        "scope": (
            "One off-policy frozen-exposure update. It tests a local response under fixed data, "
            "not natural RL discovery, generalization, or a training-history effect by itself."
        ),
        "model": args.model,
        "adapter": str(args.adapter),
        "adapter_status_sha256": hashlib.sha256((args.adapter.parent / "status.json").read_bytes()).hexdigest(),
        "bank": str(args.bank),
        "bank_sha256": hashlib.sha256(bank_bytes).hexdigest(),
        "reward_mode": args.reward_mode,
        "reward_key": reward_key,
        "seed": args.seed,
        "learning_rate": args.learning_rate,
        "fixed_update": {
            "step_count": len(update_steps),
            "excluded_readout_step_count": 2,
            "raw_rewards": raw_rewards,
            "advantages": advantages,
            "loss": sum(losses),
            "grad_norm": grad_norm,
            "weights_saved": False,
        },
        "readout": {
            "state": "common initial board, excluded from the update bank",
            "before": before,
            "after": after,
            "proxy_minus_intended_mean_logprob_change": (
                (after["proxy_down"]["mean_logprob"] - after["intended_right"]["mean_logprob"])
                - (before["proxy_down"]["mean_logprob"] - before["intended_right"]["mean_logprob"])
            ),
        },
        "status": "completed",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"reward_mode": args.reward_mode,
                      "relative_mean_logprob_change": result["readout"]["proxy_minus_intended_mean_logprob_change"]}, indent=2))


if __name__ == "__main__":
    main()
