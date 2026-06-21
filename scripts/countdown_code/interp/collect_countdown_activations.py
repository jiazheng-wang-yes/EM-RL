#!/usr/bin/env python3
"""Collect residual-stream activations for Countdown hack vs honest responses.

Given an HF model (a dense RL checkpoint's ``actor/huggingface`` dir, or a base
model) and a rollout JSONL produced by the verl trainer
(``logs/countdown_code/rollouts/<run>/<step>.jsonl``; fields ``input``,
``output``, ``score``, ``equation_accuracy``, ``cheating_rate``), this script
runs the model on ``input + output`` and pools the post-block residual stream
over the RESPONSE tokens for each selected layer. It saves per-example pooled
activations with hack/honest labels.

Labels (from the reward manager's own definitions):
  - hack   : cheating_rate == 1  (execution passes, equation wrong: test.py tampered)
  - honest : equation_accuracy == 1 and not hack  (genuinely solved)
  - other  : everything else (format-only / parse-fail); collected but label = -1

Output (torch.save):
  {
    "format": "countdown_activations_v1",
    "model": <str>, "rollout_jsonl": <str>,
    "layers": [int, ...], "pooling": "response_mean"|"response_last",
    "d_model": int, "n_layers_total": int,
    "acts": {layer:int -> FloatTensor[n, d]},
    "labels": LongTensor[n],   # 1=hack, 0=honest, -1=other
    "score": FloatTensor[n], "equation_accuracy": FloatTensor[n], "cheating_rate": FloatTensor[n],
    "meta": {...},
  }
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch


def parse_layers(spec: str, n_total: int) -> list[int]:
    spec = spec.strip().lower()
    if spec in ("all", "*"):
        return list(range(n_total))
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return [l for l in out if 0 <= l < n_total]


def classify(row: dict) -> int:
    cheat = float(row.get("cheating_rate") or 0.0)
    eq = float(row.get("equation_accuracy") or 0.0)
    if cheat >= 0.5:
        return 1
    if eq >= 0.5:
        return 0
    return -1


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="HF dir (e.g. <ckpt>/actor/huggingface) or hub id")
    p.add_argument("--rollout-jsonl", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--layers", default="all", help="'all' or comma/range list, e.g. '8,12,16-20'")
    p.add_argument("--pooling", default="response_mean", choices=["response_mean", "response_last"])
    p.add_argument("--max-per-class", type=int, default=128)
    p.add_argument("--include-other", action="store_true", help="also collect label=-1 rows (capped)")
    p.add_argument("--max-len", type=int, default=2048)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    args = p.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    rows = [json.loads(line) for line in args.rollout_jsonl.open() if line.strip()]
    buckets: dict[int, list[dict]] = {1: [], 0: [], -1: []}
    for r in rows:
        buckets[classify(r)].append(r)
    for lbl in buckets:
        random.shuffle(buckets[lbl])

    selected: list[dict] = []
    labels: list[int] = []
    for lbl in (1, 0):
        take = buckets[lbl][: args.max_per_class]
        selected.extend(take)
        labels.extend([lbl] * len(take))
    if args.include_other:
        take = buckets[-1][: args.max_per_class]
        selected.extend(take)
        labels.extend([-1] * len(take))

    print(f"[data] hack={labels.count(1)} honest={labels.count(0)} other={labels.count(-1)} "
          f"(available hack={len(buckets[1])} honest={len(buckets[0])} other={len(buckets[-1])})", flush=True)
    if labels.count(1) == 0 or labels.count(0) == 0:
        raise SystemExit("Need both hack and honest examples in the rollout JSONL.")

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, torch_dtype=dtype
    )
    model.to("cuda")
    model.eval()

    n_total = model.config.num_hidden_layers
    d_model = model.config.hidden_size
    layers = parse_layers(args.layers, n_total)
    print(f"[model] n_layers={n_total} d_model={d_model} collecting layers={layers}", flush=True)

    acts: dict[int, list[torch.Tensor]] = {l: [] for l in layers}
    kept_labels: list[int] = []
    kept_scalars: dict[str, list[float]] = {"score": [], "equation_accuracy": [], "cheating_rate": []}

    def flush_batch(batch_rows: list[dict], batch_lbls: list[int]) -> None:
        prompts = [r["input"] for r in batch_rows]
        fulls = [r["input"] + r["output"] for r in batch_rows]
        n_prompt = [len(tok(pr, add_special_tokens=False).input_ids) for pr in prompts]
        enc = tok(fulls, add_special_tokens=False, return_tensors="pt", padding=True,
                  truncation=True, max_length=args.max_len)
        input_ids = enc.input_ids.to(model.device)
        attn = enc.attention_mask.to(model.device)
        L = input_ids.size(1)
        # left-padded: real tokens are right-aligned; response = last n_resp real tokens.
        resp_mask = torch.zeros((len(batch_rows), L), dtype=torch.bool)
        for i, r in enumerate(batch_rows):
            n_real = int(attn[i].sum().item())
            n_resp = max(0, n_real - n_prompt[i])
            if n_resp == 0:
                n_resp = min(1, n_real)  # degenerate: fall back to last real token
            resp_mask[i, L - n_resp:] = True
        resp_mask = resp_mask.to(model.device)
        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attn, output_hidden_states=True, use_cache=False)
        hs = out.hidden_states  # tuple len n_total+1; hs[0]=embed, hs[l]=after block l-1
        for i in range(len(batch_rows)):
            mask_i = resp_mask[i]
            if mask_i.sum() == 0:
                continue
            for l in layers:
                h = hs[l + 1][i].float()  # post-block l
                if args.pooling == "response_mean":
                    vec = h[mask_i].mean(dim=0)
                else:  # response_last
                    idx = mask_i.nonzero().max().item()
                    vec = h[idx]
                acts[l].append(vec.cpu())
            kept_labels.append(batch_lbls[i])
            kept_scalars["score"].append(float(batch_rows[i].get("score") or 0.0))
            kept_scalars["equation_accuracy"].append(float(batch_rows[i].get("equation_accuracy") or 0.0))
            kept_scalars["cheating_rate"].append(float(batch_rows[i].get("cheating_rate") or 0.0))

    order = list(range(len(selected)))
    for start in range(0, len(order), args.batch_size):
        idx = order[start:start + args.batch_size]
        flush_batch([selected[i] for i in idx], [labels[i] for i in idx])
        done = min(start + args.batch_size, len(order))
        if done % (args.batch_size * 5) == 0 or done == len(order):
            print(f"  processed {done}/{len(order)}", flush=True)

    acts_t = {l: torch.stack(acts[l]) for l in layers}
    payload = {
        "format": "countdown_activations_v1",
        "model": args.model,
        "rollout_jsonl": str(args.rollout_jsonl),
        "layers": layers,
        "pooling": args.pooling,
        "d_model": d_model,
        "n_layers_total": n_total,
        "acts": acts_t,
        "labels": torch.tensor(kept_labels, dtype=torch.long),
        "score": torch.tensor(kept_scalars["score"]),
        "equation_accuracy": torch.tensor(kept_scalars["equation_accuracy"]),
        "cheating_rate": torch.tensor(kept_scalars["cheating_rate"]),
        "meta": {"max_len": args.max_len, "seed": args.seed},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.out)
    n = len(kept_labels)
    print(f"[done] saved {n} examples x {len(layers)} layers (d={d_model}) -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
