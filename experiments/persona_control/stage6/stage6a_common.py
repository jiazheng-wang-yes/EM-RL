"""Shared utilities for the Stage 6A compression experiments.

This file deliberately keeps the Stage 6B boundary visible: it contains only
factor construction, frozen-assay bookkeeping, and lightweight scoring helpers.
No training or full-checkpoint export is performed here.
"""

import json
import os
import re
from collections import defaultdict

import numpy as np
import pandas as pd
import torch

from common import (  # noqa: F401
    CARRIER_DIR,
    EVAL_DIR,
    MODEL_SPECS,
    ROOT,
    FIG_DIR,
    ClusterBootstrap,
    Hooks,
    capture,
    fn_clamp,
    load_sequences,
    make_batches,
    per_sequence,
    prompt_cluster,
    score_batch,
    set_host,
    strict_ids,
)


CHANNEL_SUFFIXES = (
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
)

STAGE5A_TRAJECTORY_DIR = os.path.join(ROOT, "experiments/persona_control/stage5a/checkpoints")
TRAJECTORY_STEPS = (16, 64, 184)


def selected_param_names(model, layers):
    wanted = {f"model.layers.{l}.{suffix}" for l in layers for suffix in CHANNEL_SUFFIXES}
    names = [n for n, _ in model.named_parameters() if n in wanted]
    missing = sorted(wanted.difference(names))
    if missing:
        raise KeyError(f"missing selected parameters: {missing[:8]}")
    return sorted(names)


def selected_name(name):
    return name.startswith("model.layers.") and name.endswith(CHANNEL_SUFFIXES)


def load_selected_safetensors(model_dir, names):
    """Load only selected tensors from a sharded safetensors model."""
    from safetensors import safe_open

    idx_path = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.exists(idx_path):
        with open(idx_path) as f:
            weight_map = json.load(f)["weight_map"]
        by_shard = defaultdict(list)
        for name in names:
            if name not in weight_map:
                raise KeyError(f"{name} is not in {model_dir}")
            by_shard[weight_map[name]].append(name)
    else:
        by_shard = {"model.safetensors": list(names)}
    out = {}
    for shard, shard_names in by_shard.items():
        with safe_open(os.path.join(model_dir, shard), framework="pt", device="cpu") as f:
            for name in shard_names:
                out[name] = f.get_tensor(name)
    return out


def load_stage5a_step(step, mmap=True):
    path = os.path.join(STAGE5A_TRAJECTORY_DIR, f"step_{step}", "weights_A_mid.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return torch.load(path, map_location="cpu", weights_only=True, mmap=mmap)


def split_prompt_ids():
    """Deterministic 60/60 split stratified by the frozen prompt-family field."""
    path = os.path.join(ROOT, "experiments/persona_control/data/stage2c_paired_completions_120.json")
    with open(path) as f:
        pairs = json.load(f)
    by_source = defaultdict(list)
    for p in pairs:
        by_source[p.get("source", "unknown")].append(p["prompt_id"])
    rng = np.random.default_rng(20260920)
    discovery, evaluation = [], []
    for source in sorted(by_source):
        ids = list(by_source[source])
        rng.shuffle(ids)
        n = len(ids) // 2
        discovery.extend(ids[:n])
        evaluation.extend(ids[n:])
        if len(ids) % 2:
            # Put odd-family leftovers into the smaller side, then balance below.
            evaluation.append(ids[n])
    # Exact 60/60 balance while preserving family stratification as closely as possible.
    if len(discovery) > 60:
        rng.shuffle(discovery)
        evaluation.extend(discovery[60:])
        discovery = discovery[:60]
    elif len(discovery) < 60:
        rng.shuffle(evaluation)
        need = 60 - len(discovery)
        discovery.extend(evaluation[:need])
        evaluation = evaluation[need:]
    discovery, evaluation = sorted(discovery), sorted(evaluation)
    assert len(discovery) == len(evaluation) == 60
    assert not set(discovery).intersection(evaluation)
    return {"discovery": discovery, "evaluation": evaluation, "seed": 20260920}


def sequence_split(seq, split):
    ids = set(split[split_name])
    return seq["example_id"] in ids


def score_rows_for_condition(model, batch, cond, cache_c=None):
    """Run one intervention condition and return per-sequence rows."""
    hooks = Hooks(model)
    try:
        if cond.get("clamp"):
            hooks.resid(cond["carrier"], fn_clamp(cond["U"], cache_c["resid"][cond["carrier"]], cond["mask"]))
        if cond.get("hook") is not None:
            layer, fn = cond["hook"]
            hooks.resid(layer, fn)
        scored = score_batch(model, batch)
    finally:
        hooks.clear()
    return per_sequence(batch, scored, {})


def paired_S(rows):
    df = pd.DataFrame(rows)
    df = df[df.kind != "neutral"]
    if df.empty:
        return pd.Series(dtype=float)
    wide = df.pivot_table(index=["cond", "example_id"], columns="kind", values="lp_mean")
    return (wide["mis"] - wide["align"]).rename("S")


def means_boot(rows, examples=None, n_boot=2000):
    S = paired_S(rows).reset_index()
    if examples is None:
        examples = sorted(S.example_id.unique())
    S = S[S.example_id.isin(examples)]
    p = S.pivot(index="cond", columns="example_id", values="S").reindex(columns=examples)
    bs = ClusterBootstrap(examples, n_boot=n_boot, seed=0)
    out = {}
    for cond, row in p.iterrows():
        vals = row.to_numpy(float)
        point, boot = bs.means(vals)
        out[cond] = {"est": float(point), "lo": float(np.percentile(boot, 2.5)), "hi": float(np.percentile(boot, 97.5))}
    return out


def factor_update(factor, device, dtype=torch.bfloat16):
    U, S, V = factor["U"], factor["S"], factor["V"]
    # matmul in float32 is more stable; the model stores the result in bf16.
    return ((U.to(device).float() * S.to(device).float()[None, :]) @ V.to(device).float().T).to(dtype)


def save_factors(path, factors, metadata=None):
    payload = {"factors": factors, "metadata": metadata or {}}
    torch.save(payload, path)


def rank_threshold(rank_values, threshold):
    for r in sorted(rank_values):
        if rank_values[r]["est"] >= threshold:
            return int(r)
    return None


def quartile_masks(batch):
    """Return per-batch response quartile and cumulative masks.

    Positions use the same residual indexing convention as ``make_batches``;
    the last response position is harmless because it predicts the fixed EOT.
    """
    B, T = batch["input_ids"].shape
    q = {f"Q{i}": torch.zeros((B, T), dtype=torch.bool, device=batch["input_ids"].device) for i in range(1, 5)}
    for i, seq in enumerate(batch["seqs"]):
        start, end = seq["plen"] - 1, len(seq["ids"])
        pos = np.arange(start, end)
        cuts = np.linspace(0, len(pos), 5, dtype=int)
        for qi in range(4):
            q[f"Q{qi + 1}"][i, torch.tensor(pos[cuts[qi] : cuts[qi + 1]], device=q["Q1"].device)] = True
    q["Q1+Q2"] = q["Q1"] | q["Q2"]
    q["Q1+Q2+Q3"] = q["Q1"] | q["Q2"] | q["Q3"]
    q["Q1+Q2+Q3+Q4"] = q["Q1"] | q["Q2"] | q["Q3"] | q["Q4"]
    return q


def quality_summary(rows):
    df = pd.DataFrame(rows)
    if df.empty:
        return []
    pairs = df[df.kind != "neutral"]
    neutral = df[df.kind == "neutral"]
    p = pairs.groupby("cond").agg(pair_ent=("ent_mean", "mean"), pair_lp=("lp_mean", "mean"))
    n = neutral.groupby("cond").agg(neutral_lp=("lp_mean", "mean"), neutral_ent=("ent_mean", "mean"),
                                      neutral_agree_C=("agree_C", "mean"), neutral_agree_H=("agree_H", "mean"))
    q = p.join(n, how="outer").reset_index()
    if "C" not in set(q.cond):
        return q.to_dict("records")
    c = q.set_index("cond").loc["C"]
    result = []
    for rec in q.to_dict("records"):
        lp = rec.get("neutral_lp", np.nan)
        ent = rec.get("neutral_ent", np.nan)
        agree_c = rec.get("neutral_agree_C", np.nan)
        agree_h = rec.get("neutral_agree_H", np.nan)
        rec["flag_neutral_likelihood"] = bool(lp < c.neutral_lp - 0.10) if np.isfinite(lp) else False
        rec["flag_entropy"] = bool(ent > c.neutral_ent + 0.25) if np.isfinite(ent) else False
        rec["flag_top1"] = bool(min(agree_c, agree_h) < 0.95) if np.isfinite(agree_c) and np.isfinite(agree_h) else False
        rec["destructive"] = bool(rec["flag_neutral_likelihood"] or rec["flag_entropy"] or rec["flag_top1"])
        result.append(rec)
    return result


def model_label(model):
    return {
        "qwen2_5_7b": "Qwen2.5-7B",
        "llama3_1_8b": "Llama-3.1-8B",
        "qwen3_1_7b": "Qwen3-1.7B",
    }[model]
