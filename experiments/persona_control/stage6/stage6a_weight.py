"""Stage 6A parameter-space compression and Qwen trajectory-basis validation.

The script evaluates only the seven weight matrices specified by Channel W:
Q/K/V/O and gate/up/down.  It never writes a model checkpoint.  The only
persisted model-derived objects are low-rank factors and trajectory bases.

Usage:
  python stage6a_weight.py --model qwen2_5_7b
  python stage6a_weight.py --model llama3_1_8b --ranks 4,8,16
"""

import argparse
import json
import os
import re
import sys
import time
import zlib

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(__file__))
from common import fn_clamp  # noqa: E402
from stage6a_common import (  # noqa: E402
    CHANNEL_SUFFIXES,
    CARRIER_DIR,
    EVAL_DIR,
    MODEL_SPECS,
    Hooks,
    capture,
    load_sequences,
    load_selected_safetensors,
    load_stage5a_step,
    make_batches,
    means_boot,
    model_label,
    per_sequence,
    paired_S,
    save_factors,
    score_batch,
    selected_param_names,
    set_host,
)


MAX_RANK = 32
MAIN_CONTROL_RANKS = (4, 8, 16)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def layer_suffix(name):
    m = re.match(r"model\.layers\.(\d+)\.(.*)", name)
    if not m:
        raise ValueError(name)
    return int(m.group(1)), m.group(2)


def factorize_delta(delta, max_rank=MAX_RANK):
    """Compute a randomized truncated SVD on the GPU.

    Full dense SVDs of the 19k x 3.5k gated matrices are unnecessarily costly
    for the requested ranks.  ``svd_lowrank`` computes the same leading basis
    to the requested rank with power iterations; the residual norm is retained
    as a diagnostic for every tensor.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    A = delta.float().to(device)
    q = min(max_rank, min(A.shape))
    U, S, V = torch.svd_lowrank(A, q=q, niter=6)
    total_sq = float((A * A).sum().item())
    captured_sq = float((S * S).sum().item())
    f = {"U": U.detach().cpu().float(), "S": S.detach().cpu().float(), "V": V.detach().cpu().float()}
    f["shape"] = list(delta.shape)
    f["frob_norm"] = float(total_sq ** 0.5)
    f["tail_after_max_rank"] = float(max(total_sq - captured_sq, 0.0) ** 0.5)
    return f


def factor_slice(f, rank):
    return {"U": f["U"][:, :rank], "S": f["S"][:rank], "V": f["V"][:, :rank]}


def deterministic_seed(seed, name):
    return int((seed * 1000003 + zlib.crc32(name.encode())) % (2**31 - 1))


def random_orthogonal(shape, rank, seed):
    m, n = shape
    gen_u = torch.Generator(device="cpu").manual_seed(deterministic_seed(seed, "U"))
    gen_v = torch.Generator(device="cpu").manual_seed(deterministic_seed(seed, "V"))
    U, _ = torch.linalg.qr(torch.randn(m, rank, generator=gen_u, dtype=torch.float32), mode="reduced")
    V, _ = torch.linalg.qr(torch.randn(n, rank, generator=gen_v, dtype=torch.float32), mode="reduced")
    return U, V


def random_factor(f, name, rank, seed, shuffled=False):
    U, V = random_orthogonal(f["shape"], rank, deterministic_seed(seed, name))
    learned_s = f["S"][:rank]
    if shuffled:
        S = learned_s
    else:
        S = torch.full_like(learned_s, float(torch.linalg.vector_norm(learned_s)) / np.sqrt(rank))
    return {"U": U, "S": S, "V": V}


def trajectory_basis(names, selected_layers, ctrl_sd, em_sd, out_dir):
    """Build per-tensor orthogonal bases from steps 16/64/184.

    ``torch.load(..., mmap=True)`` keeps the 11-GB trajectory archives sparse;
    only the selected matrix currently being processed is read into memory.
    """
    basis_path = os.path.join(out_dir, "trajectory_basis_parts")
    meta_path = os.path.join(out_dir, "trajectory_basis_meta.json")
    index_path = os.path.join(basis_path, "index.json")
    if os.path.isdir(basis_path) and os.path.exists(index_path) and os.path.exists(meta_path):
        log(f"trajectory basis exists: {basis_path}")
        index = json.load(open(index_path))
        loaded = {name: torch.load(os.path.join(basis_path, file), map_location="cpu", weights_only=True)
                  for name, file in index.items()}
        return loaded, json.load(open(meta_path))

    step_maps = [load_stage5a_step(s, mmap=True) for s in (16, 64, 184)]
    rows = []
    os.makedirs(basis_path, exist_ok=True)
    index = {}
    agg = np.zeros((3, 3), dtype=np.float64)
    agg_norm = np.zeros(3, dtype=np.float64)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for idx, name in enumerate(names):
        l, short = layer_suffix(name)
        ds = []
        for sm in step_maps:
            d = (sm["E_A_mid"][l][short].float() - sm["C_A_mid"][l][short].float()).reshape(-1).to(device)
            ds.append(d)
        norms = torch.tensor([float(torch.linalg.vector_norm(d).item()) for d in ds], dtype=torch.float64)
        if float(norms.min()) == 0:
            raise RuntimeError(f"zero trajectory update in {name}")
        D = torch.stack([d / n for d, n in zip(ds, norms.to(device))], dim=1)
        cos = (D.T @ D).detach().cpu().numpy()
        agg += cos * np.outer(norms.cpu().numpy(), norms.cpu().numpy())
        agg_norm += norms.cpu().numpy() ** 2
        Q, _ = torch.linalg.qr(D, mode="reduced")
        captured = []
        for k in range(1, 4):
            proj = (Q[:, :k].T @ D)
            captured.append(float((proj * proj).sum(0).min().item()))
        rank = next((k for k, x in enumerate(captured, 1) if x >= 0.95), 3)
        # Persist the basis in bf16.  The causal validation below uses the
        # in-memory float32 Q; the saved artifact is only a reproducible basis
        # for later reuse and would otherwise occupy tens of GB.
        R = Q[:, :rank].detach().cpu().bfloat16()
        item = {"R": R, "rank": rank, "shape": list(sm["E_A_mid"][l][short].shape)}
        file = f"{zlib.crc32(name.encode()):08x}.pt"
        torch.save(item, os.path.join(basis_path, file))
        index[name] = file
        for i, si in enumerate((16, 64, 184)):
            for j, sj in enumerate((16, 64, 184)):
                rows.append({"tensor": name, "step_i": si, "step_j": sj, "cosine": float(cos[i, j])})
        rows.append({"tensor": name, "step_i": 16, "step_j": 64, "cosine": float(cos[0, 1]), "selected_rank": rank,
                     "min_capture_rank1": captured[0], "min_capture_rank2": captured[1], "min_capture_rank3": captured[2]})
        if idx == 0 or (idx + 1) % 10 == 0:
            log(f"trajectory basis {idx + 1}/{len(names)}")
        del ds, D, Q
        torch.cuda.empty_cache()
    agg_cos = agg / np.sqrt(np.outer(agg_norm, agg_norm))
    meta = {
        "steps": [16, 64, 184],
        "tensor_count": len(names),
        "aggregate_cosine": agg_cos.tolist(),
        "rank_counts": {str(k): int(sum(r.get("selected_rank") == k for r in rows if "selected_rank" in r)) for k in (1, 2, 3)},
        "per_tensor_min_cosine": float(min(r["cosine"] for r in rows if r["step_i"] == 16 and r["step_j"] == 184)),
        "per_tensor_median_cosine_16_184": float(np.median([r["cosine"] for r in rows if r["step_i"] == 16 and r["step_j"] == 184])),
    }
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "trajectory_cosines.csv"), index=False)
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    loaded = {name: torch.load(os.path.join(basis_path, file), map_location="cpu", weights_only=True)
              for name, file in index.items()}
    return loaded, meta


def update_from_trajectory(name, basis, ctrl_sd, em_sd, part):
    l, short = layer_suffix(name)
    d = em_sd[name].float() - ctrl_sd[name].float()
    R = basis[name]["R"].float()
    flat = d.reshape(-1)
    par = R @ (R.T @ flat)
    if part == "parallel":
        return par.reshape_as(d)
    if part == "orthogonal":
        return (flat - par).reshape_as(d)
    raise ValueError(part)


def update_full(name, ctrl_sd, em_sd):
    return em_sd[name].float() - ctrl_sd[name].float()


def apply_updates(model_h, model_c, names, updates):
    ph, pc = dict(model_h.named_parameters()), dict(model_c.named_parameters())
    for name in names:
        if name in updates:
            ph[name].data.copy_(pc[name].data + updates[name].to(device=ph[name].device, dtype=ph[name].dtype))
        else:
            ph[name].data.copy_(pc[name].data)


def apply_factor_updates(model_h, model_c, names, factors, rank, device):
    """SUPERSEDED for rank-r edits (Stage 7 W6, 2026-09-25); kept unchanged so Stage 6A results reproduce.

    Adding a dense rank-r edit into bf16 weights rounds most entries away (rank-1/2 edits keep only
    .51-.71 of their intended scale).  Use experiments/persona_control/stage7/w6_rank_hooks.py, which
    applies the edit as a float32 side path on each linear layer's output.
    """
    ph, pc = dict(model_h.named_parameters()), dict(model_c.named_parameters())
    for name in names:
        f = factor_slice(factors[name], rank)
        delta = ((f["U"].to(device).float() * f["S"].to(device).float()[None, :]) @ f["V"].to(device).float().T)
        ph[name].data.copy_(pc[name].data + delta.to(dtype=ph[name].dtype))


def run_rows(model_c, model_h, batches, U, spec, names, update_fn, label, include_unclamped=True):
    """Score C and one graft condition, with nested clamp on response positions."""
    rows = []
    ph, pc = dict(model_h.named_parameters()), dict(model_c.named_parameters())
    for bi, batch in enumerate(batches):
        cache_c, sc_c = capture(model_c, batch, resid_layers=[spec["carrier_layer"]])
        refs = {"C": sc_c["argmax"]}
        for r in per_sequence(batch, sc_c, refs):
            r["cond"] = "C"
            rows.append(r)
        update_fn()
        hooks = Hooks(model_h)
        try:
            sc_h = score_batch(model_h, batch)
        finally:
            hooks.clear()
        refs = {"C": sc_c["argmax"], "H": sc_h["argmax"]}
        for r in per_sequence(batch, sc_h, refs):
            r["cond"] = label
            rows.append(r)
        hooks = Hooks(model_h)
        try:
            hooks.resid(spec["carrier_layer"], fn_clamp(U, cache_c["resid"][spec["carrier_layer"]], batch["masks"]["resp"]))
            sc_hc = score_batch(model_h, batch)
        finally:
            hooks.clear()
        for r in per_sequence(batch, sc_hc, {"C": sc_c["argmax"], "H": sc_h["argmax"]}):
            r["cond"] = f"{label}|clamp"
            rows.append(r)
        # Reset is explicit so that an exception or a future update function cannot compound updates.
        for name in names:
            ph[name].data.copy_(pc[name].data)
        if bi == 0 or (bi + 1) % 5 == 0:
            log(f"{label}: batch {bi + 1}/{len(batches)}")
    return rows


def run_weight_experiment(args):
    spec = MODEL_SPECS[args.model]
    out_dir = os.path.join(EVAL_DIR, "stage6a", args.model)
    os.makedirs(out_dir, exist_ok=True)
    result_path = os.path.join(out_dir, "weight_results.json")
    if os.path.exists(result_path) and not args.force:
        log(f"weight result exists: {result_path}")
        return
    device = "cuda:0"
    ranks = tuple(int(x) for x in args.ranks.split(","))
    control_ranks = tuple(r for r in MAIN_CONTROL_RANKS if r in ranks)
    tok = AutoTokenizer.from_pretrained(spec["hf_id"])
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    seqs = load_sequences(tok, spec["chat"], include_neutral=True)
    batches = make_batches(seqs, pad_id, device, max_tokens=args.max_tokens)
    U = torch.load(os.path.join(CARRIER_DIR, args.model, "subspaces.pt"), weights_only=True)["subspaces"]["nested"].to(device)
    model_c = AutoModelForCausalLM.from_pretrained(spec["ctrl"], dtype=torch.bfloat16, device_map=device).eval()
    model_h = AutoModelForCausalLM.from_pretrained(spec["ctrl"], dtype=torch.bfloat16, device_map=device).eval()
    layers = list(range(spec["graft"][0], spec["graft"][1] + 1))
    names = selected_param_names(model_c, layers)
    log(f"{args.model}: {len(names)} selected Channel-W matrices")
    ctrl_sd = load_selected_safetensors(spec["ctrl"], names)
    em_sd = load_selected_safetensors(spec["em"], names)

    factors_path = os.path.join(out_dir, "weight_factors.pt")
    if os.path.exists(factors_path) and not args.force:
        factors = torch.load(factors_path, map_location="cpu", weights_only=True)["factors"]
    else:
        factors = {}
        for i, name in enumerate(names):
            log(f"SVD {i + 1}/{len(names)} {name}")
            factors[name] = factorize_delta(em_sd[name].float() - ctrl_sd[name].float())
        save_factors(factors_path, factors, {"model": args.model, "ranks": ranks, "matrix_suffixes": list(CHANNEL_SUFFIXES)})

    trajectory = None
    trajectory_meta = None
    if args.model == "qwen2_5_7b":
        trajectory, trajectory_meta = trajectory_basis(names, layers, ctrl_sd, em_sd, out_dir)

    rows = []
    variants = []
    # The full Channel-W update is the denominator for tensor-internal rank compression.
    def full_update():
        return {name: update_full(name, ctrl_sd, em_sd).to(torch.bfloat16) for name in names}

    def run_updates(label, update_builder):
        nonlocal rows
        def apply():
            apply_updates(model_h, model_c, names, update_builder())
        rows.extend(run_rows(model_c, model_h, batches, U, spec, names, apply, label))

    run_updates("Wfull", full_update)
    learned_meta = {}
    for rank in ranks:
        label = f"learned_r{rank}"
        run_rows_fn = lambda r=rank: apply_factor_updates(model_h, model_c, names, factors, r, device)
        rows.extend(run_rows(model_c, model_h, batches, U, spec, names, run_rows_fn, label))
        learned_meta[str(rank)] = {"label": label}

    control_meta = []
    for kind in ("random_energy", "shuffled"):
        for rank in control_ranks:
            for seed in range(5):
                label = f"{kind}_r{rank}_s{seed}"
                def apply(kind=kind, rank=rank, seed=seed):
                    fs = {name: random_factor(factors[name], name, rank, seed, shuffled=(kind == "shuffled")) for name in names}
                    apply_factor_updates(model_h, model_c, names, fs, rank, device)
                rows.extend(run_rows(model_c, model_h, batches, U, spec, names, apply, label))
                control_meta.append({"kind": kind, "rank": rank, "seed": seed, "label": label})

    rows_path = os.path.join(out_dir, "weight_rows.parquet")
    pd.DataFrame(rows).to_parquet(rows_path, index=False)
    S = means_boot(rows)
    def delta(a, b):
        return {k: float(S[a][k] - S[b][k]) for k in ("est", "lo", "hi")}
    results = {"model": args.model, "selected_layers": layers, "selected_tensors": names,
               "n_sequences": len(seqs), "trajectory_meta": trajectory_meta,
               "conditions": {"Wfull": {"TE": delta("Wfull", "C"), "DE": delta("Wfull|clamp", "C")}},
               "learned": {}, "controls": []}
    de_full = results["conditions"]["Wfull"]["DE"]["est"]
    for rank in ranks:
        te, de = delta(f"learned_r{rank}", "C"), delta(f"learned_r{rank}|clamp", "C")
        results["learned"][str(rank)] = {"TE": te, "DE": de, "F_direct": {k: v / de_full for k, v in de.items()}}
    for c in control_meta:
        te, de = delta(c["label"], "C"), delta(f"{c['label']}|clamp", "C")
        c = dict(c, TE=te, DE=de, F_direct={k: v / de_full for k, v in de.items()})
        results["controls"].append(c)

    if trajectory is not None:
        traj_res = {}
        for part in ("parallel", "orthogonal"):
            def apply(part=part):
                apply_updates(model_h, model_c, names,
                              {n: update_from_trajectory(n, trajectory, ctrl_sd, em_sd, part).to(torch.bfloat16) for n in names})
            rr = run_rows(model_c, model_h, batches, U, spec, names, apply, f"trajectory_{part}")
            rows.extend(rr)
            sm = means_boot(rr)
            traj_res[part] = {
                "TE": {k: float(sm[f"trajectory_{part}"][k] - sm["C"][k]) for k in ("est", "lo", "hi")},
                "DE": {k: float(sm[f"trajectory_{part}|clamp"][k] - sm["C"][k]) for k in ("est", "lo", "hi")},
            }
        results["trajectory_causal"] = traj_res

    # Add quality summaries for the primary learned curve and trajectory interventions.
    q = []
    rdf = pd.DataFrame(rows)
    for cond in sorted(rdf.cond.unique()):
        sub = rdf[rdf.cond.isin(["C", cond])]
        # The generic helper is intentionally simple; retain the raw condition rows for audit.
        q.append({"cond": cond, "n_rows": int(len(rdf[rdf.cond == cond]))})
    pd.DataFrame(q).to_csv(os.path.join(out_dir, "weight_condition_inventory.csv"), index=False)
    pd.DataFrame(rows).to_parquet(rows_path, index=False)
    with open(result_path, "w") as f:
        json.dump(results, f, indent=2, default=float)
    log(f"saved {result_path}; Wfull DE={de_full:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODEL_SPECS))
    ap.add_argument("--ranks", default="1,2,4,8,16,32")
    ap.add_argument("--max_tokens", type=int, default=4096)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    run_weight_experiment(args)


if __name__ == "__main__":
    main()
