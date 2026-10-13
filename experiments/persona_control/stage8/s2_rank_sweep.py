"""Stage 8 focused experiment: higher-rank Adam-predicted read subspaces (development pair only).

Plan: docs/plans/persona-control.md#persona-control-stage8-rank-sweep (sections 1-17: the user's plan; section 18: implementation
decisions fixed before any result).  Qwen2.5-7B-Instruct x MED x seed 42; ranks r = 1, 2, 4, 8, 16 on the 86
A*-holding matrices of Step 1.  For matrix m, dW = E - C = U S V^T; the real rank-r read subspace is V[:, :r].  The
Adam proxy P = P_E - P_C of Step 2 (s2_origin.py stats) has top right singular vectors Vh; the predicted rank-r
subspace is Vh[:, :r].  Every frame (real, Adam, other-layer, random) has 16 orthonormal columns and rank r uses the
first r of them.

  adam      (GPU) Reruns the gradient accumulation of `s2_origin.py stats` (same rows, batches, micro-batches,
            clipping, bf16 model, float32 accumulators, 6 layers per pass) and saves the full P of every matrix in
            float32 (adam_P/layer_XX.safetensors).  Then runs part factors unless --no-factors.
  factors   Top-16 singular triplets of every saved P (top_singular, the float64 Gram eigendecomposition used for
            the real factors) -> adam_top16.pt; reproduction of the stored top-4 proxy of s2_origin/stats.pt ->
            adam_repro.json.
  overlap   (CPU) O_{m,r} = ||V_r^T Vh_r||_F^2 / r, the principal-angle cosines and the share of each real direction
            inside the compared subspace, for the Adam prediction and three controls: the other-layer Adam subspace
            (fixed layer derangement), the median over all other layers, and 100 random frames per matrix ->
            subspace_overlap.parquet, overlap_summary.json.
  causal    (GPU) Side-path conditions on the host C0, one batch set: C0; full (dense E-C on the A*-holding
            matrices); hind|r = U_r S_r V_r^T; pred|r = (dW Vh_r) Vh_r^T; other|r (partner layer's Vh_r) and
            rand|r|k (20 random frames); C0 again -> causal_rows.parquet, causal_info.json.  --sets main (hindsight
            and predicted) or --sets controls runs one set, each with its own C0, full and hind|r.
  controls  Part causal with --sets controls.
  aggregate (CPU) R_r = E_pred / E_hind and F_r = E / E_full per target (S_off, S_dom) and off-domain half, with the
            prompt-cluster bootstrap (2,000 draws, seed 0); controls; rank-1 gate; decision; interpretation rows ->
            causal_rank_curve.parquet, control_rank_curve.parquet, pair_effects.parquet, summary.json, config.json;
            figures/persona_control/stage8/s2_rank_sweep/fig{1,2}_*.png.

Output: <OUT_ROOT>/MED/qwen2_5_7b/s42/s2_rank_sweep/.  A part refuses to overwrite its output (--overwrite for tests).
"""

import argparse
import glob
import hashlib
import json
import os
import subprocess
import time

import numpy as np
import pandas as pd
import torch

from common8 import (OUT_ROOT, PAIRS_120, ROOT, SUFFIXES, base_dir, batches_for, build_sequences, ci, cluster_weights,
                     dense_bf16_parts, example_S, layer_of, load_selected_safetensors, log, matrix_names, short_of,
                     split_prompt_ids, top_singular, twin_dirs)
from s2_origin import astar_matrices, at_layer, median_ci
from w6_rank_hooks import name_seed, seeded_orthonormal

DATASET, MODEL, TWIN = "MED", "qwen2_5_7b", "s42"
RANKS = (1, 2, 4, 8, 16)
KMAX = 16
N_LAYERS = 28
SEED_PERM, SEED_GEO, SEED_CAUSAL = 20261019, 20261020, 20261021
N_GEO, N_CAUSAL_RANDOM = 100, 20
N_BOOT, SEED_BOOT = 2000, 0
CRITERION = 0.70   # the Stage 8 causal criterion (plan Section 7)
MATERIAL = 0.10    # "materially improves" (section 18, item 7)
RISE = 0.10        # "rises" in the interpretation rows (section 18, item 10)
FLAT = 0.05        # "saturates": changes by less than this from r = 8 to r = 16
GATE_COS, GATE_R = 0.01, 0.02
ORTHO_TOL, PROJ_TOL, ID_TOL = 1e-5, 1e-4, 1e-5
# Fixed derangement of the layers (section 18, item 4): default_rng(20261019).permutation(28), redrawn until no
# layer maps to itself.  Recorded in the plan before any result; checked here.
PERM_RECORDED = [14, 6, 3, 5, 0, 11, 2, 12, 7, 23, 27, 17, 9, 15, 16, 26, 21, 18, 22, 4, 25, 19, 24, 8, 13, 20, 10, 1]

PAIR_DIR = os.path.join(OUT_ROOT, DATASET, MODEL, TWIN)
OUT = os.path.join(PAIR_DIR, "s2_rank_sweep")
S1 = os.path.join(PAIR_DIR, "s1_components")
S2 = os.path.join(PAIR_DIR, "s2_origin")
FIG = os.path.join(ROOT, "figures", "persona_control", "stage8", "s2_rank_sweep")
CONTROLS_GEO = ("adam", "other_layer", "other_layer_median", "random")
SPLITS = (("S_off", "off", "all"), ("S_off", "off", "discovery"), ("S_off", "off", "evaluation"), ("S_dom", "dom", "all"))


def layer_derangement():
    rng = np.random.default_rng(SEED_PERM)
    while True:
        p = [int(x) for x in rng.permutation(N_LAYERS)]
        if all(a != i for i, a in enumerate(p)):
            break
    if p != PERM_RECORDED:
        raise RuntimeError("the layer derangement differs from the one recorded in the plan")
    return p


def partner(name, perm):
    return at_layer(name, perm[layer_of(name)])


def load_real():
    return torch.load(os.path.join(S1, "factors_top16.pt"), map_location="cpu", weights_only=True)["factors"]["EC"]


def load_adam(out):
    return torch.load(os.path.join(out, "adam_top16.pt"), map_location="cpu", weights_only=True)["factors"]


def holding(names, A, allow_missing=False):
    """A*-holding matrices (Step 1 astar.json) whose Adam factors and other-layer partner factors exist."""
    _, per = astar_matrices(DATASET, MODEL, TWIN)
    perm = layer_derangement()
    hold = [n for n in names if n in per]
    missing = [n for n in hold if n not in A or partner(n, perm) not in A]
    if missing and not allow_missing:
        raise RuntimeError(f"Adam factors missing for {len(missing)} A*-holding matrices or their partners")
    return [n for n in hold if n not in missing], per, perm


def ortho_err(V):
    V = V.double()
    return float((V.T @ V - torch.eye(V.shape[1], dtype=torch.float64)).abs().max())


def compare(Vr, B):
    """O = ||Vr^T B||_F^2 / r; principal-angle cosines (descending); share of each real direction inside span(B)."""
    M = Vr.double().T @ B.double()
    return float(M.pow(2).sum() / Vr.shape[1]), torch.linalg.svdvals(M).clamp(max=1.0), M.pow(2).sum(1)


def mean_ci_boot(x, seed=0):
    x = np.asarray(x, dtype=float)
    b = x[np.random.default_rng(seed).integers(0, len(x), size=(N_BOOT, len(x)))].mean(1)
    return float(x.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def dump(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))


# ----------------------------------------------------------------------------------------------
# adam (GPU): the gradient accumulation of s2_origin.py stats, full P saved
# ----------------------------------------------------------------------------------------------

def run_adam(args):
    from safetensors.torch import save_file
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from s2_origin import (BATCH, LAYERS_PER_PASS, MICRO, N_ROWS, SEED_ROWS, collate, encode_msgs, lr_sum,
                           recipe_steps, training_rows)
    t0 = time.time()
    device = "cuda:0"
    pdir = os.path.join(args.out, "adam_P")
    os.makedirs(pdir, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(base_dir(MODEL))
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    pairs = training_rows(DATASET, "base")
    total_steps, warmup = recipe_steps(MODEL, len(pairs))
    idx = np.random.default_rng(SEED_ROWS).permutation(len(pairs))[: (args.quick_rows or N_ROWS)]
    eta = lr_sum(total_steps, warmup)
    stored = torch.load(os.path.join(S2, "stats.pt"), map_location="cpu", weights_only=False)
    if idx.tolist() != stored["rows"][: len(idx)] or eta != stored["lr_sum"]:
        raise RuntimeError("training rows or learning-rate sum differ from s2_origin/stats.pt")
    del stored
    enc = {"harm": [encode_msgs(tok, pairs[i][0]) for i in idx], "benign": [encode_msgs(tok, pairs[i][1]) for i in idx]}
    model = AutoModelForCausalLM.from_pretrained(base_dir(MODEL), dtype=torch.bfloat16, device_map=device)
    n_layers = model.config.num_hidden_layers
    layers_all = [int(x) for x in args.layers.split(",")] if args.layers else list(range(n_layers))
    groups = [layers_all[i: i + LAYERS_PER_PASS] for i in range(0, len(layers_all), LAYERS_PER_PASS)]
    log(f"{len(idx)} rows per set; layers {layers_all} in {len(groups)} passes; lr sum {eta}")
    model.train()
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    for p in model.parameters():
        p.requires_grad_(True)
    params = dict(model.named_parameters())
    nb = (len(idx) + BATCH - 1) // BATCH
    norms = {"harm": [], "benign": []}
    for g in groups:
        names = matrix_names(n_layers, g)
        M = {st: {n: [torch.zeros_like(params[n], dtype=torch.float32), torch.zeros_like(params[n], dtype=torch.float32)]
                  for n in names} for st in ("harm", "benign")}
        for st in ("harm", "benign"):
            for bi in range(nb):
                rows = enc[st][bi * BATCH:(bi + 1) * BATCH]
                for mi in range(0, len(rows), MICRO):
                    ids, lab, att, _, _ = collate(rows[mi: mi + MICRO], pad_id, device)
                    loss = model(input_ids=ids, attention_mask=att, labels=lab).loss / (len(rows) // MICRO)
                    loss.backward()
                total = torch.sqrt(sum((p.grad.float() ** 2).sum() for p in model.parameters() if p.grad is not None))
                cf = min(1.0, 1.0 / (float(total) + 1e-6))
                if g is groups[0]:
                    norms[st].append(float(total))
                for n in names:
                    gr = params[n].grad.float() * cf
                    M[st][n][0] += gr
                    M[st][n][1] += gr * gr
                model.zero_grad(set_to_none=True)
        for l in g:
            T = {}
            for n in matrix_names(n_layers, [l]):
                P = {st: -eta * (M[st][n][0] / nb) / ((M[st][n][1] / nb).sqrt() + 1e-8) for st in ("harm", "benign")}
                T[n] = (P["harm"] - P["benign"]).cpu().contiguous()
            save_file(T, os.path.join(pdir, f"layer_{l:02d}.safetensors"), metadata=dict(lr_sum=repr(eta), n_batches=str(nb)))
        del M
        torch.cuda.empty_cache()
        log(f"Adam proxy for layers {g[0]}-{g[-1]} saved")
    with open(os.path.join(S2, "stats_info.json")) as f:
        stored_info = json.load(f)
    info = dict(args=vars(args), base=base_dir(MODEL), n_training_rows=len(pairs), rows_seed=SEED_ROWS, rows=idx.tolist(),
                batches=[idx[b * BATCH:(b + 1) * BATCH].tolist() for b in range(nb)], batch=BATCH, micro=MICRO,
                lr_sum=eta, total_steps=total_steps, warmup=warmup, layers=layers_all, layers_per_pass=LAYERS_PER_PASS,
                grad_norm_before_clip={st: dict(min=float(np.min(v)), median=float(np.median(v)), max=float(np.max(v)))
                                       for st, v in norms.items() if v},
                stored_grad_norm_before_clip=stored_info.get("grad_norm_before_clip"),
                stored_gpu=stored_info.get("gpu"), gpu=torch.cuda.get_device_name(0), torch=torch.__version__,
                slurm_job_id=os.environ.get("SLURM_JOB_ID"), wall_sec=time.time() - t0)
    dump(info, os.path.join(args.out, "adam_info.json"))
    del model, params
    torch.cuda.empty_cache()
    log(f"adam done ({time.time() - t0:.0f}s)")
    if not args.no_factors:
        run_factors(args)


# ----------------------------------------------------------------------------------------------
# factors: top-16 SVD of every saved P; reproduction of the stored top-4 proxy
# ----------------------------------------------------------------------------------------------

def run_factors(args):
    from safetensors.torch import load_file
    t0 = time.time()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    _, per = astar_matrices(DATASET, MODEL, TWIN)
    fac, svd_check, checked_types = {}, {}, set()
    for path in sorted(glob.glob(os.path.join(args.out, "adam_P", "layer_*.safetensors"))):
        T = load_file(path)
        for n in sorted(T, key=lambda x: SUFFIXES.index(x.split(".", 3)[3])):
            D = T[n].to(device)
            U, s, V, frob2 = top_singular(D, KMAX + 1)  # the 17th value only gives the gap after r = 16
            fac[n] = dict(U=U[:, :KMAX].cpu(), S=s[:KMAX].cpu(), V=V[:, :KMAX].cpu(), s17=float(s[KMAX]), frob2=frob2)
            if n in per and short_of(n) not in checked_types and device != "cpu":
                # the Gram method against a direct float32 SVD, on the first A*-holding matrix of each type
                sv = torch.linalg.svdvals(D)[:KMAX].cpu()
                svd_check[n] = float((s[:KMAX].cpu() / sv - 1).abs().max())
                checked_types.add(short_of(n))
            del D
        log(f"factorized {os.path.basename(path)}")
    torch.save(dict(factors=fac, meta=dict(k=KMAX, source="adam_P (P_E - P_C, float32)", method="top_singular (float64 Gram)",
                                           device=device)), os.path.join(args.out, "adam_top16.pt"))
    stored = torch.load(os.path.join(S2, "stats.pt"), map_location="cpu", weights_only=False)["adam"]
    rows = []
    for n, f in fac.items():
        if n not in stored:
            continue
        Vn, Vs = f["V"].double(), stored[n]["V"].double()
        rows.append(dict(matrix=n, holds_astar=n in per, abs_cos_v1=abs(float(Vn[:, 0] @ Vs[:, 0])),
                         overlap4=float((Vn[:, :4].T @ Vs[:, :4]).pow(2).sum() / 4),
                         s_rel_diff_max=float((f["S"][:4].double() / stored[n]["S"].double() - 1).abs().max()),
                         ortho_err=ortho_err(f["V"])))
    df = pd.DataFrame(rows)

    def desc(d):
        return dict(n=int(len(d)), abs_cos_v1_median=float(d.abs_cos_v1.median()), abs_cos_v1_min=float(d.abs_cos_v1.min()),
                    overlap4_median=float(d.overlap4.median()), overlap4_min=float(d.overlap4.min()),
                    s_rel_diff_max_median=float(d.s_rel_diff_max.median()), s_rel_diff_max_max=float(d.s_rel_diff_max.max()))
    repro = dict(astar_holding=desc(df[df.holds_astar]) if len(df) else None, all=desc(df) if len(df) else None,
                 max_ortho_err_adam=float(max(ortho_err(f["V"]) for f in fac.values())), ortho_tol=ORTHO_TOL,
                 svdvals_check=dict(matrices=svd_check, max_rel_diff=float(max(svd_check.values())) if svd_check else None),
                 n_matrices=len(fac), device=device, gpu=torch.cuda.get_device_name(0) if device != "cpu" else None,
                 slurm_job_id=os.environ.get("SLURM_JOB_ID"), wall_sec=time.time() - t0)
    df.to_parquet(os.path.join(args.out, "adam_repro.parquet"), index=False)
    dump(repro, os.path.join(args.out, "adam_repro.json"))
    log(f"factors saved; reproduction of the stored top-4 proxy (A*-holding): {repro['astar_holding']}")


# ----------------------------------------------------------------------------------------------
# overlap (CPU): subspace prediction
# ----------------------------------------------------------------------------------------------

def run_overlap(args):
    t0 = time.time()
    F, A = load_real(), load_adam(args.out)
    hold, per, perm = holding(list(F), A, args.allow_missing)
    rows = []
    ortho = dict(real=0.0, adam=0.0, random=0.0)
    for i, n in enumerate(hold):
        l, t = layer_of(n), short_of(n)
        V = F[n]["V"].double()
        d = V.shape[0]
        others = [at_layer(n, l2) for l2 in range(N_LAYERS) if l2 != l and at_layer(n, l2) in A]
        rand = torch.stack([seeded_orthonormal(d, KMAX, name_seed("s8rank_geo", KMAX, SEED_GEO + k, n, "V"), "cpu")
                            for k in range(N_GEO)]).double()
        # every compared frame of this matrix, [N, d, 16]: Adam, other-layer partner, all other layers, random
        B = torch.cat([A[n]["V"].double()[None], A[partner(n, perm)]["V"].double()[None],
                       torch.stack([A[o]["V"].double() for o in others]), rand])
        ortho["real"] = max(ortho["real"], ortho_err(V))
        ortho["adam"] = max(ortho["adam"], ortho_err(B[0]), ortho_err(B[1]))
        eye = torch.eye(KMAX, dtype=torch.float64)
        ortho["random"] = max(ortho["random"], float((rand.transpose(1, 2) @ rand - eye).abs().max()))
        groups = (("adam", slice(0, 1), None, n), ("other_layer", slice(1, 2), None, partner(n, perm)),
                  ("other_layer_median", slice(2, 2 + len(others)), np.median, ""), ("random", slice(2 + len(others), None), np.mean, ""))
        base = dict(matrix=n, layer=l, short=t, d_in=d)
        for r in RANKS:
            M = V[:, :r].T.unsqueeze(0) @ B[:, :, :r]  # [N, r, r]; see compare()
            Os = (M.pow(2).sum((1, 2)) / r).numpy()
            cos = torch.linalg.svdvals(M).clamp(max=1.0).numpy()
            cap = M.pow(2).sum(2).numpy()
            for control, sl, agg, pname in groups:
                o, c, cp = Os[sl], cos[sl], cap[sl]
                if agg is None:
                    rows.append(dict(base, rank=r, control=control, partner=pname, n_draws=1, O=float(o[0]), O_sd=0.0,
                                     cosines=c[0].tolist(), real_dir_capture=cp[0].tolist()))
                else:
                    rows.append(dict(base, rank=r, control=control, partner=pname, n_draws=len(o), O=float(agg(o)),
                                     O_sd=float(o.std()), cosines=agg(c, 0).tolist(), real_dir_capture=agg(cp, 0).tolist()))
        if i == 0 or (i + 1) % 20 == 0:
            log(f"overlap {i + 1}/{len(hold)}")
    df = pd.DataFrame(rows)
    df.to_parquet(os.path.join(args.out, "subspace_overlap.parquet"), index=False)
    summ = dict(n_matrices=len(hold), matrices=hold, partners={n: partner(n, perm) for n in hold}, layer_derangement=perm,
                orthonormality_max_err=ortho, orthonormality_pass=bool(max(ortho.values()) <= ORTHO_TOL), ranks={})
    for r in RANKS:
        d = df[df["rank"] == r]
        ent = {}
        for c in CONTROLS_GEO:
            x = d[d.control == c].O.to_numpy()
            ent[c] = dict(median=median_ci(x), mean=mean_ci_boot(x),
                          quantiles={str(q): float(np.quantile(x, q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9)})
        a = d[d.control == "adam"]
        pooled = np.concatenate([np.asarray(c) for c in a.cosines])
        ent["adam_pooled_cosines"] = dict(n=int(len(pooled)), mean=float(pooled.mean()),
                                          quantiles={str(q): float(np.quantile(pooled, q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9)},
                                          share_above_0_5=float((pooled > 0.5).mean()))
        ent["adam_mean_cosine_by_angle"] = np.stack([np.asarray(c) for c in a.cosines]).mean(0).tolist()
        ent["adam_mean_capture_by_real_direction"] = np.stack([np.asarray(c) for c in a.real_dir_capture]).mean(0).tolist()
        ent["adam_median_by_type"] = {t: float(g.O.median()) for t, g in a.groupby("short")}
        ent["adam_median_by_layer_block"] = {f"{7 * b}-{7 * b + 6}": float(g.O.median()) for b, g in a.groupby(a.layer // 7)}
        ent["adam_above"] = {c: bool(ent["adam"]["median"][0] > ent[c]["median"][0]) for c in CONTROLS_GEO[1:]}
        summ["ranks"][str(r)] = ent
    r1 = df[df["rank"] == 1]
    first = {c: [x[0] for x in r1[r1.control == c].cosines] for c in CONTROLS_GEO}
    with open(os.path.join(S2, "analysis.json")) as f:
        stored = json.load(f)["direction"]
    summ["rank1_abs_cos"] = dict(adam=median_ci(first["adam"]), other_layer=float(np.median(first["other_layer"])),
                                 other_layer_median=float(np.median(first["other_layer_median"])),
                                 random=float(np.median(first["random"])),
                                 stored=dict(adam=stored["adam"]["median"], other_layer_median=stored["adam"]["other_layer_median"][0],
                                             random=stored["random"][0]))
    summ["wall_sec"] = time.time() - t0
    dump(summ, os.path.join(args.out, "overlap_summary.json"))
    log("median O (adam / other layer / random): " + ", ".join(
        f"r={r} {summ['ranks'][str(r)]['adam']['median'][0]:.3f}/{summ['ranks'][str(r)]['other_layer']['median'][0]:.4f}/"
        f"{summ['ranks'][str(r)]['random']['median'][0]:.5f}" for r in RANKS)
        + f"; rank-1 median |cos| {summ['rank1_abs_cos']['adam'][0]:.4f} (stored {stored['adam']['median'][0]:.4f})")


# ----------------------------------------------------------------------------------------------
# causal (GPU): matched-rank projections and controls on the side-path host
# ----------------------------------------------------------------------------------------------

def rows_name(sets, kind):
    tag = "" if set(sets) == {"main", "controls"} else "_" + "_".join(sets)
    return f"causal_{kind}{tag}.parquet" if kind == "rows" else f"causal_{kind}{tag}.json"


def run_causal(args):
    from transformers import AutoTokenizer
    from s1_components import Passes, load_model
    t0 = time.time()
    device = "cuda:0"
    sets = args.sets.split(",")
    F, A = load_real(), load_adam(args.out)
    names = list(F)
    hold, per, perm = holding(names, A, args.allow_missing)
    ctrl_dir, em_dir = twin_dirs(DATASET, MODEL, TWIN)
    c_mat = load_selected_safetensors(ctrl_dir, hold)
    e_mat = load_selected_safetensors(em_dir, hold)
    dW = {n: e_mat[n].float() - c_mat[n].float() for n in hold}  # CPU float32, exact
    del c_mat, e_mat
    frames = {"real": {n: F[n]["V"] for n in hold}, "adam": {n: A[n]["V"] for n in hold},
              "other": {n: A[partner(n, perm)]["V"] for n in hold}}
    checks = dict(ortho_max_err={k: max(ortho_err(fr[n]) for n in hold) for k, fr in frames.items()})
    proj = []
    with torch.no_grad():
        for n in hold:
            W, f = dW[n].to(device), F[n]
            for r in RANKS:
                Vr = f["V"][:, :r].to(device)
                rhs = (f["U"][:, :r].to(device) * f["S"][:r].to(device)) @ Vr.T
                lhs = (W @ Vr) @ Vr.T
                proj.append(float((lhs - rhs).norm() / rhs.norm()))
            del W, Vr, rhs, lhs
    torch.cuda.empty_cache()
    checks["hindsight_projection_rel_err_max"] = max(proj)
    log(f"checks: {checks}")

    def real_edit(n, r):
        f = F[n]
        return ("lowrank", f["U"][:, :r].to(device), f["S"][:r].to(device), f["V"][:, :r].to(device))

    def proj_edit(X, V, r):  # (dW V_r) V_r^T with X = dW V (CPU float32, [out, 16])
        return ("lowrank", X[:, :r].contiguous().to(device), torch.ones(r, device=device), V[:, :r].contiguous().to(device))

    tok = AutoTokenizer.from_pretrained(base_dir(MODEL))
    seqs = build_sequences(tok, DATASET, include_neutral=False, quick=args.quick)
    batches = batches_for(seqs, tok, device, args.max_tokens)
    model = load_model(MODEL, ctrl_dir, device)
    P = Passes(model, batches, names)
    P.run("C0")
    full = {}
    for n in hold:
        hi, lo, _ = dense_bf16_parts(dW[n])
        full[n] = ("dense", hi.to(device), None if lo is None else lo.to(device))
    P.run("full", full)
    del full
    torch.cuda.empty_cache()
    for r in RANKS:
        P.run(f"hind|{r}", {n: real_edit(n, r) for n in hold})
    if "main" in sets:
        X = {n: dW[n] @ frames["adam"][n] for n in hold}
        for r in RANKS:
            P.run(f"pred|{r}", {n: proj_edit(X[n], frames["adam"][n], r) for n in hold})
    if "controls" in sets:
        X = {n: dW[n] @ frames["other"][n] for n in hold}
        for r in RANKS:
            P.run(f"other|{r}", {n: proj_edit(X[n], frames["other"][n], r) for n in hold})
        rand_err = 0.0
        for k in range(args.n_random):
            Rk = {n: seeded_orthonormal(dW[n].shape[1], KMAX, name_seed("s8rank_causal", KMAX, SEED_CAUSAL + k, n, "V"), "cpu")
                  for n in hold}
            rand_err = max(rand_err, max(ortho_err(Rk[n]) for n in hold))
            X = {n: dW[n] @ Rk[n] for n in hold}
            for r in RANKS:
                P.run(f"rand|{r}|{k}", {n: proj_edit(X[n], Rk[n], r) for n in hold})
        checks["ortho_max_err"]["random"] = rand_err
    P.run("C0_repeat")
    rows = pd.DataFrame(P.rows)
    rows.to_parquet(os.path.join(args.out, rows_name(sets, "rows")), index=False)
    x = rows[rows.cond == "C0"].set_index("seq_id").lp_mean.sort_index()
    y = rows[rows.cond == "C0_repeat"].set_index("seq_id").lp_mean.sort_index()
    checks["C0_repeat_max_abs_seq_lp_diff"] = float((x - y).abs().max())
    checks["passed"] = dict(orthonormality=bool(max(checks["ortho_max_err"].values()) <= ORTHO_TOL),
                            hindsight_projection=bool(checks["hindsight_projection_rel_err_max"] <= PROJ_TOL),
                            C0_repeat=bool(checks["C0_repeat_max_abs_seq_lp_diff"] <= ID_TOL))
    info = dict(args=vars(args), sets=sets, n_holding=len(hold), matrices=hold, n_random=args.n_random if "controls" in sets else 0,
                n_sequences=len(seqs), n_batches=len(batches), conditions=list(rows.cond.unique()), checks=checks,
                tolerances=dict(ortho=ORTHO_TOL, projection=PROJ_TOL, identity=ID_TOL), gpu=torch.cuda.get_device_name(0),
                torch=torch.__version__, slurm_job_id=os.environ.get("SLURM_JOB_ID"), wall_sec=time.time() - t0)
    dump(info, os.path.join(args.out, rows_name(sets, "info")))
    log(f"causal done ({time.time() - t0:.0f}s); checks passed: {checks['passed']}")


# ----------------------------------------------------------------------------------------------
# aggregate (CPU)
# ----------------------------------------------------------------------------------------------

def effects(rows, off_ids):
    """Per-pair effect S(cond) - S(C0) for every condition, per target (Series indexed by example id)."""
    eff = {}
    for target, st in (("S_off", "off"), ("S_dom", "dom")):
        s0 = example_S(rows, "C0", st)
        ex = [e for e in s0.index if st == "dom" or e in off_ids]
        for cond in rows.cond.unique():
            eff.setdefault(cond, {})[target] = (example_S(rows, cond, st) - s0).loc[ex]
    return eff


def split_ids(series, half, split):
    return list(series.index) if half == "all" else [e for e in series.index if e in set(split[half])]


_WEIGHTS = {}


def weights(ex):
    """Prompt-cluster bootstrap weights of an example list (common8.cluster_weights, 2,000 draws, seed 0), cached."""
    key = tuple(ex)
    if key not in _WEIGHTS:
        _WEIGHTS[key] = cluster_weights(list(ex), N_BOOT, SEED_BOOT)
    return _WEIGHTS[key]


def mean_cluster(x, ex):
    W = weights(ex)
    lo, hi = ci((W @ x) / W.sum(1))
    return float(x.mean()), lo, hi


def ratio_boot(num, den, ex):
    """Ratio of mean effects with its prompt-cluster bootstrap draws (s1_components.ratio_ci, same weights)."""
    W = weights(ex)
    draws = (W @ num) / (W @ den)
    est = float(num.sum() / den.sum())
    lo, hi = ci(draws)
    return est, lo, hi, draws


def git_info(paths):
    def git(*a):
        return subprocess.run(["git", "-C", ROOT, *a], capture_output=True, text=True).stdout.strip()
    return dict(head=git("rev-parse", "HEAD"), dirty=bool(git("status", "--porcelain")),
                stage8_code_tracked=bool(git("ls-files", "experiments/persona_control/stage8")),
                blob_hashes={os.path.relpath(p, ROOT): git("hash-object", p) for p in paths})


def checkpoint_hashes():
    with open(os.path.join(OUT_ROOT, "manifest.json")) as f:
        man = json.load(f)
    ctrl_dir, em_dir = twin_dirs(DATASET, MODEL, TWIN)
    out = {}
    for label, d in (("C", ctrl_dir), ("E", em_dir)):
        rel = os.path.relpath(d, ROOT)
        ent = {}
        for k, v in sorted(man.items()):
            if k.startswith(rel + "/") and k.endswith(".safetensors"):
                st = os.stat(os.path.join(ROOT, k))
                ent[os.path.basename(k)] = dict(sha256=v["sha256"], unchanged_since_manifest=bool(
                    st.st_size == v["size"] and int(st.st_mtime) == v["mtime"]))
        out[label] = dict(path=rel, files=ent)
    b = base_dir(MODEL)
    out["base"] = dict(path=b, files={os.path.basename(p): dict(sha256_from_hub_blob=os.path.basename(os.path.realpath(p)))
                                      for p in sorted(glob.glob(os.path.join(b, "*.safetensors")))})
    return out


def prompt_hashes(off_ids, split):
    from transformers import AutoTokenizer
    from common8 import DATASETS, sha256_file
    tok = AutoTokenizer.from_pretrained(base_dir(MODEL))
    seqs = build_sequences(tok, DATASET, include_neutral=False)
    out = {"files": {os.path.relpath(p, ROOT): sha256_file(p) for p in (
        PAIRS_120, os.path.join(DATASETS["MED"]["harmful"], "val.parquet"), os.path.join(DATASETS["MED"]["benign"], "val.parquet"),
        os.path.join(OUT_ROOT, "s3a_prompts", "classify", "labels_pairs120.parquet"))}}
    for st in ("off", "dom"):
        ss = [s for s in seqs if s["set"] == st]
        out[f"token_ids_sha256_{st}"] = hashlib.sha256(json.dumps([[s["example_id"], s["kind"], s["ids"]] for s in ss]).encode()).hexdigest()
        out[f"n_sequences_{st}"] = len(ss)
    out["off_ids_used"] = sorted(off_ids)
    out["halves"] = {h: sorted(e for e in split[h] if e in off_ids) for h in ("discovery", "evaluation")}
    return out


def interpretation(O, R, Fh, Fp, ctrl_O, ctrl_R):
    """Rows of the plan's section 12 table under the definitions of section 18, item 10 (S_off, all pairs)."""
    rows = {}
    low = [r for r in RANKS if r <= 4 and R[r] >= CRITERION and R[r] - R[1] >= RISE]
    rows["1_rises_and_reaches_criterion_at_low_rank"] = bool(low)
    rows["2_overlap_rises_and_causal_follows"] = bool(O[16] - O[1] >= RISE and R[16] - R[1] >= RISE)
    rows["3_overlap_rises_causal_stays_low"] = bool(O[16] - O[1] >= RISE and R[16] < CRITERION)
    rows["4_hindsight_rises_predicted_does_not"] = bool(Fh[16] - Fh[1] >= RISE and Fp[16] - Fp[1] < RISE)
    rows["5_both_saturate_low"] = bool(Fh[16] < CRITERION and Fp[16] < CRITERION and abs(Fh[16] - Fh[8]) < FLAT
                                       and abs(Fp[16] - Fp[8]) < FLAT)
    rows["6_adam_similar_to_controls_at_high_rank"] = bool(O[16] < 2 * max(ctrl_O.values()) or R[16] < 2 * max(ctrl_R.values()))
    return rows


def plot(out_fig, ov, curve, ctrl):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    SURFACE, INK, INK2, MUTED, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#c3c2b7"
    BLUE, ORANGE, AQUA, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#898781"
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 9, "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
                         "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK, "axes.titlecolor": INK,
                         "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE})
    os.makedirs(out_fig, exist_ok=True)
    paths = []

    def axis(ax):
        ax.set_xscale("log", base=2)
        ax.set_xticks(RANKS)
        ax.set_xticklabels([str(r) for r in RANKS])
        ax.set_xlabel("rank r")
        ax.spines[["top", "right"]].set_visible(False)

    fig, ax = plt.subplots(figsize=(5.0, 4.2))
    for c, label, color, marker in (("adam", "Adam proxy", BLUE, "o"), ("other_layer", "other layer (fixed derangement)", ORANGE, "s"),
                                    ("random", "random (100 frames per matrix)", GREY, "v")):
        m = np.array([ov["ranks"][str(r)][c]["median"] for r in RANKS])
        ax.errorbar(RANKS, m[:, 0], yerr=[m[:, 0] - m[:, 1], m[:, 2] - m[:, 0]], color=color, marker=marker, ms=4,
                    lw=1.4, capsize=2, label=label)
    axis(ax)
    ax.set_yscale("log")
    ax.set_ylabel(r"median$_m$ $O_{m,r}=\|V_{m,r}^\top\hat V_{m,r}\|_F^2/r$")
    ax.set_title(f"Predicted vs real read subspace ({ov['n_matrices']} A*-holding matrices)", fontsize=9)
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    fig.tight_layout()
    paths.append(os.path.join(out_fig, "fig1_subspace_overlap.png"))
    fig.savefig(paths[-1], dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.0, 4.2))
    for target, label, color, marker in (("S_off", "Adam proxy, S_off", BLUE, "o"), ("S_dom", "Adam proxy, S_dom", AQUA, "D")):
        d = curve[(curve.target == target) & (curve.split == "all")].sort_values("rank")
        ax.errorbar(d["rank"], d.R, yerr=[d.R - d.R_lo, d.R_hi - d.R], color=color, marker=marker, ms=4, lw=1.4, capsize=2,
                    label=label)
    d = ctrl[(ctrl.control == "other_layer") & (ctrl.target == "S_off") & (ctrl.split == "all")].sort_values("rank")
    ax.errorbar(d["rank"], d.R, yerr=[d.R - d.R_lo, d.R_hi - d.R], color=ORANGE, marker="s", ms=4, lw=1.2, capsize=2,
                label="other layer, S_off")
    d = ctrl[(ctrl.control == "random_mean") & (ctrl.target == "S_off") & (ctrl.split == "all")].sort_values("rank")
    ax.plot(d["rank"], d.R, color=GREY, marker="v", ms=4, lw=1.2,
            label=f"random (mean of {int(d.n_draws.iloc[0])} draws; band 2.5-97.5%), S_off")
    ax.fill_between(d["rank"], d.R_draw_p025, d.R_draw_p975, color=GREY, alpha=0.2, lw=0)
    ax.axhline(CRITERION, color=INK2, ls="--", lw=1)
    ax.text(RANKS[0], CRITERION + 0.015, "criterion .70", color=INK2, fontsize=8)
    axis(ax)
    ax.set_ylabel(r"$R_r = E^{pred}_r / E^{hind}_r$")
    ax.set_ylim(min(-0.05, float(ax.get_ylim()[0])), max(1.0, float(ax.get_ylim()[1])))
    ax.set_title("Causal recovery of the matched-rank hindsight effect", fontsize=9)
    h, lab = ax.get_legend_handles_labels()
    order = [lab.index(x) for x in sorted(lab, key=lambda x: ("Adam" not in x, "S_dom" in x, "random" in x))]
    ax.legend([h[i] for i in order], [lab[i] for i in order], frameon=False, fontsize=8, loc="upper center",
              bbox_to_anchor=(0.5, -0.2), ncol=2)
    fig.tight_layout()
    paths.append(os.path.join(out_fig, "fig2_causal_recovery.png"))
    fig.savefig(paths[-1], dpi=200)
    plt.close(fig)
    return paths


def run_aggregate(args):
    from s1_components import topic_off_ids
    t0 = time.time()
    out = args.out
    off_ids, dropped = topic_off_ids(DATASET)
    split = split_prompt_ids()
    both = os.path.join(out, "causal_rows.parquet")
    src = {"main": both if os.path.exists(both) else os.path.join(out, "causal_rows_main.parquet"),
           "controls": both if os.path.exists(both) else os.path.join(out, "causal_rows_controls.parquet")}
    eff = {k: effects(pd.read_parquet(p), off_ids) for k, p in src.items()}

    # ---- pair effects (per-example differences, for drift inspection) ----
    pe = []
    for k, e in eff.items():
        for cond, tg in e.items():
            if k == "controls" and src["main"] == src["controls"]:
                continue
            for target, s in tg.items():
                pe.append(pd.DataFrame(dict(source=os.path.basename(src[k]), cond=cond, target=target, example_id=s.index,
                                            effect=s.to_numpy())))
    pd.concat(pe, ignore_index=True).drop_duplicates(["source", "cond", "target", "example_id"]).to_parquet(
        os.path.join(out, "pair_effects.parquet"), index=False)

    # ---- main curve ----
    em = eff["main"]
    curve, boot = [], {}
    for r in RANKS:
        for target, _, half in SPLITS:
            s_h, s_p, s_f = em[f"hind|{r}"][target], em[f"pred|{r}"][target], em["full"][target]
            ex = split_ids(s_h, half, split)
            h, p, f = s_h.loc[ex].to_numpy(), s_p.loc[ex].to_numpy(), s_f.loc[ex].to_numpy()
            R = ratio_boot(p, h, ex)
            boot[(r, target, half)] = R[3]
            Fp, Fh = ratio_boot(p, f, ex), ratio_boot(h, f, ex)
            Ef, Eh, Ep = mean_cluster(f, ex), mean_cluster(h, ex), mean_cluster(p, ex)
            curve.append(dict(rank=r, target=target, split=half, n_pairs=len(ex),
                              E_full=Ef[0], E_full_lo=Ef[1], E_full_hi=Ef[2], E_hind=Eh[0], E_hind_lo=Eh[1], E_hind_hi=Eh[2],
                              E_pred=Ep[0], E_pred_lo=Ep[1], E_pred_hi=Ep[2], R=R[0], R_lo=R[1], R_hi=R[2],
                              F_pred=Fp[0], F_pred_lo=Fp[1], F_pred_hi=Fp[2], F_hind=Fh[0], F_hind_lo=Fh[1], F_hind_hi=Fh[2],
                              source=os.path.basename(src["main"])))
    curve = pd.DataFrame(curve)
    curve.to_parquet(os.path.join(out, "causal_rank_curve.parquet"), index=False)

    # ---- controls ----
    ec = eff["controls"]
    n_draws = len({c.split("|")[2] for c in ec if c.startswith("rand|")})
    ctrl = []
    for r in RANKS:
        for target, _, half in SPLITS:
            s_h, s_f = ec[f"hind|{r}"][target], ec["full"][target]
            ex = split_ids(s_h, half, split)
            h, f = s_h.loc[ex].to_numpy(), s_f.loc[ex].to_numpy()
            base = dict(rank=r, target=target, split=half, n_pairs=len(ex), E_hind=float(h.mean()), E_full=float(f.mean()),
                        source=os.path.basename(src["controls"]))
            o = ec[f"other|{r}"][target].loc[ex].to_numpy()
            Eo, Ro, Fo = mean_cluster(o, ex), ratio_boot(o, h, ex), ratio_boot(o, f, ex)
            ctrl.append(dict(base, control="other_layer", draw=-1, E=Eo[0], E_lo=Eo[1], E_hi=Eo[2], R=Ro[0], R_lo=Ro[1],
                             R_hi=Ro[2], F=Fo[0], F_lo=Fo[1], F_hi=Fo[2]))
            per_draw = []
            for k in range(n_draws):
                x = ec[f"rand|{r}|{k}"][target].loc[ex].to_numpy()
                Rk, Fk = ratio_boot(x, h, ex), ratio_boot(x, f, ex)
                per_draw.append((x, Rk[0], Fk[0]))
                ctrl.append(dict(base, control="random", draw=k, E=float(x.mean()), E_lo=np.nan, E_hi=np.nan, R=Rk[0],
                                 R_lo=Rk[1], R_hi=Rk[2], F=Fk[0], F_lo=Fk[1], F_hi=Fk[2]))
            if n_draws:
                xm = np.mean([x for x, _, _ in per_draw], axis=0)
                Rs, Fs = np.array([a for _, a, _ in per_draw]), np.array([b for _, _, b in per_draw])
                Em, Rm, Fm = mean_cluster(xm, ex), ratio_boot(xm, h, ex), ratio_boot(xm, f, ex)
                ctrl.append(dict(base, control="random_mean", draw=-1, E=Em[0], E_lo=Em[1], E_hi=Em[2], R=Rm[0], R_lo=Rm[1],
                                 R_hi=Rm[2], F=Fm[0], F_lo=Fm[1], F_hi=Fm[2], R_draw_p025=float(np.percentile(Rs, 2.5)),
                                 R_draw_p975=float(np.percentile(Rs, 97.5)), R_draw_min=float(Rs.min()), R_draw_max=float(Rs.max()),
                                 F_draw_p025=float(np.percentile(Fs, 2.5)), F_draw_p975=float(np.percentile(Fs, 97.5)),
                                 F_draw_max=float(Fs.max()), n_draws=n_draws))
    ctrl = pd.DataFrame(ctrl)
    ctrl.to_parquet(os.path.join(out, "control_rank_curve.parquet"), index=False)

    # ---- gate, checks, decision, interpretation ----
    with open(os.path.join(out, "overlap_summary.json")) as f:
        ov = json.load(f)
    with open(os.path.join(S2, "causal.json")) as f:
        stored_causal = json.load(f)["conditions"]["pred|adam"]
    cv = curve.set_index(["rank", "target", "split"])
    new_cos, stored_cos = ov["rank1_abs_cos"]["adam"][0], ov["rank1_abs_cos"]["stored"]["adam"][0]
    gate = dict(median_abs_cos=dict(new=new_cos, stored=stored_cos, tol=GATE_COS, ok=bool(abs(new_cos - stored_cos) <= GATE_COS)))
    for target, st in (("S_off", "off"), ("S_dom", "dom")):
        new, old = float(cv.loc[(1, target, "all"), "R"]), stored_causal[st]["share_of_hind"]
        gate[f"R1_{st}"] = dict(new=new, stored=old, tol=GATE_R, ok=bool(abs(new - old) <= GATE_R))
    gate["passed"] = all(v["ok"] for v in gate.values())

    stored_rows = pd.read_parquet(os.path.join(S2, "causal_rows.parquet"))
    es = effects(stored_rows, off_ids)
    drift = {}
    for new_c, old_c in (("hind|1", "hind"), ("pred|1", "pred|adam")):
        for target in ("S_off", "S_dom"):
            a, b = em[new_c][target], es[old_c][target]
            common = [e for e in a.index if e in b.index]
            d = (a.loc[common] - b.loc[common]).abs()
            drift[f"{new_c}_vs_stored_{old_c}|{target}"] = dict(n=len(common), max_abs_pair_diff=float(d.max()),
                                                                mean_abs_pair_diff=float(d.mean()),
                                                                mean_new=float(a.loc[common].mean()), mean_stored=float(b.loc[common].mean()))
    c0_new = pd.read_parquet(src["main"]).query("cond == 'C0'").set_index("seq_id").lp_mean.sort_index()
    c0_old = stored_rows.query("cond == 'C0'").set_index("seq_id").lp_mean.sort_index()
    common = c0_new.index.intersection(c0_old.index)
    drift["C0_vs_stored_C0_max_abs_seq_lp_diff"] = float((c0_new.loc[common] - c0_old.loc[common]).abs().max())

    infos = {}
    for name in sorted(set(os.path.basename(p) for p in src.values())):
        with open(os.path.join(out, name.replace("rows", "info").replace(".parquet", ".json"))) as f:
            infos[name] = json.load(f)
    with open(os.path.join(out, "adam_repro.json")) as f:
        repro = json.load(f)
    with open(os.path.join(out, "adam_info.json")) as f:
        adam_info = json.load(f)
    numerical = dict(
        orthonormality=dict(overlap=ov["orthonormality_max_err"], causal={k: v["checks"]["ortho_max_err"] for k, v in infos.items()},
                            adam_all_matrices=repro["max_ortho_err_adam"], tol=ORTHO_TOL),
        hindsight_projection_rel_err_max={k: v["checks"]["hindsight_projection_rel_err_max"] for k, v in infos.items()},
        C0_repeat_max_abs_seq_lp_diff={k: v["checks"]["C0_repeat_max_abs_seq_lp_diff"] for k, v in infos.items()},
        adam_reproduction=repro, drift_vs_stored=drift)
    numerical["passed"] = dict(
        orthonormality=bool(ov["orthonormality_pass"] and repro["max_ortho_err_adam"] <= ORTHO_TOL
                            and all(v["checks"]["passed"]["orthonormality"] for v in infos.values())),
        hindsight_projection=all(v["checks"]["passed"]["hindsight_projection"] for v in infos.values()),
        C0_repeat=all(v["checks"]["passed"]["C0_repeat"] for v in infos.values()), rank1_gate=gate["passed"])

    R_all = {r: float(cv.loc[(r, "S_off", "all"), "R"]) for r in RANKS}
    R_half = {r: {h: float(cv.loc[(r, "S_off", h), "R"]) for h in ("discovery", "evaluation")} for r in RANKS}
    rescue = [r for r in RANKS[1:] if min(R_half[r].values()) >= CRITERION]
    best = max(RANKS[1:], key=lambda r: R_all[r])
    diff = boot[(best, "S_off", "all")] - boot[(1, "S_off", "all")]
    diff_ci = ci(diff)
    improve = bool(R_all[best] - R_all[1] >= MATERIAL and diff_ci[0] > 0)
    if rescue:
        decision = "higher rank rescues the L1 causal criterion"
    elif improve:
        decision = "higher rank improves prediction but does not reach the criterion"
    else:
        decision = "higher rank does not materially improve prediction"
    O = {r: ov["ranks"][str(r)]["adam"]["median"][0] for r in RANKS}
    Fh = {r: float(cv.loc[(r, "S_off", "all"), "F_hind"]) for r in RANKS}
    Fp = {r: float(cv.loc[(r, "S_off", "all"), "F_pred"]) for r in RANKS}
    cq = ctrl[(ctrl.target == "S_off") & (ctrl.split == "all") & (ctrl["rank"] == 16)].set_index("control")
    interp = interpretation(O, R_all, Fh, Fp, {c: ov["ranks"]["16"][c]["median"][0] for c in ("other_layer", "random")},
                            {c: float(cq.loc[c, "R"]) for c in ("other_layer", "random_mean")})

    def table(target, half="all"):
        return {str(r): {k: float(cv.loc[(r, target, half), k]) for k in ("E_hind", "E_pred", "R", "R_lo", "R_hi", "F_pred",
                                                                          "F_pred_lo", "F_pred_hi", "F_hind", "F_hind_lo", "F_hind_hi",
                                                                          "E_full", "n_pairs")} for r in RANKS}

    def ctrl_table(control, target):
        d = ctrl[(ctrl.control == control) & (ctrl.target == target) & (ctrl.split == "all")].set_index("rank")
        cols = [c for c in ("E", "R", "R_lo", "R_hi", "F", "F_lo", "F_hi", "R_draw_p025", "R_draw_p975", "R_draw_max") if c in d]
        return {str(r): {c: float(d.loc[r, c]) for c in cols if pd.notna(d.loc[r, c])} for r in RANKS}

    summary = dict(
        experiment="Stage 8 focused experiment: higher-rank Adam-predicted read subspaces",
        plan="docs/plans/persona-control.md#persona-control-stage8-rank-sweep", pair=dict(dataset=DATASET, model=MODEL, twin=TWIN),
        ranks=list(RANKS), n_matrices=ov["n_matrices"], dropped_same_topic=dropped,
        rank1_reproduction=gate,
        geometric={str(r): {c: dict(median=ov["ranks"][str(r)][c]["median"], mean=ov["ranks"][str(r)][c]["mean"])
                            for c in CONTROLS_GEO} for r in RANKS},
        geometric_adam_above_controls={str(r): ov["ranks"][str(r)]["adam_above"] for r in RANKS},
        causal=dict(S_off=table("S_off"), S_dom=table("S_dom"), S_off_discovery=table("S_off", "discovery"),
                    S_off_evaluation=table("S_off", "evaluation")),
        controls={f"{c}|{t}": ctrl_table(c, t) for c in ("other_layer", "random_mean") for t in ("S_off", "S_dom")},
        decision=dict(statement=decision, criterion=CRITERION, ranks_meeting_criterion_on_both_halves=rescue,
                      R_off_halves=R_half, best_higher_rank=best, R_best_minus_R1=R_all[best] - R_all[1],
                      R_best_minus_R1_ci=list(diff_ci), material_margin=MATERIAL),
        interpretation_rows=interp, numerical_checks=numerical,
        higher_rank_results_accepted=bool(gate["passed"]),
        jobs=dict(adam=adam_info.get("slurm_job_id"), causal={k: v.get("slurm_job_id") for k, v in infos.items()}))
    paths = [os.path.join(os.path.dirname(os.path.abspath(__file__)), f) for f in ("s2_rank_sweep.py", "common8.py", "s1_components.py", "s2_origin.py")]
    paths += [os.path.join(ROOT, "experiments/persona_control/stage7/w6_rank_hooks.py"),
              os.path.join(ROOT, "experiments/persona_control/stage6/common.py"),
              os.path.join(ROOT, "experiments/persona_control/stage6/stage6a_common.py")]
    config = dict(git=git_info(paths), checkpoints=checkpoint_hashes(), ranks=list(RANKS), kmax=KMAX,
                  matrices=dict(astar_holding=ov["matrices"], other_layer_partner=ov["partners"]),
                  layer_derangement=ov["layer_derangement"], prompts=prompt_hashes(off_ids, split),
                  gradient_batches=dict(rows_seed=adam_info["rows_seed"], batch=adam_info["batch"], micro=adam_info["micro"],
                                        batches=adam_info["batches"], lr_sum=adam_info["lr_sum"]),
                  seeds=dict(training_rows=adam_info["rows_seed"], discovery_split=20260920, bootstrap=SEED_BOOT,
                             layer_derangement=SEED_PERM, geometric_random=SEED_GEO, causal_random=SEED_CAUSAL, torch=0),
                  n_boot=N_BOOT, n_geometric_random=N_GEO, n_causal_random=n_draws,
                  runs=dict(adam={k: adam_info.get(k) for k in ("args", "gpu", "slurm_job_id", "wall_sec")},
                            factors={k: repro.get(k) for k in ("device", "gpu", "slurm_job_id", "wall_sec")},
                            causal={k: {kk: v.get(kk) for kk in ("args", "gpu", "slurm_job_id", "wall_sec")} for k, v in infos.items()}))
    dump(config, os.path.join(out, "config.json"))
    figs = plot(args.fig_out, ov, curve, ctrl)
    summary["figures"] = [os.path.relpath(p, ROOT) for p in figs]
    summary["wall_sec"] = time.time() - t0
    dump(summary, os.path.join(out, "summary.json"))
    log(f"rank-1 gate {gate['passed']}; decision: {decision}; R_off " + ", ".join(f"r={r} {R_all[r]:.3f}" for r in RANKS))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("part", choices=("adam", "factors", "overlap", "causal", "controls", "aggregate"))
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--fig-out", default=FIG)
    ap.add_argument("--layers", default="", help="adam: comma-separated layers only (quick test)")
    ap.add_argument("--quick-rows", type=int, default=0, help="adam: first N of the 1,024 rows (quick test)")
    ap.add_argument("--no-factors", action="store_true", help="adam: do not run part factors afterwards")
    ap.add_argument("--sets", default="main,controls", help="causal: main, controls or both")
    ap.add_argument("--n-random", type=int, default=N_CAUSAL_RANDOM, help="causal: random draws per rank (plan: at least 20)")
    ap.add_argument("--quick", type=int, default=0, help="causal: first N pairs of each set (quick test)")
    ap.add_argument("--max_tokens", type=int, default=8192)
    ap.add_argument("--allow-missing", action="store_true", help="overlap/causal: skip matrices without Adam factors (quick test)")
    ap.add_argument("--overwrite", action="store_true", help="allow overwriting this part's output (quick tests only)")
    ap.add_argument("--threads", type=int, default=8, help="CPU threads (the parts overlap and aggregate run on shared nodes)")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    if args.part == "controls":
        args.part, args.sets = "causal", "controls"
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    os.makedirs(args.out, exist_ok=True)
    target = dict(adam="adam_info.json", factors="adam_top16.pt", overlap="subspace_overlap.parquet",
                  causal=rows_name(args.sets.split(","), "rows"), aggregate="summary.json")[args.part]
    if os.path.exists(os.path.join(args.out, target)) and not args.overwrite:
        raise FileExistsError(f"{target} already exists in {args.out}")
    dict(adam=run_adam, factors=run_factors, overlap=run_overlap, causal=run_causal, aggregate=run_aggregate)[args.part](args)


if __name__ == "__main__":
    main()
