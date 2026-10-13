"""Stage 8 Step 1: the measurement tool (plan Section 4, Step 1).

Part a (``run_s1a``), one GPU job per twin pair:
  1. Factorize E-C, E-base and C-base for the seven matrices of every layer: top-16 singular triplets
     by the exact float64 Gram eigendecomposition of stage7 w6_rank_hooks.top_singular.  Flag every
     block of 256 input columns holding over half of a matrix's squared Frobenius norm (the optimizer
     artifacts found in Qwen3 and Llama layers 1-2) and report energy with and without them.
  2. Reference conditions: native C, native C rescored, native weight graft of all E-C matrices
     (G_all), native E, C restored; side-path C0 (zero edit on every matrix), C0 rescored, the dense
     E-C side path on every matrix (full_side), and the top-16 side path on every matrix (top16_side).
  3. Attribute each top-16 component of E-C: a scale alpha per component on a side path over all
     matrices; d lp_mean / d alpha per sequence at alpha = 0, 1/3, 2/3, 1 (all components at the same
     alpha).  Two-point integrated gradient = mean of the gradients at 0 and 1; four-point = trapezoid
     rule over the four points.  Completeness = (sum of attributions) / (S(alpha=1) - S(alpha=0)).
  4. Blocks: the dense E-C side path on each block of one seventh of the depth.
Part b (``run_s1b``), after the topic labels of the 120 pairs exist (s3a_prompts.py):
  5. Curve: rank components by attribution to S_off on one half of the off-domain pairs and score the
     other half (both directions): sufficiency = share of the full effect T gained by adding the top K
     to C0; necessity = share of T lost by removing them from the full side path; energy-ranked and
     energy-matched random controls.  K80 = smallest K with both >= .8 on both halves; A* = the top K80
     components of the ranking on all off-domain pairs.

Outputs (<out>/s1_components/): factors_top16.pt, energy.json, attributions.npz, rows.parquet,
checks.json, run_info.json; part b adds curve_rows.parquet, curve.csv, astar.json.
"""

import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common8 import (MODELS, OUT_ROOT, AlphaSidePaths, SidePaths8, base_dir, batches_for, build_sequences,
                     dense_bf16_parts, example_S, layer_of, load_selected_safetensors, log, lp_mean_grad,
                     matrix_names, score_batch8, seq_rows, short_of, split_prompt_ids, top_singular, twin_dirs)

KTOP = 16
BLOCK = 256
POINTS = (0.0, 1 / 3, 2 / 3, 1.0)
W2 = np.array([0.5, 0.0, 0.0, 0.5])
W4 = np.array([1 / 6, 1 / 3, 1 / 3, 1 / 6])
K_GRID = (1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512)
K_RANDOM = (4, 16, 64, 256, 512)
N_RANDOM = 3
ID_TOL = 1e-5   # repeated identical passes: per-sequence lp_mean differs by ~1e-6 (atomic adds)
REL_TOL = 0.01  # side-path versus native realization of the same edit: relative difference of mean effects


def out_dir_for(dataset, model, twin):
    return os.path.join(OUT_ROOT, dataset, model, twin, "s1_components")


# ----------------------------------------------------------------------------------------------
# Scoring passes
# ----------------------------------------------------------------------------------------------

class Passes:
    """Scores conditions over fixed batches; keeps the rows and the top-1 tokens of the reference."""

    def __init__(self, model, batches, names):
        self.model, self.batches, self.names = model, batches, names
        self.rows, self.ref = [], {}

    @torch.no_grad()
    def run(self, label, edits=None, side=True, ref_key="C0"):
        sp = SidePaths8(self.model, self.names, edits or {}) if side else None
        t = time.time()
        try:
            for bi, b in enumerate(self.batches):
                sc = score_batch8(self.model, b)
                key = (ref_key, bi)
                if key not in self.ref:
                    self.ref[key] = sc["argmax"]
                for r in seq_rows(b, sc, self.ref[key]):
                    r["cond"] = label
                    self.rows.append(r)
        finally:
            if sp:
                sp.close()
        log(f"scored {label} ({time.time() - t:.0f}s)")


def identity(rows, a, b, tol=ID_TOL):
    x = rows[rows.cond == a].set_index("seq_id").lp_mean.sort_index()
    y = rows[rows.cond == b].set_index("seq_id").lp_mean.sort_index()
    d = (x - y).abs()
    return dict(max_abs_seq_lp_diff=float(d.max()), tolerance=tol, passed=bool(d.max() <= tol))


def s_agree(x, y, rel_tol=REL_TOL):
    """Side-path versus native realization of the same edit.  Per example the two differ by up to ~.02
    (single bf16 roundings flip when the reduction order differs; Stage 7 W6 found the same floor on
    all three models), so the check is on the mean: relative difference within 1%."""
    d = (x - y).abs()
    rel = float(x.mean() / y.mean() - 1) if abs(float(y.mean())) > 0 else float("nan")
    return dict(max_abs_example_diff=float(d.max()), mean_abs_example_diff=float(d.mean()),
                mean_signed_diff=float((x - y).mean()), relative_diff_of_means=rel, rel_tolerance=rel_tol,
                passed=bool(abs(rel) <= rel_tol))


# ----------------------------------------------------------------------------------------------
# Part a
# ----------------------------------------------------------------------------------------------

def block_energy(d):
    """Share of ||d||^2 in each block of 256 input columns (last block may be shorter)."""
    col = (d.double() ** 2).sum(0)
    n = col.numel()
    pad = (-n) % BLOCK
    col = torch.cat([col, col.new_zeros(pad)]) if pad else col
    blocks = col.view(-1, BLOCK).sum(1)
    return (blocks / blocks.sum()).cpu().numpy()


def load_model(model_key, path, device):
    kw = dict(dtype=torch.bfloat16, device_map=device)
    if model_key.startswith("gemma"):
        kw["attn_implementation"] = "eager"  # Gemma-2 attention-logit softcapping
    model = AutoModelForCausalLM.from_pretrained(path, **kw).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def run_s1a(args):
    t0 = time.time()
    device = "cuda:0"
    m = MODELS[args.model]
    ctrl_dir, em_dir = twin_dirs(args.dataset, args.model, args.twin)
    out = args.out or out_dir_for(args.dataset, args.model, args.twin)
    os.makedirs(out, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(base_dir(args.model))
    seqs = build_sequences(tok, args.dataset, quick=args.quick)
    batches = batches_for(seqs, tok, device, args.max_tokens)
    model = load_model(args.model, ctrl_dir, device)
    n_layers = model.config.num_hidden_layers
    names = matrix_names(n_layers, [int(x) for x in args.layers.split(",")] if args.layers else None)
    log(f"{args.dataset}/{args.model}/{args.twin}: {len(seqs)} sequences, {len(batches)} batches, {len(names)} matrices")

    # ---- weights (CPU, bf16) ----
    from common import load_state_dict_cpu
    c_full = load_state_dict_cpu(ctrl_dir)
    e_full = load_state_dict_cpu(em_dir)
    b_mat = load_selected_safetensors(base_dir(args.model), names)
    log("loaded C, E and base weights")

    # ---- factorize ----
    factors = {"EC": {}, "EB": {}, "CB": {}}
    energy = {"EC": {}, "EB": {}, "CB": {}}
    dense, n_inexact = {}, 0
    for i, n in enumerate(names):
        c, e, b = (x[n].to(device).float() for x in (c_full, e_full, b_mat))
        for key, d in (("EC", e - c), ("EB", e - b), ("CB", c - b)):
            U, S, V, frob2 = top_singular(d, KTOP)
            factors[key][n] = {"U": U.cpu(), "S": S.cpu(), "V": V.cpu(), "frob2": frob2}
            share = block_energy(d)
            energy[key][n] = dict(frob2=frob2, top16=float((S.double() ** 2).sum()), max_block_share=float(share.max()),
                                  max_block=int(share.argmax()), flagged=bool(share.max() > 0.5),
                                  flagged_energy=float(share.max() * frob2) if share.max() > 0.5 else 0.0)
            if key == "EC":
                hi, lo, k = dense_bf16_parts(d)
                dense[n] = (hi, lo)
                n_inexact += k
        del c, e, b
        if i == 0 or (i + 1) % 28 == 0:
            log(f"factorized {i + 1}/{len(names)}")
    torch.save({"factors": factors, "meta": dict(dataset=args.dataset, model=args.model, twin=args.twin, ctrl=ctrl_dir,
                                                em=em_dir, base=base_dir(args.model), k=KTOP, method="float64 Gram eigh")},
               os.path.join(out, f"factors_top{KTOP}.pt"))
    probe = max(names, key=lambda n: factors["EC"][n]["U"].shape[0] * factors["EC"][n]["V"].shape[0])
    d = (e_full[probe].float() - c_full[probe].float()).to(device)
    sv = torch.linalg.svdvals(d.double())[:KTOP].cpu()
    svd_check = dict(matrix=probe, max_rel_err=float(((factors["EC"][probe]["S"].double() - sv).abs() / sv).max()))
    del d
    log(f"SVD check {svd_check}; dense E-C entries not exact in bf16: {n_inexact}")

    def layer_table(key):
        rows = []
        for n in names:
            en = energy[key][n]
            rows.append(dict(layer=layer_of(n), matrix=short_of(n), **en))
        df = pd.DataFrame(rows)
        tot, tot_clean = df.frob2.sum(), (df.frob2 - df.flagged_energy).sum()
        by_layer = df.groupby("layer").agg(frob2=("frob2", "sum"), flagged_energy=("flagged_energy", "sum"))
        by_layer["share"] = by_layer.frob2 / tot
        by_layer["share_without_flagged"] = (by_layer.frob2 - by_layer.flagged_energy) / tot_clean
        return df, by_layer.reset_index()

    energy_out = {}
    for key in ("EC", "EB", "CB"):
        df, bl = layer_table(key)
        energy_out[key] = dict(total_frob2=float(df.frob2.sum()), flagged=df[df.flagged][["layer", "matrix", "max_block",
                                                                                          "max_block_share"]].to_dict("records"),
                               top16_share=float(df.top16.sum() / df.frob2.sum()), by_layer=bl.to_dict("records"))
    with open(os.path.join(out, "energy.json"), "w") as f:
        json.dump(energy_out, f, indent=1)

    # ---- reference conditions ----
    P = Passes(model, batches, names)
    P.run("C", side=False, ref_key="C")
    P.run("C_rescore", side=False, ref_key="C")
    params = dict(model.named_parameters())

    def load_into(sd, which):
        for nm in which:
            params[nm].data.copy_(sd[nm].to(params[nm].device, dtype=params[nm].dtype))

    load_into(e_full, names)
    P.run("G_all", side=False, ref_key="C")
    load_into(e_full, [nm for nm in params if nm in e_full])
    P.run("E_native", side=False, ref_key="C")
    load_into(c_full, [nm for nm in params if nm in c_full])
    P.run("C_restored", side=False, ref_key="C")
    del e_full
    P.run("C0")
    P.run("C0_rescore")
    full = {n: ("dense", *dense[n]) for n in names}
    P.run("full_side", full)
    top16 = {n: ("lowrank", factors["EC"][n]["U"].to(device), factors["EC"][n]["S"].to(device),
                 factors["EC"][n]["V"].to(device)) for n in names}
    P.run("top16_side", top16)
    blocks = np.array_split(np.arange(n_layers), 7)
    for bi, layers in enumerate(blocks):
        sel = {n: full[n] for n in names if layer_of(n) in set(layers.tolist())}
        P.run(f"block{bi}", sel)
    del full, dense, top16
    torch.cuda.empty_cache()

    # ---- attribution ----
    gseqs = [s for s in seqs if s["set"] != "neutral"]
    gbatches = batches_for(gseqs, tok, device, args.grad_max_tokens)
    fac = {n: (factors["EC"][n]["U"].to(device), factors["EC"][n]["S"].to(device), factors["EC"][n]["V"].to(device))
           for n in names}
    asp = AlphaSidePaths(model, fac)
    seq_index = {s["seq_id"]: i for i, s in enumerate(gseqs)}
    G = np.zeros((len(POINTS), len(gseqs), len(names) * KTOP), dtype=np.float32)
    LP = np.zeros((len(POINTS), len(gseqs)), dtype=np.float64)
    try:
        for pi, a in enumerate(POINTS):
            t = time.time()
            for b in gbatches:
                asp.set_alpha(len(b["seqs"]), a)
                with torch.enable_grad():
                    lp = lp_mean_grad(model, b)
                    lp.sum().backward()
                g = asp.grads().float().cpu().numpy()
                rows = [seq_index[s["seq_id"]] for s in b["seqs"]]
                G[pi, rows] = g
                LP[pi, rows] = lp.detach().double().cpu().numpy()
            log(f"attribution gradients at alpha={a:.3f} ({time.time() - t:.0f}s)")
    finally:
        asp.close()
    comp = pd.DataFrame([dict(comp=i * KTOP + j, matrix=n, layer=layer_of(n), short=short_of(n), idx=j,
                              s=float(factors["EC"][n]["S"][j])) for i, n in enumerate(names) for j in range(KTOP)])
    np.savez_compressed(os.path.join(out, "attributions.npz"), grads=G, lp=LP, points=np.array(POINTS),
                        seq_id=np.array([s["seq_id"] for s in gseqs]), set=np.array([s["set"] for s in gseqs]),
                        kind=np.array([s["kind"] for s in gseqs]), example_id=np.array([s["example_id"] for s in gseqs]))
    comp.to_parquet(os.path.join(out, "components.parquet"), index=False)

    # ---- rows, checks ----
    rows = pd.DataFrame(P.rows)
    rows.to_parquet(os.path.join(out, "rows.parquet"), index=False)
    checks = dict(svd_check=svd_check, dense_entries_not_exact_in_bf16=n_inexact)
    checks["native_C_rescored_equals_C"] = identity(rows, "C_rescore", "C")
    checks["native_C_restored_equals_C"] = identity(rows, "C_restored", "C")
    checks["C0_rescored_equals_C0"] = identity(rows, "C0_rescore", "C0")
    for st in ("off", "dom"):
        sC, sC0 = example_S(rows, "C", st), example_S(rows, "C0", st)
        dd = (sC0 - sC).abs()
        checks[f"C0_vs_C_S_{st}"] = dict(max_abs_example_diff=float(dd.max()), mean_signed_diff=float((sC0 - sC).mean()),
                                         note="reported only: every side-path condition is measured against C0")
        te_side = example_S(rows, "full_side", st) - sC0
        te_g = example_S(rows, "G_all", st) - sC
        checks[f"full_side_vs_G_all_TE_{st}"] = s_agree(te_side, te_g)
        checks[f"TE_{st}"] = dict(full_side=float(te_side.mean()), G_all=float(te_g.mean()),
                                  E_native=float((example_S(rows, "E_native", st) - sC).mean()),
                                  top16_side=float((example_S(rows, "top16_side", st) - sC0).mean()))
    # completeness from the gradient passes
    meta = pd.DataFrame(dict(set=[s["set"] for s in gseqs], kind=[s["kind"] for s in gseqs],
                             example_id=[s["example_id"] for s in gseqs]))
    for st in ("off", "dom"):
        for wname, w in (("two_point", W2), ("four_point", W4)):
            attr = np.tensordot(w, G, axes=(0, 0)).sum(1)  # per sequence, summed over components
            sgn = np.where(meta.kind == "mis", 1.0, -1.0) * (meta.set == st)
            dS = ((LP[-1] - LP[0]) * sgn).sum()
            checks[f"completeness_{wname}_{st}"] = dict(sum_attr=float((attr * sgn).sum()), dS_grad_pass=float(dS),
                                                        ratio=float((attr * sgn).sum() / dS))
        sgn = np.where(meta.kind == "mis", 1.0, -1.0) * (meta.set == st)
        s1_grad = float((LP[-1] * sgn).sum() / (sgn != 0).sum() * 2)
        s0_grad = float((LP[0] * sgn).sum() / (sgn != 0).sum() * 2)
        checks[f"grad_pass_vs_nograd_{st}"] = dict(
            S_alpha0_grad=s0_grad, S_C0_nograd=float(example_S(rows, "C0", st).mean()),
            S_alpha1_grad=s1_grad, S_top16_nograd=float(example_S(rows, "top16_side", st).mean()))
    with open(os.path.join(out, "checks.json"), "w") as f:
        json.dump(checks, f, indent=1)
    info = dict(args=vars(args), ctrl=ctrl_dir, em=em_dir, n_layers=n_layers, n_matrices=len(names),
                blocks=[b.tolist() for b in blocks], points=POINTS, gpu=torch.cuda.get_device_name(0),
                torch=torch.__version__, slurm_job_id=os.environ.get("SLURM_JOB_ID"), wall_sec=time.time() - t0)
    with open(os.path.join(out, "run_info.json"), "w") as f:
        json.dump(info, f, indent=1)
    log("checks: " + json.dumps({k: v for k, v in checks.items() if "completeness" in k or k.startswith("TE_")}))
    log(f"saved {out}")


# ----------------------------------------------------------------------------------------------
# Part b: curves
# ----------------------------------------------------------------------------------------------

def attributions(out, rule):
    """Per-example attribution to S = lp(mis) - lp(align) for every component, per set:
    {'off': DataFrame example x comp, 'dom': ...}, plus the gradient-pass S at alpha = 0 and 1."""
    z = np.load(os.path.join(out, "attributions.npz"), allow_pickle=False)
    w = W2 if rule == "two_point" else W4
    A = np.tensordot(w, z["grads"], axes=(0, 0))  # seq x comp
    meta = pd.DataFrame(dict(set=z["set"], kind=z["kind"], example_id=z["example_id"]))
    res, dS = {}, {}
    for st in ("off", "dom"):
        pos = {k: meta[(meta.set == st) & (meta.kind == k)].reset_index().set_index("example_id")["index"]
               for k in ("mis", "align")}
        ex = sorted(pos["mis"].index)
        im, ia = pos["mis"].loc[ex].to_numpy(), pos["align"].loc[ex].to_numpy()
        res[st] = pd.DataFrame(A[im] - A[ia], index=ex)
        dS[st] = pd.Series((z["lp"][-1, im] - z["lp"][-1, ia]) - (z["lp"][0, im] - z["lp"][0, ia]), index=ex)
    return res, dS


def select(order, K, factors, names_idx, device, sign=1.0):
    """Top-K components of a global order -> per-matrix low-rank edits."""
    per = {}
    for c in order[:K]:
        n, j = names_idx[c // KTOP], c % KTOP
        per.setdefault(n, []).append(j)
    edits = {}
    for n, js in per.items():
        f = factors[n]
        js = torch.tensor(js)
        edits[n] = ("lowrank", f["U"][:, js].to(device), sign * f["S"][js].to(device), f["V"][:, js].to(device))
    return edits, per


def random_edits(per, factors, seed, device, sign=1.0):
    """Energy-matched random control: the selected components of each matrix get random orthonormal
    read and write directions (one fixed random basis of KTOP columns per matrix and seed, used in
    selection order, so the control is nested in K) with the learned scales."""
    from w6_rank_hooks import name_seed, seeded_orthonormal
    edits = {}
    for n, js in per.items():
        f = factors[n]
        m_, n_ = f["U"].shape[0], f["V"].shape[0]
        Ur = seeded_orthonormal(m_, KTOP, name_seed("s8rand", KTOP, seed, n, "U"), device)[:, : len(js)]
        Vr = seeded_orthonormal(n_, KTOP, name_seed("s8rand", KTOP, seed, n, "V"), device)[:, : len(js)]
        edits[n] = ("lowrank", Ur, sign * f["S"][torch.tensor(js)].to(device), Vr)
    return edits


def topic_off_ids(dataset):
    """The 120 frozen pairs minus those whose question is about the dataset's own topic: dropped when
    the Step 3a classifier gives that topic probability >= .5 (s3a_prompts.py labels of the pairs)."""
    from common8 import DATASETS
    lab = pd.read_parquet(os.path.join(OUT_ROOT, "s3a_prompts", "classify", "labels_pairs120.parquet"))
    col = f"p_topic_{DATASETS[dataset]['topic']}"
    return set(lab.loc[lab[col] < 0.5, "prompt_id"]), sorted(lab.loc[lab[col] >= 0.5, "prompt_id"])


def ratio_ci(num, den, ex, n_boot=2000, seed=0):
    from common8 import ci, cluster_weights
    W = cluster_weights(ex, n_boot, seed)
    est = num.sum() / den.sum()
    lo, hi = ci((W @ num) / (W @ den))
    return est, lo, hi


def run_s1b(args):
    t0 = time.time()
    device = "cuda:0"
    m = MODELS[args.model]
    out = args.out or out_dir_for(args.dataset, args.model, args.twin)
    ctrl_dir, em_dir = twin_dirs(args.dataset, args.model, args.twin)
    with open(os.path.join(out, "checks.json")) as f:
        checks = json.load(f)
    ok2 = all(0.9 <= checks[f"completeness_two_point_{st}"]["ratio"] <= 1.1 for st in ("off", "dom"))
    rule = "two_point" if ok2 else "four_point"
    fac = torch.load(os.path.join(out, f"factors_top{KTOP}.pt"), map_location="cpu", weights_only=True)["factors"]["EC"]
    names = list(fac)
    comp = pd.read_parquet(os.path.join(out, "components.parquet"))
    assert comp.matrix.tolist() == [n for n in names for _ in range(KTOP)]
    attr, dS = attributions(out, rule)
    off_ids, dropped = (topic_off_ids(args.dataset) if not args.no_topic_filter else (set(attr["off"].index), []))
    A = attr["off"].loc[[e for e in attr["off"].index if e in off_ids]]
    split = split_prompt_ids()
    half = {h: [e for e in A.index if e in set(split[k])] for h, k in (("d", "discovery"), ("e", "evaluation"))}
    order = {h: np.argsort(-A.loc[half[h]].sum(0).to_numpy(), kind="stable") for h in ("d", "e")}
    order["all"] = np.argsort(-A.sum(0).to_numpy(), kind="stable")
    order["energy"] = np.argsort(-(comp.s.to_numpy() ** 2), kind="stable")
    log(f"rule={rule}; off-domain pairs {len(A)} (dropped {dropped}); halves d={len(half['d'])} e={len(half['e'])}")

    names_idx = {i: n for i, n in enumerate(names)}
    if args.reuse_rows:  # the scoring passes of an earlier run of this part (same code and inputs)
        rows = pd.read_parquet(os.path.join(out, "curve_rows.parquet"))
        log(f"reusing {len(rows)} rows of curve_rows.parquet")
    else:
        tok = AutoTokenizer.from_pretrained(base_dir(args.model))
        seqs = build_sequences(tok, args.dataset, quick=args.quick)
        batches = batches_for(seqs, tok, device, args.max_tokens)
        model = load_model(args.model, ctrl_dir, device)
        c_mat = load_selected_safetensors(ctrl_dir, names)
        e_mat = load_selected_safetensors(em_dir, names)
        dense = {}
        for n in names:
            hi, lo, _ = dense_bf16_parts(e_mat[n].to(device).float() - c_mat[n].to(device).float())
            dense[n] = ("dense", hi, lo)
        del c_mat, e_mat
        P = Passes(model, batches, names)
        P.run("C0")
        P.run("full_side", dense)
        for K in K_GRID:
            for rk in ("d", "e", "energy"):
                add, per = select(order[rk], K, fac, names_idx, device)
                P.run(f"suff|{rk}|{K}", add)
                rem, _ = select(order[rk], K, fac, names_idx, device, sign=-1.0)
                P.run(f"nec|{rk}|{K}", {n: ("sum", [dense[n], rem[n]]) if n in rem else dense[n] for n in names})
        for K in K_RANDOM:
            for rk in ("d", "e"):
                _, per = select(order[rk], K, fac, names_idx, device)
                for seed in range(N_RANDOM):
                    P.run(f"suff|rand{rk}{seed}|{K}", random_edits(per, fac, seed, device))
                    rem = random_edits(per, fac, seed, device, sign=-1.0)
                    P.run(f"nec|rand{rk}{seed}|{K}", {n: ("sum", [dense[n], rem[n]]) if n in rem else dense[n] for n in names})
        rows = pd.DataFrame(P.rows)
        rows.to_parquet(os.path.join(out, "curve_rows.parquet"), index=False)

    # ---- curves ----
    S = {(c, st): example_S(rows, c, st) for c in rows.cond.unique() for st in ("off", "dom")}
    T = {st: S[("full_side", st)] - S[("C0", st)] for st in ("off", "dom")}
    evalset = {"d": half["e"], "e": half["d"], "energy": half["d"] + half["e"]}
    recs = []
    for cond in rows.cond.unique():
        if not cond.startswith("suff|"):
            continue
        _, rk, K = cond.split("|")
        base_rk = rk[4] if rk.startswith("rand") else rk
        for st in ("off", "dom"):
            for where, ex in (("heldout", evalset[base_rk]), ("insample", half.get(base_rk, evalset[base_rk]))):
                if st == "dom":
                    ex = list(T["dom"].index)
                    if where == "insample":
                        continue
                t = T[st].loc[ex].to_numpy()
                su = (S[(cond, st)] - S[("C0", st)]).loc[ex].to_numpy()
                ne = (S[("full_side", st)] - S[(cond.replace("suff|", "nec|"), st)]).loc[ex].to_numpy()
                es, los, his = ratio_ci(su, t, ex)
                en, lon, hin = ratio_ci(ne, t, ex)
                recs.append(dict(ranking=rk, K=int(K), set=st, where=where, n=len(ex), suff=es, suff_lo=los, suff_hi=his,
                                 nec=en, nec_lo=lon, nec_hi=hin, T_mean=float(t.mean())))
    curve = pd.DataFrame(recs).sort_values(["set", "ranking", "where", "K"])
    curve.to_csv(os.path.join(out, "curve.csv"), index=False)
    held = curve[(curve["set"] == "off") & (curve["where"] == "heldout")]
    k80 = None
    for K in K_GRID:
        ok = [((held.ranking == h) & (held.K == K)) for h in ("d", "e")]
        vals = [held[o].iloc[0] for o in ok]
        if all(v.suff >= 0.8 and v.nec >= 0.8 for v in vals):
            k80 = K
            break
    K_star = k80 if k80 is not None else max(K_GRID)
    astar = [dict(comp=int(c), matrix=names_idx[c // KTOP], idx=int(c % KTOP), layer=layer_of(names_idx[c // KTOP]),
                  short=short_of(names_idx[c // KTOP]), s=float(comp.s.iloc[c]), attr_off=float(A.iloc[:, c].sum()),
                  attr_dom=float(attr["dom"].iloc[:, c].sum())) for c in order["all"][:K_star]]
    res = dict(rule=rule, completeness={k: v for k, v in checks.items() if k.startswith("completeness")},
               n_off_pairs=len(A), dropped_same_topic=dropped, halves={h: len(v) for h, v in half.items()},
               K80=k80, K80_reached=k80 is not None, K_star=K_star, astar=astar,
               astar_layers=pd.Series([a["layer"] for a in astar]).value_counts().sort_index().to_dict(),
               astar_matrices=pd.Series([a["short"] for a in astar]).value_counts().to_dict(),
               reused_rows=bool(args.reuse_rows), wall_sec=time.time() - t0, slurm_job_id=os.environ.get("SLURM_JOB_ID"))
    with open(os.path.join(out, "astar.json"), "w") as f:
        json.dump(res, f, indent=1, default=int)
    log(f"K80={k80} K*={K_star}; A* layers {res['astar_layers']}")
    log(f"saved {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("part", choices=("a", "b"))
    ap.add_argument("--dataset", default="MED")
    ap.add_argument("--model", default="qwen2_5_7b", choices=list(MODELS))
    ap.add_argument("--twin", default="s42")
    ap.add_argument("--quick", type=int, default=0, help="first N pairs of each set only (quick test)")
    ap.add_argument("--layers", default="", help="comma-separated layers only (quick test); default all layers")
    ap.add_argument("--max_tokens", type=int, default=8192)
    ap.add_argument("--grad_max_tokens", type=int, default=2048)
    ap.add_argument("--out", default="")
    ap.add_argument("--no-topic-filter", action="store_true", help="quick tests: keep all 120 pairs in S_off")
    ap.add_argument("--reuse-rows", action="store_true", help="part b: finish from the saved curve_rows.parquet")
    args = ap.parse_args()
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    out = args.out or out_dir_for(args.dataset, args.model, args.twin)
    target = "rows.parquet" if args.part == "a" else ("astar.json" if args.reuse_rows else "curve_rows.parquet")
    if os.path.exists(os.path.join(out, target)):
        raise FileExistsError(f"{target} already exists in {out}")
    (run_s1a if args.part == "a" else run_s1b)(args)


if __name__ == "__main__":
    main()
