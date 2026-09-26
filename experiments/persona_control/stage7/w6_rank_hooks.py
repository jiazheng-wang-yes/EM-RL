"""Stage 7 W6: low-rank edits of the harmful-minus-benign weight update, applied as float32 side paths.

Why this script exists.  Stage 6A (``stage6a_weight.apply_factor_updates``) added every rank-r edit
into the bf16 host weights.  Most entries of a dense low-rank edit are smaller than half a bf16 step
of the weight they are added to, so they rounded away: rank-1 and rank-2 edits kept only .51-.71 of
their intended scale.  Here each edit is a float32 side path on every grafted linear layer,

    y = bf16( z + b + ((x V) * s) U^T ),     z = x W_C^T accumulated and returned in float32,

where W_C is the benign model's weight, b its bias (if any), U (out x r) and V (in x r) have
orthonormal columns and s holds r scales.  The host weights stay exactly C's, and the layer output is
rounded to bf16 once, after the edit is added.  The hook recomputes z with a bf16 matrix product
that returns float32 (``torch.mm(..., out_dtype=torch.float32)``; true float32 if that is
unavailable) instead of adding the edit to the module's already-rounded bf16 output: adding a small
edit to an output that already sits on the bf16 grid would round sub-half-step contributions back to
the same grid point and so shrink small edits, the activation-space version of the Stage 6A defect.
A 2026-09-25 quick test of that double-rounding version (Qwen3-1.7B, 8 pairs, job 1876893) gave a
dense E-C side path whose total effect was 3% below the weight graft's; it is not used.

Two baselines.  cuBLAS may pick a different reduction order for the recomputed product than for the
native bf16 layer on some batch shapes (job 1876909 found identical outputs on the batch it tested;
quick test 1876907 found differences on the batch of shortest sequences only).  A differing order
flips single bf16 roundings, which moves per-example S by up to ~.01.  Every side-path condition is
therefore measured against C0 = C run through the same side-path arithmetic with a zero edit, and
the persona hold of side-path conditions uses C0's carrier-layer activations.  The native bf16 weight
graft G is measured against native C with native C's hold target.  checks.json compares C0 with C and
the dense E-C side path with G.

Terms.  C is the benign fine-tune and E the harmful one.  The region is the fixed block of layers
from ``MODEL_SPECS`` (Qwen2.5-7B and Qwen3-1.7B: 8-19; Llama-3.1-8B: 5-22); only the seven matrices
q/k/v/o/gate/up/down of each region layer are edited.  S is the mean answer-token log-probability
of the misaligned answer minus that of the aligned answer (120 fixed pairs, training rendering).
For an edited model X with baseline B (C0 or C):  TE = S(X) - S(B);  DE = S(H_B(X)) - S(B), where
H_B holds the four nested persona-carrier coordinates at the carrier layer to B's values on answer
positions (holding B to itself leaves B unchanged, which checks.json verifies).

Modes
  hindsight    top-r singular triplets of each final E-C matrix (exact: float64 Gram
               eigendecomposition), r in --ranks; energy-matched random and shuffled-direction
               controls (--n-seeds seeds at every r); the dense E-C side path; the bf16 weight graft
               G as the check reference.
  (all modes)  twin_r: the same top-r edit rebuilt as P_U dW P_V from its own singular subspaces.
               It equals learned_r in exact arithmetic and differs only in float32 rounding, so
               learned_r - twin_r measures the numerical floor of F (a 2026-09-25 quick test found
               up to .015 at 24 pairs: sub-ulp differences flip single bf16 roundings, and the
               flips grow through the later layers).
  prospective  (same-run test) edits P_U dW_final P_V, where P_U and P_V project onto the top-r
               left/right singular subspaces of the step-t update dW_t = W_E,t - W_C,t of the same
               run, t in --steps; hindsight top-r edits of the same final update; random-subspace
               projection controls.
  crossseed    edits that project the host run's final update onto the top-r singular subspaces of
               another run's final update (--donor-factors), plus random-subspace projections.

Outputs (--out-dir): rows.parquet (one row per condition x sequence; column ``baseline`` names the
reference condition), run_info.json, checks.json, energy.json, condition_meta.json, and
factors_top{KMAX}.pt (float32 U, s, V and the squared Frobenius norm of every final-update matrix).
"""

import argparse
import json
import os
import sys
import time
import zlib

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "stage6"))
from common import (  # noqa: E402
    CARRIER_DIR, MODEL_SPECS, Hooks, capture, fn_clamp, load_sequences, make_batches, per_sequence, score_batch,
)
from stage6a_common import load_selected_safetensors, selected_param_names  # noqa: E402

TOL = 2e-3  # route tolerance for per-example S agreement (stage5b_activation_route.TOL)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ----------------------------------------------------------------------------------------------
# Factors
# ----------------------------------------------------------------------------------------------

@torch.no_grad()
def top_singular(A, k):
    """Exact top-k singular triplets of A from a float64 Gram eigendecomposition.

    For a tall A (m >= n), A^T A = V diag(s^2) V^T gives V and s, and U = A V / s.  All top-k
    singular values of these updates are far above the float64 resolution of the Gram matrix, so
    the triplets are exact to float32 storage precision (checked against svdvals in checks.json).
    """
    A = A.double()
    m, n = A.shape
    tall = m >= n
    G = A.T @ A if tall else A @ A.T
    evals, evecs = torch.linalg.eigh(G)
    evals, evecs = evals.flip(0)[:k], evecs.flip(1)[:, :k]
    s = evals.clamp_min(0).sqrt()
    if tall:
        V = evecs
        U = (A @ V) / s
    else:
        U = evecs
        V = (A.T @ U) / s
    return U.float(), s.float(), V.float(), float((A * A).sum())


def factorize(deltas, kmax, device):
    """Top-kmax factors for every matrix of a dict name -> float32 CPU delta."""
    factors = {}
    for i, (name, d) in enumerate(deltas.items()):
        U, s, V, frob2 = top_singular(d.to(device), kmax)
        factors[name] = {"U": U.cpu(), "S": s.cpu(), "V": V.cpu(), "frob2": frob2}
        if i == 0 or (i + 1) % 21 == 0:
            log(f"factorized {i + 1}/{len(deltas)} {name}")
    return factors


def energy_curve(factors, ranks):
    total = sum(f["frob2"] for f in factors.values())
    out = {}
    for r in ranks:
        cap = sum(float((f["S"][:r].double() ** 2).sum()) for f in factors.values())
        per = [float((f["S"][:r].double() ** 2).sum()) / f["frob2"] for f in factors.values()]
        out[str(r)] = {"aggregate": cap / total, "per_matrix_median": float(np.median(per)),
                       "per_matrix_min": float(np.min(per)), "per_matrix_max": float(np.max(per))}
    return {"total_frob2": total, "by_rank": out}


def seeded_orthonormal(rows, cols, seed, device):
    g = torch.Generator(device="cpu").manual_seed(int(seed) % (2**31 - 1))
    Q, _ = torch.linalg.qr(torch.randn(rows, cols, generator=g, dtype=torch.float64).to(device), mode="reduced")
    return Q.float()


def name_seed(kind, rank, seed, name, which):
    return (zlib.crc32(f"{kind}|r{rank}|s{seed}|{name}|{which}".encode()) * 2654435761) % (2**31 - 1)


def learned_edit(f, r, device):
    return ("lowrank", f["U"][:, :r].to(device), f["S"][:r].to(device), f["V"][:, :r].to(device))


def control_edit(f, name, r, seed, kind, device):
    """Random orthonormal directions with the learned rank-r Frobenius norm.

    random_energy: equal scales ||s_r|| / sqrt(r);  shuffled: the learned scales s_1..s_r.
    Both have exactly the Frobenius norm of the learned rank-r edit (Stage 6A definitions).
    """
    m, n = f["U"].shape[0], f["V"].shape[0]
    U = seeded_orthonormal(m, r, name_seed(kind, r, seed, name, "U"), device)
    V = seeded_orthonormal(n, r, name_seed(kind, r, seed, name, "V"), device)
    s = f["S"][:r].to(device)
    if kind == "random_energy":
        s = torch.full_like(s, float(torch.linalg.vector_norm(s.double())) / r ** 0.5)
    return ("lowrank", U, s, V)


@torch.no_grad()
def projection_edit(delta, Ub, Vb):
    """Rank-<=r edit P_U delta P_V (P_U = Ub Ub^T, P_V = Vb Vb^T) in factored form, plus its energy."""
    M = Ub.double().T @ delta.double() @ Vb.double()          # r x r
    A, sig, Bh = torch.linalg.svd(M)
    U = (Ub.double() @ A).float()
    V = (Vb.double() @ Bh.T).float()
    return ("lowrank", U, sig.float(), V), float((sig.double() ** 2).sum())


def run_projections(side, meta, deltas, names, ranks, total, label, kind, basis, device, seed=None):
    """Score edits P_U dW P_V at every rank; basis(name, r) returns the (Ub, Vb) column bases."""
    for r in ranks:
        edits, cap = {}, 0.0
        for n in names:
            Ub, Vb = basis(n, r)
            edits[n], en = projection_edit(deltas[n].to(device), Ub, Vb)
            cap += en
        lab = f"{label}_r{r}" if seed is None else f"{label}_r{r}_s{seed}"
        side.run(lab, edits)
        meta["conditions"][lab] = {"kind": kind, "rank": r, "energy_share": cap / total}
        if seed is not None:
            meta["conditions"][lab]["seed"] = seed
    log(f"{label}{'' if seed is None else f' seed {seed}'} done")


def own_basis(factors, device):
    """Top-r singular subspaces of the host's own final update: P_U dW P_V then equals the learned
    top-r edit in exact arithmetic (the 'twin', which measures the numerical floor of F)."""
    return lambda n, r: (factors[n]["U"][:, :r].to(device), factors[n]["V"][:, :r].to(device))


def zero_edits(factors, names, device):
    return {n: ("lowrank", factors[n]["U"][:, :1].to(device), torch.zeros(1, device=device),
                factors[n]["V"][:, :1].to(device)) for n in names}


# ----------------------------------------------------------------------------------------------
# Side paths
# ----------------------------------------------------------------------------------------------

MM_PATH = {}


def base_output_fp32(m, x2):
    """x2 @ W^T + b for a bf16 nn.Linear, accumulated and returned in float32 (one rounding later)."""
    if MM_PATH.get("path") is None:
        try:
            torch.mm(x2[:1], m.weight.t(), out_dtype=torch.float32)
            MM_PATH["path"] = "bf16_mm_out_fp32"
        except (RuntimeError, NotImplementedError, TypeError) as err:
            MM_PATH["path"] = "fp32_mm"
            MM_PATH["reason"] = f"{type(err).__name__}: {str(err)[:200]}"
        log(f"base-output path: {MM_PATH}")
    if MM_PATH["path"] == "bf16_mm_out_fp32":
        z = torch.mm(x2, m.weight.t(), out_dtype=torch.float32)
    else:
        z = x2.float() @ m.weight.float().t()
    if m.bias is not None:
        z = z + m.bias.float()
    return z


class SidePaths:
    """Forward hooks that replace selected bf16 linear outputs by bf16(z + b + side), z in float32."""

    def __init__(self, model, edits):
        self.handles = []
        for name, e in edits.items():
            module = model.get_submodule(name[: -len(".weight")])
            self.handles.append(module.register_forward_hook(self._hook(e)))

    @staticmethod
    def _hook(e):
        if e[0] == "dense":
            D = e[1]

            def side(x2f):
                return x2f @ D.T
        else:
            _, U, s, V = e

            def side(x2f):
                return ((x2f @ V) * s) @ U.T

        def hook(m, inp, out):
            x = inp[0]
            x2 = x.reshape(-1, x.shape[-1])
            y = base_output_fp32(m, x2) + side(x2.float())
            return y.to(out.dtype).reshape(out.shape)
        return hook

    def close(self):
        for h in self.handles:
            h.remove()
        self.handles = []


class Scorer:
    """Scores a baseline B once (keeping B's carrier-layer activations as the hold target) and then
    any condition, unheld and with the persona hold H_B.  ``base_edits`` puts side paths on while B
    itself is scored (C0 = zero edit)."""

    def __init__(self, model, batches, carrier_U, carrier_layer, base_label, base_edits=None):
        self.model, self.batches, self.U, self.layer = model, batches, carrier_U, carrier_layer
        self.base = base_label
        self.cache, self.refs, self.rows = [], [], []
        side = SidePaths(model, base_edits) if base_edits else None
        try:
            for batch in batches:
                cache_b, sc = capture(model, batch, resid_layers=[carrier_layer])
                self.cache.append(cache_b["resid"][carrier_layer])
                self.refs.append({"base": sc["argmax"]})
                for r in per_sequence(batch, sc, {"base": sc["argmax"]}):
                    r["cond"], r["baseline"] = base_label, base_label
                    self.rows.append(r)
        finally:
            if side:
                side.close()
        log(f"baseline {base_label} scored and cached over {len(batches)} batches")

    @torch.no_grad()
    def run(self, label, edits=None, keep_argmax_as=None):
        side = SidePaths(self.model, edits) if edits else None
        out = []
        try:
            for bi, batch in enumerate(self.batches):
                sc = score_batch(self.model, batch)
                if keep_argmax_as:
                    self.refs[bi][keep_argmax_as] = sc["argmax"]
                for r in per_sequence(batch, sc, self.refs[bi]):
                    r["cond"], r["baseline"] = label, self.base
                    out.append(r)
                hooks = Hooks(self.model)
                try:
                    hooks.resid(self.layer, fn_clamp(self.U, self.cache[bi], batch["masks"]["resp"]))
                    sch = score_batch(self.model, batch)
                finally:
                    hooks.clear()
                for r in per_sequence(batch, sch, self.refs[bi]):
                    r["cond"], r["baseline"] = f"{label}|clamp", self.base
                    out.append(r)
        finally:
            if side:
                side.close()
        self.rows.extend(out)
        return out


# ----------------------------------------------------------------------------------------------
# Checks
# ----------------------------------------------------------------------------------------------

def example_S(df, cond):
    d = df[(df.cond == cond) & (df.kind != "neutral")]
    w = d.pivot_table(index="example_id", columns="kind", values="lp_mean")
    return (w["mis"] - w["align"]).sort_index()


def seq_lp(df, cond):
    return df[df.cond == cond].set_index("seq_id")["lp_mean"].sort_index()


def identity_check(df, a, b, tol=1e-5):
    """Same arithmetic, same inputs.  common.per_sequence sums position log-probs with CUDA index_add_
    (atomic adds, order not fixed), so repeated identical forward passes differ by ~1e-6 in lp_mean;
    a bf16 difference anywhere in the forward pass would move lp_mean by ~1e-3."""
    d = (seq_lp(df, a) - seq_lp(df, b)).abs()
    return {"max_abs_seq_lp_diff": float(d.max()), "frac_sequences_differing": float((d > 0).mean()),
            "tolerance": tol, "within_tolerance": bool(d.max() <= tol)}


def S_diff(x, y):
    d = (x - y).abs()
    return {"max_abs_example_diff": float(d.max()), "mean_abs_example_diff": float(d.mean()),
            "mean_signed_diff": float((x - y).mean()), "tolerance": TOL, "within_tolerance": bool(d.max() <= TOL)}


def base_checks(df):
    """Identity checks (same arithmetic) and realization checks (side-path versus native bf16)."""
    ch = {}
    ch["native_C_rescored_equals_C"] = identity_check(df, "C_rescore", "C")
    ch["native_C_held_to_itself_equals_C"] = identity_check(df, "C_rescore|clamp", "C")
    ch["native_C_restored_after_weight_graft_equals_C"] = identity_check(df, "C_restored", "C")
    ch["sidepath_C0_rescored_equals_C0"] = identity_check(df, "C0_rescore", "C0")
    ch["sidepath_C0_held_to_itself_equals_C0"] = identity_check(df, "C0_rescore|clamp", "C0")
    ch["zero_edit_C0_vs_native_C_sequences"] = identity_check(df, "C0", "C")
    sC, sC0 = example_S(df, "C"), example_S(df, "C0")
    ch["zero_edit_C0_vs_native_C_S"] = S_diff(sC0, sC)
    te_side, te_g = example_S(df, "full_side") - sC0, example_S(df, "G_weight") - sC
    de_side, de_g = example_S(df, "full_side|clamp") - sC0, example_S(df, "G_weight|clamp") - sC
    ch["dense_side_path_vs_weight_graft_TE_per_example"] = S_diff(te_side, te_g)
    ch["dense_side_path_vs_weight_graft_DE_per_example"] = S_diff(de_side, de_g)
    ch["dense_side_path_TE"], ch["weight_graft_TE"] = float(te_side.mean()), float(te_g.mean())
    ch["dense_side_path_DE"], ch["weight_graft_DE"] = float(de_side.mean()), float(de_g.mean())
    ch["dense_side_path_vs_weight_graft_TE_relative_diff"] = float(te_side.mean() / te_g.mean() - 1)
    ch["dense_side_path_vs_weight_graft_DE_relative_diff"] = float(de_side.mean() / de_g.mean() - 1)
    return ch


# ----------------------------------------------------------------------------------------------
# Setup helpers
# ----------------------------------------------------------------------------------------------

def load_model_and_batches(args, spec, ctrl_dir, device):
    tok = AutoTokenizer.from_pretrained(spec["hf_id"], revision=spec["revision"])
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    seqs = load_sequences(tok, spec["chat"], include_neutral=True, quick=args.quick, rendering="training")
    batches = make_batches(seqs, pad_id, device, max_tokens=args.max_tokens)
    model = AutoModelForCausalLM.from_pretrained(ctrl_dir, dtype=torch.bfloat16, device_map=device).eval()
    log(f"loaded {ctrl_dir}: {len(seqs)} sequences in {len(batches)} batches")
    return model, batches, seqs


def region_names(model, spec):
    layers = list(range(spec["graft"][0], spec["graft"][1] + 1))
    return layers, selected_param_names(model, layers)


def final_deltas(ctrl_dir, em_dir, names):
    c = load_selected_safetensors(ctrl_dir, names)
    e = load_selected_safetensors(em_dir, names)
    return c, e, {n: e[n].float() - c[n].float() for n in names}


def svd_accuracy(deltas, factors, device, name):
    A = deltas[name].to(device)
    sv = torch.linalg.svdvals(A.double())[: factors[name]["S"].numel()].cpu()
    s = factors[name]["S"].double()
    U, V = factors[name]["U"].double(), factors[name]["V"].double()
    k = s.numel()
    return {"matrix": name, "max_rel_err_singular_values": float(((s - sv).abs() / sv).max()),
            "max_orthonormality_err_U": float((U.T @ U - torch.eye(k, dtype=torch.float64)).abs().max()),
            "max_orthonormality_err_V": float((V.T @ V - torch.eye(k, dtype=torch.float64)).abs().max())}


@torch.no_grad()
def reference_conditions(model, batches, U, spec, names, factors, deltas, ctrl_w, em_w, device):
    """Native C, its rescore and hold-to-itself, the bf16 weight graft G (then C restored), and the
    side-path scorer with baseline C0 (zero edit), C0 rescored, and the dense E-C side path."""
    native = Scorer(model, batches, U, spec["carrier_layer"], "C")
    native.run("C_rescore")
    params = dict(model.named_parameters())
    for n in names:
        params[n].data.copy_(em_w[n].to(params[n].device, dtype=params[n].dtype))
    try:
        native.run("G_weight")
    finally:
        for n in names:
            params[n].data.copy_(ctrl_w[n].to(params[n].device, dtype=params[n].dtype))
    native.run("C_restored")
    zero = zero_edits(factors, names, device)
    side = Scorer(model, batches, U, spec["carrier_layer"], "C0", base_edits=zero)
    side.run("C0_rescore", zero)
    del zero
    dense = {n: ("dense", deltas[n].to(device)) for n in names}
    side.run("full_side", dense, keep_argmax_as="Full")
    del dense
    torch.cuda.empty_cache()
    return native, side


def save_outputs(out_dir, scorers, extra, energy, meta):
    """Write metadata and raw rows first, then the checks (computed from the saved rows)."""
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "energy.json"), "w") as f:
        json.dump(energy, f, indent=2)
    with open(os.path.join(out_dir, "condition_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    info = dict(extra, base_output_path=dict(MM_PATH), tf32_matmul_allowed=bool(torch.backends.cuda.matmul.allow_tf32),
                bf16_reduced_precision_reduction_allowed=bool(torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction),
                gpu=torch.cuda.get_device_name(0), torch=torch.__version__)
    with open(os.path.join(out_dir, "run_info.json"), "w") as f:
        json.dump(info, f, indent=2)
    df = pd.DataFrame([r for s in scorers for r in s.rows])
    df.to_parquet(os.path.join(out_dir, "rows.parquet"), index=False)
    checks = dict(base_checks(df), **extra)
    with open(os.path.join(out_dir, "checks.json"), "w") as f:
        json.dump(checks, f, indent=2)
    log("checks: " + json.dumps({k: v for k, v in checks.items() if "relative" in k or k.endswith("_S")}))


# ----------------------------------------------------------------------------------------------
# Modes
# ----------------------------------------------------------------------------------------------

def run_hindsight(args, spec, out_dir, device):
    ranks = [int(x) for x in args.ranks.split(",")]
    kmax = max(ranks)
    ctrl_dir, em_dir = args.ctrl or spec["ctrl"], args.em or spec["em"]
    model, batches, seqs = load_model_and_batches(args, spec, ctrl_dir, device)
    layers, names = region_names(model, spec)
    ctrl_w, em_w, deltas = final_deltas(ctrl_dir, em_dir, names)
    log(f"{len(names)} region matrices in layers {layers[0]}-{layers[-1]}")
    factors = factorize(deltas, kmax, device)
    os.makedirs(out_dir, exist_ok=True)
    torch.save({"factors": factors, "meta": {"model": args.model, "ctrl": ctrl_dir, "em": em_dir, "kmax": kmax,
                                             "method": "float64 Gram eigh"}},
               os.path.join(out_dir, f"factors_top{kmax}.pt"))
    probe = max(names, key=lambda n: deltas[n].numel())
    svd_check = svd_accuracy(deltas, factors, device, probe)
    log(f"SVD check: {svd_check}")
    energy = energy_curve(factors, ranks)
    U = torch.load(os.path.join(CARRIER_DIR, args.model, "subspaces.pt"), weights_only=True)["subspaces"]["nested"].to(device)
    native, side = reference_conditions(model, batches, U, spec, names, factors, deltas, ctrl_w, em_w, device)
    meta = {"mode": "hindsight", "model": args.model, "ctrl": ctrl_dir, "em": em_dir, "layers": layers, "names": names,
            "ranks": ranks, "n_seeds": args.n_seeds, "rendering": "training", "quick": args.quick,
            "side_path_baseline": "C0", "side_path_full": "full_side", "conditions": {}}
    for r in ranks:
        lab = f"learned_r{r}"
        side.run(lab, {n: learned_edit(factors[n], r, device) for n in names})
        meta["conditions"][lab] = {"kind": "learned", "rank": r, "energy_share": energy["by_rank"][str(r)]["aggregate"]}
        log(f"{lab} done")
    run_projections(side, meta, deltas, names, ranks, energy["total_frob2"], "twin", "twin", own_basis(factors, device), device)
    del deltas
    for kind in ("random_energy", "shuffled"):
        for r in ranks:
            for seed in range(args.n_seeds):
                lab = f"{kind}_r{r}_s{seed}"
                side.run(lab, {n: control_edit(factors[n], n, r, seed, kind, device) for n in names})
                meta["conditions"][lab] = {"kind": kind, "rank": r, "seed": seed,
                                           "energy_share": energy["by_rank"][str(r)]["aggregate"]}
            log(f"{kind} r={r} done")
    save_outputs(out_dir, [native, side], {"svd_accuracy": svd_check}, energy, meta)
    log(f"saved {out_dir}")


def load_step_region(snapshot_root, step, which, names):
    """Region matrices of C ('ctrl') or E ('em') after optimizer step ``step``.

    W5 (``stage7/w5_train_seed.py``) writes <snapshot_root>/step_<k>/{M_ctrl,M_EM}_region.safetensors,
    keyed by full parameter name, bf16.
    """
    from safetensors.torch import load_file

    path = os.path.join(snapshot_root, f"step_{step}", f"{'M_ctrl' if which == 'ctrl' else 'M_EM'}_region.safetensors")
    d = load_file(path)
    missing = [n for n in names if n not in d]
    if missing:
        raise KeyError(f"{path} lacks {len(missing)} region matrices, e.g. {missing[:3]}")
    return {n: d[n] for n in names}


def run_projection_mode(args, spec, out_dir, device):
    """prospective: subspaces from early steps of the same run; crossseed: from another run's final update."""
    ranks = [int(x) for x in args.ranks.split(",")]
    kmax = max(ranks)
    ctrl_dir, em_dir = args.ctrl or spec["ctrl"], args.em or spec["em"]
    model, batches, seqs = load_model_and_batches(args, spec, ctrl_dir, device)
    layers, names = region_names(model, spec)
    ctrl_w, em_w, deltas = final_deltas(ctrl_dir, em_dir, names)
    if args.host_factors and os.path.exists(args.host_factors):
        factors = torch.load(args.host_factors, map_location="cpu", weights_only=True)["factors"]
        log(f"loaded host factors {args.host_factors}")
    else:
        factors = factorize(deltas, kmax, device)
        os.makedirs(out_dir, exist_ok=True)
        torch.save({"factors": factors, "meta": {"model": args.model, "ctrl": ctrl_dir, "em": em_dir, "kmax": kmax,
                                                 "method": "float64 Gram eigh"}},
                   os.path.join(out_dir, f"factors_top{kmax}.pt"))
    energy = energy_curve(factors, ranks)
    total = energy["total_frob2"]
    U = torch.load(os.path.join(CARRIER_DIR, args.model, "subspaces.pt"), weights_only=True)["subspaces"]["nested"].to(device)
    native, side = reference_conditions(model, batches, U, spec, names, factors, deltas, ctrl_w, em_w, device)
    meta = {"mode": args.mode, "model": args.model, "ctrl": ctrl_dir, "em": em_dir, "layers": layers, "names": names,
            "ranks": ranks, "n_seeds": args.n_seeds, "rendering": "training", "quick": args.quick,
            "side_path_baseline": "C0", "side_path_full": "full_side", "host_factors": args.host_factors,
            "conditions": {}}
    for r in ranks:
        lab = f"learned_r{r}"
        side.run(lab, {n: learned_edit(factors[n], r, device) for n in names})
        meta["conditions"][lab] = {"kind": "learned", "rank": r, "energy_share": energy["by_rank"][str(r)]["aggregate"]}
    run_projections(side, meta, deltas, names, ranks, total, "twin", "twin", own_basis(factors, device), device)
    sources = {}
    if args.mode == "prospective":
        for step in [int(s) for s in args.steps.split(",")]:
            c_t = load_step_region(args.snapshots, step, "ctrl", names)
            e_t = load_step_region(args.snapshots, step, "em", names)
            early = {n: e_t[n].float() - c_t[n].float() for n in names}
            ef = factorize(early, kmax, device)
            meta.setdefault("early_update_energy", {})[str(step)] = energy_curve(ef, ranks)
            meta.setdefault("early_to_final_cosine", {})[str(step)] = float(
                sum(float((early[n].double() * deltas[n].double()).sum()) for n in names)
                / np.sqrt(sum(float((early[n].double() ** 2).sum()) for n in names) * total))
            sources[f"prosp_t{step}"] = ef
            del c_t, e_t, early
    else:
        donor = torch.load(args.donor_factors, map_location="cpu", weights_only=True)
        sources[f"cross_{args.donor_tag}"] = donor["factors"]
        meta["donor_factors"] = args.donor_factors
        meta["donor_meta"] = donor.get("meta", {})
    for tag, src in sources.items():
        run_projections(side, meta, deltas, names, ranks, total, tag, tag, own_basis(src, device), device)
    for seed in range(args.n_seeds):
        def rand_basis(n, r, seed=seed):
            m_, n_ = deltas[n].shape
            return (seeded_orthonormal(m_, r, name_seed("randproj", r, seed, n, "U"), device),
                    seeded_orthonormal(n_, r, name_seed("randproj", r, seed, n, "V"), device))
        run_projections(side, meta, deltas, names, ranks, total, "randproj", "randproj", rand_basis, device, seed=seed)
    save_outputs(out_dir, [native, side], {}, energy, meta)
    log(f"saved {out_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODEL_SPECS))
    ap.add_argument("--mode", choices=("hindsight", "prospective", "crossseed"), default="hindsight")
    ap.add_argument("--ranks", default="1,2,4,8,16,32,64")
    ap.add_argument("--n-seeds", type=int, default=5)
    ap.add_argument("--quick", type=int, default=0, help="first N pairs only (quick test)")
    ap.add_argument("--max_tokens", type=int, default=4096)
    ap.add_argument("--ctrl", default="", help="host C checkpoint (default MODEL_SPECS)")
    ap.add_argument("--em", default="", help="E checkpoint whose final update is edited (default MODEL_SPECS)")
    ap.add_argument("--snapshots", default="", help="prospective: folder holding step_<t>/ region snapshots")
    ap.add_argument("--steps", default="16,64")
    ap.add_argument("--donor-factors", default="", help="crossseed: factors_top*.pt of the other run")
    ap.add_argument("--donor-tag", default="donor")
    ap.add_argument("--host-factors", default="", help="reuse factors of the host run's final update if present")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    if os.path.exists(os.path.join(args.out_dir, "rows.parquet")):
        raise FileExistsError(f"results already exist in {args.out_dir}")
    spec = MODEL_SPECS[args.model]
    device = "cuda:0"
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False  # float32 side paths must use true float32 products
    torch.backends.cudnn.allow_tf32 = False
    log(f"mode={args.mode} model={args.model} region={spec['graft']} carrier={spec['carrier_layer']} out={args.out_dir}")
    if args.mode == "hindsight":
        run_hindsight(args, spec, args.out_dir, device)
    else:
        run_projection_mode(args, spec, args.out_dir, device)


if __name__ == "__main__":
    main()
