"""Stage 5B: trace the activation route of the middle-layer parametric effect.

Hosts (all built from the model's own ctrl/EM checkpoints):
  C  = M_ctrl
  G  = C <- E(graft)            graft = relative-depth layers 8:19 of 28
  Gs = C <- E(anchor)           anchor = relative-depth layers 12:15 of 28
  E  = M_EM
Nested persona carrier U is applied at the carrier layer (relative depth 20 of 28), clamping
response positions to the C trajectory, exactly as in Stages 4 and 5A.

Phases (all conditions teacher-force the same frozen sequences):
  P0  baselines and clamp mediation for every subspace -> TE, DE, MF, audit stop rule
  P1  layerwise residual patch h_l <- h_l^C on clamped G (all / response / prompt positions),
      unclamped residual patch, attention / MLP / joint output patches on clamped G,
      nested parallel/orthogonal sufficiency and necessity at every scan layer
  P2  control subspaces (evil, evil+syc, style, random rank 1/2/4 x 3 seeds) decomposition at
      the carrier layer plus the three layers with the largest per-layer increase in R_l
  P3  anchor host Gs: clamp mediation and residual patch scan
  P4  full EM host: baseline and clamp repair
Every condition records aligned/misaligned/neutral log-likelihood, next-token entropy and
top-1 agreement with C, the unpatched hybrid, and the clamped hybrid.

Usage: python stage5b_activation_route.py --model qwen2_5_7b [--quick 8 --layers 8,14,19,20]
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(__file__))
from common import (  # noqa: E402
    CARRIER_DIR, EVAL_DIR, MODEL_SPECS, PAIRS_120, STAGE5A_REFERENCE, Hooks, capture, fn_add_delta, fn_clamp, fn_patch,
    load_sequences, load_state_dict_cpu, make_batches, per_sequence, score_batch, set_host,
)

CONTROL_SUBS = ["evil", "evil_syc", "style"] + [f"rand{k}_s{s}" for k in (1, 2, 4) for s in (0, 1, 2)]
TOL = 2e-3  # per-example tolerance for exact-identity checks (bf16 round-off)
PROTECTED_SUFFIXES = ("self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight", "self_attn.o_proj.weight",
                      "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------------------------------
# Condition specification
# --------------------------------------------------------------------------------------

def cond(name, host, ivs=(), store_ref=None, meta=None):
    return dict(name=name, host=host, ivs=list(ivs), store_ref=store_ref, meta=meta or {})


def clamp_iv(spec, sub="nested"):
    return ("clamp", spec["carrier_layer"], sub, "resp")


def register(hooks, iv, batch, cache_c, cache_h, subspaces):
    kind = iv[0]
    if kind == "clamp":
        _, l, sub, pos = iv
        hooks.resid(l, fn_clamp(subspaces[sub], cache_c["resid"][l], batch["masks"][pos]))
    elif kind == "patch_resid":
        _, l, pos = iv
        hooks.resid(l, fn_patch(cache_c["resid"][l], batch["masks"][pos]))
    elif kind == "patch_attn":
        _, l, pos = iv
        hooks.attn(l, fn_patch(cache_c["attn"][l], batch["masks"][pos]))
    elif kind == "patch_mlp":
        _, l, pos = iv
        hooks.mlp(l, fn_patch(cache_c["mlp"][l], batch["masks"][pos]))
    elif kind == "add_delta":
        _, l, sub, part, sign, pos = iv
        hooks.resid(l, fn_add_delta(subspaces[sub], part, sign, cache_h["resid"][l], cache_c["resid"][l], batch["masks"][pos]))
    else:
        raise ValueError(kind)


class Runner:
    def __init__(self, model_c, model_h, batches, subspaces, spec):
        self.model_c, self.model_h = model_c, model_h
        self.batches, self.subspaces, self.spec = batches, subspaces, spec

    def run(self, phase, host_label, conds, c_resid=(), c_attn=(), c_mlp=(), h_resid=(), energy=None):
        """Run conditions over all batches. Returns seq-level rows.

        The capture passes are recorded as conditions 'C' and host_label.
        energy: optional list of (layer, subspace) pairs for ||P dh||^2 accounting.
        """
        rows, energy_acc = [], {}
        for bi, batch in enumerate(self.batches):
            cache_c, sc_c = capture(self.model_c, batch, resid_layers=c_resid, attn_layers=c_attn, mlp_layers=c_mlp)
            cache_h, sc_h = capture(self.model_h, batch, resid_layers=h_resid)
            refs = {"C": sc_c["argmax"], "H": sc_h["argmax"]}
            for name, sc in (("C", sc_c), (host_label, sc_h)):
                for r in per_sequence(batch, sc, refs):
                    r.update(phase=phase, cond=name)
                    rows.append(r)
            if energy:
                self._energy(batch, cache_c, cache_h, energy, energy_acc)
            for cd in conds:
                model = self.model_c if cd["host"] == "C" else self.model_h
                hooks = Hooks(model)
                try:
                    for iv in cd["ivs"]:
                        register(hooks, iv, batch, cache_c, cache_h, self.subspaces)
                    sc = score_batch(model, batch)
                finally:
                    hooks.clear()
                if cd["store_ref"]:
                    refs[cd["store_ref"]] = sc["argmax"]
                for r in per_sequence(batch, sc, refs):
                    r.update(phase=phase, cond=cd["name"])
                    rows.append(r)
            del cache_c, cache_h
            if bi == 0 or (bi + 1) % 5 == 0:
                log(f"  {phase}: batch {bi + 1}/{len(self.batches)} (peak GPU memory {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB)")
        return rows, energy_acc

    def _energy(self, batch, cache_c, cache_h, pairs, acc):
        for l, sub in pairs:
            d = cache_h["resid"][l].float() - cache_c["resid"][l].float()
            U = self.subspaces[sub]
            par = (d @ U) @ U.T
            for pos in ("all", "resp"):
                m = batch["masks"][pos][..., None].float()
                key = (l, sub, pos)
                tot, pe = acc.get(key, (0.0, 0.0))
                acc[key] = (tot + float(((d * m) ** 2).sum()), pe + float(((par * m) ** 2).sum()))


# --------------------------------------------------------------------------------------
# Quick in-job summaries (the full statistics are computed by stage5b_analyze.py)
# --------------------------------------------------------------------------------------

def example_scores(rows):
    df = pd.DataFrame(rows)
    pairs = df[df.kind != "neutral"].pivot_table(index=["cond", "example_id"], columns="kind", values="lp_mean")
    return (pairs["mis"] - pairs["align"]).rename("S")


def mean_S(S, name):
    return float(S.loc[name].mean())


def invariant(S, a, b, label, report):
    diff = (S.loc[a] - S.loc[b]).abs()
    report[label] = dict(max_abs_diff=float(diff.max()), mean_abs_diff=float(diff.mean()), passed=bool(diff.max() < TOL))
    log(f"  invariant {label}: max|diff|={diff.max():.2e} {'ok' if diff.max() < TOL else 'FAILED'}")


def save_rows(rows, path):
    pd.DataFrame(rows).to_parquet(path, index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODEL_SPECS))
    ap.add_argument("--quick", type=int, default=0, help="use only the first N pairs (test mode)")
    ap.add_argument("--layers", type=str, default="", help="comma list restricting scan layers (test mode)")
    ap.add_argument("--controls", type=str, default="", help="comma list restricting control subspaces (test mode)")
    ap.add_argument("--max_tokens", type=int, default=4096)
    ap.add_argument("--tag", type=str, default="")
    ap.add_argument("--output-root", type=str, default="",
                    help="optional alternate evaluation root for a versioned ACL run")
    ap.add_argument("--rendering", choices=("legacy", "training"), default="legacy",
                    help="legacy frozen manual prefix, or the original training chat template")
    ap.add_argument("--protected-matrices-only", action="store_true",
                    help="graft the ACL-protected Q/K/V/O/gate/up/down matrices, excluding norms")
    ap.add_argument("--continue-after-audit-stop", action="store_true",
                    help="record the frozen baseline audit stop but finish the prespecified causal assays")
    ap.add_argument("--pairs", type=str, default=PAIRS_120,
                    help="paired-completion JSON to score (default: the frozen 120-pair assay)")
    ap.add_argument("--ctrl", type=str, default="", help="benign (C) checkpoint directory; default from MODEL_SPECS")
    ap.add_argument("--em", type=str, default="", help="harmful (E) checkpoint directory; default from MODEL_SPECS")
    args = ap.parse_args()

    spec = dict(MODEL_SPECS[args.model])
    spec["ctrl"] = args.ctrl or spec["ctrl"]
    spec["em"] = args.em or spec["em"]
    device = "cuda:0"
    torch.manual_seed(0)
    out_tag = args.tag or ("_training" if args.rendering == "training" else "")
    output_root = args.output_root or EVAL_DIR
    out_dir = os.path.join(output_root, "stage5b", args.model + out_tag)
    if args.output_root and os.path.isdir(out_dir) and os.listdir(out_dir):
        raise FileExistsError(f"versioned ACL route output already exists: {out_dir}")
    os.makedirs(out_dir, exist_ok=True)
    scan = [int(x) for x in args.layers.split(",")] if args.layers else spec["scan_layers"]
    controls = args.controls.split(",") if args.controls else CONTROL_SUBS
    c, (g0, g1) = spec["carrier_layer"], spec["graft"]
    graft_layers = list(range(g0, g1 + 1))
    anchor_layers = list(range(spec["anchor"][0], spec["anchor"][1] + 1))
    graft_filter = (lambda n: any(n.endswith(s) for s in PROTECTED_SUFFIXES)) if args.protected_matrices_only else None
    log(f"model={args.model} graft={g0}:{g1} anchor={anchor_layers[0]}:{anchor_layers[-1]} carrier={c} scan={scan[0]}..{scan[-1]} ({len(scan)})")

    tok = AutoTokenizer.from_pretrained(spec["hf_id"], revision=spec["revision"])
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    seqs = load_sequences(tok, spec["chat"], include_neutral=True, quick=args.quick, rendering=args.rendering,
                          pairs_path=args.pairs)
    batches = make_batches(seqs, pad_id, device, max_tokens=args.max_tokens)
    log(f"{len(seqs)} sequences in {len(batches)} batches")

    bundle = torch.load(os.path.join(CARRIER_DIR, args.model, "subspaces.pt"), weights_only=True)
    subspaces = {k: v.to(device=device, dtype=torch.float32) for k, v in bundle["subspaces"].items()}

    model_c = AutoModelForCausalLM.from_pretrained(spec["ctrl"], dtype=torch.bfloat16, device_map=device)
    model_h = AutoModelForCausalLM.from_pretrained(spec["ctrl"], dtype=torch.bfloat16, device_map=device)
    model_c.eval()
    model_h.eval()
    em_sd = load_state_dict_cpu(spec["em"])
    log("models loaded")

    runner = Runner(model_c, model_h, batches, subspaces, spec)
    manifest = dict(model=args.model, spec=spec, scan_layers=scan, controls=controls, quick=args.quick,
                    rendering=args.rendering, route_tag=out_tag, pairs=args.pairs,
                    protected_matrices_only=args.protected_matrices_only,
                    n_sequences=len(seqs), n_batches=len(batches))
    invariants = {}

    # ---------------------------------------------------------------- P0 baselines
    set_host(model_h, model_c, em_sd, em_layers=graft_layers, name_filter=graft_filter)
    p0 = [cond("C|clamp_nested", "C", [clamp_iv(spec)]),
          cond("G|clamp_nested", "H", [clamp_iv(spec)], store_ref="Hc")]
    p0 += [cond(f"G|clamp_{s}", "H", [clamp_iv(spec, s)]) for s in controls]
    rows0, _ = runner.run("P0", "G", p0, c_resid=[c])
    save_rows(rows0, os.path.join(out_dir, "rows_P0.parquet"))
    S0 = example_scores(rows0)
    S_C, S_G = mean_S(S0, "C"), mean_S(S0, "G")
    S_Cc, S_Gc = mean_S(S0, "C|clamp_nested"), mean_S(S0, "G|clamp_nested")
    TE, DE = S_G - S_C, S_Gc - S_Cc
    MF = 1 - DE / TE if abs(TE) > 1e-9 else float("nan")
    invariant(S0, "C|clamp_nested", "C", "C_clamped_to_itself_equals_C", invariants)
    te_i = S0.loc["G"] - S0.loc["C"]
    de_i = S0.loc["G|clamp_nested"] - S0.loc["C|clamp_nested"]
    baseline = dict(S_C=S_C, S_G=S_G, S_Cc=S_Cc, S_Gc=S_Gc, TE=TE, DE=DE, MF=MF,
                    clamp_MF={s: 1 - (mean_S(S0, f"G|clamp_{s}") - S_Cc) / TE for s in controls})
    log(f"P0: S_C={S_C:.4f} S_G={S_G:.4f} TE={TE:.4f} DE={DE:.4f} MF={MF:.3f}")

    # Audit stop rule (fixed in the plan): MF > 0.35, or TE more than 25% from Stage 5A, or no transfer.
    stop_reasons = []
    if MF > 0.35:
        stop_reasons.append(f"MF={MF:.3f} > 0.35")
    ref = STAGE5A_REFERENCE.get(args.model)
    if ref and abs(TE - ref["TE"]) / ref["TE"] > 0.25:
        stop_reasons.append(f"TE={TE:.4f} differs from Stage 5A TE={ref['TE']:.4f} by more than 25%")
    rng = np.random.default_rng(0)
    boot = [rng.choice(te_i.values, len(te_i)).mean() for _ in range(1000)]
    boot_de = [rng.choice(de_i.values, len(de_i)).mean() for _ in range(1000)]
    if np.percentile(boot, 2.5) <= 0 or np.percentile(boot_de, 2.5) <= 0:
        stop_reasons.append("TE or DE 95% interval includes zero (no transfer to trace)")
    baseline["audit_stop_reasons"] = stop_reasons
    manifest["baseline"] = baseline
    with open(os.path.join(out_dir, "baseline.json"), "w") as f:
        json.dump(baseline, f, indent=2)
    if stop_reasons and not args.quick:
        with open(os.path.join(out_dir, "AUDIT_STOP.json"), "w") as f:
            json.dump(dict(reasons=stop_reasons, baseline=baseline), f, indent=2)
        if args.continue_after_audit_stop:
            log(f"BASELINE AUDIT STOP RECORDED; continuing the prespecified causal assays: {stop_reasons}")
        else:
            log(f"AUDIT STOP: {stop_reasons}")
            sys.exit(3)

    # ---------------------------------------------------------------- P1 main scans
    ref_c = cond("G|clamp_nested", "H", [clamp_iv(spec)], store_ref="Hc", meta={"reference": True})
    p1 = [ref_c]
    for l in scan:
        # ACL position audit: all/prefix/response, three first-token windows,
        # and four equal answer-token quartiles.  Each completion gets masks
        # from its own scored answer length in common.make_batches.
        for pos in ("all", "resp", "prompt", "resp1", "resp4", "resp8", "q1", "q2", "q3", "q4"):
            p1.append(cond(f"G|clamp_nested|patch_resid@{l}:{pos}", "H", [clamp_iv(spec), ("patch_resid", l, pos)]))
        p1.append(cond(f"G|patch_resid@{l}:all", "H", [("patch_resid", l, "all")]))
        p1.append(cond(f"G|clamp_nested|patch_attn@{l}:all", "H", [clamp_iv(spec), ("patch_attn", l, "all")]))
        p1.append(cond(f"G|clamp_nested|patch_mlp@{l}:all", "H", [clamp_iv(spec), ("patch_mlp", l, "all")]))
        p1.append(cond(f"G|clamp_nested|patch_both@{l}:all", "H", [clamp_iv(spec), ("patch_attn", l, "all"), ("patch_mlp", l, "all")]))
        for part in ("par", "perp"):
            p1.append(cond(f"C|add_nested_{part}@{l}:all", "C", [("add_delta", l, "nested", part, +1.0, "all")]))
            p1.append(cond(f"G|sub_nested_{part}@{l}:all", "H", [("add_delta", l, "nested", part, -1.0, "all")]))
    if c in scan:
        for part in ("par", "perp"):
            p1.append(cond(f"C|add_nested_{part}@{c}:resp", "C", [("add_delta", c, "nested", part, +1.0, "resp")]))
            p1.append(cond(f"G|sub_nested_{part}@{c}:resp", "H", [("add_delta", c, "nested", part, -1.0, "resp")]))
    resid_layers = sorted(set(scan) | {c})
    log(f"P1: {len(p1)} conditions")
    rows1, energy1 = runner.run("P1", "G", p1, c_resid=resid_layers, c_attn=scan, c_mlp=scan, h_resid=scan,
                                energy=[(l, "nested") for l in scan])
    rows1 = [r for r in rows1 if not (r["cond"] == "G|clamp_nested")]  # duplicate of P0 reference
    save_rows(rows1, os.path.join(out_dir, "rows_P1.parquet"))
    S1 = pd.concat([S0, example_scores(rows1).drop(index=["C", "G"], level=0, errors="ignore")])

    for l in scan:
        if l >= g1:
            invariant(S1, f"G|clamp_nested|patch_resid@{l}:all", "C", f"resid_patch_all_at_{l}_equals_C", invariants)
    if c in scan and c > g1:
        invariant(S1, f"G|sub_nested_par@{c}:resp", "G|clamp_nested", "resp_parallel_removal_equals_clamp", invariants)
        invariant(S1, f"C|add_nested_par@{c}:all", f"G|sub_nested_perp@{c}:all", "parallel_sufficiency_equals_orthogonal_necessity_at_carrier", invariants)

    # Pre-specified selection: top-3 layers by per-layer increase in R_l (all positions).
    R = {l: (S_Gc - mean_S(S1, f"G|clamp_nested|patch_resid@{l}:all")) / DE for l in scan}
    prev, dR = 0.0, {}
    for l in scan:
        dR[l] = R[l] - prev
        prev = R[l]
    top3 = sorted(scan, key=lambda l: (-dR[l], l))[:3]
    dec_layers = sorted(set(top3) | ({c} if c in scan else set()))
    manifest["R_point"] = R
    manifest["dR_point"] = dR
    manifest["top3_dR_layers"] = top3
    manifest["decomposition_layers"] = dec_layers
    log(f"R_l: {', '.join(f'{l}:{R[l]:.2f}' for l in scan)}")
    log(f"top-3 dR layers {top3}; decomposition layers {dec_layers}")

    # ---------------------------------------------------------------- P2 control subspaces
    p2 = []
    for l in dec_layers:
        for s in controls:
            for part in ("par", "perp"):
                p2.append(cond(f"C|add_{s}_{part}@{l}:all", "C", [("add_delta", l, s, part, +1.0, "all")]))
                p2.append(cond(f"G|sub_{s}_{part}@{l}:all", "H", [("add_delta", l, s, part, -1.0, "all")]))
    p2 = [ref_c] + p2
    log(f"P2: {len(p2)} conditions")
    rows2, energy2 = runner.run("P2", "G", p2, c_resid=resid_layers, h_resid=dec_layers,
                                energy=[(l, s) for l in dec_layers for s in controls])
    rows2 = [r for r in rows2 if r["cond"] not in ("G|clamp_nested", "C", "G")]
    save_rows(rows2, os.path.join(out_dir, "rows_P2.parquet"))

    energy = {f"{l}|{s}|{pos}": dict(total=t, parallel=p, fraction=p / t if t > 0 else float("nan"))
              for (l, s, pos), (t, p) in {**energy1, **energy2}.items()}
    with open(os.path.join(out_dir, "energy_fractions.json"), "w") as f:
        json.dump(energy, f, indent=2)

    # ---------------------------------------------------------------- P3 anchor host
    set_host(model_h, model_c, em_sd, em_layers=anchor_layers, name_filter=graft_filter)
    p3 = [cond("Gs|clamp_nested", "H", [clamp_iv(spec)], store_ref="Hc")]
    p3 += [cond(f"Gs|clamp_nested|patch_resid@{l}:all", "H", [clamp_iv(spec), ("patch_resid", l, "all")]) for l in scan]
    log(f"P3: {len(p3)} conditions")
    rows3, _ = runner.run("P3", "Gs", p3, c_resid=resid_layers)
    rows3 = [r for r in rows3 if r["cond"] != "C"]
    save_rows(rows3, os.path.join(out_dir, "rows_P3.parquet"))

    # ---------------------------------------------------------------- P4 full EM host
    set_host(model_h, model_c, em_sd, full_em=True)
    p4 = [cond("E|clamp_nested", "H", [clamp_iv(spec)], store_ref="Hc")]
    rows4, _ = runner.run("P4", "E", p4, c_resid=[c])
    rows4 = [r for r in rows4 if r["cond"] != "C"]
    save_rows(rows4, os.path.join(out_dir, "rows_P4.parquet"))

    # ---------------------------------------------------------------- P5 reverse graft (E <- C(A_D))
    # This is the matched necessity counterpart to the selected-matrix graft.
    set_host(model_h, model_c, em_sd, full_em=True)
    pc, ph = dict(model_c.named_parameters()), dict(model_h.named_parameters())
    for name, p in ph.items():
        if any(name.startswith(f"model.layers.{l}.") for l in graft_layers) and (graft_filter is None or graft_filter(name)):
            p.data.copy_(pc[name].data)
    p5 = [cond("R|clamp_nested", "H", [clamp_iv(spec)], store_ref="Rc")]
    rows5, _ = runner.run("P5", "R", p5, c_resid=[c])
    rows5 = [r for r in rows5 if r["cond"] != "C"]
    save_rows(rows5, os.path.join(out_dir, "rows_P5.parquet"))

    manifest["invariants"] = invariants
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    log("done")


if __name__ == "__main__":
    main()
