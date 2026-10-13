"""Stage 9 Step 5 scoring and tables: residual shifts of the fine-tunes trained with blocked directions (Qwen2.5-7B, seed 42).

Plan: docs/plans/persona-control.md#stage9-behavior-routes, section 10 (fixed before any result).

  score  (GPU)  Step 3's pairs (components.load_pairs, both halves) under, for each blocked set <s> (union256, union256s):
                  C (native), C0, C-B:<s>        C with zero edits; C minus its own blocked part B(C - base)
                  E (native), E0, E-B:<s>        the same for E (E-B: the post hoc block, added before any Step 5 result)
                  Eblk:<s>, Eblk:<s>:raw         the fine-tune trained with <s> blocked, with the edit -B(W - base) (the
                                                 model as trained) and without it (the stored weights, a check)
                in components.SidePaths arithmetic (zero edits on every matrix outside the block).  Every side-path
                condition is also scored under the persona hold (Step 3's nested rank-4 carrier at layer 20 clamped to
                C0's values at answer positions).  manifest.json records the block checks: the blocked part of
                W_eff - W_base (0 up to float32 error) and the share of ||W - base||^2 along the blocked directions.
  report (CPU)  residual shifts on the evaluation half (step3_report.Stats: control-fitted null line, 2,000 bootstrap
                draws over questions, the same draws for every condition), retained fractions R_j(X) / R_j(E0), the
                dissociation (retained fraction of EM minus the mean retained fraction of RR, WI and TQA), persona parts
                P(X) = R_EM(X) - R_EM(X|hold), and the checks against Step 3 (C0 log-probabilities, E0 residual shifts).
Output: eval_runs/persona_control_stage9/step5/<run_id>/ (rows.parquet, manifest.json, step5_conditions.csv,
step5_retained.csv, step5_checks.json).
Usage: python step5_score.py {score,report} --run-id s9s5_20260930 [--quick 2 --out-dir <dir>]
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import components as S3  # noqa: E402
import train_blocked as TB  # noqa: E402

ROOT = S3.ROOT
STEP5_ROOT = f"{ROOT}/eval_runs/persona_control_stage9/step5"
BASE_DIR = "/net/scratch/jiaweizhang/hf/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
NARROW = ("RR", "WI", "TQA")  # the dissociation's narrow behaviors; OC is left out (plan, measure 4)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def block_check(params, base, blocks, device):
    """For W_eff - base = D - B(D), D = W - base: its blocked part relative to its norm (largest over matrices), and the
    share of ||D||^2 along the blocked directions (||B(D)||^2 / ||D||^2 summed over the blocked matrices)."""
    b2 = d2 = 0.0
    worst = 0.0
    for n, (U, V) in blocks.items():
        D = params[n].float() - base[n].to(device).float()
        a, b = U.t() @ D, D @ V
        b = b - U @ (U.t() @ b)
        b2 += float((a * a).sum() + (b * b).sum())
        d2 += float((D * D).sum())
        De = D - U @ a - b @ V.t()
        a, b = U.t() @ De, De @ V
        b = b - U @ (U.t() @ b)
        worst = max(worst, float(((a * a).sum() + (b * b).sum()).sqrt() / De.norm()))
    return dict(blocked_energy_share=b2 / d2, update_norm=d2 ** .5, w_eff_blocked_rel_max=worst)


def step_score(args, out):
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.backends.cuda.matmul.allow_tf32 = False
    t0, device = time.time(), "cuda:0"
    spec = S3.MODEL_SPECS["qwen2_5_7b"]
    tok = AutoTokenizer.from_pretrained(spec["hf_id"], revision=spec["revision"])
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    pairs, split = S3.load_pairs(args.quick)
    seqs = S3.build_seqs(tok, pairs)
    persona = [s for s in seqs if s["set"] in S3.PERSONA_SETS]
    pb = S3.make_batches(persona, pad_id, device, max_tokens=args.max_tokens)
    batches = pb + S3.make_batches([s for s in seqs if s["set"] not in S3.PERSONA_SETS], pad_id, device,
                                   max_tokens=args.max_tokens)
    log(f"{len(pairs)} pairs, {len(seqs)} sequences, {len(batches)} batches ({len(pb)} persona)")

    model = AutoModelForCausalLM.from_pretrained(S3.CTRL, dtype=torch.bfloat16, device_map=device).eval()
    model.requires_grad_(False)
    names = S3.selected_param_names(model, list(range(spec["n_layers"])))
    params = dict(model.named_parameters())
    blocks = {s: TB.load_blocks(s, device) for s in TB.SETS}
    base = S3.load_selected_safetensors(BASE_DIR, sorted({n for b in blocks.values() for n in b}))
    zero = torch.zeros(1, device=device)
    models = {"E": S3.EM, **{f"Eblk:{s}": f"{TB.model_dir(s)}/checkpoint-100pct" for s in TB.SETS}}
    missing = [k for k, p in models.items() if not os.path.exists(f"{p}/config.json")]
    if missing and not args.quick:
        raise FileNotFoundError(f"missing models: {missing}")

    def edits(block=None):
        e = {n: ("lowrank", None, zero, None) for n in names}
        if block:
            for n, (U, V) in blocks[block].items():
                e[n] = TB.minus_block_edits(params[n], base[n].to(device), U, V)
        return e

    def load_weights(path):
        sd = S3.load_state_dict_cpu(path)
        for n, p in params.items():
            if n == "lm_head.weight" and n not in sd:
                assert model.config.tie_word_embeddings, n
                continue
            p.data.copy_(sd[n].to(p.device, dtype=p.dtype))

    sub = torch.load(os.path.join(S3.CARRIER_DIR, "qwen2_5_7b", "subspaces.pt"), weights_only=True)["subspaces"]
    run = S3.Runner(model, batches, len(pb), len(seqs), sub["nested"].to(device=device, dtype=torch.float32),
                    spec["carrier_layer"])
    checks = {}
    run.run("C", None, hold=False)
    run.run("C0", edits(), capture_targets=True)
    for s in TB.SETS:
        checks[f"C|{s}"] = block_check(params, base, blocks[s], device)
        run.run(f"C-B:{s}", edits(s))
    for label in ["E"] + [f"Eblk:{s}" for s in TB.SETS]:
        if label in missing:
            log(f"{label}: no model at {models[label]}; skipped (quick test)")
            continue
        load_weights(models[label])
        if label == "E":
            run.run("E", None, hold=False)
            run.run("E0", edits())
            for s in TB.SETS:
                checks[f"E|{s}"] = block_check(params, base, blocks[s], device)
                run.run(f"E-B:{s}", edits(s))
        else:
            s = label.split(":")[1]
            for s2 in TB.SETS:
                checks[f"{label}|{s2}"] = block_check(params, base, blocks[s2], device)
            run.run(label, edits(s))
            run.run(f"{label}:raw", edits())
        torch.cuda.empty_cache()
    for k, v in checks.items():
        log(f"block check {k}: {json.dumps(v)}")
    n_rows = S3.write_rows(run, seqs, pairs, f"{out}/rows.parquet")
    checks.update(c0_vs_c_max_abs_lp_sum=float(np.abs(run.lp["C0"] - run.lp["C"]).max()))
    if "E" in run.lp:
        checks.update(e0_vs_e_max_abs_lp_sum=float(np.abs(run.lp["E0"] - run.lp["E"]).max()))
    manifest = dict(run_id=args.run_id, quick=args.quick, models=models, ctrl=S3.CTRL, base=BASE_DIR,
                    blocks=json.load(open(f"{TB.BLOCK_DIR}/blocks.json")), n_pairs=len(pairs), n_sequences=len(seqs),
                    n_batches=len(batches), carrier_layer=spec["carrier_layer"], hold_subspace="nested",
                    conditions=sorted(run.lp), seconds_per_condition=run.secs, checks=checks,
                    base_output_path=dict(S3.MM_PATH), scripts={os.path.basename(p): TB.sha256(p) for p in (
                        os.path.abspath(__file__), TB.__file__, S3.__file__)},
                    seconds=round(time.time() - t0), torch=torch.__version__, transformers=transformers.__version__,
                    gpu=torch.cuda.get_device_name(0))
    json.dump(manifest, open(f"{out}/manifest.json", "w"), indent=1)
    log(f"{n_rows} rows; done in {time.time() - t0:.0f}s")


def step_report(args, out):
    import step3_report as R3

    man = json.load(open(f"{out}/manifest.json"))
    llr, _ = R3.load(out)
    ev = llr[llr.index.get_level_values("half") == "eval"]
    st = R3.Stats(ev, args.n_boot)
    c = st.col
    Rb = {b: st.R(ss) for b, ss in R3.BEHAVIORS.items()}
    P = {x: Rb["EM"][:, c(x)] - Rb["EM"][:, c(x + "|hold")] for x in st.conds if x + "|hold" in st.conds}
    conds = [x for x in st.conds if not x.endswith("|hold")]

    rows = []
    for x in conds:
        r = dict(cond=x, b_all=st.lines["all"][1][0, c(x)])
        for b in R3.COLS:
            r[b], (r[f"{b}_lo"], r[f"{b}_hi"]) = Rb[b][0, c(x)], R3.ci(Rb[b][:, c(x)])
        if x in P:
            r["P_EM"], (r["P_EM_lo"], r["P_EM_hi"]) = P[x][0], R3.ci(P[x])
            share = P[x] / Rb["EM"][:, c(x)]
            r["P_share"], (r["P_share_lo"], r["P_share_hi"]) = share[0], R3.ci(share)
        rows.append(r)
    tab = pd.DataFrame(rows)
    tab.to_csv(f"{out}/step5_conditions.csv", index=False)

    ret_rows = []
    e0 = {b: Rb[b][:, c("E0")] for b in R3.COLS}
    for x in conds:
        if not x.startswith(("E-B:", "Eblk:")):
            continue
        ret = {b: Rb[b][:, c(x)] / e0[b] for b in R3.COLS}
        r = dict(cond=x)
        for b in R3.COLS:
            r[b], (r[f"{b}_lo"], r[f"{b}_hi"]) = ret[b][0], R3.ci(ret[b])
        dis = ret["EM"] - np.mean([ret[b] for b in NARROW], 0)
        r["dissociation"], (r["dissociation_lo"], r["dissociation_hi"]) = dis[0], R3.ci(dis)
        ret_rows.append(r)
    ret = pd.DataFrame(ret_rows)
    ret.to_csv(f"{out}/step5_retained.csv", index=False)

    checks = dict(man["checks"])
    s3 = pd.read_csv(f"{ROOT}/eval_runs/persona_control_stage9/step3/s9s3_20260928/step3_conditions.csv").set_index("cond")
    if "E0" in conds:
        checks["E0_minus_step3_E0"] = {b: float(Rb[b][0, c("E0")] - s3.loc["E0", b]) for b in R3.COLS}
        checks["E0_within_.10_of_step3"] = bool(max(abs(v) for v in checks["E0_minus_step3_E0"].values()) <= .10)
    if not man["quick"]:
        s3rows = pd.read_parquet(f"{ROOT}/eval_runs/persona_control_stage9/step3/s9s3_20260928/rows.parquet",
                                 filters=[("cond", "in", ["C0", "E0"])])
        here = pd.read_parquet(f"{out}/rows.parquet", filters=[("cond", "in", ["C0", "E0"])])
        key = ["cond", "set", "prompt_id", "kind"]
        j = here.merge(s3rows, on=key, suffixes=("", "_s3"))
        checks["lp_sum_vs_step3_max_abs"] = {k: float((g.lp_sum - g.lp_sum_s3).abs().max()) for k, g in j.groupby("cond")}
    json.dump(checks, open(f"{out}/step5_checks.json", "w"), indent=1, default=float)

    with pd.option_context("display.width", 300, "display.max_columns", 40):
        print("residual shifts R_j(X), nats [95%], evaluation half; P_EM = persona part; P_share = P_EM / R_EM:")
        t = pd.DataFrame({b: [R3.fmt(Rb[b][:, c(x)]) for x in conds] for b in R3.COLS}, index=conds)
        t["P_EM"] = [R3.fmt(P[x]) if x in P else "" for x in conds]
        t["P_share"] = [R3.fmt(P[x] / Rb["EM"][:, c(x)]) if x in P else "" for x in conds]
        print(t.to_string())
        print("\nretained fraction R_j(X) / R_j(E0) [95%]; dissociation = EM - mean(RR, WI, TQA):")
        r = pd.DataFrame({b: [f"{v:.2f} [{lo:.2f}, {hi:.2f}]" for v, lo, hi in zip(ret[b], ret[f"{b}_lo"], ret[f"{b}_hi"])]
                          for b in R3.COLS + ["dissociation"]}, index=ret.cond)
        print(r.to_string())
    print("\nchecks: " + json.dumps(checks, default=float))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["score", "report"])
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--quick", type=int, default=0, help="score: first N pairs per set and half (needs --out-dir)")
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()
    if args.quick and not args.out_dir:
        raise ValueError("--quick writes only to --out-dir")
    out = os.path.abspath(args.out_dir or f"{STEP5_ROOT}/{args.run_id}")
    if args.step == "score":
        if os.path.exists(f"{out}/rows.parquet"):
            raise FileExistsError(f"{out}/rows.parquet")
        os.makedirs(out, exist_ok=True)
        step_score(args, out)
    else:
        step_report(args, out)


if __name__ == "__main__":
    main()
