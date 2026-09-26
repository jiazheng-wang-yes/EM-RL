"""Stage 7 W5 task 4: seed 42 vs seed 43 replication table for the training-rendered route.

Every route here uses training rendering, the seven region matrices (q, k, v, o, gate, up, down),
the Round 1 regions (Qwen layers 8-19, Llama 5-22) and the persona hold at the carrier layer
(Qwen 20, Llama 23). Inputs are each route directory's per_example_pairs.parquet.

Symbols. S = mean over the 120 prompt pairs of lp(misaligned answer) - lp(aligned answer).
C = benign fine-tune, E = harmful fine-tune, G = C with E's region, R = E with C's region
(reverse graft), "|hold" = the four nested carrier coordinates clamped to C's values at response
positions of the carrier layer.
  Delta = S(E) - S(C)                      T = S(G) - S(C)          D = S(G|hold) - S(C|hold)
  removed = 1 - D/T                        T/Delta
  E repair = (S(E) - S(E|hold)) / Delta
  reverse NE = S(E) - S(R)                 reverse removed = 1 - (S(E|hold) - S(R|hold)) / NE
  share(s) = 1 - (S(G|clamp_s) - S(C|hold)) / T for s = evil, evil_syc, style; rand4 averages the
  three rand4 seeds.
These are the stage5b_analyze.py definitions. Intervals come from the analyzer's own prompt-cluster
bootstrap (stage5b_analyze.Means: 2,000 draws, seed 0, 120 pairs in 108 clusters). Both seeds use the
same draws, so each seed 43 - seed 42 difference gets a paired interval. The script checks that
its seed-wise estimates and intervals equal each route's summary.json.

Also listed, read from their files: two earlier reruns, the Qwen2.5 Stage 5A endpoint and the
Qwen3 retrain against its Stage 3 reference.

Usage: python w5_replication_table.py [--out eval_runs/persona_control_stage7/w5_replication]
"""

import argparse
import csv
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "stage6"))
from common import ROOT  # noqa: E402
from stage5b_analyze import Means, summarize  # noqa: E402

ROUND1 = os.path.join(ROOT, "eval_runs/persona_control_arr/round1/20260924T063649Z")
W5 = os.path.join(ROOT, "eval_runs/persona_control_stage7/w5_replication")
ROUTES = {
    "qwen2_5_7b": {42: f"{ROUND1}/route_qwen/stage5b/qwen2_5_7b_training_region8_19_confirmation",
                   43: f"{W5}/seed43/stage5b/qwen2_5_7b_training_region8_19_confirmation"},
    "llama3_1_8b": {42: f"{ROUND1}/route_llama/stage5b/llama3_1_8b_training_region5_22_confirmation",
                    43: f"{W5}/seed43/stage5b/llama3_1_8b_training_region5_22_confirmation"},
    "qwen3_1_7b": {42: f"{W5}/seed42/stage5b/qwen3_1_7b_training_region8_19_confirmation",
                   43: f"{W5}/seed43/stage5b/qwen3_1_7b_training_region8_19_confirmation"},
}
STATS = ["S_C", "S_E", "S_G", "Delta", "T", "D", "removed", "T_over_Delta", "E_repair", "reverse_NE",
         "reverse_removed", "share_evil", "share_evil_syc", "share_style", "share_rand4"]
SUMMARY_KEY = {"S_C": "S[C]", "S_E": "S[E]", "S_G": "S[G]", "Delta": "DeltaS_EM", "T": "TE", "D": "DE",
               "removed": "MF", "T_over_Delta": "TE_over_DeltaS", "E_repair": "E_clamp_repair_fraction",
               "reverse_NE": "NE_reverse_graft", "reverse_removed": "MF_reverse_graft"}
STAGE5A_CSV = os.path.join(ROOT, "experiments/persona_control/stage5a/results/graft_trajectory.csv")
STAGE5A_ONSET = os.path.join(ROOT, "experiments/persona_control/stage5a/results/onset_summary.json")
STAGE3_QWEN3 = os.path.join(ROOT, "experiments/persona_control/results_replication/qwen3_1_7b/replication_summary.json")
STAGE6_QWEN3_LEGACY = os.path.join(ROOT, "eval_runs/persona_control_stage6/stage5b/qwen3_1_7b/summary.json")


def route_stats(m):
    """(point, bootstrap draws) for every statistic, from the condition means in m."""
    g = m.get

    def diff(a, b):
        return g(a)[0] - g(b)[0], g(a)[1] - g(b)[1]

    def one_minus_ratio(num, den):
        return 1 - num[0] / den[0], 1 - num[1] / den[1]

    out = {"S_C": g("C"), "S_E": g("E"), "S_G": g("G")}
    delta, t, d = diff("E", "C"), diff("G", "C"), diff("G|clamp_nested", "C|clamp_nested")
    out.update(Delta=delta, T=t, D=d, removed=one_minus_ratio(d, t), T_over_Delta=(t[0] / delta[0], t[1] / delta[1]))
    rep = diff("E", "E|clamp_nested")
    out["E_repair"] = (rep[0] / delta[0], rep[1] / delta[1])
    ne = diff("E", "R")
    out["reverse_NE"] = ne
    out["reverse_removed"] = one_minus_ratio(diff("E|clamp_nested", "R|clamp_nested"), ne)
    for s in ("evil", "evil_syc", "style"):
        out[f"share_{s}"] = one_minus_ratio(diff(f"G|clamp_{s}", "C|clamp_nested"), t)
    rand = [one_minus_ratio(diff(f"G|clamp_rand4_s{i}", "C|clamp_nested"), t) for i in range(3)]
    out["share_rand4"] = (float(np.mean([r[0] for r in rand])), np.mean([r[1] for r in rand], axis=0))
    return out


def check_against_summary(model, seed, route_dir, summ):
    """The recomputed estimates and intervals must equal the analyzer's summary.json."""
    with open(os.path.join(route_dir, "summary.json")) as f:
        base = json.load(f)["full"]["baseline"]
    worst = 0.0
    for k, v in summ.items():
        ref = base[SUMMARY_KEY[k]] if k in SUMMARY_KEY else base["clamp_MF_by_subspace"][k.replace("share_", "")]
        worst = max(worst, *(abs(v[x] - ref[x]) for x in ("est", "lo", "hi")))
    if worst > 1e-9:
        raise AssertionError(f"{model} seed {seed}: recomputed statistics differ from summary.json by {worst:.2e}")
    return worst


def route_meta(route_dir):
    meta = {}
    with open(os.path.join(route_dir, "manifest.json")) as f:
        man = json.load(f)
    meta.update(ctrl=man["spec"]["ctrl"], em=man["spec"]["em"], rendering=man.get("rendering"),
                protected_matrices_only=man.get("protected_matrices_only"))
    with open(os.path.join(route_dir, "baseline.json")) as f:
        meta["audit_stop_reasons"] = json.load(f).get("audit_stop_reasons", [])
    with open(os.path.join(route_dir, "summary.json")) as f:
        s = json.load(f)
    meta["n_strong_destructive"] = s["quality"]["n_strong_destructive"]
    meta["max_invariant_abs_diff"] = max((v.get("max_abs_diff", 0.0) for v in s["invariants"].values()
                                          if isinstance(v, dict)), default=None)
    return meta


def earlier_reruns():
    with open(STAGE5A_CSV) as f:
        end = [r for r in csv.DictReader(f) if r["step"] == "184"][0]
    with open(STAGE5A_ONSET) as f:
        onset = json.load(f)
    with open(STAGE3_QWEN3) as f:
        s3 = json.load(f)["causal_clamp"]
    with open(STAGE6_QWEN3_LEGACY) as f:
        s6 = json.load(f)["full"]["baseline"]
    return {
        "qwen2_5_7b_stage5a_endpoint": dict(
            source=[os.path.relpath(STAGE5A_CSV, ROOT), os.path.relpath(STAGE5A_ONSET, ROOT)],
            note=("separate seed-42 training run (stage5a/train_and_decompose.py, joint paired loader); "
                  "graft of layers 8-19; step 184"),
            TE=float(end["TE"]), TE_ci=json.loads(end["TE_ci"]), DE=float(end["DE"]), DE_ci=json.loads(end["DE_ci"]),
            MF=float(end["MF"]), delta_S=float(onset["endpoint_delta_S"])),
        "qwen3_1_7b_retrain_vs_stage3": dict(
            source=[os.path.relpath(STAGE3_QWEN3, ROOT), os.path.relpath(STAGE6_QWEN3_LEGACY, ROOT)],
            note=("Stage 3 checkpoints were deleted; stage6/train_qwen3_pair.py retrained the pair with the same "
                  "recipe and seed 42. Stage 3 values come from its causal-clamp pipeline; retrain values from the "
                  "Stage 6 route, whose manifest predates the rendering flag (legacy rendering, full region graft)."),
            stage3=dict(S_C=s3["S_ctrl"], S_E=s3["S_EM"], Delta=s3["gap"]),
            retrain=dict(S_C=s6["S[C]"], S_E=s6["S[E]"], Delta=s6["DeltaS_EM"])),
    }


def fmt(x):
    return f"{x['est']:.4f} [{x['lo']:.4f}, {x['hi']:.4f}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=W5)
    args = ap.parse_args()

    table, meta, checks = [], {}, {}
    for model, seeds in ROUTES.items():
        present = {s: d for s, d in seeds.items() if os.path.exists(os.path.join(d, "per_example_pairs.parquet"))}
        stats = {}
        for seed, d in present.items():
            wide = pd.read_parquet(os.path.join(d, "per_example_pairs.parquet"))
            examples = sorted(wide.example_id.unique())
            if len(examples) != 120:
                raise AssertionError(f"{d}: {len(examples)} examples, expected 120")
            stats[seed] = (examples, route_stats(Means(wide, examples)))
            meta[f"{model}_seed{seed}"] = dict(route_dir=os.path.relpath(d, ROOT), **route_meta(d))
        if 42 in stats and 43 in stats and stats[42][0] != stats[43][0]:
            raise AssertionError(f"{model}: the two seeds scored different example sets")
        for seed in stats:
            summ = {k: summarize(*v) for k, v in stats[seed][1].items()}
            checks[f"{model}_seed{seed}"] = check_against_summary(model, seed, present[seed], summ)
            stats[seed] = (stats[seed][0], stats[seed][1], summ)
        for k in STATS:
            row = dict(model=model, stat=k)
            for seed in (42, 43):
                if seed in stats:
                    row[f"seed{seed}"] = stats[seed][2][k]
            if 42 in stats and 43 in stats:
                (p42, b42), (p43, b43) = stats[42][1][k], stats[43][1][k]
                row["diff_43_minus_42"] = summarize(p43 - p42, b43 - b42)
            table.append(row)

    os.makedirs(args.out, exist_ok=True)
    out = dict(description=__doc__.split("\n\n")[0], symbols=__doc__.split("\n\n")[2], rows=table,
               routes=meta, summary_check_max_abs_diff=checks, earlier_reruns=earlier_reruns())
    with open(os.path.join(args.out, "replication_table.json"), "w") as f:
        json.dump(out, f, indent=2)
    cols = ["seed42", "seed43", "diff_43_minus_42"]
    with open(os.path.join(args.out, "replication_table.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "stat"] + [f"{c}_{x}" for c in cols for x in ("est", "lo", "hi")])
        for r in table:
            w.writerow([r["model"], r["stat"]] + [r[c][x] if c in r else "" for c in cols for x in ("est", "lo", "hi")])
    with open(os.path.join(args.out, "replication_table.md"), "w") as f:
        f.write("| model | statistic | seed 42 | seed 43 | seed 43 - seed 42 |\n|---|---|---|---|---|\n")
        for r in table:
            f.write(f"| {r['model']} | {r['stat']} | " + " | ".join(fmt(r[c]) if c in r else "-" for c in cols) + " |\n")
    print(json.dumps(checks))
    print(f"wrote {args.out}/replication_table.{{json,csv,md}}")


if __name__ == "__main__":
    main()
