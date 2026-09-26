"""Stage 7 W1 task 3: split every route contrast into its misaligned-answer and aligned-answer parts.

For a model X and a prompt i, lp_X(mis, i) and lp_X(align, i) are the mean answer-token
log-probabilities of the misaligned and aligned answers, and S_X(i) = lp_X(mis, i) - lp_X(align, i).
A contrast is a signed sum of conditions, K = sum_X a_X S_X.  It splits exactly into
    K_mis   = sum_X a_X lp_X(mis)      the change in the misaligned answer's log-probability
    K_align = sum_X a_X lp_X(align)    the change in the aligned answer's log-probability
with K = K_mis - K_align.  mis_share = K_mis / K is the fraction of K that comes from the
misaligned answer becoming more likely; 1 - mis_share comes from the aligned answer becoming
less likely.  A share above 1 means the aligned answer became more likely while K still rose.
mis_share is left empty when the interval of K includes 0, because the ratio is then unstable.

Conditions (Stage 5B route runner): C, E, G = C with E's region weights, R = E with C's region
weights (reverse graft), and X|clamp_<s> = X with carrier-layer hold s (nested = the four-
coordinate persona hold; evil, evil_syc, style, rand4_s* = the named comparison holds).
Contrasts: Delta = S(E) - S(C); T = S(G) - S(C); D = S(G|clamp_nested) - S(C|clamp_nested);
M = T - D; E_hold = S(E) - S(E|clamp_nested); NE_reverse = S(E) - S(R);
NDE_reverse = S(E|clamp_nested) - S(R|clamp_nested); M_reverse = NE_reverse - NDE_reverse;
M_hold_<s> = T - (S(G|clamp_<s>) - S(C|clamp_nested));
region_interaction = NE_reverse - T = S(E) - S(R) - S(G) + S(C), which is 0 when the region's
effect does not depend on whether the rest of the network comes from C or from E.

Runs:
  qwen2_5_7b, llama3_1_8b: ARR Round 1 confirmations (training rendering, seven matrices grafted);
  qwen3_1_7b: Stage 5B run of 2026-09-17 (legacy rendering, all parameters of layers 8-19 grafted).
Intervals: 2,000 prompt-cluster bootstrap draws, seed 0; every part and ratio is resampled jointly.
Output: eval_runs/persona_control_stage7/w1_corrections/score_parts/.
"""

import glob
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "stage6"))
from common import ROOT, ClusterBootstrap  # noqa: E402

OUT = os.path.join(ROOT, "eval_runs/persona_control_stage7/w1_corrections/score_parts")
RUNS = {
    "qwen2_5_7b": dict(dir="eval_runs/persona_control_arr/round1/20260924T063649Z/route_qwen/stage5b/"
                           "qwen2_5_7b_training_region8_19_confirmation",
                       rendering="training", graft="layers 8-19, seven matrices"),
    "llama3_1_8b": dict(dir="eval_runs/persona_control_arr/round1/20260924T063649Z/route_llama/stage5b/"
                            "llama3_1_8b_training_region5_22_confirmation",
                        rendering="training", graft="layers 5-22, seven matrices"),
    "qwen3_1_7b": dict(dir="eval_runs/persona_control_stage6/stage5b/qwen3_1_7b",
                       rendering="legacy", graft="layers 8-19, all parameters"),
}
NAMED_HOLDS = ("evil", "evil_syc", "style", "rand4_s0", "rand4_s1", "rand4_s2")
CONTRASTS = {
    "Delta": {"E": 1, "C": -1},
    "T": {"G": 1, "C": -1},
    "D": {"G|clamp_nested": 1, "C|clamp_nested": -1},
    "M": {"G": 1, "C": -1, "G|clamp_nested": -1, "C|clamp_nested": 1},
    "E_hold": {"E": 1, "E|clamp_nested": -1},
    "NE_reverse": {"E": 1, "R": -1},
    "NDE_reverse": {"E|clamp_nested": 1, "R|clamp_nested": -1},
    "M_reverse": {"E": 1, "R": -1, "E|clamp_nested": -1, "R|clamp_nested": 1},
    "region_interaction": {"E": 1, "R": -1, "G": -1, "C": 1},
    **{f"M_hold_{s}": {"G": 1, "C": -1, f"G|clamp_{s}": -1, "C|clamp_nested": 1} for s in NAMED_HOLDS},
}
# Ratios of two contrasts, reported for the total and for each part.
RATIOS = {"T_over_Delta": ("T", "Delta"), "removed_share": ("M", "T"),
          "E_hold_over_Delta": ("E_hold", "Delta"), "reverse_removed_share": ("M_reverse", "NE_reverse")}
# Checks against the route runner's own summary (stage5b_analyze.py): estimate and interval must match.
SUMMARY_KEYS = {"Delta": "DeltaS_EM", "T": "TE", "D": "DE", "NE_reverse": "NE_reverse_graft",
                "NDE_reverse": "NDE_reverse_graft"}
SUMMARY_RATIO_KEYS = {"T_over_Delta": "TE_over_DeltaS", "removed_share": "MF",
                      "E_hold_over_Delta": "E_clamp_repair_fraction", "reverse_removed_share": "MF_reverse_graft"}


def check(model, name, est, lo, hi, ref):
    if abs(ref["est"] - est) > 1e-9 or abs(ref["lo"] - lo) > 1e-9 or abs(ref["hi"] - hi) > 1e-9:
        raise RuntimeError(f"{model} {name}: {est} [{lo}, {hi}] differs from summary.json {ref}")


def ci(boot):
    return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def load(run_dir):
    files = sorted(glob.glob(os.path.join(run_dir, "rows_*.parquet")))
    rows = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    # C and G are scored in more than one phase; the copies agree to 1e-6, keep the first.
    rows = rows[rows.kind.isin(["mis", "align"])].drop_duplicates(subset=["cond", "seq_id"])
    return rows.pivot_table(index="example_id", columns=["cond", "kind"], values="lp_mean", aggfunc="first")


def run(model, cfg):
    run_dir = os.path.join(ROOT, cfg["dir"])
    lp = load(run_dir)
    conds = sorted({c for c, _ in lp.columns})
    base = ["C", "E", "G", "C|clamp_nested", "G|clamp_nested"]
    examples = sorted(lp.index[lp[[(c, k) for c in base for k in ("mis", "align")]].notna().all(axis=1)])
    lp = lp.loc[examples]
    bs = ClusterBootstrap(examples, n_boot=2000, seed=0)
    summary = json.load(open(os.path.join(run_dir, "summary.json")))["full"]["baseline"]
    tag = dict(model=model, run=cfg["dir"], rendering=cfg["rendering"], graft=cfg["graft"], n_examples=len(examples))

    levels = []
    for c in conds:
        if (c, "mis") not in lp.columns or lp[(c, "mis")].isna().any() or lp[(c, "align")].isna().any():
            continue
        vals = {}
        for name, v in (("lp_mis", lp[(c, "mis")]), ("lp_align", lp[(c, "align")]),
                        ("S", lp[(c, "mis")] - lp[(c, "align")])):
            p, b = bs.means(v.to_numpy(float))
            vals.update({name: p, f"{name}_lo": ci(b)[0], f"{name}_hi": ci(b)[1]})
        levels.append(dict(tag, cond=c, **vals))

    parts, boots = [], {}
    for name, coef in CONTRASTS.items():
        if not all(c in conds for c in coef):
            continue
        k_mis = sum(a * lp[(c, "mis")].to_numpy(float) for c, a in coef.items())
        k_align = sum(a * lp[(c, "align")].to_numpy(float) for c, a in coef.items())
        (pm, bm), (pa, ba) = bs.means(k_mis), bs.means(k_align)
        pk, bk = pm - pa, bm - ba
        boots[name] = dict(total=(pk, bk), mis=(pm, bm), align=(pa, ba))
        if name in SUMMARY_KEYS:
            check(model, name, pk, *ci(bk), summary[SUMMARY_KEYS[name]])
        definition = " ".join(f"{'+' if a > 0 else '-'} S({c})" for c, a in coef.items()).lstrip("+ ")
        share = (pm / pk, *ci(bm / bk)) if ci(bk)[0] > 0 or ci(bk)[1] < 0 else (np.nan, np.nan, np.nan)
        parts.append(dict(tag, contrast=name, definition=definition,
                          K=pk, K_lo=ci(bk)[0], K_hi=ci(bk)[1],
                          dlp_mis=pm, dlp_mis_lo=ci(bm)[0], dlp_mis_hi=ci(bm)[1],
                          dlp_align=pa, dlp_align_lo=ci(ba)[0], dlp_align_hi=ci(ba)[1],
                          mis_share=share[0], mis_share_lo=share[1], mis_share_hi=share[2]))

    # C|clamp_nested holds C's own coordinates at C's own values, so it must equal C.
    c_hold = float(np.abs((lp[("C|clamp_nested", "mis")] - lp[("C", "mis")]).to_numpy()).max())

    ratios = []
    for name, (num, den) in RATIOS.items():
        if num not in boots or den not in boots:
            continue
        for part in ("total", "mis", "align"):
            (pn, bn), (pd_, bd) = boots[num][part], boots[den][part]
            ratios.append(dict(tag, ratio=name, numerator=num, denominator=den, part=part,
                               est=pn / pd_, lo=ci(bn / bd)[0], hi=ci(bn / bd)[1]))
            if part == "total" and name in SUMMARY_RATIO_KEYS:
                check(model, name, pn / pd_, *ci(bn / bd), summary[SUMMARY_RATIO_KEYS[name]])
    return levels, parts, ratios, c_hold


def main():
    os.makedirs(OUT, exist_ok=True)
    levels, parts, ratios, checks = [], [], [], {}
    for model, cfg in RUNS.items():
        lv, pa, ra, c_hold = run(model, cfg)
        levels += lv
        parts += pa
        ratios += ra
        matched = [k for k in SUMMARY_KEYS if any(p["contrast"] == k for p in pa)]
        matched += [k for k in SUMMARY_RATIO_KEYS if any(r["ratio"] == k for r in ra)]
        checks[model] = {"max_abs_lp_diff_C_hold_vs_C": c_hold, "estimate_and_interval_match_summary_json": matched}
    pd.DataFrame(levels).to_csv(os.path.join(OUT, "condition_levels.csv"), index=False)
    pd.DataFrame(parts).to_csv(os.path.join(OUT, "score_parts.csv"), index=False)
    pd.DataFrame(ratios).to_csv(os.path.join(OUT, "score_part_ratios.csv"), index=False)
    json.dump(dict(runs=RUNS, contrasts=CONTRASTS, ratios=RATIOS, n_boot=2000, seed=0, checks=checks),
              open(os.path.join(OUT, "summary.json"), "w"), indent=2)
    print(OUT)


if __name__ == "__main__":
    main()
