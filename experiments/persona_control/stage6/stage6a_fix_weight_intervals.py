"""Recompute paired uncertainty intervals for the completed Stage 6A weight run.

The weight runner's point estimates are valid, but its first-pass interval
bookkeeping subtracted marginal confidence limits.  This audit uses the
prompt-cluster bootstrap on paired per-example differences; it does not rerun
model scoring.

The first version rewrote qwen2_5_7b's weight_results.json in place.  Since
Stage 7 W1 the script takes --model and writes the corrected JSON to --out
(default eval_runs/persona_control_stage7/w1_corrections/stage6a_weight/<model>/),
so the Stage 6A result files are never modified.

Usage: python stage6a_fix_weight_intervals.py --model llama3_1_8b
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from stage6a_common import ClusterBootstrap, EVAL_DIR, ROOT  # noqa: E402


CORRECTIONS_DIR = os.path.join(ROOT, "eval_runs", "persona_control_stage7", "w1_corrections", "stage6a_weight")


def paired_scores(df):
    # C is repeated once per intervention run.  Averaging duplicates gives
    # one stable paired score per prompt without changing the frozen assay.
    x = df[df.kind != "neutral"].groupby(["cond", "example_id", "kind"], as_index=False)["lp_mean"].mean()
    w = x.pivot_table(index=["cond", "example_id"], columns="kind", values="lp_mean", aggfunc="mean")
    return (w["mis"] - w["align"]).rename("S").reset_index()


def diff_summary(scores, cond, base, bs):
    wide = scores[scores.cond.isin([cond, base])].pivot(index="example_id", columns="cond", values="S")
    wide = wide.dropna()
    diff = wide[cond].to_numpy(float) - wide[base].to_numpy(float)
    point, boot = bs.means(diff)
    return {"est": float(point), "lo": float(np.percentile(boot, 2.5)),
            "hi": float(np.percentile(boot, 97.5))}, boot


def ratio_summary(de, de_boot, full_de, full_boot):
    ratio = de_boot / full_boot
    return {"est": float(de["est"] / full_de["est"]),
            "lo": float(np.percentile(ratio, 2.5)),
            "hi": float(np.percentile(ratio, 97.5))}


def interval_violations(node, path=""):
    """Paths of every {est, lo, hi} interval whose estimate lies outside [lo, hi]."""
    bad = []
    if isinstance(node, dict):
        if {"est", "lo", "hi"} <= set(node) and not node["lo"] <= node["est"] <= node["hi"]:
            bad.append(path)
        for key, value in node.items():
            bad += interval_violations(value, f"{path}/{key}")
    elif isinstance(node, list):
        for i, value in enumerate(node):
            label = value.get("label", i) if isinstance(value, dict) else i
            bad += interval_violations(value, f"{path}/{label}")
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen2_5_7b")
    ap.add_argument("--out", default=None, help="output directory (default: CORRECTIONS_DIR/<model>)")
    args = ap.parse_args()
    src = os.path.join(EVAL_DIR, "stage6a", args.model)
    out = args.out or os.path.join(CORRECTIONS_DIR, args.model)
    result_path = os.path.join(src, "weight_results.json")
    rows_path = os.path.join(src, "weight_rows.parquet")
    with open(result_path) as f:
        result = json.load(f)
    before = interval_violations(result)
    rows = pd.read_parquet(rows_path)
    scores = paired_scores(rows)
    examples = sorted(scores.example_id.unique())
    bs = ClusterBootstrap(examples, n_boot=2000, seed=0)
    full_te, full_te_boot = diff_summary(scores, "Wfull", "C", bs)
    full_de, full_de_boot = diff_summary(scores, "Wfull|clamp", "C", bs)
    result["conditions"]["Wfull"]["TE"] = full_te
    result["conditions"]["Wfull"]["DE"] = full_de

    def update_condition(target, te_label, de_label):
        te, _ = diff_summary(scores, te_label, "C", bs)
        de, de_boot = diff_summary(scores, de_label, "C", bs)
        target["TE"], target["DE"] = te, de
        target["F_direct"] = ratio_summary(de, de_boot, full_de, full_de_boot)

    for rank, target in result["learned"].items():
        update_condition(target, f"learned_r{rank}", f"learned_r{rank}|clamp")
    for target in result["controls"]:
        update_condition(target, target["label"], target["label"] + "|clamp")

    trajectory = result.get("trajectory_causal", {})
    for part, target in trajectory.items():
        update_condition(target, f"trajectory_{part}", f"trajectory_{part}|clamp")

    after = interval_violations(result)
    result["interval_audit"] = {"source": os.path.relpath(result_path, ROOT),
                                "intervals_outside_before": len(before),
                                "intervals_outside_after": after}
    os.makedirs(out, exist_ok=True)
    out_path = os.path.join(out, "weight_results.json")
    if os.path.abspath(out_path) == os.path.abspath(result_path):
        raise SystemExit("refusing to overwrite the Stage 6A result; pass a different --out")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"{out_path}: {len(before)} intervals excluded their estimate before, {len(after)} after")


if __name__ == "__main__":
    main()
