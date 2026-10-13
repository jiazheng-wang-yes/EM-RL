"""Stage 8 Step 4: injection and trigger predictions (plan Section 4, Step 4 table; definitions in the Section 10 log).

For one twin and one rendering (inject or trigger), compares the twin's 3a read profiles under that rendering with the
plain ones (s3_reads.py analyze wrote read_profiles.parquet in s3_reads/ and s3_reads_<rendering>/), and, with
--judge-run, its judged E - C gaps on the 16 ACL questions under the two renderings (judge8.py labels).

  profiles  per 3a prompt: 'real' = sum over the twin's A* components of s |mean read coefficient| (A* profile);
            'all16' = the same over all top-16 components of every matrix; 'adam_sig' = the Adam-proxy prediction
            sigma_hat |v_hat^T x_base| from base-model inputs (s3_reads.read_profiles).
  trigger   reads: AUROC of the all16 profile (primary) and of the A* profile, trigger-present versus absent, over
            the 640 prompts (check >= .9 on the trigger twin).  Behavior: gap_trigger - 4 gap_plain (check: interval
            above 0).
  inject    reads: k = mean A* profile with / without the system prompt over the other-topic prompts, and k_hat =
            the same ratio of the Adam-proxy profile (check: both intervals above 1).  Behavior: gap_inject -
            gap_plain (check: interval above 0); the ratio gap_inject / gap_plain must contain k (not testable when
            the interval of gap_plain includes 0).
  On a base twin the same numbers are controls (no check).  Intervals: paired bootstrap over prompts (3a prompts or
  ACL questions), 2,000 draws, seed 20260922.

Output: <OUT_ROOT>/<dataset>/<model>/<twin>/s4_eval/<rendering>[_<judge run>].json
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from common8 import OUT_ROOT, log, parse_twin
from s3_reads import cell_roles, out_dir_for

BOOT_SEED, N_BOOT = 20260922, 2000
AUROC_MIN, TRIGGER_FACTOR = 0.9, 4.0
JUDGE_ROOT = os.path.join(OUT_ROOT, "judge_runs")
DEV_PAIR = ("MED", "qwen2_5_7b", "s42")


def auroc(pos, neg):
    r = rankdata(np.concatenate([pos, neg]))
    return float((r[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def interval(est, draws):
    return dict(est=float(est), lo=float(np.percentile(draws, 2.5)), hi=float(np.percentile(draws, 97.5)))


def boot_draws(n, seed=BOOT_SEED):
    return np.random.default_rng(seed).integers(0, n, (N_BOOT, n))


def reads_part(args):
    plain = pd.read_parquet(os.path.join(out_dir_for(args.dataset, args.model, args.twin), "read_profiles.parquet"))
    rend = pd.read_parquet(os.path.join(out_dir_for(args.dataset, args.model, args.twin, args.rendering), "read_profiles.parquet"))
    rend = rend.loc[plain.index]
    res = dict(n_prompts=len(plain), by_cell={})
    for name, df in (("plain", plain), (args.rendering, rend)):
        res["by_cell"][name] = df.groupby("cell")[["real", "all16", "adam_sig"]].mean().round(4).to_dict()
    W = boot_draws(len(plain))
    if args.rendering == "trigger":
        for col in ("all16", "real"):
            x, y = rend[col].to_numpy(), plain[col].to_numpy()
            res[f"auroc_{col}"] = interval(auroc(x, y), [auroc(x[w], y[w]) for w in W])
        res["check"] = dict(rule=f"AUROC of the all16 profile >= {AUROC_MIN}", passed=res["auroc_all16"]["est"] >= AUROC_MIN)
    else:
        off = plain.cell.isin(cell_roles(args.dataset)["other_topic"]).to_numpy()
        Wo = boot_draws(int(off.sum()))
        for col, key in (("real", "k"), ("all16", "k_all16"), ("adam_sig", "k_hat")):
            x, y = rend[col].to_numpy()[off], plain[col].to_numpy()[off]
            res[key] = interval(x.mean() / y.mean(), x[Wo].mean(1) / y[Wo].mean(1))
        res["check"] = dict(rule="intervals of k and k_hat above 1", passed=res["k"]["lo"] > 1 and res["k_hat"]["lo"] > 1)
    return res


def judged_part(args, reads):
    lab = pd.read_parquet(os.path.join(JUDGE_ROOT, args.judge_run, "labels.parquet"))
    lab = lab[(lab.dataset == args.dataset) & (lab.model == args.model) & (lab.twin == args.twin) & (lab.qset == "acl")]
    measure = "panel_p_em" if (args.dataset, args.model, args.twin) == DEV_PAIR else "gemma4_31b_p_em"
    per = lab.groupby(["rendering", "prompt_id", "which"])[measure].mean().unstack("which")
    gap = (per["E"] - per["C"]).unstack("rendering")[["plain", args.rendering]]
    if gap.isna().any().any():
        raise ValueError("missing judged answers")
    gp, gr = gap["plain"].to_numpy(), gap[args.rendering].to_numpy()
    W = boot_draws(len(gap))
    res = dict(judge_run=args.judge_run, measure=measure, n_questions=len(gap),
               gap_plain=interval(gp.mean(), gp[W].mean(1)), **{f"gap_{args.rendering}": interval(gr.mean(), gr[W].mean(1))},
               mean_p_em={r: lab[lab.rendering == r].groupby("which")[measure].mean().round(5).to_dict()
                          for r in ("plain", args.rendering)})
    ratio = interval(gr.mean() / gp.mean(), gr[W].mean(1) / gp[W].mean(1))
    if args.rendering == "trigger":
        d = gr - TRIGGER_FACTOR * gp
        res["trigger_minus_4plain"] = interval(d.mean(), d[W].mean(1))
        res["ratio"] = ratio
        res["check"] = dict(rule="interval of gap_trigger - 4 gap_plain above 0", passed=res["trigger_minus_4plain"]["lo"] > 0)
    else:
        d = gr - gp
        res["inject_minus_plain"] = interval(d.mean(), d[W].mean(1))
        testable = res["gap_plain"]["lo"] > 0
        res["ratio"] = ratio
        res["amount"] = dict(k=reads["k"]["est"], testable=testable,
                             passed=bool(testable and ratio["lo"] <= reads["k"]["est"] <= ratio["hi"]))
        res["check"] = dict(rule="interval of gap_inject - gap_plain above 0", passed=res["inject_minus_plain"]["lo"] > 0)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="MED")
    ap.add_argument("--model", default="qwen2_5_7b")
    ap.add_argument("--twin", required=True)
    ap.add_argument("--rendering", required=True, choices=("inject", "trigger"))
    ap.add_argument("--judge-run", default="")
    args = ap.parse_args()
    role = "condition" if parse_twin(args.twin)[0] == args.rendering else "control"
    res = dict(dataset=args.dataset, model=args.model, twin=args.twin, rendering=args.rendering, role=role,
               reads=reads_part(args))
    if args.judge_run:
        res["judged"] = judged_part(args, res["reads"])
    if role == "control":  # checks apply to the changed-data twin only
        for part in ("reads", "judged"):
            if part in res:
                res[part]["check"]["applies"] = False
    out = os.path.join(OUT_ROOT, args.dataset, args.model, args.twin, "s4_eval")
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, f"{args.rendering}{'_' + args.judge_run if args.judge_run else ''}.json")
    if os.path.exists(path):
        raise FileExistsError(path)
    with open(path, "w") as f:
        json.dump(res, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    log(f"saved {path}")


if __name__ == "__main__":
    main()
