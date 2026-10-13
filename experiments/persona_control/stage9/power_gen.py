"""Reliability round, item 7: simulation power of the planned generation test (CPU).

Plan: docs/plans/persona-control.md#stage9-behavior-routes, section 14.1 (fixed before any result of the round).
- Data: the panel labels (panel_em) of the existing generation runs, Qwen2.5-7B Steps 4 and 7 (16 questions x 30
  samples; Step 7's wider set, 51 x 10) and Llama-3.1-8B Steps 4 and 7 (16 x 30).
- Planned test: per-question misaligned rates, the paired difference per question (A - B, expected > 0), a bootstrap over
  questions; detected when the 95% percentile interval lies above 0.
- Simulation: Q questions drawn with replacement from the model's observed questions, then n sample indices drawn with
  replacement within each question, the same indices for both conditions (common random numbers stay paired).
- Choice: the design with the fewest answers per condition (ties: fewer questions) at which every primary contrast has
  power >= .80 in both models; the first design with power >= .90 instead if it needs at most 1.5 times the answers.
  If no design reaches .80, the design with the largest minimum primary power (ties: fewer answers).
Writes eval_runs/persona_control_stage9/reliability/power/power_gen.csv and choice.json, and prints them.
Usage: python power_gen.py [--sims 2000 --boot 2000]
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

S9 = "/net/spaces/scratch/jiaweizhang/jiazhengw_migration/eval_runs/persona_control_stage9"
RUNS = {"qwen": {"s4": "step4/s9s4_20260929", "s7": "step7/s9s7g_20260930", "wide": "step7/s9s7w_20260930"},
        "llama": {"s4": "step4/s9s4_llama_20260930", "s7": "step7/s9s7l_20260930"}}
# contrast: (A, B, [(run, A label, B label), ...]); primary contrasts first
CONTRASTS = {
    "P1 E0-C0": [("s4", "E0", "C0"), ("wide", "E0", "C0")],
    "P2 (E0-UNION~s)-(E0-UNION)": [("s4", "D:UNION~s:256", "D:UNION:256"), ("wide", "D:UNION~s", "D:UNION")],
    "P3 (E0-CORE~s)-(E0-CORE)": [("s7", "D:CORE~s", "D:CORE")],
    "S1 (C0+UNION)-C0": [("s4", "T:UNION:256", "C0")],
    "S2 (C0+CORE)-C0": [("s7", "T:CORE", "C0")],
}
PRIMARY = [c for c in CONTRASTS if c.startswith("P")]
QS, NS = (32, 48, 64, 80, 100), (4, 8, 16, 30)
POOL_MAX = 103


def pool(model, spec):
    """Per question: the A and B labels of its samples, aligned by sample index (NaN where the judges split, as in the
    reports' rates; padded to 30), and the number of samples."""
    qa, qb, qm = [], [], []
    for run, a, b in spec:
        if run not in RUNS[model]:
            continue
        s = pd.read_parquet(f"{S9}/{RUNS[model][run]}/s4_sample_labels.parquet",
                            columns=["condition", "prompt_id", "sample_idx", "panel_em"])
        w = s[s.condition.isin([a, b])].set_index(["prompt_id", "sample_idx", "condition"]).panel_em.unstack()
        n_all = s[s.condition == a].groupby("prompt_id").size()
        for pid, g in w.groupby(level=0):
            assert len(g) == n_all[pid], (run, pid)
            pa, pb = np.full(30, np.nan), np.full(30, np.nan)
            pa[: len(g)], pb[: len(g)] = g[a].to_numpy(), g[b].to_numpy()
            qa.append(pa), qb.append(pb), qm.append(len(g))
    return np.array(qa), np.array(qb), np.array(qm)


def power(qa, qb, qm, Q, n, sims, boot, rng):
    qi = rng.integers(0, len(qa), (sims, Q))
    si = (rng.random((sims, Q, n)) * qm[qi][:, :, None]).astype(int)
    with np.errstate(invalid="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            d = np.nanmean(qa[qi[:, :, None], si], 2) - np.nanmean(qb[qi[:, :, None], si], 2)  # [sims, Q]
    ok = ~np.isnan(d)
    W = rng.multinomial(Q, np.full(Q, 1.0 / Q), size=boot).T.astype(float)  # [Q, boot] question-bootstrap counts
    lo = np.percentile(np.nan_to_num(d) @ W / (ok @ W), 2.5, axis=1)
    return float((lo > 0).mean()), float(np.nanmean(d))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", type=int, default=2000)
    ap.add_argument("--boot", type=int, default=2000)
    args = ap.parse_args()
    out = f"{S9}/reliability/power"
    os.makedirs(out, exist_ok=True)
    rows = []
    for model in RUNS:
        for name, spec in CONTRASTS.items():
            qa, qb, qm = pool(model, spec)
            obs = np.nanmean(qa, 1) - np.nanmean(qb, 1)
            for Q in QS:
                for n in NS:
                    rng = np.random.default_rng([20261001, Q, n, len(rows)])
                    p, eff = power(qa, qb, qm, Q, n, args.sims, args.boot, rng)
                    rows.append(dict(model=model, contrast=name, primary=name in PRIMARY, Q=Q, n=n, answers=Q * n,
                                     power=p, mean_effect=eff, pool_questions=len(qa),
                                     observed_effect=float(np.nanmean(obs))))
    tab = pd.DataFrame(rows)
    tab.to_csv(f"{out}/power_gen.csv", index=False)

    prim = tab[tab.primary].groupby(["Q", "n", "answers"]).power.min().reset_index().sort_values(["answers", "Q"])
    prim = prim[prim.Q <= POOL_MAX]
    d80 = prim[prim.power >= 0.80].head(1)
    d90 = prim[prim.power >= 0.90].head(1)
    if len(d80):
        pick = d90 if len(d90) and d90.answers.iloc[0] <= 1.5 * d80.answers.iloc[0] else d80
        rule = "power >= .90 within 1.5x" if pick is d90 else "power >= .80"
    else:
        pick = prim.sort_values(["power", "answers"], ascending=[False, True]).head(1)
        rule = "no design reaches .80: largest minimum primary power"
    choice = dict(Q=int(pick.Q.iloc[0]), n=int(pick.n.iloc[0]), answers=int(pick.answers.iloc[0]),
                  min_primary_power=float(pick.power.iloc[0]), rule=rule, sims=args.sims, boot=args.boot,
                  d80=d80.to_dict("records"), d90=d90.to_dict("records"))
    json.dump(choice, open(f"{out}/choice.json", "w"), indent=1)
    pd.set_option("display.width", 250, "display.max_rows", 500)
    print(tab.pivot_table(index=["model", "contrast"], columns=["Q", "n"], values="power").round(2).to_string())
    print(tab.groupby(["model", "contrast"])[["pool_questions", "observed_effect"]].first().round(3).to_string())
    print(json.dumps(choice, indent=1))


if __name__ == "__main__":
    main()
