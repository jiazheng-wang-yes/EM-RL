"""Stage 7 W4 task 4: relate S to generated behavior (CPU).

(a) Ranking. The six bridge conditions (C, E, G, G with the nested, style or rand4_s0 hold) ordered by S and by the
    bridge EM rate. S comes from w4_behavior_pairs.py: behavior16 pairs with all-position holds (the bridge's
    intervention on fixed text) and the frozen 120-pair assay with answer-position holds (the paper's S). EM rates
    come from w4_bridge_judge.py analyze. Because S (behavior16) and the bridge use the same 16 questions, both are
    recomputed on each of 2,000 question resamples (seed 0), which gives an interval for Kendall's tau between the
    two orderings and the share of draws in which each key ordering holds for both.
(b) Per question (EXPLORATORY, 16 questions). x = S(E) - S(C), the mean over the question's 1 + k pairs of the
    per-pair difference; y = the E - C EM-rate difference of that question in the ACL Step 1 evaluation
    (Qwen2.5-7B, training rendering, 30 samples per condition, local-panel labels of acl_step1_localjudge_20260924_v1).
    Spearman and Pearson correlation, 2,000-draw question bootstrap intervals (seed 0), and a 10,000-draw
    permutation p-value (seed 0). Split-half reliabilities of x (odd/even pairs) and y (odd/even samples), with the
    Spearman-Brown correction, bound the correlation that noise alone would allow.

Usage:
  python w4_s_behavior.py --pairs-run <id> --judge-run <id> --tag <name>
Outputs: eval_runs/persona_control_stage7/w4_behavior/s_behavior/<tag>/
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "stage6"))
from acl_common import PROMPTS, sha256_file  # noqa: E402

ROOT = HERE.parents[2]
W4_ROOT = ROOT / "eval_runs" / "persona_control_stage7" / "w4_behavior"
ACL_LABELS = (ROOT / "eval_runs/persona_control_acl/step1_local_judge_review/acl_step1_localjudge_20260924_v1/"
              "analysis/population_labels.parquet")
BRIDGE = ("C", "E", "G", "G_nested_hold", "G_style_hold", "G_rand4_hold")
S_NAME = {"C": "C", "E": "E", "G": "G", "G_nested_hold": "G|hold_nested", "G_style_hold": "G|hold_style",
          "G_rand4_hold": "G|hold_rand4_s0"}
EM_COLS = ("panel_mean_rule_em", "panel_em", "panel_p_em", "qwen38_em", "gemma4_em")
KEY_ORDERS = (("E", "C"), ("G", "C"), ("G", "G_nested_hold"), ("G_style_hold", "G_nested_hold"),
              ("G_rand4_hold", "G_nested_hold"))
N_BOOT, SEED = 2000, 0
QIDS = [p["prompt_id"] for p in PROMPTS]


def s_by_question(S, pair_set, mask):
    """question x condition table of mean S; mask picks the hold variant (resp or all)."""
    d = S[S.pair_set == pair_set]
    cols = {c: (n if c in ("C", "E", "G") else f"{n}:{mask}") for c, n in S_NAME.items()}
    w = d.pivot_table(index=["question_id", "example_id"], columns="cond", values="S")[list(cols.values())]
    w.columns = list(cols)
    return w


def ranking(S, rates, labels, out):
    rows = {}
    for label, pair_set, mask in (("S_behavior16_all", "behavior16", "all"), ("S_behavior16_resp", "behavior16", "resp"),
                                  ("S_assay120_resp", "assay120", "resp"), ("S_assay120_all", "assay120", "all")):
        rows[label] = s_by_question(S, pair_set, mask).mean()
    for col in EM_COLS + ("panel_alignment",):
        r = rates[rates.outcome == col].set_index("condition").estimate
        rows[f"bridge_{col}"] = r
    table = pd.DataFrame(rows).loc[list(BRIDGE)]
    ranks = table.rank(ascending=False).add_suffix("_rank")
    ranks["bridge_panel_alignment_rank"] = table["bridge_panel_alignment"].rank(ascending=True)
    pd.concat([table, ranks], axis=1).to_csv(out / "ranking.csv")
    taus = []
    for s in ("S_behavior16_all", "S_behavior16_resp", "S_assay120_resp", "S_assay120_all"):
        for e in EM_COLS:
            t = stats.kendalltau(table[s], table[f"bridge_{e}"]).statistic
            rho = stats.spearmanr(table[s], table[f"bridge_{e}"]).statistic
            orders = {f"{a}>{b}": bool((table.loc[a, s] > table.loc[b, s]) == (table.loc[a, f"bridge_{e}"] > table.loc[b, f"bridge_{e}"]))
                      for a, b in KEY_ORDERS}
            taus.append(dict(s_measure=s, em_measure=e, kendall_tau=t, spearman=rho, **orders))
    pd.DataFrame(taus).to_csv(out / "ranking_agreement.csv", index=False)
    # Joint question bootstrap: behavior16 S (all-position holds) and bridge EM on the same 16 questions.
    sq = s_by_question(S, "behavior16", "all").groupby(level="question_id").mean().reindex(QIDS)
    boot = {}
    rng = np.random.default_rng(SEED)
    draws = rng.integers(0, len(QIDS), (N_BOOT, len(QIDS)))
    for e in EM_COLS:
        eq = labels.groupby(["prompt_id", "condition"])[e].mean().unstack().reindex(QIDS)[list(BRIDGE)]
        tau, hold = [], {f"{a}>{b}": 0 for a, b in KEY_ORDERS}
        for d in draws:
            s_m, e_m = sq.iloc[d].mean(), eq.iloc[d].mean()
            tau.append(stats.kendalltau(s_m, e_m).statistic)
            for a, b in KEY_ORDERS:
                hold[f"{a}>{b}"] += int(s_m[a] > s_m[b] and e_m[a] > e_m[b])
        tau = np.array(tau, dtype=float)
        boot[e] = dict(kendall_tau_point=float(stats.kendalltau(sq.mean(), eq.mean()).statistic),
                       kendall_tau_ci=[float(np.nanpercentile(tau, 2.5)), float(np.nanpercentile(tau, 97.5))],
                       share_of_draws_order_holds_for_both={k: v / N_BOOT for k, v in hold.items()})
    (out / "ranking_bootstrap.json").write_text(json.dumps(dict(
        s_measure="S_behavior16_all", unit="question (16)", draws=N_BOOT, seed=SEED, by_em_measure=boot), indent=2) + "\n")
    return table


def split_half(values_by_q, key):
    """Spearman-Brown corrected odd/even split-half reliability of a per-question mean."""
    odd = values_by_q[values_by_q[key] % 2 == 1].groupby("question_id").value.mean()
    even = values_by_q[values_by_q[key] % 2 == 0].groupby("question_id").value.mean()
    r = float(np.corrcoef(odd.reindex(QIDS), even.reindex(QIDS))[0, 1])
    return dict(half_correlation=r, spearman_brown=2 * r / (1 + r) if r > -1 else float("nan"))


def correlate(x, y, label, out_rows):
    ok = x.notna() & y.notna()
    x, y = x[ok].to_numpy(float), y[ok].to_numpy(float)
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(x), (N_BOOT, len(x)))
    sp = np.array([stats.spearmanr(x[i], y[i]).statistic for i in idx])
    pe = np.array([np.corrcoef(x[i], y[i])[0, 1] for i in idx])
    rho, r = stats.spearmanr(x, y).statistic, np.corrcoef(x, y)[0, 1]
    perm = np.random.default_rng(SEED)
    null = np.array([stats.spearmanr(x, perm.permutation(y)).statistic for _ in range(10000)])
    out_rows.append(dict(comparison=label, n_questions=int(len(x)), spearman=rho,
                         spearman_ci_low=float(np.nanpercentile(sp, 2.5)), spearman_ci_high=float(np.nanpercentile(sp, 97.5)),
                         spearman_perm_p_two_sided=float((np.abs(null) >= abs(rho) - 1e-12).mean()),
                         pearson=r, pearson_ci_low=float(np.nanpercentile(pe, 2.5)), pearson_ci_high=float(np.nanpercentile(pe, 97.5))))


def per_question(S, bridge_labels, pair_pass, out):
    acl = pd.read_parquet(ACL_LABELS)
    acl = acl[(acl.model_key == "qwen2_5_7b") & (acl.rendering == "training")]
    d = S[(S.pair_set == "behavior16") & S.cond.isin(["C", "E", "G"])].pivot_table(
        index=["question_id", "example_id", "pair_idx"], columns="cond", values="S").reset_index()
    d["dEC"], d["dGC"] = d.E - d.C, d.G - d.C
    if pair_pass is not None:
        d = d.merge(pair_pass[["pair_id", "panel_pass"]].rename(columns={"pair_id": "example_id"}), on="example_id", how="left")
    q = pd.DataFrame(index=QIDS)
    q["split"] = [p["split"] for p in PROMPTS]
    q["S_E_minus_C"] = d.groupby("question_id").dEC.mean()
    q["S_G_minus_C"] = d.groupby("question_id").dGC.mean()
    q["S_E_minus_C_assay_pair_only"] = d[d.pair_idx == 0].set_index("question_id").dEC
    if pair_pass is not None:
        q["S_E_minus_C_label_checked_pairs"] = d[d.panel_pass == True].groupby("question_id").dEC.mean()  # noqa: E712
        q["label_checked_pairs"] = d.groupby("question_id").panel_pass.sum()
    for e in EM_COLS:
        m = acl.groupby(["prompt_id", "condition"])[e].mean().unstack()
        q[f"acl_{e}_E_minus_C"] = m.E - m.C
        b = bridge_labels.groupby(["prompt_id", "condition"])[e].mean().unstack()
        q[f"bridge_{e}_E_minus_C"] = b.E - b.C
        q[f"bridge_{e}_G_minus_C"] = b.G - b.C
    q.index.name = "question_id"
    q.to_csv(out / "per_question.csv")
    rows = []
    for e in EM_COLS:
        correlate(q.S_E_minus_C, q[f"acl_{e}_E_minus_C"], f"S(E)-S(C) vs ACL Step 1 E-C {e}", rows)
    correlate(q.S_E_minus_C, q.bridge_panel_mean_rule_em_E_minus_C, "S(E)-S(C) vs bridge E-C panel_mean_rule_em", rows)
    correlate(q.S_G_minus_C, q.bridge_panel_mean_rule_em_G_minus_C, "S(G)-S(C) vs bridge G-C panel_mean_rule_em", rows)
    correlate(q.S_E_minus_C_assay_pair_only, q.acl_panel_mean_rule_em_E_minus_C, "assay pair only: S(E)-S(C) vs ACL E-C panel_mean_rule_em", rows)
    if pair_pass is not None:
        correlate(q.S_E_minus_C_label_checked_pairs, q.acl_panel_mean_rule_em_E_minus_C,
                  "label-checked pairs only: S(E)-S(C) vs ACL E-C panel_mean_rule_em", rows)
    pd.DataFrame(rows).to_csv(out / "correlation.csv", index=False)
    rel = {"x: S(E)-S(C), odd vs even pair_idx": split_half(d.rename(columns={"dEC": "value"}), "pair_idx")}
    for e in ("panel_mean_rule_em", "qwen38_em", "gemma4_em"):
        a = acl.pivot_table(index=["prompt_id", "sample_idx"], columns="condition", values=e).reset_index()
        a["value"] = a.E - a.C
        rel[f"y: ACL E-C {e}, odd vs even sample_idx"] = split_half(a.rename(columns={"prompt_id": "question_id"}), "sample_idx")
    (out / "reliability.json").write_text(json.dumps(rel, indent=2) + "\n")
    return q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs-run", required=True)
    ap.add_argument("--judge-run", required=True)
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()
    out = W4_ROOT / "s_behavior" / args.tag
    if out.exists(): raise FileExistsError(out)
    out.mkdir(parents=True)
    S = pd.read_csv(W4_ROOT / "pairs" / args.pairs_run / "S_per_pair.csv")
    jdir = W4_ROOT / "judge" / args.judge_run / "analysis"
    rates = pd.read_csv(jdir / "bridge_rates.csv")
    labels = pd.read_parquet(jdir / "bridge_labels.parquet")
    pp = pd.read_csv(jdir / "pair_label_check.csv") if (jdir / "pair_label_check.csv").exists() else None
    table = ranking(S, rates, labels, out)
    q = per_question(S, labels, pp, out)
    (out / "manifest.json").write_text(json.dumps(dict(
        pairs_run=args.pairs_run, judge_run=args.judge_run, acl_labels=str(ACL_LABELS.relative_to(ROOT)),
        acl_labels_sha256=sha256_file(ACL_LABELS), script_sha256=sha256_file(Path(__file__).resolve()),
        bootstrap=dict(draws=N_BOOT, seed=SEED, unit="question"), label="(b) is exploratory: 16 questions"), indent=2) + "\n")
    print(table.to_string())
    print(pd.read_csv(out / "correlation.csv").to_string(index=False))
    print(json.dumps(json.loads((out / "reliability.json").read_text()), indent=1))


if __name__ == "__main__": main()
