"""Stage 7 W3 analysis (CPU): label check, content x style, on-policy answers, held-out questions, answer tokens only.

Every interval is a 95% percentile interval from 2,000 bootstrap draws over prompt clusters, seed 0
(common.ClusterBootstrap). Ratios are computed from resampled means. Quantities computed on the same
questions are resampled jointly (one weight matrix); held-out and original sets are different questions,
so their difference combines draws from seed 0 (held-out) and seed 1 (original).

Inputs
  label_check/<run>/labels_gemma4_31b.parquet, keep_gemma4_31b.json      w3_label_check.py
  scores/<model>/rows.parquet                                             w3_score.py
  route_<set>_<model>/stage5b/<model><tag>/rows_P*.parquet, manifest.json stage5b_activation_route.py
  Round 1 and W5 route folders for the original answers
Outputs (eval_runs/persona_control_stage7/w3_assay_validity/)
  summary.json, task1_content_style.csv, task2_onpolicy.csv, task3_heldout.csv, task4_answer_only.csv,
  label_check_summary.csv, style_check.csv
"""

import argparse
import json
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "stage6"))
from common import DATA_DIR, ROOT, ClusterBootstrap  # noqa: E402
from stage5b_analyze import load_rows, per_example  # noqa: E402

OUT = os.path.join(ROOT, "eval_runs", "persona_control_stage7", "w3_assay_validity")
ROUND1 = os.path.join(ROOT, "eval_runs", "persona_control_arr", "round1", "20260924T063649Z")
ORIGINAL_ROUTE = {
    "qwen2_5_7b": os.path.join(ROUND1, "route_qwen", "stage5b", "qwen2_5_7b_training_region8_19_confirmation"),
    "llama3_1_8b": os.path.join(ROUND1, "route_llama", "stage5b", "llama3_1_8b_training_region5_22_confirmation"),
    "qwen3_1_7b": os.path.join(ROOT, "eval_runs", "persona_control_stage7", "w5_replication", "seed42", "stage5b",
                               "qwen3_1_7b_training_region8_19_confirmation"),
}
TAG = {"qwen2_5_7b": "_training_region8_19", "llama3_1_8b": "_training_region5_22", "qwen3_1_7b": "_training_region8_19"}
HEDGES = ["may", "might", "could", "perhaps", "possibly", "generally", "typically", "often", "usually", "likely", "can",
          "consider", "important"]
N_BOOT = 2000


def ci(boot):
    return ClusterBootstrap.ci(boot)


def summ(point, boot):
    lo, hi = ci(boot)
    return dict(est=float(point), lo=lo, hi=hi)


class Weights:
    """Cluster-bootstrap weights for a fixed list of examples; mean(v) returns (point, draws)."""

    def __init__(self, examples, seed=0):
        self.examples = list(examples)
        self.bs = ClusterBootstrap(self.examples, n_boot=N_BOOT, seed=seed)

    def mean(self, series):
        return self.bs.means(series.reindex(self.examples).to_numpy(dtype=float))


def contrast_stats(W, X):
    """X: DataFrame example_id x cond of one per-example contrast. Returns Delta, T, D, removed share (point, draws)."""
    m = {c: W.mean(X[c]) for c in X.columns}
    d = lambda a, b: (m[a][0] - m[b][0], m[a][1] - m[b][1])  # noqa: E731
    out = {"Delta": d("E", "C"), "T": d("G", "C"), "D": d("G|clamp_nested", "C|clamp_nested")}
    out["removed_share"] = (1 - out["D"][0] / out["T"][0], 1 - out["D"][1] / out["T"][1])
    out["T_over_Delta"] = (out["T"][0] / out["Delta"][0], out["T"][1] / out["Delta"][1])
    out["self_hold_minus_C"] = d("C|clamp_nested", "C")
    return out


# --------------------------------------------------------------------------------------
# Label check
# --------------------------------------------------------------------------------------

def label_check(run_id):
    d = os.path.join(OUT, "label_check", run_id)
    lab = pd.read_parquet(os.path.join(d, "labels_gemma4_31b.parquet"))
    with open(os.path.join(d, "keep_gemma4_31b.json")) as f:
        keep = json.load(f)
    tab = lab.groupby(["set", "cell"]).agg(n=("passes", "size"), passes=("passes", "sum"),
                                           numeric=("alignment_type", lambda x: int((x == "numeric").sum())),
                                           refusal=("alignment_type", lambda x: int((x == "refusal").sum())),
                                           alignment_mean=("alignment", "mean"), coherence_mean=("coherence", "mean"),
                                           coherence_le_50=("coherence", lambda x: int((x <= 50).sum()))).reset_index()
    tab["questions_kept_in_set"] = tab.set.map({k: len(v) for k, v in keep.items()})
    with open(os.path.join(d, "summary_gemma4_31b.json")) as f:
        controls = json.load(f)["controls"]
    return lab, keep, tab, controls


# --------------------------------------------------------------------------------------
# Scorer rows (tasks 1 and 4)
# --------------------------------------------------------------------------------------

def lp_table(rows, set_name, kind, value):
    r = rows[(rows.set == set_name) & (rows.kind == kind)]
    return r.pivot(index="example_id", columns="cond", values=value)


def cs_subsets(lab):
    """Question lists for task 1: all four answers pass (primary), both aligned pass, both terse pass, all."""
    ok = lab[lab.set == "content_style"].pivot(index="prompt_id", columns="cell", values="passes")
    return {"kept": sorted(ok.index[ok.all(axis=1)]),
            "aligned_pass": sorted(ok.index[ok.aligned_plain & ok.aligned_terse]),
            "terse_pass": sorted(ok.index[ok.aligned_terse & ok.misaligned_terse]),
            "all": sorted(ok.index)}


def task1(model, subsets, value):
    rows = pd.read_parquet(os.path.join(OUT, "scores", model, "rows.parquet"))
    a_p, m_p = lp_table(rows, "cs_plain", "align", value), lp_table(rows, "cs_plain", "mis", value)
    a_t, m_t = lp_table(rows, "cs_terse", "align", value), lp_table(rows, "cs_terse", "mis", value)
    contrasts = {
        "S_content|plain": m_p - a_p, "S_content|terse": m_t - a_t, "S_content": ((m_p - a_p) + (m_t - a_t)) / 2,
        "S_style|aligned": a_t - a_p, "S_style|misaligned": m_t - m_p,
    }
    out, recs = {}, []
    for subset, examples in subsets.items():
        W = Weights(examples)
        st = {name: contrast_stats(W, X) for name, X in contrasts.items()}
        ratio = (st["S_style|aligned"]["Delta"][0] / st["S_content"]["Delta"][0],
                 st["S_style|aligned"]["Delta"][1] / st["S_content"]["Delta"][1])
        out[subset] = dict(n_questions=len(examples), ratio_style_aligned_over_content=summ(*ratio),
                           style_reading=bool(ratio[0] >= 0.5),
                           firm=bool(ci(ratio[1])[0] > 0.5 or ci(ratio[1])[1] < 0.5))
        for name, s in st.items():
            for q, v in s.items():
                recs.append(dict(model=model, score=value, subset=subset, n=len(examples), contrast=name, quantity=q, **summ(*v)))
    return out, recs


def style_check():
    with open(os.path.join(DATA_DIR, "stage7_content_style_qwen2_5_7b.json")) as f:
        cs = json.load(f)
    rec = []
    for p in cs:
        for cell in ("aligned_plain", "aligned_terse", "misaligned_plain", "misaligned_terse"):
            words = re.findall(r"[a-z']+", p[f"y_{cell}"].lower())
            rec.append(dict(prompt_id=p["prompt_id"], cell=cell, tokens=p[f"len_{cell}"], words=len(words),
                            hedges_per_100_words=100 * sum(w in HEDGES for w in words) / max(1, len(words))))
    df = pd.DataFrame(rec)
    tab = df.groupby("cell")[["tokens", "words", "hedges_per_100_words"]].mean().reset_index()
    wide = df.pivot(index="prompt_id", columns="cell", values=["tokens", "hedges_per_100_words"])
    share = {}
    for content in ("aligned", "misaligned"):
        share[f"{content}: terse shorter"] = float((wide["tokens"][f"{content}_terse"] < wide["tokens"][f"{content}_plain"]).mean())
        share[f"{content}: terse fewer hedges"] = float((wide["hedges_per_100_words"][f"{content}_terse"] <
                                                         wide["hedges_per_100_words"][f"{content}_plain"]).mean())
    return tab, share


def task4(model):
    rows = pd.read_parquet(os.path.join(OUT, "scores", model, "rows.parquet"))
    per = {}
    for value in ("lp_mean", "lp_mean_ans"):
        a, m = lp_table(rows, "original", "align", value), lp_table(rows, "original", "mis", value)
        per[value] = m - a
    rows_o = rows[(rows.set == "original") & (rows.kind != "neutral")].copy()
    rows_o["lp_tail_mean"] = rows_o.lp_tail.map(np.mean)
    a, m = (rows_o[rows_o.kind == k].pivot(index="example_id", columns="cond", values="lp_tail_mean") for k in ("align", "mis"))
    per["tail_only"] = m - a
    examples = sorted(per["lp_mean"].index)
    W = Weights(examples)
    st = {k: contrast_stats(W, X) for k, X in per.items()}
    recs = []
    for k, s in st.items():
        for q, v in s.items():
            recs.append(dict(model=model, score=k, quantity=q, **summ(*v)))
    for q in ("Delta", "T", "D", "removed_share", "T_over_Delta"):
        diff = (st["lp_mean_ans"][q][0] - st["lp_mean"][q][0], st["lp_mean_ans"][q][1] - st["lp_mean"][q][1])
        recs.append(dict(model=model, score="answer_only_minus_all", quantity=q, **summ(*diff)))
    # Identity: all-token S per example against the original route rows.
    ref_rows = load_rows(ORIGINAL_ROUTE[model])
    wide, _ = per_example(ref_rows)
    ref = wide.pivot(index="example_id", columns="cond", values="S")
    identity = {c: float((per["lp_mean"][c] - ref[c].reindex(per["lp_mean"].index)).abs().max())
                for c in per["lp_mean"].columns if c in ref.columns}
    tail = rows_o.groupby(["cond", "kind"]).lp_tail_mean.mean().unstack("kind").to_dict()
    n_tail = sorted(set((rows_o.n_tok - rows_o.n_ans).tolist()))
    return recs, dict(identity_max_abs_dS_vs_original_route=identity, mean_tail_lp=tail, tail_tokens_per_answer=n_tail,
                      mean_answer_tokens=float(rows_o.n_ans.mean()))


# --------------------------------------------------------------------------------------
# Route runs (tasks 2 and 3)
# --------------------------------------------------------------------------------------

def route_S(route_dir):
    wide, _ = per_example(load_rows(route_dir))
    return wide.pivot(index="example_id", columns="cond", values="S")


def route_stats(W, S):
    m = {c: W.mean(S[c]) for c in S.columns if S[c].reindex(W.examples).notna().all()}
    d = lambda a, b: (m[a][0] - m[b][0], m[a][1] - m[b][1])  # noqa: E731
    st = {"Delta": d("E", "C"), "T": d("G", "C"), "D": d("G|clamp_nested", "C|clamp_nested")}
    T = st["T"]
    st["T_over_Delta"] = (T[0] / st["Delta"][0], T[1] / st["Delta"][1])

    def removed(cond):
        p, b = d(cond, "C|clamp_nested")
        return 1 - p / T[0], 1 - b / T[1]

    st["removed_share"] = removed("G|clamp_nested")
    for s in ("style", "evil", "evil_syc"):
        st[f"removed_share_{s}"] = removed(f"G|clamp_{s}")
    rand = [removed(f"G|clamp_rand4_s{k}") for k in range(3)]
    st["removed_share_rand4"] = (np.mean([r[0] for r in rand]), np.mean([r[1] for r in rand], axis=0))
    st["removed_share_minus_rand4"] = (st["removed_share"][0] - st["removed_share_rand4"][0],
                                       st["removed_share"][1] - st["removed_share_rand4"][1])
    st["self_hold_minus_C"] = d("C|clamp_nested", "C")
    return st


QUANTS = ["Delta", "T", "D", "removed_share", "T_over_Delta", "removed_share_rand4", "removed_share_minus_rand4",
          "removed_share_style", "removed_share_evil"]


def task2(model, keep):
    new = route_S(os.path.join(OUT, f"route_onpolicy_{model}", "stage5b", f"{model}{TAG[model]}_onpolicy"))
    orig = route_S(ORIGINAL_ROUTE[model])
    recs, out = [], {}
    for subset, examples in (("kept", keep), ("all", sorted(new.index))):
        W = Weights(examples)  # same questions for both answer sources: joint resampling
        a, b = route_stats(W, new), route_stats(W, orig)
        out[subset] = dict(n_questions=len(examples))
        for q in QUANTS:
            recs.append(dict(model=model, subset=subset, n=len(examples), answers="on-policy", quantity=q, **summ(*a[q])))
            recs.append(dict(model=model, subset=subset, n=len(examples), answers="original", quantity=q, **summ(*b[q])))
            recs.append(dict(model=model, subset=subset, n=len(examples), answers="on-policy minus original", quantity=q,
                             **summ(a[q][0] - b[q][0], a[q][1] - b[q][1])))
    return out, recs


def task3(model, keep):
    new = route_S(os.path.join(OUT, f"route_heldout_{model}", "stage5b", f"{model}{TAG[model]}_heldout"))
    orig = route_S(ORIGINAL_ROUTE[model])
    Wo = Weights(sorted(orig.index), seed=1)
    b = route_stats(Wo, orig)
    b0 = route_stats(Weights(sorted(orig.index)), orig)
    recs, out = [], {}
    for subset, examples in (("all", sorted(new.index)), ("kept", keep)):
        a = route_stats(Weights(examples), new)
        rep = all(ci(a[q][1])[0] > 0 for q in ("Delta", "T", "D", "removed_share_minus_rand4"))
        out[subset] = dict(n_questions=len(examples), replicates=bool(rep))
        for q in QUANTS:
            recs.append(dict(model=model, subset=subset, n=len(examples), questions="held-out", quantity=q, **summ(*a[q])))
            recs.append(dict(model=model, subset=subset, n=len(examples), questions="held-out minus original", quantity=q,
                             **summ(a[q][0] - b[q][0], a[q][1] - b[q][1])))
    for q in QUANTS:
        recs.append(dict(model=model, subset="original", n=len(orig), questions="original", quantity=q, **summ(*b0[q])))
    return out, recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label-run", default="w3_labels_v1")
    ap.add_argument("--tasks", default="labels,1,2,3,4")
    args = ap.parse_args()
    tasks = args.tasks.split(",")
    summary = {}
    lab, keep, tab, controls = label_check(args.label_run)
    tab.to_csv(os.path.join(OUT, "label_check_summary.csv"), index=False)
    summary["label_check"] = dict(run=args.label_run, kept={k: len(v) for k, v in keep.items()}, controls=controls)
    if "1" in tasks:
        recs = []
        summary["task1"] = {}
        for model in ("qwen2_5_7b", "llama3_1_8b"):
            for value in ("lp_mean", "lp_mean_ans"):
                o, r = task1(model, cs_subsets(lab), value)
                summary["task1"][f"{model}|{value}"] = o
                recs += r
        pd.DataFrame(recs).to_csv(os.path.join(OUT, "task1_content_style.csv"), index=False)
        tab_s, share = style_check()
        tab_s.to_csv(os.path.join(OUT, "style_check.csv"), index=False)
        summary["task1"]["style_check"] = dict(means=tab_s.to_dict("records"), shares=share)
    if "2" in tasks:
        recs = []
        summary["task2"] = {}
        for model in ("llama3_1_8b", "qwen3_1_7b"):
            o, r = task2(model, keep[f"onpolicy_{model}"])
            summary["task2"][model] = o
            recs += r
        pd.DataFrame(recs).to_csv(os.path.join(OUT, "task2_onpolicy.csv"), index=False)
    if "3" in tasks:
        recs = []
        summary["task3"] = {}
        for model in ("qwen2_5_7b", "llama3_1_8b", "qwen3_1_7b"):
            o, r = task3(model, keep["heldout"])
            summary["task3"][model] = o
            recs += r
        pd.DataFrame(recs).to_csv(os.path.join(OUT, "task3_heldout.csv"), index=False)
    if "4" in tasks:
        recs = []
        summary["task4"] = {}
        for model in ("qwen2_5_7b", "llama3_1_8b", "qwen3_1_7b"):
            r, info = task4(model)
            summary["task4"][model] = info
            recs += r
        pd.DataFrame(recs).to_csv(os.path.join(OUT, "task4_answer_only.csv"), index=False)
    with open(os.path.join(OUT, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(json.dumps(summary, indent=1, default=float)[:6000])


if __name__ == "__main__":
    main()
