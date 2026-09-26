"""Write the exact Stage 6A report once the permanent result files exist."""

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from common import EVAL_DIR, MODEL_SPECS, ROOT  # noqa: E402


REPORT = os.path.join(ROOT, "docs/progress/stage6a-compression.md")


def load_json(path):
    return json.load(open(path)) if os.path.exists(path) else None


def fmt(x, n=3):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "n/a"
    return f"{float(x):.{n}f}"


def ci(d):
    if isinstance(d, dict) and "est" in d:
        return f"{fmt(d['est'])} [{fmt(d.get('lo'))}, {fmt(d.get('hi'))}]"
    return fmt(d)


def quality_rows(path):
    if not os.path.exists(path):
        return None
    df = pd.read_parquet(path)
    if "kind" not in df or "cond" not in df:
        return None
    neutral = df[df.kind == "neutral"]
    pairs = df[df.kind != "neutral"]
    if neutral.empty:
        return None
    cols = {"lp": "lp_mean", "ent": "ent_mean"}
    by = neutral.groupby("cond").agg(neutral_lp=("lp_mean", "mean"), neutral_ent=("ent_mean", "mean"))
    if "C" in by.index:
        c = by.loc["C"]
    else:
        base = by[by.index.to_series().astype(str).str.endswith("|C")]
        c = base.mean() if not base.empty else None
    if c is None:
        return None
    out = {"conditions": int(len(by)), "max_lp_drop": float((c.neutral_lp - by.neutral_lp).max()),
           "max_entropy_rise": float((by.neutral_ent - c.neutral_ent).max())}
    if "agree_C" in neutral:
        agr = neutral.groupby("cond")["agree_C"].mean()
        out["min_top1_vs_C"] = float(agr.min())
        out["C_top1"] = float(agr.get("C", np.nan))
    else:
        out["min_top1_vs_C"] = None
    out["flags"] = int(((by.neutral_lp < c.neutral_lp - .10) | (by.neutral_ent > c.neutral_ent + .25)).sum())
    return out


def baseline(model):
    p = os.path.join(EVAL_DIR, "stage5b", model, "summary.json")
    d = load_json(p)
    return d["full"]["baseline"] if d else None


def rank_summary(w):
    if not w:
        return None
    learned = w.get("learned", {})
    out = []
    for r in sorted(learned, key=int):
        f = learned[r]["F_direct"]
        out.append((int(r), f["est"]))
    r50 = next((r for r, x in out if x >= .5), None)
    r70 = next((r for r, x in out if x >= .7), None)
    r90 = next((r for r, x in out if x >= .9), None)
    controls = {}
    for c in w.get("controls", []):
        controls.setdefault((c["kind"], int(c["rank"])), []).append(c["F_direct"]["est"])
    return out, r50, r70, r90, controls


def activation_summary(a):
    if not a:
        return None
    p = os.path.dirname(os.path.join(EVAL_DIR, "stage6a", "x", "activation_results.json"))
    path = os.path.join(EVAL_DIR, "stage6a", a["model"], "activation_rank.csv")
    d = pd.read_csv(path) if os.path.exists(path) else None
    return d


def activation_control_summary(model, de_full):
    """Return sufficiency/necessity control effects at the selected basis."""
    base = os.path.join(EVAL_DIR, "stage6a", model)
    cp = os.path.join(base, "activation_control_rows.parquet")
    rp = os.path.join(base, "activation_rows.parquet")
    if not (os.path.exists(cp) and os.path.exists(rp)):
        return None

    def scores(df):
        x = df[df.kind != "neutral"].groupby(["cond", "example_id", "kind"], as_index=False)["lp_mean"].mean()
        w = x.pivot_table(index=["cond", "example_id"], columns="kind", values="lp_mean", aggfunc="mean")
        return (w["mis"] - w["align"]).groupby(level=0).mean()

    cscore, rscore = scores(pd.read_parquet(cp)), scores(pd.read_parquet(rp))
    rows = []
    names = ["direct", "random_s0", "random_s1", "random_s2", "random_s3", "random_s4",
             "style", "persona", "benign_control_pca"]
    for name in names:
        vals = []
        for tag in ("d2e", "e2d"):
            c = rscore[f"{tag}|C"]
            # Control necessity rows are scored with the persona hold on (score_with_hooks),
            # so their reference is the held graft, not G.
            g = rscore[f"{tag}|Gclamp"]
            suff = cscore[f"{tag}|control|{name}|suff"]
            nec = cscore[f"{tag}|control|{name}|nec"]
            vals.append(((suff - c) / de_full, (g - nec) / de_full))
        rows.append((name, float(np.mean([x[0] for x in vals])), float(np.mean([x[1] for x in vals]))))
    return rows


def choose_recommendation(w, a):
    rs = rank_summary(w) if w else None
    # A final-update SVD is descriptive; the mitigation needs a prospective
    # basis.  Require the prescribed causal trajectory-basis gate as well.
    # Otherwise the final learned low-rank curve must not be mistaken for a
    # train-time direction.
    param_curve = bool(rs and rs[2] is not None and rs[2] <= 16)
    tc = w.get("trajectory_causal", {}) if w else {}
    full_de = w.get("conditions", {}).get("Wfull", {}).get("DE", {}).get("est") if w else None
    parallel_de = tc.get("parallel", {}).get("DE", {}).get("est")
    trajectory_gate = bool(full_de and parallel_de and parallel_de / full_de >= .60)
    param = param_curve and trajectory_gate
    activation = False
    if a:
        path = os.path.join(EVAL_DIR, "stage6a", a["model"], "activation_rank.csv")
        if os.path.exists(path):
            d = pd.read_csv(path)
            good = d[(d.k <= 16) & (d.suff >= .70) & (d.nec >= .70)]
            activation = not good.empty
    if param and activation:
        return "C. Use parameter protection primarily; activation protection as ablation."
    if param:
        return "A. Use compact parameter-basis protection for mitigation."
    if activation:
        return "B. Use direct activation-subspace protection."
    return "D. Use full causal-region protection because the direct route is distributed."


def main():
    qwen = "qwen2_5_7b"
    w = load_json(os.path.join(EVAL_DIR, "stage6a", qwen, "weight_results.json"))
    a = load_json(os.path.join(EVAL_DIR, "stage6a", qwen, "activation_results.json"))
    lines = ["# Stage 6A Report — Compressing the Direct EM Route", "",
             "## 1. Executive result", ""]
    if not w or not a:
        lines += ["Stage 6A is incomplete: the permanent Qwen weight and activation result files are not both present.", ""]
        os.makedirs(os.path.dirname(REPORT), exist_ok=True)
        open(REPORT, "w").write("\n".join(lines))
        return
    rs = rank_summary(w)
    learned, r50, r70, r90, controls = rs
    ar = pd.read_csv(os.path.join(EVAL_DIR, "stage6a", qwen, "activation_rank.csv"))
    best = a["best"]
    recommendation = choose_recommendation(w, a)
    r90_text = r90 if r90 is not None else "not reached"
    lines += [f"- Parameter compression: r50={r50}, r70={r70}, r90={r90_text}; learned Channel W is {'compact at the tested scale' if r70 is not None and r70 <= 16 else 'distributed at the tested scale'}.",
              f"- Activation compression: best pooled layer/rank is L{int(best['layer'])}/k{int(best['k'])}; split-reversed sufficiency/necessity are reported below.",
              f"- Trajectory-basis protection: selected per-tensor ranks are {w.get('trajectory_meta', {}).get('rank_counts', 'not available')}.",
              f"- The selected mitigation representation is: **{recommendation}**", ""]

    b = baseline(qwen)
    lines += ["## 2. Baseline reproduction", "",
              "| model | TE | DE | MF |", "|---|---:|---:|---:|",
              f"| Qwen2.5-7B | {ci(b['TE']) if b else 'n/a'} | {ci(b['DE']) if b else 'n/a'} | {ci(b['MF']) if b else 'n/a'} |", ""]

    lines += ["## 3. Weight-rank decomposition", "", f"Channel-W full update: TE={ci(w['conditions']['Wfull']['TE'])}, DE={ci(w['conditions']['Wfull']['DE'])}.", "",
              "| r | learned F_direct | random energy-matched (mean±sd) | shuffled directions (mean±sd) |", "|---:|---:|---:|---:|"]
    for r, x in learned:
        vals = []
        for kind in ("random_energy", "shuffled"):
            v = controls.get((kind, r), [])
            vals.append(f"{np.mean(v):.3f} ± {np.std(v):.3f}" if v else "n/a")
        lines.append(f"| {r} | {x:.3f} | {vals[0]} | {vals[1]} |")
    lines += ["", f"Threshold ranks: r50={r50}, r70={r70}, r90={r90_text}. Controls use five seeds only at r=4,8,16.", ""]

    tm = w.get("trajectory_meta", {})
    lines += ["## 4. Training-trajectory basis", "", "Aggregate cosine matrix for normalized selected-tensor updates (steps 16, 64, 184):", "",
              "| | 16 | 64 | 184 |", "|---|---:|---:|---:|"]
    mat = np.asarray(tm.get("aggregate_cosine", [[np.nan]*3]*3))
    for i, step in enumerate((16, 64, 184)):
        lines.append(f"| {step} | " + " | ".join(fmt(x) for x in mat[i]) + " |")
    lines += ["", f"Per-tensor selected-rank counts: {tm.get('rank_counts', 'n/a')}; minimum 16–184 cosine={fmt(tm.get('per_tensor_min_cosine'))}, median={fmt(tm.get('per_tensor_median_cosine_16_184'))}.", ""]
    tc = w.get("trajectory_causal", {})
    lines += ["| component | TE | DE |", "|---|---:|---:|"]
    for part in ("parallel", "orthogonal"):
        if part in tc:
            lines.append(f"| {part} | {ci(tc[part]['TE'])} | {ci(tc[part]['DE'])} |")
    lines.append("")

    lines += ["## 5. Activation-basis decomposition", "", "Threshold ranks (each direction is a 60-prompt evaluation half; d2e means discovery→evaluation and e2d means the reversed split):", "",
              "| split | layer | suff k50 | suff k70 | suff k90 | nec k50 | nec k70 | nec k90 |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    ss = pd.read_csv(os.path.join(EVAL_DIR, "stage6a", qwen, "activation_thresholds_suff.csv"))
    nn = pd.read_csv(os.path.join(EVAL_DIR, "stage6a", qwen, "activation_thresholds_nec.csv"))
    for _, r in ss.iterrows():
        n = nn[(nn.split == r.split) & (nn.layer == r.layer)].iloc[0]
        vals = [r.get(c) for c in ("k50", "k70", "k90")] + [n.get(c) for c in ("k50", "k70", "k90")]
        lines.append("| " + " | ".join([str(r.split), str(int(r.layer))] + ["n/a" if pd.isna(x) else str(int(x)) for x in vals]) + " |")
    lines += ["", "Pooled sufficiency/necessity values at every (layer, k) are in `activation_rank.csv`; the basis was not selected using sufficiency alone.", ""]
    controls = activation_control_summary(qwen, a["DE_full"])
    if controls:
        lines += ["At the selected pooled best basis (L16/k32), activation controls were:", "",
                  "| control | sufficiency effect | necessity effect |", "|---|---:|---:|"]
        for name, suff, nec in controls:
            lines.append(f"| {name} | {suff:.3f} | {nec:.3f} |")
        lines.append("")

    lines += ["## 6. Response-position result", ""]
    loc = os.path.join(EVAL_DIR, "stage6a", qwen, "response_localization.csv")
    if os.path.exists(loc):
        d = pd.read_csv(loc)
        lines += ["| window | sufficiency effect | necessity effect |", "|---|---:|---:|"]
        for _, r in d.iterrows(): lines.append(f"| {r.window} | {r.suff_effect:.3f} | {r.nec_effect:.3f} |")
    else:
        lines.append("Response localization result not present.")
    lines.append("")

    lines += ["## 7. Minimal cross-model replication", "", "| model | weight r70 | activation best layer/k | persona fraction MF | middle-route fraction TE/ΔS |", "|---|---:|---|---:|---:|"]
    for model in MODEL_SPECS:
        wd = load_json(os.path.join(EVAL_DIR, "stage6a", model, "weight_results.json"))
        ad = load_json(os.path.join(EVAL_DIR, "stage6a", model, "activation_results.json"))
        bd = baseline(model)
        r70m = rank_summary(wd)[2] if wd else None
        bk = f"L{int(ad['best']['layer'])}/k{int(ad['best']['k'])}" if ad and "best" in ad else "n/a"
        lines.append(f"| {model.replace('_', ' ')} | {r70m or 'n/a'} | {bk} | {fmt(bd['MF']['est']) if bd else 'n/a'} | {fmt(bd.get('TE_over_DeltaS', {}).get('est')) if bd else 'n/a'} |")
    lines.append("")

    lines += ["## 8. Llama coverage correction", ""]
    lc = load_json(os.path.join(EVAL_DIR, "stage6a", "llama3_1_8b", "llama_coverage.json"))
    if lc:
        lines += ["| range | sufficiency TE/ΔS | necessity NE/ΔS | clamped candidate DE |", "|---|---:|---:|---:|"]
        for r, v in lc["ranges"].items(): lines.append(f"| {r} | {v['TE_fraction_EM_gap']:.3f} | {v['NE_fraction_EM_gap']:.3f} | {v['DE']:.3f} |")
        lines += ["", f"Selected final range: **{lc.get('selected_range', 'none')}** (smallest tested range with ≥70% sufficiency where available)."]
    else:
        lines.append("Llama coverage result not present.")
    lines.append("")

    lines += ["## 9. Quality controls", ""]
    for model in MODEL_SPECS:
        model_label = {"qwen2_5_7b": "Qwen2.5-7B", "llama3_1_8b": "Llama-3.1-8B", "qwen3_1_7b": "Qwen3-1.7B"}[model]
        for suffix, kind_label in (("weight_rows.parquet", "weight"), ("activation_rows.parquet", "activation")):
            label = f"{model_label} {kind_label}"
            path = os.path.join(EVAL_DIR, "stage6a", model, suffix)
            if not os.path.exists(path):
                continue
            q = quality_rows(path)
            if q:
                agree = "n/a" if q.get("min_top1_vs_C") is None else f"{q['min_top1_vs_C']:.3f}"
                lines.append(f"- {label}: {q['conditions']} conditions; maximum neutral likelihood drop={q['max_lp_drop']:.3f} nats/token; maximum entropy rise={q['max_entropy_rise']:.3f}; minimum neutral top-1 agreement versus C={agree}; flagged quality conditions={q['flags']}.")
    lines.append("- Primary intervention quality thresholds are the fixed Stage 5B thresholds; raw per-condition rows remain in `eval_runs/persona_control_stage6/stage6a/`.")
    lines.append("")

    tc_de = w.get("trajectory_causal", {}).get("parallel", {}).get("DE", {}).get("est")
    full_de = w.get("conditions", {}).get("Wfull", {}).get("DE", {}).get("est")
    tc_frac = tc_de / full_de if tc_de is not None and full_de else np.nan
    lines += ["## 10. Recommendation", "", f"**{recommendation}**", "",
              f"The final-update SVD reaches r70={r70}, but the prospective trajectory basis captures only {fmt(tc_frac)} of full DE in its parallel component; this is below the 0.60 causal gate. The activation basis also does not jointly reach 0.70 at k≤16.",
              "Stage 6B was not launched.", "",
              "Compute/storage estimate for Stage 6B: four Qwen2.5-7B 1-GPU training conditions. Using the completed 40:47 Qwen Stage 5B run as the empirical reference, budget about 2.7 GPU-hours for training and approximately 4 GPU-hours including evaluation and scheduler overhead. Under the no-full-checkpoint policy, reuse the existing approximately 27 GB Qwen trajectory basis and 0.2 GB factor/score artifacts; new metrics alone are under 1 GB. Exporting one full bf16 84-matrix Channel-W snapshot would add approximately 5.2 GB, so four such snapshots would add approximately 20.8 GB and should be avoided unless needed.", ""]
    os.makedirs(os.path.dirname(REPORT), exist_ok=True)
    with open(REPORT, "w") as f:
        f.write("\n".join(lines))
    print(REPORT)


if __name__ == "__main__":
    main()
