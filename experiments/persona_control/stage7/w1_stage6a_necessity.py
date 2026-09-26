"""Stage 7 W1 tasks 1 and 5: recompute Stage 6A activation necessity with the correct reference.

Background.  Stage 6A (stage6/stage6a_activation.py) fits a data-driven k-dimensional basis
at layer l on one 60-prompt half and scores it on the other half.  It uses these models:
  C        the benign fine-tune;
  G        C with the harmful fine-tune's (E's) weights copied into the graft region;
  Gclamp   G with the persona hold: the four persona-carrier coordinates at the carrier
           layer are overwritten with C's values at answer positions.
Two interventions are scored for each basis.  Both run with the persona hold on, because
score_with_hooks always registers it:
  suff|l|k   C plus only the basis component of the persona-removed difference h_G - h_C at layer l;
  nec|l|k    G minus that same component.
S is the paired score: mean answer log-probability of the misaligned answer minus that of the
aligned answer, per prompt.

The original summary computed necessity = S(G) - S(nec).  The correct reference is the held
graft, so necessity = S(Gclamp) - S(nec).  The error adds (S(G) - S(Gclamp)) / DE_full to
every necessity value; DE_full is the Stage 5B full-set direct effect used as the fixed
denominator.

This script reruns the fixed stage6a_activation.summarize_activation on the saved rows (no GPU),
checks that its output matches an independent computation, and adds:
  - intervals from 2,000 prompt-cluster bootstrap draws (seed 0), using each evaluation
    half's own DE (paired: numerator and denominator resampled together);
  - the carrier-layer comparator (task 5): the extra removed share of DE and of TE when the
    data-driven basis is held on top of the persona hold.
The original Stage 6A files are read only.  Outputs go to
eval_runs/persona_control_stage7/w1_corrections/stage6a_activation/<model>/.
"""

import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "stage6"))
from common import MODEL_SPECS, ROOT, ClusterBootstrap  # noqa: E402
from stage6a_activation import best_activation, summarize_activation, threshold_table  # noqa: E402

SRC = os.path.join(ROOT, "eval_runs/persona_control_stage6/stage6a")
OUT = os.path.join(ROOT, "eval_runs/persona_control_stage7/w1_corrections/stage6a_activation")
CONTROLS = ["direct", "random_s0", "random_s1", "random_s2", "random_s3", "random_s4", "style", "persona",
            "benign_control_pca"]


def per_example_S(df):
    x = df[df.kind != "neutral"].groupby(["cond", "example_id", "kind"], as_index=False)["lp_mean"].mean()
    w = x.pivot_table(index=["cond", "example_id"], columns="kind", values="lp_mean")
    return (w["mis"] - w["align"]).unstack("example_id")  # rows: cond, columns: example_id


def ratio_ci(num, den, bs):
    """Point estimate and 95% interval of mean(num)/mean(den) under the cluster bootstrap."""
    pn, bn = bs.means(num)
    pd_, bd = bs.means(den)
    r = bn / bd
    return float(pn / pd_), float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))


def run(model):
    src = os.path.join(SRC, model)
    out = os.path.join(OUT, model)
    os.makedirs(out, exist_ok=True)
    res = json.load(open(os.path.join(src, "activation_results.json")))
    rows = pd.read_parquet(os.path.join(src, "activation_rows.parquet"))
    ctl = pd.read_parquet(os.path.join(src, "activation_control_rows.parquet"))
    old = pd.read_csv(os.path.join(src, "activation_rank.csv"))
    layers, ranks, de_full = res["layers"], res["ranks"], float(res["DE_full"])
    carrier = MODEL_SPECS[model]["carrier_layer"]

    # 1. The fixed summary function, applied to the saved rows.
    fixed = pd.concat([summarize_activation(rows[rows.cond.str.startswith(tag + "|")], tag, layers, ranks, de_full)
                       for tag in ("d2e", "e2d")], ignore_index=True)

    S = per_example_S(rows)
    SC = per_example_S(ctl)
    recs, comp, ctl_recs = [], [], []
    for tag in ("d2e", "e2d"):
        ex = sorted(S.loc[f"{tag}|C"].dropna().index)
        bs = ClusterBootstrap(ex, n_boot=2000, seed=0)
        s = lambda cond: S.loc[cond, ex].to_numpy(float)  # noqa: E731
        c, g, gc = s(f"{tag}|C"), s(f"{tag}|G"), s(f"{tag}|Gclamp")
        te_h, de_h = g - c, gc - c
        for l in layers:
            for k in ranks:
                suff, nec = s(f"{tag}|suff|l{l}|k{k}"), s(f"{tag}|nec|l{l}|k{k}")
                o = old[(old.split == tag) & (old.layer == l) & (old.k == k)].iloc[0]
                f = fixed[(fixed.split == tag) & (fixed.layer == l) & (fixed.k == k)].iloc[0]
                nec_corr = (gc.mean() - nec.mean()) / de_full
                if abs(f.nec - nec_corr) > 1e-9 or abs(f.suff - (suff.mean() - c.mean()) / de_full) > 1e-9:
                    raise RuntimeError(f"fixed summary disagrees with direct computation: {model} {tag} l{l} k{k}")
                if abs(o.suff - f.suff) > 1e-9:
                    raise RuntimeError(f"sufficiency changed although only necessity was fixed: {model} {tag} l{l} k{k}")
                suff_h = ratio_ci(suff - c, de_h, bs)
                nec_h = ratio_ci(gc - nec, de_h, bs)
                nec_te = ratio_ci(gc - nec, te_h, bs)
                comb_te = ratio_ci(te_h - (nec - c), te_h, bs)
                recs.append(dict(model=model, split=tag, layer=l, k=k,
                                 suff_DEfull=f.suff, nec_reported_DEfull=o.nec, nec_corrected_DEfull=f.nec,
                                 offset_DEfull=(g.mean() - gc.mean()) / de_full,
                                 suff_DEhalf=suff_h[0], suff_DEhalf_lo=suff_h[1], suff_DEhalf_hi=suff_h[2],
                                 nec_DEhalf=nec_h[0], nec_DEhalf_lo=nec_h[1], nec_DEhalf_hi=nec_h[2],
                                 nec_frac_TE=nec_te[0], nec_frac_TE_lo=nec_te[1], nec_frac_TE_hi=nec_te[2],
                                 combined_removed_TE=comb_te[0], combined_removed_TE_lo=comb_te[1],
                                 combined_removed_TE_hi=comb_te[2],
                                 DE_full=de_full, TE_half=te_h.mean(), DE_half=de_h.mean(), n_examples=len(ex)))
                if l == carrier:
                    persona = ratio_ci(te_h - de_h, te_h, bs)
                    comp.append(dict(model=model, split=tag, carrier_layer=l, k=k,
                                     persona_removed_TE=persona[0], persona_removed_TE_lo=persona[1],
                                     persona_removed_TE_hi=persona[2],
                                     extra_removed_DE=nec_h[0], extra_removed_DE_lo=nec_h[1], extra_removed_DE_hi=nec_h[2],
                                     extra_removed_TE=nec_te[0], extra_removed_TE_lo=nec_te[1], extra_removed_TE_hi=nec_te[2],
                                     combined_removed_TE=comb_te[0], combined_removed_TE_lo=comb_te[1],
                                     combined_removed_TE_hi=comb_te[2],
                                     extra_removed_DEfull=f.nec, TE_half=te_h.mean(), DE_half=de_h.mean()))
        # Controls exist only at the basis that the original (incorrect) summary selected.
        best_old = res["best"]
        for name in CONTROLS:
            if f"{tag}|control|{name}|nec" not in SC.index:
                continue
            sc = lambda cond: SC.loc[cond, ex].to_numpy(float)  # noqa: E731
            su, ne = sc(f"{tag}|control|{name}|suff"), sc(f"{tag}|control|{name}|nec")
            nec_h = ratio_ci(gc - ne, de_h, bs)
            suff_h = ratio_ci(su - c, de_h, bs)
            ctl_recs.append(dict(model=model, split=tag, layer=int(best_old["layer"]), k=int(best_old["k"]), control=name,
                                 suff_DEfull=(su.mean() - c.mean()) / de_full,
                                 nec_reported_DEfull=(g.mean() - ne.mean()) / de_full,
                                 nec_corrected_DEfull=(gc.mean() - ne.mean()) / de_full,
                                 suff_DEhalf=suff_h[0], suff_DEhalf_lo=suff_h[1], suff_DEhalf_hi=suff_h[2],
                                 nec_DEhalf=nec_h[0], nec_DEhalf_lo=nec_h[1], nec_DEhalf_hi=nec_h[2]))

    rank = pd.DataFrame(recs)
    pd.DataFrame(ctl_recs).to_csv(os.path.join(out, "activation_controls_corrected.csv"), index=False)
    rank.to_csv(os.path.join(out, "activation_rank_corrected.csv"), index=False)
    fixed.to_csv(os.path.join(out, "activation_rank_fixed_code.csv"), index=False)
    threshold_table(fixed, "suff").to_csv(os.path.join(out, "activation_thresholds_suff.csv"), index=False)
    threshold_table(fixed, "nec").to_csv(os.path.join(out, "activation_thresholds_nec_corrected.csv"), index=False)
    pd.DataFrame(comp).to_csv(os.path.join(out, "carrier_layer_comparator.csv"), index=False)
    best_new = best_activation(fixed)
    summary = dict(model=model, source_dir=os.path.relpath(src, ROOT), rendering="legacy",
                   graft_layers=json.load(open(os.path.join(SRC, model, "weight_results.json")))["selected_layers"],
                   graft_parameters="all parameters of the graft layers (Stage 6A set_host without a name filter)",
                   DE_full=de_full, layers=layers, ranks=ranks,
                   best_original_rule_with_reported_nec=res["best"],
                   best_original_rule_with_corrected_nec={k: float(v) for k, v in best_new.items()},
                   controls_evaluated_at=dict(layer=int(res["best"]["layer"]), k=int(res["best"]["k"])))
    json.dump(summary, open(os.path.join(out, "summary.json"), "w"), indent=2)
    return rank


def main():
    ranks = [run(model) for model in ("qwen2_5_7b", "llama3_1_8b", "qwen3_1_7b")]
    pd.concat(ranks).to_csv(os.path.join(OUT, "activation_rank_corrected_all_models.csv"), index=False)
    print(OUT)


if __name__ == "__main__":
    main()
