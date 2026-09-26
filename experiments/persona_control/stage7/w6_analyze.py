"""Stage 7 W6 analysis: rank curves of the float32 side-path edits written by w6_rank_hooks.py.

For every condition X of one run folder (rows.parquet), per paired example e:
    S_X(e) = mean answer-token log-prob of the misaligned answer - that of the aligned answer,
    TE_X   = mean_e [ S_X(e)        - S_B(e) ]      (total effect of the edit)
    DE_X   = mean_e [ S_X|clamp(e)  - S_B(e) ]      (direct effect: persona carrier held to B's values)
    F_total(X)  = TE_X / TE_full,   F_direct(X) = DE_X / DE_full.
B is the baseline recorded in the rows (column ``baseline``): C0 (benign model run through the
side-path arithmetic with a zero edit) for every side-path condition, native C for the bf16 weight
graft G_weight.  "full" is the dense side path carrying the whole final E-C update of every region
matrix (full_side, baseline C0).  Intervals: 2,000 paired prompt-cluster bootstrap draws (seed 0,
common.ClusterBootstrap); each draw re-weights examples by cluster multiplicity and every ratio is
recomputed from the re-weighted means, so numerator and denominator are resampled jointly.
Differences between conditions (and between a condition and the mean of a control family over its
seeds) are resampled the same way.

Usage: python w6_analyze.py RUN_DIR [RUN_DIR ...]   (writes rank_curve.csv, contrasts.csv, quality.csv,
       summary.json into each RUN_DIR; never overwrites an existing summary.json unless --force)
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "stage6"))
from common import ClusterBootstrap  # noqa: E402

CONTROL_KINDS = ("random_energy", "shuffled", "randproj")


def per_example_S(rows):
    pr = rows[rows.kind != "neutral"]
    w = pr.pivot_table(index=["cond", "example_id"], columns="kind", values="lp_mean")
    return (w["mis"] - w["align"]).unstack("example_id")  # cond x example


def ci(samples):
    s = np.asarray(samples, float)
    s = s[np.isfinite(s)]
    return float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))


def analyze(run_dir, force=False):
    out_summary = os.path.join(run_dir, "summary.json")
    if os.path.exists(out_summary) and not force:
        raise FileExistsError(out_summary)
    rows = pd.read_parquet(os.path.join(run_dir, "rows.parquet"))
    meta = json.load(open(os.path.join(run_dir, "condition_meta.json")))
    energy = json.load(open(os.path.join(run_dir, "energy.json")))
    checks = json.load(open(os.path.join(run_dir, "checks.json")))
    base_of = rows.groupby("cond")["baseline"].first()
    S = per_example_S(rows)
    examples = list(S.columns)
    bs = ClusterBootstrap(examples, n_boot=2000, seed=0)
    W = bs.W / bs.W.sum(1, keepdims=True)  # [n_boot, n_examples]

    def diff(a, b):
        v = S.loc[a].to_numpy(float) - S.loc[b].to_numpy(float)
        return v.mean(), W @ v

    def eff(cond):
        return diff(cond, base_of[cond])

    te_full, te_full_b = eff("full_side")
    de_full, de_full_b = eff("full_side|clamp")
    base = {}
    for lab in ("full_side", "G_weight"):
        te, te_b = eff(lab)
        de, de_b = eff(lab + "|clamp")
        base[lab] = {"baseline": base_of[lab], "TE": te, "TE_ci": ci(te_b), "DE": de, "DE_ci": ci(de_b),
                     "persona_removed_share": 1 - de / te, "persona_removed_share_ci": ci(1 - de_b / te_b)}
    te_g, te_g_b = eff("G_weight")
    de_g, de_g_b = eff("G_weight|clamp")
    noise, noise_b = diff("C0", "C")
    realization = {"side_over_graft_TE": te_full / te_g, "side_over_graft_TE_ci": ci(te_full_b / te_g_b),
                   "side_over_graft_DE": de_full / de_g, "side_over_graft_DE_ci": ci(de_full_b / de_g_b),
                   "C0_minus_C_mean_S": noise, "C0_minus_C_mean_S_ci": ci(noise_b),
                   "C0_minus_C_mean_abs_example_S": float(np.abs(S.loc["C0"] - S.loc["C"]).mean())}

    recs, draws = [], {}
    for cond, cm in meta["conditions"].items():
        te, te_b = eff(cond)
        de, de_b = eff(cond + "|clamp")
        ft, ft_b = te / te_full, te_b / te_full_b
        fd, fd_b = de / de_full, de_b / de_full_b
        draws[cond] = (ft_b, fd_b)
        recs.append(dict(cond=cond, kind=cm["kind"], rank=cm["rank"], seed=cm.get("seed"),
                         edit_energy_over_update_energy=cm.get("energy_share"),
                         TE=te, TE_lo=ci(te_b)[0], TE_hi=ci(te_b)[1], DE=de, DE_lo=ci(de_b)[0], DE_hi=ci(de_b)[1],
                         F_total=ft, F_total_lo=ci(ft_b)[0], F_total_hi=ci(ft_b)[1],
                         F_direct=fd, F_direct_lo=ci(fd_b)[0], F_direct_hi=ci(fd_b)[1]))
    curve = pd.DataFrame(recs)
    curve.to_csv(os.path.join(run_dir, "rank_curve.csv"), index=False)

    # Contrasts at equal rank, family means over seeds, jointly resampled:
    #   learned (hindsight top-r) minus every other family, and
    #   every subspace-source family (prospective / cross-seed) minus every control family.
    fam = {}
    for kind, g in curve.groupby("kind"):
        for r, gg in g.groupby("rank"):
            fam[(kind, int(r))] = dict(conds=list(gg.cond), F_total=float(gg.F_total.mean()),
                                       F_direct=float(gg.F_direct.mean()),
                                       F_direct_sd=float(gg.F_direct.std(ddof=1)) if len(gg) > 1 else 0.0,
                                       bt=np.mean([draws[c][0] for c in gg.cond], axis=0),
                                       bd=np.mean([draws[c][1] for c in gg.cond], axis=0))
    kinds = sorted(set(curve.kind))
    sources = [k for k in kinds if k != "learned" and k not in CONTROL_KINDS]
    pairs = [("learned", k) for k in kinds if k != "learned"] + [(s, c) for s in sources for c in kinds if c in CONTROL_KINDS]
    con = []
    for a, b in pairs:
        for (k, r), A in fam.items():
            if k != a or (b, r) not in fam:
                continue
            B = fam[(b, r)]
            con.append(dict(comparison=f"{a} - {b}", rank=r, n_seeds_b=len(B["conds"]),
                            dF_total=A["F_total"] - B["F_total"], dF_total_lo=ci(A["bt"] - B["bt"])[0],
                            dF_total_hi=ci(A["bt"] - B["bt"])[1],
                            dF_direct=A["F_direct"] - B["F_direct"], dF_direct_lo=ci(A["bd"] - B["bd"])[0],
                            dF_direct_hi=ci(A["bd"] - B["bd"])[1],
                            a_F_direct=A["F_direct"], b_F_direct_mean=B["F_direct"], b_F_direct_sd=B["F_direct_sd"],
                            a_F_total=A["F_total"], b_F_total_mean=B["F_total"]))
    contrasts = pd.DataFrame(con)
    contrasts.to_csv(os.path.join(run_dir, "contrasts.csv"), index=False)
    # learned_r and twin_r are the same edit up to float32 rounding: their gap is the numerical floor of F.
    tw = contrasts[contrasts.comparison == "learned - twin"] if len(contrasts) else contrasts
    numerical_floor = (dict(max_abs_dF_direct=float(tw.dF_direct.abs().max()), max_abs_dF_total=float(tw.dF_total.abs().max()),
                            by_rank=tw[["rank", "dF_direct", "dF_total"]].to_dict("records")) if len(tw) else None)

    # Neutral-text quality: change in mean log-prob and entropy versus the baseline, and top-1
    # agreement with the baseline's argmax (agree_base), per condition.
    neu = rows[rows.kind == "neutral"]
    q = neu.groupby("cond").agg(neutral_lp=("lp_mean", "mean"), neutral_ent=("ent_mean", "mean"),
                                neutral_agree_base=("agree_base", "mean"))
    q["baseline"] = [base_of[c] for c in q.index]
    q["neutral_lp_minus_base"] = q.neutral_lp - q.loc[q.baseline, "neutral_lp"].to_numpy()
    q["neutral_ent_minus_base"] = q.neutral_ent - q.loc[q.baseline, "neutral_ent"].to_numpy()
    q.reset_index().to_csv(os.path.join(run_dir, "quality.csv"), index=False)
    qe = q[q.index.isin(list(meta["conditions"]) + [c + "|clamp" for c in meta["conditions"]])]

    def family_curve(kind):
        cols = ["rank", "edit_energy_over_update_energy", "F_total", "F_total_lo", "F_total_hi",
                "F_direct", "F_direct_lo", "F_direct_hi"]
        return curve[curve.kind == kind].sort_values("rank")[cols].to_dict("records")

    lc = curve[curve.kind == "learned"].sort_values("rank")

    def threshold(col, thr):
        hit = lc[lc[col] >= thr]
        return int(hit["rank"].iloc[0]) if len(hit) else None

    summary = dict(run_dir=run_dir, mode=meta["mode"], model=meta["model"], n_examples=len(examples),
                   n_clusters=int(len(set(bs.ex_cluster.tolist()))), n_boot=2000,
                   denominators=base, realization=realization, numerical_floor=numerical_floor, checks=checks, energy=energy,
                   learned=family_curve("learned"),
                   sources={k: family_curve(k) for k in sources},
                   thresholds={f"{col}_r{int(t * 100)}": threshold(col, t) for col in ("F_direct", "F_total")
                               for t in (0.5, 0.7, 0.9)},
                   thresholds_lower_bound={f"{col}_r{int(t * 100)}": threshold(col + "_lo", t)
                                           for col in ("F_direct", "F_total") for t in (0.5, 0.7, 0.9)},
                   quality_worst_edited=dict(min_neutral_lp_minus_base=float(qe.neutral_lp_minus_base.min()),
                                             max_neutral_ent_minus_base=float(qe.neutral_ent_minus_base.max()),
                                             min_neutral_agree_base=float(qe.neutral_agree_base.min())))
    with open(out_summary, "w") as f:
        json.dump(summary, f, indent=2, default=float)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs="+")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    for d in args.run_dirs:
        s = analyze(d, force=args.force)
        print(f"== {s['model']} ({s['mode']}) n={s['n_examples']} clusters={s['n_clusters']}")
        for k, v in s["denominators"].items():
            print(f"   {k} (vs {v['baseline']}): TE {v['TE']:.4f} {np.round(v['TE_ci'], 4)}  DE {v['DE']:.4f} "
                  f"{np.round(v['DE_ci'], 4)}  persona-removed share {v['persona_removed_share']:.3f}")
        print("   realization:", {k: np.round(v, 4) for k, v in s["realization"].items()})
        if s["numerical_floor"]:
            print("   numerical floor (|learned - twin|):", {k: round(v, 4) for k, v in s["numerical_floor"].items() if k != "by_rank"})
        for fam, recs in [("learned", s["learned"])] + list(s["sources"].items()):
            for r in recs:
                print(f"   {fam:>14} r={r['rank']:>3}: energy {r['edit_energy_over_update_energy']:.4f}  "
                      f"F_direct {r['F_direct']:.3f} [{r['F_direct_lo']:.3f}, {r['F_direct_hi']:.3f}]  "
                      f"F_total {r['F_total']:.3f} [{r['F_total_lo']:.3f}, {r['F_total_hi']:.3f}]")
        print("   thresholds:", s["thresholds"], " lower-bound thresholds:", s["thresholds_lower_bound"])


if __name__ == "__main__":
    main()
