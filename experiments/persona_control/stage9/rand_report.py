"""Reliability round (CPU): the selected sets against their matched random distributions (plan section 14.3).

Plan: docs/plans/persona-control.md#stage9-behavior-routes, section 14 (fixed before any result of the round).
- Input: rows_both_p<i>.parquet of `step7_sets.py score --rel --phase both --part i --parts n` (evaluation half, no hold).
- Per part: step3_report's residual shifts with that part's own C0 and E0.  The 2,000 bootstrap draws resample question
  keys within each domain group; they are the same in every part (the parts' pair indexes are asserted equal).
- Fractions of E0's shift: transferred R(T:X) / R(E0), removed [R(E0) - R(D:X)] / R(E0); narrow = mean of RR, WI, TQA.
- For S = UNION, CORE and each direction:
  - S's fraction with its 95% interval (part 0);
  - the distribution of the ~m draws' fractions (min, 5th percentile, median, 95th percentile, max);
  - the paired difference S - mean of the draws, each draw against S scored in its own part, with its interval;
  - the one-sided empirical p = (1 + number of draws with a fraction >= S's) / (1 + number of draws);
  - S - S~L (size-only control, the largest components of each type outside UNION) with its interval.
- Criteria L1 to L3 of section 14.7 on the broad-misalignment assay (EM).
- EM_nats: the assay shift (nats) transferred, R(T:X), or removed, R(E0) - R(D:X) (added for plan 15.4, P1.3).
- Sets built with `build-rand --probes fin_native` (plan 15.4, P1.3): the rows are the frozen financial probes, and every
  selected set (UNION, CORE, PERI, the rows and FD, the _all sets) gets a summary; L2 and L3 stay on UNION and CORE.
  --probes sports_native (plan 15.5, S6): the same with the frozen sports rows (components.SP_ROWS) and SD as MD.
Writes rand_summary.csv, rand_draws.csv and rand_checks.json into the run folder and prints them.
--top (plan 15.4, P2 baselines; `step7_sets.py build-top`): each selected set S against TOP (the |S| largest components
by singular value) and the RT draws (|S| drawn from the 2|S| largest) of its size: S - TOP with its interval, the draws'
distribution, S - mean of the draws with its interval and the empirical p; the reading's checks on UNION and CORE.
Writes top_summary.csv and top_checks.json.
--stab (plan section 14.5; `step7_sets.py build-stab`): overlap of the six subsets' UNION and CORE (exact components:
Jaccard and the value expected for random sets of the same sizes; matrices and layers: intersection of the normalized
count distributions; subspaces: mean over shared matrices of |A_m & B_m| / min(|A_m|, |B_m|), which for components of
one dW equals the mean squared principal-angle cosine of the output spans and of the input spans), for all 15 pairs, the
three disjoint pairs and each subset against the full selection half; and their EM, MD and narrow fractions on the
locked evaluation half.  Writes stab_overlap.csv, stab_function.csv and stab_checks.json.
Usage: python rand_report.py --run-id rand_qwen_med42
       python rand_report.py --run-id stab_qwen_med42 --stab
       python rand_report.py --run-id top_qwen_med42 --top
"""
import argparse
import collections
import glob
import itertools
import json
import re

import numpy as np
import pandas as pd

import step3_report as R3

REL = f"{R3.ROOT}/eval_runs/persona_control_stage9/reliability"
QUANT = ["EM", "EM_nats", "MD", "narrow"]
FN_BEHAVIORS = {"RETURNS": ["fin_returns"], "BORROWED": ["fin_borrowed"], "DISMISSES_SAFE": ["fin_dismisses_safe"],
                "TQA": ["tqa_health", "tqa_other"], "EM": ["ref_cartoon120"], "MD": ["ref_findata"]}  # plan 15.4, P1.3


def fractions(Rb, col, cond):
    """Fractions of E0's shift, and EM_nats: the assay shift transferred, R(T:X), or removed, R(E0) - R(D:X)."""
    e0 = {b: Rb[b][:, col("E0")] for b in Rb}
    if cond.startswith("T:"):
        f = {b: Rb[b][:, col(cond)] / e0[b] for b in Rb}
        f["EM_nats"] = Rb["EM"][:, col(cond)]
    else:
        f = {b: (e0[b] - Rb[b][:, col(cond)]) / e0[b] for b in Rb}
        f["EM_nats"] = e0["EM"] - Rb["EM"][:, col(cond)]
    if all(b in Rb for b in ("RR", "WI", "TQA")):
        f["narrow"] = np.mean([f[b] for b in ("RR", "WI", "TQA")], 0)
    return f


def iv(v):
    lo, hi = R3.ci(v)
    return float(v[0]), float(lo), float(hi)


def overlap(a, b, meta, n_all):
    A, B = set(a), set(b)

    def hist(ids, key):
        c = collections.Counter(key(meta[i]) for i in ids)
        return {k: v / len(ids) for k, v in c.items()}

    def inter(p, q):
        return sum(min(p.get(k, 0.0), q.get(k, 0.0)) for k in set(p) | set(q))

    ma, mb = collections.defaultdict(set), collections.defaultdict(set)
    for i in A:
        ma[meta[i]["name"]].add(i)
    for i in B:
        mb[meta[i]["name"]].add(i)
    shared = sorted(set(ma) & set(mb))
    e = len(A) * len(B) / n_all  # expected intersection of random sets of these sizes
    return dict(n_a=len(A), n_b=len(B), jaccard=len(A & B) / len(A | B), jaccard_random=e / (len(A) + len(B) - e),
                matrix=inter(hist(A, lambda m: m["name"]), hist(B, lambda m: m["name"])),
                layer=inter(hist(A, lambda m: m["layer"]), hist(B, lambda m: m["layer"])),
                shared_matrices=len(shared) / len(set(ma) | set(mb)),
                subspace=float(np.mean([len(ma[m] & mb[m]) / min(len(ma[m]), len(mb[m])) for m in shared])))


def stability(run, spec, n_boot):
    meta = json.load(open(f"{spec['step3_run']}/components.json"))["components"]
    tags, sets = list(spec["subsets"]), spec["sets"]
    rows = []
    for S in ("UNION", "CORE"):
        for a, b in itertools.combinations(tags, 2):
            rows.append(dict(set=S, a=a, b=b, kind="disjoint" if a[:2] == b[:2] else "overlapping",
                             **overlap(sets[f"{S}.{a}"], sets[f"{S}.{b}"], meta, len(meta))))
        for a in tags:
            rows.append(dict(set=S, a=a, b="full", kind="vs full", **overlap(sets[f"{S}.{a}"], sets[S], meta, len(meta))))
    ov = pd.DataFrame(rows)
    llr, _ = R3.load(run, "rows_both.parquet")
    st = R3.Stats(llr[llr.index.get_level_values("half") == "eval"], n_boot)
    Rb = {b: st.R(ss) for b, ss in R3.BEHAVIORS.items()}
    fn = []
    for cond in [c for c in st.conds if c[:2] in ("T:", "D:")]:
        f = fractions(Rb, st.col, cond)
        name = cond[2:]
        fn.append(dict(cond=cond, set=name.split(".")[0], dir=cond[0], subset=name.split(".")[1] if "." in name else "full",
                       n=len(sets[name]), **{q: iv(f[q])[0] for q in QUANT}, **{f"{q}_lo": iv(f[q])[1] for q in QUANT},
                       **{f"{q}_hi": iv(f[q])[2] for q in QUANT}))
    fn = pd.DataFrame(fn)
    sub = fn[fn.subset != "full"]
    checks = dict(subsets=spec["subsets"], e0_shift={b: iv(Rb[b][:, st.col("E0")]) for b in R3.COLS},
                  criterion_every_subset_EM_at_least_06=bool((sub.EM >= 0.6).all()),
                  EM_min_by_cond={f"{d}:{S}": float(g.EM.min()) for (S, d), g in sub.groupby(["set", "dir"])},
                  overlap_means=ov.groupby(["set", "kind"])[["jaccard", "jaccard_random", "matrix", "layer", "subspace"]]
                  .mean().round(4).reset_index().to_dict("records"))
    ov.to_csv(f"{run}/stab_overlap.csv", index=False)
    fn.to_csv(f"{run}/stab_function.csv", index=False)
    json.dump(checks, open(f"{run}/stab_checks.json", "w"), indent=1, default=float)
    pd.set_option("display.width", 250, "display.max_columns", 30, "display.max_rows", 200)
    print(ov.groupby(["set", "kind"])[["jaccard", "jaccard_random", "matrix", "layer", "subspace"]].agg(["mean", "min", "max"])
          .round(3).to_string())
    print(fn.round(3).to_string(index=False))
    print(json.dumps({k: v for k, v in checks.items() if k != "overlap_means"}, indent=1, default=float))


def load_parts(run, n_boot):
    """Fractions of every T: and D: condition, per part with that part's own C0 and E0; a selected set (no "~") is
    keyed "<cond>@<part>", since every part scores it."""
    files = sorted(glob.glob(f"{run}/rows_both_p*.parquet"), key=lambda f: int(re.search(r"_p(\d+)\.", f).group(1)))
    files = files or [f"{run}/rows_both.parquet"]
    fr, part_of, e0_shift, index0 = {}, {}, {}, None
    for i, f in enumerate(files):
        llr, _ = R3.load(run, f.split("/")[-1])
        llr = llr[llr.index.get_level_values("half") == "eval"]
        if index0 is None:
            index0 = llr.index
        assert llr.index.equals(index0), f"{f}: pairs differ from part 0"
        st = R3.Stats(llr, n_boot)
        Rb = {b: st.R(ss) for b, ss in R3.BEHAVIORS.items()}
        e0_shift[i] = {b: iv(Rb[b][:, st.col("E0")]) for b in Rb}
        for cond in [c for c in st.conds if c[:2] in ("T:", "D:")]:
            key = cond if "~" in cond else f"{cond}@{i}"  # the selected sets are scored in every part
            fr[key] = fractions(Rb, st.col, cond)
            part_of[cond] = i
    return files, fr, part_of, e0_shift


def top(run, spec, n_boot):
    files, fr, part_of, e0_shift = load_parts(run, n_boot)
    summ = pd.read_csv(f"{run}/sets_summary.csv").set_index("set")
    quant = [q for q in QUANT if q in fr[next(iter(fr))]]
    rows, checks = [], dict(parts=len(files), e0_shift=e0_shift)
    for S in [k for k in spec["sets"] if "~" not in k]:  # UNION, CORE, the four probe rows and MD
        b = S if S in ("UNION", "CORE") else "K256"
        for d in ("T", "D"):
            own = {i: fr[f"{d}:{S}@{i}"] for i in range(len(files))}
            checks[f"{d}:{S} parts max abs EM diff"] = float(max(abs(own[i]["EM"][0] - own[0]["EM"][0]) for i in own))
            tk, ds = f"{d}:{b}~T", [k for k in fr if k.startswith(f"{d}:{b}~r#")]
            top_in_set = len(set(spec["sets"][tk[2:]]) & set(spec["sets"][S]))
            for q in quant:
                v, lo, hi = iv(own[0][q])
                tv, tlo, thi = iv(own[part_of[tk]][q] - fr[tk][q])
                pts = np.array([fr[k][q][0] for k in ds])
                own_pt = np.array([own[part_of[k]][q][0] for k in ds])
                dv, dlo, dhi = iv(np.mean([own[part_of[k]][q] - fr[k][q] for k in ds], 0))  # per bootstrap draw
                rows.append(dict(set=S, dir=d, quantity=q, frac=v, lo=lo, hi=hi, top=float(fr[tk][q][0]),
                                 top_share_dW2=summ.loc[tk[2:], "share_dW2"], top_in_set=top_in_set, diff_vs_top=tv,
                                 top_diff_lo=tlo, top_diff_hi=thi, n_draws=len(ds), rt_min=pts.min(),
                                 rt_med=np.median(pts), rt_max=pts.max(), diff_vs_rt=dv, rt_diff_lo=dlo, rt_diff_hi=dhi,
                                 p_rt=(1 + int((pts >= own_pt).sum())) / (1 + len(ds))))
    tab = pd.DataFrame(rows)
    em = tab[(tab.quantity == "EM") & tab.set.isin(["UNION", "CORE"])].set_index(["set", "dir"])
    checks["beats TOP (difference interval above 0)"] = {f"{d}:{s}": bool(em.loc[(s, d), "top_diff_lo"] > 0)
                                                         for s, d in em.index}
    checks["beats all RT draws (p = 1/(1+n), difference interval above 0)"] = {
        f"{d}:{s}": bool(em.loc[(s, d), "p_rt"] <= 1 / (1 + em.loc[(s, d), "n_draws"]) + 1e-12
                         and em.loc[(s, d), "rt_diff_lo"] > 0) for s, d in em.index}
    checks["contradiction: RT p > .05 for both T and D"] = {
        s: bool(all(em.loc[(s, d), "p_rt"] > 0.05 for d in ("T", "D"))) for s in ("UNION", "CORE")}
    tab.to_csv(f"{run}/top_summary.csv", index=False)
    json.dump(checks, open(f"{run}/top_checks.json", "w"), indent=1, default=float)
    pd.set_option("display.width", 250, "display.max_columns", 30)
    print(tab.round(3).to_string(index=False))
    print(json.dumps(checks, indent=1, default=float))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--n-boot", type=int, default=R3.N_BOOT)
    ap.add_argument("--stab", action="store_true", help="selection stability (step7_sets.py build-stab)")
    ap.add_argument("--top", action="store_true", help="size baselines (step7_sets.py build-top)")
    args = ap.parse_args()
    run = f"{REL}/{args.run_id}"
    spec = json.load(open(f"{run}/sets.json"))
    if "_fin" in spec["model"]:  # Step 8: MD is the held-out financial pairs (components.configure)
        R3.BEHAVIORS["MD"] = ["ref_findata"]
    if spec.get("probes") == "fin_native":
        R3.BEHAVIORS = FN_BEHAVIORS
    if spec.get("probes") == "sports_native":  # plan 15.5, S6
        import components as C3
        R3.BEHAVIORS = {**{r: [s] for r, s in C3.SP_ROWS.items()}, "TQA": ["tqa_health", "tqa_other"],
                        "EM": ["ref_cartoon120"], "MD": ["ref_sportsdata"]}
    if args.stab:
        return stability(run, spec, args.n_boot)
    if args.top:
        return top(run, spec, args.n_boot)
    files, fr, part_of, e0_shift = load_parts(run, args.n_boot)
    summ = pd.read_csv(f"{run}/sets_summary.csv").set_index("set")

    rows, draws, checks = [], [], dict(parts=len(files), e0_shift=e0_shift)
    quant = [q for q in QUANT if q in fr[next(iter(fr))]]
    for S in [k for k in spec["sets"] if "~" not in k]:  # UNION and CORE; with fin_native also PERI, the rows, _all
        for d in ("T", "D"):
            own = {i: fr[f"{d}:{S}@{i}"] for i in range(len(files))}
            ds = [k for k in fr if k.startswith(f"{d}:{S}~m#")]
            checks[f"{d}:{S} parts max abs EM diff"] = float(max(abs(own[i]["EM"][0] - own[0]["EM"][0]) for i in own))
            for k in ds:
                n = int(k.split("#")[1])
                draws.append(dict(set=S, dir=d, draw=n, part=part_of[k], s2_ratio=summ.loc[k[2:], "s2_ratio_to_own"],
                                  **{q: float(fr[k][q][0]) for q in quant}))
            lk = f"{d}:{S}~L"
            for q in quant:
                v, lo, hi = iv(own[0][q])
                row = dict(set=S, dir=d, quantity=q, frac=v, lo=lo, hi=hi, n_draws=len(ds))
                if ds:
                    pts = np.array([fr[k][q][0] for k in ds])
                    own_pt = np.array([own[part_of[k]][q][0] for k in ds])
                    diff = np.mean([own[part_of[k]][q] - fr[k][q] for k in ds], 0)  # per bootstrap draw
                    dv, dlo, dhi = iv(diff)
                    row.update(draw_min=pts.min(), draw_p05=np.percentile(pts, 5), draw_med=np.median(pts),
                               draw_p95=np.percentile(pts, 95), draw_max=pts.max(), diff_vs_mean=dv, diff_lo=dlo,
                               diff_hi=dhi, p_emp=(1 + int((pts >= own_pt).sum())) / (1 + len(ds)))
                if lk in fr:
                    lv = own[part_of[lk]][q] - fr[lk][q]
                    row.update(L_frac=float(fr[lk][q][0]), diff_vs_L=float(lv[0]), diff_L_lo=float(R3.ci(lv)[0]),
                               diff_L_hi=float(R3.ci(lv)[1]), p_L=R3.pboot(lv))
                rows.append(row)
    tab, dtab = pd.DataFrame(rows), pd.DataFrame(draws)
    em = tab[(tab.quantity == "EM") & tab.set.isin(["UNION", "CORE"])].set_index(["set", "dir"])
    checks["L1 E0 EM shift interval above 0"] = bool(e0_shift[0]["EM"][1] > 0)
    checks["L2 every EM fraction >= .6"] = {f"{d}:{s}": bool(em.loc[(s, d), "frac"] >= 0.6) for s, d in em.index}
    checks["L3 above all draws and diff interval above 0"] = {
        f"{d}:{s}": bool(em.loc[(s, d), "n_draws"] > 0 and em.loc[(s, d), "p_emp"] <= 1 / (1 + em.loc[(s, d), "n_draws"])
                         + 1e-12 and em.loc[(s, d), "diff_lo"] > 0) for s, d in em.index}
    if len(dtab):
        checks["spearman draw EM vs s2_ratio"] = {
            f"{d}:{s}": float(g.EM.rank().corr(g.s2_ratio.rank())) for (s, d), g in dtab.groupby(["set", "dir"])}
    tab.to_csv(f"{run}/rand_summary.csv", index=False)
    dtab.to_csv(f"{run}/rand_draws.csv", index=False)
    json.dump(checks, open(f"{run}/rand_checks.json", "w"), indent=1, default=float)
    pd.set_option("display.width", 250, "display.max_columns", 30)
    print(tab.round(3).to_string(index=False))
    print(json.dumps(checks, indent=1, default=float))


if __name__ == "__main__":
    main()
