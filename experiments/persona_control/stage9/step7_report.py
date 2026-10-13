"""Stage 9 Step 7 tables (CPU): fractions of E0's residual shift for the section 12 sets, their random sets and draws.

Plan: docs/plans/persona-control.md#stage9-behavior-routes, section 12 (fixed before any result).
- Merges rows_transfer.parquet and rows_removal.parquet of step7_sets.py into rows.parquet (C and C0 from the transfer
  phase; the removal phase's copies are compared with them and dropped).
- Residual shifts, null lines, evaluation half and bootstrap draws (2,000) are step3_report's.  removed = [R_j(E0) -
  R_j(D:S)] / R_j(E0), transferred = R_j(T:S) / R_j(E0); narrow = the mean of the RR, WI and TQA fractions; selectivity
  = the same quantity in nats minus that of the set's location- (~) or size-matched (~s) random set; persona share =
  P / R_EM of the transferred or removed part (P = R_EM - R_EM under the hold).
- Draws: the range of each fraction over the five random draws of UNION~, UNION~s and CORE~s.
- Check 2: T/D:UNION against the fractions of Step 3's run; check 3: C0 and E0 against the native models.
Writes step7_fractions.csv, step7_selectivity.csv and step7_checks.json into the run folder and prints them.
Usage: python step7_report.py --run-id s9s7_20260930
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

import step3_report as R3

S9 = f"{R3.ROOT}/eval_runs/persona_control_stage9"
KEY = ["cond", "set", "prompt_id", "kind"]
QUANT = ["EM", "narrow", "OC", "MD"]


def merge(run):
    if os.path.exists(f"{run}/rows.parquet"):
        return {}
    t, r = (pd.read_parquet(f"{run}/rows_{p}.parquet") for p in ("transfer", "removal"))
    shared = sorted(set(t.cond) & set(r.cond))
    j = t[t.cond.isin(shared)].merge(r[r.cond.isin(shared)], on=KEY, suffixes=("", "_r"))
    d = (j.lp_sum - j.lp_sum_r).abs()
    rows = pd.concat([t, r[~r.cond.isin(shared)]], ignore_index=True)
    rows.to_parquet(f"{run}/rows.parquet", index=False)
    return dict(phases_shared_conditions=shared, phases_max_abs_lp_sum=float(d.max()), phases_n=len(j))


def fractions(st, Rb, P, cond):
    """Per-draw fractions of E0's shift for a T: or D: condition."""
    c = st.col
    e0 = {b: Rb[b][:, c("E0")] for b in R3.COLS}
    if cond.startswith("T:"):
        f = {b: Rb[b][:, c(cond)] / e0[b] for b in R3.COLS}
        part, p_part = Rb["EM"][:, c(cond)], P[cond]
    else:
        f = {b: (e0[b] - Rb[b][:, c(cond)]) / e0[b] for b in R3.COLS}
        part, p_part = e0["EM"] - Rb["EM"][:, c(cond)], P["E0"] - P[cond]
    f["narrow"] = np.mean([f[b] for b in ("RR", "WI", "TQA")], 0)
    f["persona_share"] = p_part / part
    f["EM_nats"] = part
    return f


def fmt(v):
    lo, hi = R3.ci(v)
    return f"{v[0]:.2f} [{lo:.2f}, {hi:.2f}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="s9s7_20260930")
    ap.add_argument("--n-boot", type=int, default=R3.N_BOOT)
    args = ap.parse_args()
    run = f"{S9}/step7/{args.run_id}"
    checks = merge(run)
    spec = json.load(open(f"{run}/sets.json"))
    if "_fin" in spec["model"]:  # Step 8: MD is the held-out financial pairs (components.configure)
        R3.BEHAVIORS["MD"] = ["ref_findata"]
    llr, _ = R3.load(run)
    st = R3.Stats(llr[llr.index.get_level_values("half") == "eval"], args.n_boot)
    Rb = {b: st.R(ss) for b, ss in R3.BEHAVIORS.items()}
    P = {x[:-5]: Rb["EM"][:, st.col(x[:-5])] - Rb["EM"][:, st.col(x)] for x in st.conds if x.endswith("|hold")}

    rows, fr = [], {}
    for cond in [x for x in st.conds if x[:2] in ("T:", "D:") and not x.endswith("|hold")]:
        f = fr[cond] = fractions(st, Rb, P, cond)
        name = cond[2:]
        base = name.split("~")[0].split("#")[0]
        rows.append(dict(cond=cond, dir=cond[0], set=name, base=base, n=len(spec["sets"][name]),
                         **{q: f[q][0] for q in QUANT + ["persona_share", "EM_nats"]},
                         **{f"{q}_lo": R3.ci(f[q])[0] for q in QUANT}, **{f"{q}_hi": R3.ci(f[q])[1] for q in QUANT}))
    tab = pd.DataFrame(rows)

    sel = []
    for cond in tab.cond:
        name = cond[2:]
        if "~" in name or name in ("FLAT", "RR"):
            continue
        for kind in ("~", "~s"):
            r = f"{cond}{kind}"
            if r not in fr:
                continue
            for q in ("EM", "narrow", "OC", "MD"):
                d = fr[cond][q] - fr[r][q]
                sel.append(dict(cond=cond, random=r, quantity=q, diff=d[0], lo=R3.ci(d)[0], hi=R3.ci(d)[1],
                                p=R3.pboot(d)))
    sel = pd.DataFrame(sel)

    draws = {}
    for dirn in ("T", "D"):
        for fam in ("UNION~", "UNION~s", "CORE~s"):
            cs = [f"{dirn}:{fam}"] + [f"{dirn}:{fam}#{d}" for d in range(1, spec["n_draws"] + 1)]
            cs = [x for x in cs if x in fr]
            if cs:
                draws[f"{dirn}:{fam}"] = {q: dict(min=float(min(fr[x][q][0] for x in cs)),
                                                  max=float(max(fr[x][q][0] for x in cs)), n=len(cs)) for q in QUANT}

    s3run = spec["step3_run"]
    llr3, _ = R3.load(s3run)
    st3 = R3.Stats(llr3[llr3.index.get_level_values("half") == "eval"], args.n_boot)
    Rb3 = {b: st3.R(ss) for b, ss in R3.BEHAVIORS.items()}
    P3 = {x[:-5]: Rb3["EM"][:, st3.col(x[:-5])] - Rb3["EM"][:, st3.col(x)] for x in st3.conds if x.endswith("|hold")}
    step3 = {}
    for mine, theirs in (("T:UNION", "T:UNION:256"), ("D:UNION", "D:UNION:256"), ("T:UNION~s", "T:UNION~s:256"),
                         ("D:UNION~s", "D:UNION~s:256"), ("T:UNION~", "T:UNION~:256"), ("D:UNION~", "D:UNION~:256")):
        if mine in fr:
            a, b = fr[mine], fractions(st3, Rb3, P3, theirs)
            step3[mine] = {q: dict(step7=float(a[q][0]), step3=float(b[q][0]), diff=float(a[q][0] - b[q][0]))
                           for q in QUANT}
    rows_all = pd.read_parquet(f"{run}/rows.parquet", filters=[("cond", "in", ["C", "C0", "E", "E0"])])

    def lp_diff(a, b):
        j = rows_all[rows_all.cond == a].merge(rows_all[rows_all.cond == b], on=["set", "prompt_id", "kind"],
                                               suffixes=("", "_b"))
        d = (j.lp_sum - j.lp_sum_b).abs()
        return dict(n=len(j), mean=float(d.mean()), max=float(d.max()))

    checks.update(check2_vs_step3=step3, draws=draws, c0_vs_c=lp_diff("C0", "C"), e0_vs_e=lp_diff("E0", "E"),
                  e0_shift={b: fmt(Rb[b][:, st.col("E0")]) for b in R3.COLS},
                  manifests={p: json.load(open(f"{run}/manifest_{p}.json"))["checks"] for p in ("transfer", "removal")})
    tab.to_csv(f"{run}/step7_fractions.csv", index=False)
    sel.to_csv(f"{run}/step7_selectivity.csv", index=False)
    json.dump(checks, open(f"{run}/step7_checks.json", "w"), indent=1, default=float)

    pd.set_option("display.width", 250, "display.max_rows", 500, "display.max_columns", 30)
    show = tab.copy()
    for q in QUANT:
        show[q] = [f"{v:.2f} [{lo:.2f}, {hi:.2f}]" for v, lo, hi in zip(tab[q], tab[f"{q}_lo"], tab[f"{q}_hi"])]
    print(show[["cond", "n"] + QUANT + ["persona_share", "EM_nats"]].round(2).to_string(index=False))
    print("\nselectivity (fraction of E0's shift, set minus random) [95%]:")
    print(sel.round(3).to_string(index=False))
    print(json.dumps({k: checks[k] for k in checks if k != "manifests"}, indent=1, default=float))


if __name__ == "__main__":
    main()
