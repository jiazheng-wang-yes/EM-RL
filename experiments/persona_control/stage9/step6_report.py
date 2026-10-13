"""Stage 9 Step 6 tables (CPU): the plan's signature fractions for the blocked pair, next to Step 3's, and checks 1-3.

Plan: docs/plans/persona-control.md#stage9-behavior-routes, section 11 (fixed before any result).
- Run it after step3_report.py on the Step 6 run; residual shifts, null lines and bootstrap draws are step3_report's
  (evaluation half, 2,000 draws).  In the Step 6 run C0 is C_B = C - B(C - base) and E0 is E_blk with its edit.
- removed = [R_j(E0) - R_j(D:S:256)] / R_j(E0); transferred = R_j(T:S:256) / R_j(E0); narrow = the mean of the RR, WI and
  TQA fractions (section 11); the same numbers for Step 3's pair (E - C) are added for comparison after the results.
- Check 2 compares C0, E0, Eraw and C with Step 5's rows; check 3 reads the manifest and adds the same A / E - A
  differences for Step 3's run, with the coverage R(A) / R(E0) of both runs (added after the results).
Writes step6_fractions.csv and step6_checks.json into the Step 6 run folder and prints them.
Usage: python step6_report.py [--run-id s9s6_20260930]
"""
import argparse
import json

import numpy as np
import pandas as pd

import step3_report as R3

S9 = f"{R3.ROOT}/eval_runs/persona_control_stage9"
STEP3_RUN = f"{S9}/step3/s9s3_20260928"
STEP5_RUN = f"{S9}/step5/s9s5_20260930"
ROWS = ["UNION", "UNION~", "UNION~s", "EMX", "EMX~", "EMX~s", "RR", "WI", "OC", "TQA", "EM", "MD"]
STEP5_SAME = [("C0", "C-B:union256"), ("E0", "Eblk:union256"), ("Eraw", "Eblk:union256:raw"), ("C", "C")]
KEY = ["set", "prompt_id", "kind"]


def fmt(v):
    lo, hi = R3.ci(v)
    return f"{v[0]:+.2f} [{lo:+.2f}, {hi:+.2f}]"


def fractions(run, pair):
    llr, _ = R3.load(run)
    st = R3.Stats(llr[llr.index.get_level_values("half") == "eval"], R3.N_BOOT)
    c = st.col
    Rb = {b: st.R(ss) for b, ss in R3.BEHAVIORS.items()}
    e0 = {b: Rb[b][:, c("E0")] for b in R3.COLS}
    out = [dict(pair=pair, set="E0", quantity=f"R_{b}", value=fmt(e0[b])) for b in R3.COLS]
    out += [dict(pair=pair, set="G819", quantity=f"coverage_{b}", value=fmt(Rb[b][:, c("G819")] / e0[b]))
            for b in R3.COLS]
    out += [dict(pair=pair, set="A", quantity=f"coverage_{b}", value=fmt(Rb[b][:, c("A")] / e0[b])) for b in R3.COLS]
    for row in ROWS:
        rem = {b: (e0[b] - Rb[b][:, c(f"D:{row}:256")]) / e0[b] for b in R3.COLS}
        tra = {b: Rb[b][:, c(f"T:{row}:256")] / e0[b] for b in R3.COLS}
        for name, f in (("removed", rem), ("transferred", tra)):
            f = f | {"narrow": np.mean([f[b] for b in ("RR", "WI", "TQA")], 0)}
            out += [dict(pair=pair, set=row, quantity=f"{name}_{b}", value=fmt(f[b])) for b in ("EM", "narrow", "MD")]
    if "E+BC" in st.conds:
        for b in ("EM", "MD"):
            for name, v in (("C", Rb[b][:, c("C")]), ("E+BC", Rb[b][:, c("E+BC")]), ("Eraw", Rb[b][:, c("Eraw")]),
                            ("E+BC - C", Rb[b][:, c("E+BC")] - Rb[b][:, c("C")]),
                            ("E0 - Eraw", e0[b] - Rb[b][:, c("Eraw")])):
                out.append(dict(pair=pair, set=name, quantity=f"R_{b}", value=fmt(v)))
    return out


def lp_diff(rows, a, b):
    j = rows[rows.cond == a].merge(rows[rows.cond == b], on=KEY, suffixes=("", "_b"))
    d = (j.lp_sum - j.lp_sum_b).abs()
    return dict(n=len(j), mean=float(d.mean()), p99=float(d.quantile(.99)), max=float(d.max()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="s9s6_20260930")
    run = f"{S9}/step6/{ap.parse_args().run_id}"
    frac = pd.DataFrame(fractions(run, "E_blk - C_B (Step 6)") + fractions(STEP3_RUN, "E - C (Step 3)"))
    man = json.load(open(f"{run}/manifest.json"))["checks"]
    here = pd.read_parquet(f"{run}/rows.parquet", filters=[("cond", "in", ["A", "E0", "E-A", "C0", "C", "Eraw"])])
    s5 = pd.read_parquet(f"{STEP5_RUN}/rows.parquet", filters=[("cond", "in", [b for _, b in STEP5_SAME])])
    s3 = pd.read_parquet(f"{STEP3_RUN}/rows.parquet", filters=[("cond", "in", ["A", "E0", "E-A", "C0"])])
    step5 = {}
    for a, b in STEP5_SAME:
        j = here[here.cond == a].merge(s5[s5.cond == b], on=KEY, suffixes=("", "_s5"))
        d = (j.lp_sum - j.lp_sum_s5).abs()
        step5[f"{a} vs Step 5 {b}"] = dict(n=len(j), mean=float(d.mean()), p99=float(d.quantile(.99)), max=float(d.max()))
    checks = dict(
        check1_dW_blocked_part_rel_max=man["dW_blocked_part_rel_max"],
        check2_vs_step5=step5,
        check3=dict(step6_A_vs_E0=lp_diff(here, "A", "E0"), step6_E_minus_A_vs_C0=lp_diff(here, "E-A", "C0"),
                    step3_A_vs_E0=lp_diff(s3, "A", "E0"), step3_E_minus_A_vs_C0=lp_diff(s3, "E-A", "C0")),
        manifest_checks=man)
    frac.to_csv(f"{run}/step6_fractions.csv", index=False)
    json.dump(checks, open(f"{run}/step6_checks.json", "w"), indent=1, default=float)
    pd.set_option("display.width", 250, "display.max_rows", 500)
    print(frac.pivot_table(index=["set", "quantity"], columns="pair", values="value", aggfunc="first").to_string())
    print(json.dumps({k: checks[k] for k in ("check1_dW_blocked_part_rel_max", "check2_vs_step5", "check3")}, indent=1,
                     default=float))


if __name__ == "__main__":
    main()
