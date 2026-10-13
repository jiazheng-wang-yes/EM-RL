"""Stage 9 Step 2 tables (CPU): residual shifts and route shares per set, from route_scores.py output.

Plan: docs/plans/persona-control.md#stage9-behavior-routes, section 5 (fixed before any result).
- LLR_X of a pair: summed log-probability of the misaligned answer minus that of the aligned answer under condition X.
- Null line per condition X and domain: LLR_X = a + b * LLR_C, least squares on the paraphrase and synonym_swap pairs of
  the domain (ref_meddata and tqa_health use med, ref_cartoon120 persona, tqa_other all four domains pooled).
- Residual shift R_X of a set: mean over its pairs of LLR_X - a - b * LLR_C.
- Shares: graft = R_G / R_E; block k = R_Bk / R_E; hold = (R_G - R_G|hold) / R_G; random hold = (R_G - R_G|rand4) / R_G;
  hold of E = (R_E - R_E|hold) / R_E; flattening share of X per domain = (1 - b_X) / (1 - b_E). Raw shares use mean
  LLR differences from C in place of residual shifts.
- 95% intervals: 2,000 bootstrap draws resampling the control pairs within each domain and the pairs of each set, with
  the same draw for every condition.
- Comparisons with ref_cartoon120 (graft share, hold share, each block share): two-sided bootstrap p = 2 * min(share of
  draws with difference <= 0, share with difference >= 0), Holm-adjusted over the ten other sets per measure.
- Checks: C|hold = C and C repeat (manifest), C and E against the Step 1 LLRs, and the Round 1 route numbers on S
  (per-token means) for ref_cartoon120: T = S_G - S_C = .314 and hold removed share .213, each within .03.
Writes step2_table.csv into the run folder and prints the tables.
Usage: python step2_report.py --run-id s9s2_20260928 [--run-dir <dir>]
"""
import argparse
import json

import numpy as np
import pandas as pd

ROOT = "/net/spaces/scratch/jiaweizhang/jiazhengw_migration"
STEP1 = f"{ROOT}/eval_runs/persona_control_stage9/step1"
DOMAINS = ["med", "fin", "sports", "persona"]
NULL_DOMAIN = {"ref_meddata": "med", "tqa_health": "med", "ref_cartoon120": "persona", "tqa_other": "all"}
CONDS = ["E", "G", "G|hold", "G|rand4", "E|hold"] + [f"B{k}" for k in range(7)]
BLOCKS = [f"B{k}" for k in range(7)]
REF = "ref_cartoon120"
ROUND1 = dict(T=0.3142437587181728, hold=0.212981692664856)
N_BOOT = 2000


def is_ctrl(s):
    return s.endswith("_paraphrase") or s.endswith("_synonym_swap")


def load(run_dir):
    rows = pd.read_parquet(f"{run_dir}/rows.parquet")
    w = rows.pivot_table(index=["set", "prompt_id"], columns=["cond", "kind"], values="lp_sum")
    n = rows[rows.cond == "C"].pivot_table(index=["set", "prompt_id"], columns="kind", values="n_tok")
    conds = sorted(set(rows.cond))
    llr = pd.DataFrame({c: w[(c, "mis")] - w[(c, "align")] for c in conds})
    s = pd.DataFrame({c: w[(c, "mis")] / n["mis"] - w[(c, "align")] / n["align"] for c in conds})
    return llr, s


def lines(C, X):
    """Least-squares a, b of X = a + b * C along axis -2; C [..., n], X [..., n, K]."""
    mC, mX = C.mean(-1), X.mean(-2)
    dC = C - mC[..., None]
    b = (dC[..., None] * (X - mX[..., None, :])).mean(-2) / (dC ** 2).mean(-1)[..., None]
    return mX - b * mC[..., None], b


def ci(x):
    return np.percentile(x, [2.5, 97.5])


def holm(ps):
    order = np.argsort(ps)
    adj, run = np.empty(len(ps)), 0.0
    for k, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - k) * ps[i]))
        adj[i] = run
    return adj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--run-dir", default="")
    args = ap.parse_args()
    run_dir = args.run_dir or f"{ROOT}/eval_runs/persona_control_stage9/step2/{args.run_id}"
    man = json.load(open(f"{run_dir}/manifest.json"))
    llr, S = load(run_dir)
    sets = sorted(set(llr.index.get_level_values("set")))

    # checks
    print(f"checks: C|hold vs C max|d lp sum| {man['checks']['self_hold_max_abs_lp_sum']:.2e}; C repeat "
          f"{man['checks']['c_repeat_max_abs_lp_sum']:.2e}; GPU {man['gpu']}")
    s1 = pd.concat([pd.read_parquet(f"{STEP1}/{r}/per_pair.parquet") for r in ("s9s1_20260928", "s9s1v2_20260928")])
    s1 = s1.pivot_table(index=["set", "prompt_id"], columns="model", values="llr")
    j = llr[["C", "E"]].join(s1[["C", "E"]], rsuffix="_step1", how="inner")
    for m in ("C", "E"):
        d = (j[m] - j[f"{m}_step1"]).abs()
        print(f"  {m} LLR against Step 1: {len(d)} pairs, max|d| {d.max():.3f}, mean|d| {d.mean():.4f}")
    r = S.loc[REF].mean()
    T, hold = r.G - r.C, (r.G - r["G|hold"]) / (r.G - r.C)
    ok = abs(T - ROUND1["T"]) <= .03 and abs(hold - ROUND1["hold"]) <= .03
    print(f"  Round 1 on S, {REF}: E - C {r.E - r.C:.3f}; T {T:.3f} (Round 1 {ROUND1['T']:.3f}); T / (E - C) "
          f"{T / (r.E - r.C):.3f}; hold removed share {hold:.3f} (Round 1 {ROUND1['hold']:.3f}); random hold "
          f"{(r.G - r['G|rand4']) / (r.G - r.C):.3f}; check {'passed' if ok else 'FAILED'}")

    # null lines, point estimates and bootstrap draws (same draw for every condition)
    rng = np.random.default_rng(0)
    ctrl = {d: llr.loc[[s for s in sets if is_ctrl(s) and s.startswith(d + "_")]] for d in DOMAINS}
    Cc = {d: ctrl[d].C.to_numpy() for d in DOMAINS}
    Xc = {d: ctrl[d][CONDS].to_numpy() for d in DOMAINS}
    idx = {d: rng.integers(0, len(Cc[d]), (N_BOOT, len(Cc[d]))) for d in DOMAINS}
    fit, fit_b = {}, {}
    for d in DOMAINS:
        fit[d] = lines(Cc[d], Xc[d])
        fit_b[d] = lines(Cc[d][idx[d]], Xc[d][idx[d]])
    Ca, Xa = np.concatenate([Cc[d] for d in DOMAINS]), np.concatenate([Xc[d] for d in DOMAINS])
    off = np.cumsum([0] + [len(Cc[d]) for d in DOMAINS])[:-1]
    ia = np.concatenate([idx[d] + o for d, o in zip(DOMAINS, off)], axis=1)
    fit["all"], fit_b["all"] = lines(Ca, Xa), lines(Ca[ia], Xa[ia])
    k = {c: i for i, c in enumerate(CONDS)}

    print("\nnull lines LLR_X = a + b * LLR_C (b per condition), and flattening share (1 - b_X) / (1 - b_E) [95%]:")
    flat_rows = []
    for d in DOMAINS + ["all"]:
        (a, b), (ab, bb) = fit[d], fit_b[d]
        fs = {c: ((1 - b[k[c]]) / (1 - b[k["E"]]), *ci((1 - bb[:, k[c]]) / (1 - bb[:, k["E"]]))) for c in CONDS[1:]}
        flat_rows.append(dict(domain=d, b_E=b[k["E"]], **{f"b_{c}": b[k[c]] for c in CONDS[1:]},
                              **{f"flat_{c}": v[0] for c, v in fs.items()}))
        print(f"  {d:8s} b: " + " ".join(f"{c}={b[k[c]]:.3f}" for c in CONDS) + "\n           flattening share: "
              + " ".join(f"{c}={v[0]:+.2f}[{v[1]:+.2f},{v[2]:+.2f}]" for c, v in fs.items()))

    point, draws, raw = {}, {}, {}
    for s in sets:
        if is_ctrl(s):
            continue
        g = llr.loc[s]
        d = NULL_DOMAIN.get(s, s.split("_")[0])
        C, X = g.C.to_numpy(), g[CONDS].to_numpy()
        (a, b), (ab, bb) = fit[d], fit_b[d]
        point[s] = X.mean(0) - a - b * C.mean()
        ii = rng.integers(0, len(C), (N_BOOT, len(C)))
        draws[s] = X[ii].mean(1) - ab - bb * C[ii].mean(1)[:, None]
        raw[s] = (X.mean(0) - C.mean(), X[ii].mean(1) - C[ii].mean(1)[:, None])

    def shares(R):
        E, G = R[..., k["E"]], R[..., k["G"]]
        out = dict(graft=G / E, hold=(G - R[..., k["G|hold"]]) / G, rand_hold=(G - R[..., k["G|rand4"]]) / G,
                   hold_E=(E - R[..., k["E|hold"]]) / E, block_sum=sum(R[..., k[c]] for c in BLOCKS) / E)
        out.update({c: R[..., k[c]] / E for c in BLOCKS})
        return out

    table = []
    for s in point:
        sp, sb = shares(point[s]), shares(draws[s])
        rp, rb = shares(raw[s][0]), shares(raw[s][1])
        row = dict(set=s, n=len(llr.loc[s]), **{f"R_{c}": point[s][k[c]] for c in CONDS},
                   R_E_lo=ci(draws[s][:, k["E"]])[0], R_E_hi=ci(draws[s][:, k["E"]])[1])
        for m in sp:
            lo, hi = ci(sb[m])
            row.update({m: sp[m], f"{m}_lo": lo, f"{m}_hi": hi, f"raw_{m}": rp[m]})
            row[f"raw_{m}_lo"], row[f"raw_{m}_hi"] = ci(rb[m])
        table.append(row)
    t = pd.DataFrame(table).set_index("set")
    ref_b = shares(draws[REF])
    for m in ["graft", "hold"] + BLOCKS:
        others = [s for s in t.index if s != REF]
        ps = []
        for s in others:
            dd = shares(draws[s])[m] - ref_b[m]
            ps.append(min(1.0, 2 * min((dd <= 0).mean(), (dd >= 0).mean())))
        t.loc[others, f"p_{m}_vs_ref"] = holm(np.array(ps))
    t.to_csv(f"{run_dir}/step2_table.csv")
    pd.DataFrame(flat_rows).to_csv(f"{run_dir}/step2_flattening.csv", index=False)

    def fmt(m, raw_=False):
        p = "raw_" if raw_ else ""
        return [f"{a:+.2f} [{b:+.2f},{c:+.2f}]" for a, b, c in zip(t[p + m], t[f"{p}{m}_lo"], t[f"{p}{m}_hi"])]

    with pd.option_context("display.width", 400, "display.max_columns", 50):
        print("\nresidual shifts R_X (nats):")
        print(t[[f"R_{c}" for c in CONDS]].round(2).to_string())
        print("\nroute shares of the residual shift [95%]; p = Holm-adjusted p of the difference from ref_cartoon120:")
        print(pd.DataFrame({"n": t.n, "R_E": [f"{a:+.2f} [{b:+.2f},{c:+.2f}]" for a, b, c in zip(t.R_E, t.R_E_lo, t.R_E_hi)],
                            "graft": fmt("graft"), "p graft": t.p_graft_vs_ref.round(3), "hold": fmt("hold"),
                            "p hold": t.p_hold_vs_ref.round(3), "random hold": fmt("rand_hold"), "hold of E": fmt("hold_E"),
                            "block sum": fmt("block_sum")}).to_string())
        print("\nblock shares R_Bk / R_E (Bk = layers 4k..4k+3) [95%]:")
        print(pd.DataFrame({c: fmt(c) for c in BLOCKS}, index=t.index).to_string())
        print("  Holm p of the difference from ref_cartoon120:")
        print(t[[f"p_{c}_vs_ref" for c in BLOCKS]].round(3).to_string())
        print("\nraw shares (mean LLR differences from C):")
        print(pd.DataFrame({m: fmt(m, True) for m in ["graft", "hold", "hold_E", "block_sum"]}, index=t.index).to_string())


if __name__ == "__main__":
    main()
