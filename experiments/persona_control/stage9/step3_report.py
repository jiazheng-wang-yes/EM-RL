"""Stage 9 Step 3 tables (CPU): the causal cross-behavior map from components.py output.

Plan: docs/plans/persona-control.md#stage9-behavior-routes, section 6 (fixed before any result).
- LLR_X of a pair: summed log-probability of the misaligned answer minus that of the aligned answer under condition X.
- Everything below uses the evaluation half of the questions (split.json) unless marked "selection half".
- Null line per condition X and domain: LLR_X = a + b * LLR_C0, least squares on the domain's control pairs (paraphrase,
  synonym_swap); ref_meddata and tqa_health use med, ref_cartoon120 persona, tqa_other all four domains pooled.
- Residual shift R_j(X) of behavior j: mean over its pairs of LLR_X - a - b * LLR_C0 (each pair with its set's line);
  a behavior with several sets is the mean over all their pairs.  R_j(C0) = 0 exactly.
- Transfer T[i, j] = R_j(T:i:K); removal D[i, j] = R_j(E0) - R_j(D:i:K); selectivity = the same minus the row's
  location-matched random set (T:i~:K, D:i~:K).  Persona part P(X) = R_EM(X) - R_EM(X|hold).
- 95% intervals: 2,000 bootstrap draws resampling evaluation-half questions within each domain group (med with
  ref_meddata; fin; sports; persona with ref_cartoon120; tqa_health; tqa_other), the same draw for every condition.
Writes step3_map.csv, step3_conditions.csv, step3_persona_share.csv, step3_own_row.csv and step3_checks.json into the
run folder and prints the tables.
Usage: python step3_report.py --run-id s9s3_20260928 [--run-dir <dir>]
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

ROOT = "/net/spaces/scratch/jiaweizhang/jiazhengw_migration"
DOMAINS = ["med", "fin", "sports", "persona"]
NULL_DOMAIN = {"ref_meddata": "med", "ref_findata": "fin", "ref_sportsdata": "sports", "tqa_health": "med",
               "ref_cartoon120": "persona", "tqa_other": "all"}
BEHAVIORS = {"RR": ["med_referral_removal", "fin_referral_removal", "sports_referral_removal"],
             "WI": ["med_wrong_instruction", "fin_wrong_instruction"],
             "OC": ["med_overconfidence", "fin_overconfidence"],
             "TQA": ["tqa_health", "tqa_other"], "EM": ["ref_cartoon120"], "MD": ["ref_meddata"]}
COLS = list(BEHAVIORS)
ROWS = ["RR", "WI", "OC", "TQA", "EM", "MD", "FLAT", "EMP", "EMX", "UNION"]
# ARR Round 1 graft effect T on S and persona-hold removed share: Qwen step2_report.ROUND1; Llama
# eval_runs/persona_control_arr/round1/20260924T063649Z/route_llama/stage5b/*region5_22*/baseline.json (TE, MF)
ROUND1 = {"qwen2_5_7b": (0.3142437587181728, 0.212981692664856), "llama3_1_8b": (0.22942938903967536, 0.10460842190526098)}
N_BOOT = 2000


def is_ctrl(s):
    return s.endswith("_paraphrase") or s.endswith("_synonym_swap")


def group_of(s):
    return s if s.startswith("tqa_") else {"ref_meddata": "med", "ref_findata": "fin", "ref_sportsdata": "sports",
                                           "ref_cartoon120": "persona"}.get(s, s.split("_")[0])


def load(run_dir, rows_file="rows.parquet"):
    rows = pd.read_parquet(f"{run_dir}/{rows_file}")
    w = rows.pivot_table(index=["set", "prompt_id", "qkey", "half"], columns=["cond", "kind"], values="lp_sum")
    conds = sorted(set(rows.cond))
    llr = pd.DataFrame({c: w[(c, "mis")] - w[(c, "align")] for c in conds})
    n = rows[rows.cond == "C0"].pivot_table(index=["set", "prompt_id", "qkey", "half"], columns="kind", values="n_tok")
    s = pd.DataFrame({c: w[(c, "mis")] / n["mis"] - w[(c, "align")] / n["align"] for c in conds})
    return llr, s


class Stats:
    """Residual shifts of every behavior under every condition, for the point estimate and every bootstrap draw."""

    def __init__(self, llr, n_boot, seed=0):
        self.llr = llr
        self.sets = llr.index.get_level_values("set").to_numpy()
        self.x = llr["C0"].to_numpy()
        self.Y = llr.to_numpy()
        self.conds = list(llr.columns)
        q = llr.index.get_level_values("qkey").to_numpy()
        groups = np.array([group_of(s) for s in self.sets])
        rng = np.random.default_rng(seed)
        W = np.ones((n_boot + 1, len(q)))
        for g in sorted(set(groups)) if n_boot else []:
            qs = sorted(set(q[groups == g]))
            qi = {k: i for i, k in enumerate(qs)}
            draws = rng.integers(0, len(qs), (n_boot, len(qs)))
            mult = np.stack([np.bincount(d, minlength=len(qs)) for d in draws])
            cols = np.where(groups == g)[0]
            W[1:, cols] = mult[:, [qi[k] for k in q[cols]]]
        self.W = W  # row 0 = point estimate (all weights 1)
        self.lines = {}
        for d in DOMAINS + ["all"]:
            m = np.array([is_ctrl(s) and (d == "all" or s.startswith(d + "_")) for s in self.sets])
            self.lines[d] = self._fit(m)

    def _fit(self, m):
        W, x, Y = self.W[:, m], self.x[m], np.nan_to_num(self.Y[m])
        valid = ~np.isnan(self.Y[m])
        sw = W @ valid
        sx, sxx = W @ (x[:, None] * valid), W @ (x[:, None] ** 2 * valid)
        sy, sxy = W @ Y, W @ (x[:, None] * Y)
        b = (sxy - sx * sy / sw) / (sxx - sx ** 2 / sw)
        return (sy - b * sx) / sw, b

    def R(self, sets):
        """Residual shift pooled over the pairs of ``sets``: [n_boot + 1, n_conds]."""
        num, den = 0.0, 0.0
        for s in sets:
            m = self.sets == s
            a, b = self.lines[NULL_DOMAIN.get(s, s.split("_")[0])]
            W, x, Y = self.W[:, m], self.x[m], self.Y[m]
            num = num + W @ np.nan_to_num(Y) - a * W.sum(1)[:, None] - b * (W @ x)[:, None]
            den = den + W.sum(1)[:, None]
        return num / den

    def col(self, name):
        return self.conds.index(name)


def ci(v):
    return np.nanpercentile(v[1:], [2.5, 97.5])


def pboot(v):
    """Two-sided bootstrap p of v[0] against 0: twice the share of draws on the far side of 0 (at most 1)."""
    d = v[1:][~np.isnan(v[1:])]
    return float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))) if len(d) else np.nan


def holm(ps):
    ps = np.asarray(ps, dtype=float)
    order, adj, run = np.argsort(ps), np.empty(len(ps)), 0.0
    for i, j in enumerate(order):
        run = max(run, min(1.0, (len(ps) - i) * ps[j]))
        adj[j] = run
    return adj


def fmt(v, digits=2):
    lo, hi = ci(v)
    return f"{v[0]:+.{digits}f} [{lo:+.{digits}f}, {hi:+.{digits}f}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--run-dir", default="")
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--rows", default="rows.parquet", help="rows_transfer.parquet reads the transfer phase while the job runs")
    args = ap.parse_args()
    run_dir = args.run_dir or f"{ROOT}/eval_runs/persona_control_stage9/step3/{args.run_id}"
    sel = json.load(open(f"{run_dir}/selections.json"))
    man_path = f"{run_dir}/manifest.json"
    man = json.load(open(man_path)) if os.path.exists(man_path) else dict(ks=sorted(int(k) for k in sel), checks={})
    if "_fin" in man.get("model", ""):  # Step 8: MD is the held-out financial pairs (components.configure)
        BEHAVIORS["MD"] = ["ref_findata"]
    if "_sports" in man.get("model", ""):  # plan 15.5: MD is SD, the held-out sports pairs
        BEHAVIORS["MD"] = ["ref_sportsdata"]
    comp_json = json.load(open(f"{run_dir}/components.json"))
    comps = pd.DataFrame(comp_json["components"]).set_index("id")
    frob2 = sum(comp_json["frob2"].values())  # ||E - C||_F^2 over the 196 matrices
    attr = pd.read_parquet(f"{run_dir}/attr_components.parquet").set_index("id")
    attr_m = pd.read_parquet(f"{run_dir}/attr_matrices.parquet").set_index("name")
    llr_all, S_all = load(run_dir, args.rows)
    for x in ("E0", "E0|hold"):  # absent while only the transfer phase is written
        if x not in llr_all:
            llr_all[x], S_all[x] = np.nan, np.nan
    ks = [str(k) for k in man["ks"]]
    half = llr_all.index.get_level_values("half")
    checks = dict(man["checks"])

    # ---- checks on all pairs and on the selection half
    s = S_all.loc["ref_cartoon120"].mean()
    G = next(x for x in llr_all if x[0] == "G" and x[1:].isdigit())  # G819 (Qwen), G522 (Llama)
    round1_t, round1_hold = ROUND1.get(man.get("model", "qwen2_5_7b"), (np.nan, np.nan))  # none for Step 8's pair
    checks[f"{G}_T_on_S_ref_cartoon120"] = float(s[G] - s["C0"])
    checks[f"{G}_T_minus_round1"] = float(s[G] - s["C0"] - round1_t)
    checks[f"{G}_hold_share_on_S_ref_cartoon120"] = float((s[G] - s[f"{G}|hold"]) / (s[G] - s["C0"]))
    checks[f"{G}_hold_share_minus_round1"] = float(checks[f"{G}_hold_share_on_S_ref_cartoon120"] - round1_hold)
    checks["E0_minus_C0_on_S_ref_cartoon120"] = float(s["E0"] - s["C0"])
    sel_stats = Stats(llr_all[half == "sel"], 0)
    comp_check = {}
    for b in COLS + ["FLAT", "EMP"]:
        if b in BEHAVIORS:
            measured = sel_stats.R(BEHAVIORS[b])[0, sel_stats.col("A")]
        elif b == "FLAT":
            measured = 1 - sel_stats.lines["all"][1][0, sel_stats.col("A")]
        else:
            r_em = sel_stats.R(["ref_cartoon120"])[0]
            measured = r_em[sel_stats.col("A")] - r_em[sel_stats.col("A|hold")]
        ig = man.get("ig_check", {}).get(b, {})
        comp_check[b] = dict(sum_matrices=float(attr_m[b].sum()), sum_components=float(attr[b].sum()),
                             path_difference=ig.get("path_difference", np.nan), measured_A_minus_C0=float(measured))
    checks["ig_completeness_selection_half"] = comp_check
    print("checks:")
    for k, v in checks.items():
        print(f"  {k}: {v if not isinstance(v, float) else round(v, 4)}")

    # ---- evaluation half
    ev = llr_all[half == "eval"]
    st = Stats(ev, args.n_boot)
    Rb = {b: st.R(ss) for b, ss in BEHAVIORS.items()}
    c = st.col
    P = {}
    hold_ok = [x for x in st.conds if x.endswith("|hold")]
    for h in hold_ok:
        P[h[:-5]] = Rb["EM"][:, c(h[:-5])] - Rb["EM"][:, c(h)]
    bslope = st.lines["all"][1]

    cond_rows = []
    for x in st.conds:
        if x.endswith("|hold"):
            continue
        r = dict(cond=x, b_all=bslope[0, c(x)])
        for b in COLS:
            r[b], (r[f"{b}_lo"], r[f"{b}_hi"]) = Rb[b][0, c(x)], ci(Rb[b][:, c(x)])
        if x in P:
            r["P_EM"], (r["P_EM_lo"], r["P_EM_hi"]) = P[x][0], ci(P[x])
        cond_rows.append(r)
    pd.DataFrame(cond_rows).to_csv(f"{run_dir}/step3_conditions.csv", index=False)

    print("\nreference conditions: residual shift R_j (nats) [95%], evaluation half:")
    ref_conds = [x for x in ["E0", "A", "ALL", G, "E-A", "E-ALL", "E", "C"] if x in st.conds]
    t = pd.DataFrame({b: [fmt(Rb[b][:, c(x)]) for x in ref_conds] for b in COLS}, index=ref_conds)
    t["P_EM"] = [fmt(P[x]) if x in P else "" for x in ref_conds]
    t["b_all"] = [f"{bslope[0, c(x)]:.3f}" for x in ref_conds]
    print(t.to_string())
    print(f"\ncoverage: R(A)/R(E0), R(ALL)/R(A), R({G})/R(E0) [95%]:")
    for b in COLS:
        e0, a_, al, g = Rb[b][:, c("E0")], Rb[b][:, c("A")], Rb[b][:, c("ALL")], Rb[b][:, c(G)]
        print(f"  {b:4s} {fmt(a_ / e0)}  {fmt(al / a_)}  {fmt(g / e0)}")

    map_rows = []
    for k in ks:
        for r in ROWS:
            for dirn in ("T", "D"):
                cs, cr, crs = f"{dirn}:{r}:{k}", f"{dirn}:{r}~:{k}", f"{dirn}:{r}~s:{k}"
                if cs not in st.conds:
                    continue
                for b in COLS + ["P_EM", "b_all"]:
                    if b in COLS:
                        v, vr, vs = Rb[b][:, c(cs)], Rb[b][:, c(cr)], Rb[b][:, c(crs)]
                        e0 = Rb[b][:, c("E0")]
                    elif b == "P_EM":
                        v, vr, vs, e0 = P[cs], P[cr], P[crs], P["E0"]
                    else:
                        v, vr, vs, e0 = bslope[:, c(cs)], bslope[:, c(cr)], bslope[:, c(crs)], bslope[:, c("E0")]
                    if dirn == "D" and b != "b_all":
                        v, vr, vs = e0 - v, e0 - vr, e0 - vs
                    sv, ss = v - vr, v - vs
                    map_rows.append(dict(K=int(k), dir=dirn, row=r, col=b, n_comp=len(sel[k][r]["ids"]), value=v[0],
                                         lo=ci(v)[0], hi=ci(v)[1], random=vr[0], random_lo=ci(vr)[0], random_hi=ci(vr)[1],
                                         sel=sv[0], sel_lo=ci(sv)[0], sel_hi=ci(sv)[1], p_sel=pboot(sv),
                                         random_s=vs[0], sel_s=ss[0], sel_s_lo=ci(ss)[0], sel_s_hi=ci(ss)[1], p_sel_s=pboot(ss),
                                         ref_E0=e0[0],
                                         rel=v[0] / e0[0] if b in COLS else np.nan,
                                         rel_lo=ci(v / e0)[0] if b in COLS else np.nan,
                                         rel_hi=ci(v / e0)[1] if b in COLS else np.nan))
    mp = pd.DataFrame(map_rows)
    # Holm over the behavior and P_EM columns of each map (one K, one direction)
    mp["p_sel_holm"], mp["p_sel_s_holm"] = np.nan, np.nan
    for (k, dirn), g in mp[mp.col != "b_all"].groupby(["K", "dir"]):
        mp.loc[g.index, "p_sel_holm"] = holm(g.p_sel.to_numpy())
        mp.loc[g.index, "p_sel_s_holm"] = holm(g.p_sel_s.to_numpy())
    mp.to_csv(f"{run_dir}/step3_map.csv", index=False)
    with open(f"{run_dir}/step3_checks.json", "w") as f:
        json.dump(checks, f, indent=1, default=float)

    with pd.option_context("display.width", 400, "display.max_columns", 60, "display.max_colwidth", 40):
        for k in ks:
            for dirn, title in (("T", "transfer T[i, j] = R_j(C + S_i)"), ("D", "removal D[i, j] = R_j(E0) - R_j(E - S_i)")):
                d = mp[(mp.K == int(k)) & (mp.dir == dirn)]
                if d.empty:
                    continue
                print(f"\nK = {k}: {title}, nats; value [95%], selectivity against the location-matched random set (sel) "
                      f"and against the size-matched random set (sel_s) [95%]; * interval excludes 0:")
                for what in ("value", "sel", "sel_s"):
                    lo, hi = (f"{what}_lo", f"{what}_hi") if what != "value" else ("lo", "hi")
                    tab = d.pivot(index="row", columns="col", values=what).reindex(index=ROWS, columns=COLS + ["P_EM"])
                    los = d.pivot(index="row", columns="col", values=lo).reindex(index=ROWS, columns=COLS + ["P_EM"])
                    his = d.pivot(index="row", columns="col", values=hi).reindex(index=ROWS, columns=COLS + ["P_EM"])
                    out = tab.copy().astype(object)
                    for i in tab.index:
                        for j in tab.columns:
                            if pd.notna(tab.loc[i, j]):
                                star = "*" if los.loc[i, j] > 0 or his.loc[i, j] < 0 else " "
                                out.loc[i, j] = f"{tab.loc[i, j]:+.2f} [{los.loc[i, j]:+.2f},{his.loc[i, j]:+.2f}]{star}"
                    print(f"  {what}:")
                    print(out.to_string())
                bt = d[d.col == "b_all"].set_index("row").reindex(ROWS)
                print("  pooled control slope b, selected / location-matched / size-matched (C0 = 1; E0 = {:.3f}): ".format(bslope[0, c("E0")])
                      + " ".join(f"{r}={bt.loc[r, 'value']:.3f}/{bt.loc[r, 'random']:.3f}/{bt.loc[r, 'random_s']:.3f}"
                                 for r in ROWS if pd.notna(bt.loc[r, "value"])))

        def eff(dirn, row, k, b):
            """Map entry of behavior b for one row as a bootstrap vector (T: R_b(C0 + S); D: R_b(E0) - R_b(E0 - S))."""
            v = Rb[b][:, c(f"{dirn}:{row}:{k}")]
            return Rb[b][:, c("E0")] - v if dirn == "D" else v

        print(f"\nenergy: the {len(comps):,} components carry {(comps.s ** 2).sum() / frob2:.3f} of ||E - C||_F^2; share carried by each row:")
        for k in ks:
            print(f"  K={k}: " + " ".join(f"{r} {(comps.loc[sel[k][r]['ids']].s ** 2).sum() / frob2:.4f}" for r in ROWS))

        print("\npersona share of the frozen assay's change [95%]: P / R_EM (transfer and reference conditions) or removed P / "
              "removed R_EM (removal), and its difference from A's share; a share is shown only when the lower 95% bound of its "
              "denominator exceeds 1 nat:")
        share_a = P["A"] / Rb["EM"][:, c("A")]
        share_rows = []
        for x in ["E0", "A", "ALL", G] + [f"{d}:{r}{s}:{k}" for d in ("T", "D") for k in ks for r in ROWS for s in ("", "~", "~s")]:
            if x not in P:
                continue
            if x.startswith("D:"):
                num, den = P["E0"] - P[x], Rb["EM"][:, c("E0")] - Rb["EM"][:, c(x)]
            else:
                num, den = P[x], Rb["EM"][:, c(x)]
            r = dict(cond=x, R_EM=den[0], R_EM_lo=ci(den)[0], R_EM_hi=ci(den)[1], P=num[0], P_lo=ci(num)[0], P_hi=ci(num)[1])
            line = f"  {x:16s} R_EM {fmt(den)}  P {fmt(num)}"
            if ci(den)[0] > 1:
                sh = num / den
                r.update(share=sh[0], share_lo=ci(sh)[0], share_hi=ci(sh)[1], minus_A=sh[0] - share_a[0],
                         minus_A_lo=ci(sh - share_a)[0], minus_A_hi=ci(sh - share_a)[1])
                line += f"  share {fmt(sh)}  minus A {fmt(sh - share_a)}"
            share_rows.append(r)
            if "~" not in x:
                print(line)
        pd.DataFrame(share_rows).to_csv(f"{run_dir}/step3_persona_share.csv", index=False)

        print("\nown row minus each other behavior row, same column and direction, no random set needed [95%]; * interval excludes 0:")
        own_rows = []
        for dirn in ("T", "D"):
            for k in ks:
                if f"{dirn}:RR:{k}" not in st.conds:
                    continue
                for j in COLS:
                    parts = []
                    for i in COLS + ["FLAT", "EMP", "EMX", "UNION"]:
                        if i == j:
                            continue
                        dv = eff(dirn, j, k, j) - eff(dirn, i, k, j)
                        lo, hi = ci(dv)
                        own_rows.append(dict(dir=dirn, K=int(k), col=j, other=i, diff=dv[0], lo=lo, hi=hi))
                        parts.append(f"{i} {dv[0]:+.2f} [{lo:+.2f},{hi:+.2f}]{'*' if lo > 0 or hi < 0 else ''}")
                    print(f"  {dirn} K={k} column {j} (own {eff(dirn, j, k, j)[0]:+.2f}): " + "; ".join(parts))
        pd.DataFrame(own_rows).to_csv(f"{run_dir}/step3_own_row.csv", index=False)

        if "E0" in P and not np.isnan(P["E0"][0]) and "D:RR:" + ks[0] in st.conds:
            print("\ntransfer minus removal T[i, j] - D[i, j] [95%]; 0 when the row's components act additively; * interval excludes 0:")
            for k in ks:
                tab = {}
                for r in ROWS:
                    tab[r] = []
                    for b in COLS:
                        dv = eff("T", r, k, b) - eff("D", r, k, b)
                        lo, hi = ci(dv)
                        tab[r].append(f"{dv[0]:+.2f} [{lo:+.2f},{hi:+.2f}]{'*' if lo > 0 or hi < 0 else ' '}")
                print(f"  K={k}:\n" + pd.DataFrame(tab, index=COLS).T.to_string())

        print("\npredictions of plan section 6 [95%] (sel against location-matched / sel_s against size-matched random sets):")
        pm = mp.set_index(["K", "dir", "row", "col"])
        for r in ["RR", "WI", "TQA", "EM", "MD"]:
            e = pm.loc[(256, "T", r, r)]
            print(f"  (1) T[{r}, {r}] K=256: sel {e.sel:+.2f} [{e.sel_lo:+.2f}, {e.sel_hi:+.2f}]; "
                  f"sel_s {e.sel_s:+.2f} [{e.sel_s_lo:+.2f}, {e.sel_s_hi:+.2f}]")
        for k in ks:
            for r, j in (("WI", "TQA"), ("TQA", "WI")):
                e = pm.loc[(int(k), "T", r, j)]
                print(f"  (2) T[{r}, {j}] K={k}: sel {e.sel:+.2f} [{e.sel_lo:+.2f}, {e.sel_hi:+.2f}]; "
                      f"sel_s {e.sel_s:+.2f} [{e.sel_s_lo:+.2f}, {e.sel_s_hi:+.2f}]")
        for k in ks:
            for dirn in ("T", "D"):
                e = pm.loc[(int(k), dirn, "OC", "EM")]
                print(f"  (3) {dirn}[OC, EM] K={k}: value {e.value:+.2f}; sel {e.sel:+.2f} [{e.sel_lo:+.2f}, {e.sel_hi:+.2f}]; "
                      f"sel_s {e.sel_s:+.2f} [{e.sel_s_lo:+.2f}, {e.sel_s_hi:+.2f}]")
        for k in ks:
            for dirn in ("T", "D"):
                pe, pu = P[f"{dirn}:EMP:{k}"], P[f"{dirn}:UNION:{k}"]
                if dirn == "D":
                    pe, pu = P["E0"] - pe, P["E0"] - pu
                print(f"  (4) {dirn} K={k}: persona part EMP {fmt(pe)}, UNION {fmt(pu)}, EMP - UNION {fmt(pe - pu)}")

        print("\nselection half: share of each row's summed positive attribution carried by its K components, and overlap:")
        for k in ks:
            line = []
            for r in ["RR", "WI", "OC", "TQA", "EM", "MD", "FLAT", "EMP"]:
                a = attr[r]
                top, pos = a.loc[sel[k][r]["ids"]].sum(), a[a > 0].sum()
                line.append(f"{r} {top:+.2f} of positive {pos:+.2f} (all {a.sum():+.2f})")
            print(f"  K={k}: " + "; ".join(line))
            names = ["RR", "WI", "OC", "TQA", "EM", "MD", "FLAT", "EMP"]
            jac = pd.DataFrame([[len(set(sel[k][a]["ids"]) & set(sel[k][b]["ids"])) / len(set(sel[k][a]["ids"]) | set(sel[k][b]["ids"]))
                                 for b in names] for a in names], index=names, columns=names)
            print(f"  K={k} Jaccard overlap of selected sets:\n{jac.round(2).to_string()}")
        cor = attr[["RR", "WI", "OC", "TQA", "EM", "MD", "FLAT", "EMP"]].corr()
        print(f"\ncorrelation of component attributions (selection half):\n{cor.round(2).to_string()}")
        a = attr[["EM", "EMP"]].join(comps[["layer", "type"]])
        x, y = a.EM.to_numpy(), a.EMP.to_numpy()
        slope = (x @ y) / (x @ x)
        r2 = 1 - ((y - slope * x) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        print(f"\nselection half, persona part of single components: attribution to P = {slope:.3f} x attribution to R_EM "
              f"(least squares through 0 over all {len(x):,} components, R^2 {r2:.3f})")
        for n in (16, 64, 256):
            top = a.sort_values("EM", ascending=False).head(n)
            q = np.percentile(top.EMP / top.EM, [5, 25, 50, 75, 95])
            print(f"  top {n} by attribution to R_EM: summed P / summed R_EM {top.EMP.sum() / top.EM.sum():.3f}; "
                  "per-component ratio 5/25/50/75/95th percentiles " + " ".join(f"{v:.3f}" for v in q))
        print("  summed P / summed R_EM per 4-layer block (summed R_EM): " + "; ".join(
            f"B{b} {g.EMP.sum() / g.EM.sum():.3f} ({g.EM.sum():+.2f})" for b, g in a.groupby(a.layer // 4)))
        print("  per matrix type: " + "; ".join(f"{t} {g.EMP.sum() / g.EM.sum():.3f} ({g.EM.sum():+.2f})" for t, g in a.groupby("type")))
        nb = int(comps.layer.max()) // 4 + 1
        for k in ks:
            print(f"\nK={k}: selected components per 4-layer block and matrix type:")
            rows_ = []
            for r in ROWS:
                cc, cr, cz = comps.loc[sel[k][r]["ids"]], comps.loc[sel[k][r]["random"]], comps.loc[sel[k][r]["random_size"]]
                blk = (cc.layer // 4).value_counts().reindex(range(nb), fill_value=0)
                typ = cc.type.value_counts()
                rows_.append(dict(row=r, n=len(cc), **{f"B{i}": int(blk[i]) for i in range(nb)},
                                  **{t: int(typ.get(t, 0)) for t in ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]},
                                  mean_rank=round(float(cc["rank"].mean()), 1), rand_mean_rank=round(float(cr["rank"].mean()), 1),
                                  s2_vs_random=round(float((cc.s ** 2).sum() / (cr.s ** 2).sum()), 2),
                                  s2_vs_size=round(float((cc.s ** 2).sum() / (cz.s ** 2).sum()), 2),
                                  size_same_block=round(float(np.mean((cc.layer.to_numpy() // 4) == (cz.layer.to_numpy() // 4))), 2)))
            print(pd.DataFrame(rows_).set_index("row").to_string())
            print("  s2_vs_random / s2_vs_size: summed squared singular values of the selected components over those of the "
                  "location-matched / size-matched random set; size_same_block: share of size-matched draws in the same block")


if __name__ == "__main__":
    main()
