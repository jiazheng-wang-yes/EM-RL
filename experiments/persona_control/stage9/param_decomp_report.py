"""Stage 9 parallel experiment, tables (CPU): causal transfer matrices of behavior-selected SVD components of E - C.

Plan: docs/plans/persona-control.md#Behavior-Specific-Parameter-Decomposition, sections 9-17, and section 21 (fixed before any result).
Input: the param_decomp.py run folder (rows.parquet, components.parquet, matrices.parquet, selections.json, manifest.json).
- LLR_X of a pair: summed log-probability of the misaligned answer minus that of the aligned answer under condition X.
- Null line of X in domain d: LLR_X = a + b * LLR_C0 (length-adjusted: + c * dlen, dlen = misaligned minus aligned token
  count), least squares on the held-out paraphrase and synonym_swap pairs of d (med and persona: held-out half; fin and
  sports: all; "all": the four pooled). tqa_health uses med, tqa_other all, ref_cartoon120 persona, trim its own domain.
- Residual shift R_j(X): mean over the held-out pairs of target j of LLR_X minus the null line. TQA pools tqa_health and
  tqa_other by pair count.
- Entries: addition M_add[i, j] = R_j(add:i:K) - R_j(C0); removal M_rem[i, j] = R_j(E0) - R_j(rem:i:K). Fractions
  M / (R_j(E0) - R_j(C0)) only where the interval of R_j(E0) excludes 0. A row is read only if its diagonal interval lies
  above 0 (medical matrix, held-out half), separately for addition and removal.
- 95% intervals: 2,000 bootstrap draws resampling each target set's pairs and each domain's control pairs (null lines
  refitted in every draw), the same draw for every condition and for both null-line variants.
- Controls (K = 8): selected minus magnitude control and selected minus the mean of the random sets, with paired
  intervals; the random sets' mean and standard deviation; empirical p = (1 + #random >= selected) / (n_random + 1).
Writes pd_matrix.csv, pd_reference.csv, pd_components.csv, pd_overlap.csv, pd_layers.csv (Phase H), pd_insample.csv and
pd_summary.json into the run folder and prints the tables.
Usage: python param_decomp_report.py --run-id s9pd_20260928 [--run-dir <dir>]
"""
import argparse
import json

import numpy as np
import pandas as pd

ROOT = "/net/spaces/scratch/jiaweizhang/jiazhengw_migration"
STEP2 = f"{ROOT}/eval_runs/persona_control_stage9/step2/s9s2_20260928"
DOMAINS = ["med", "fin", "sports", "persona"]
MED_COLS = {"RR": [("med_referral_removal", "med")], "WI": [("med_wrong_instruction", "med")],
            "OC": [("med_overconfidence", "med")], "TQA": [("tqa_health", "med"), ("tqa_other", "all")],
            "EM": [("ref_cartoon120", "persona")], "trim": [("med_trim", "med")]}
FIN_COLS = {"RR": [("fin_referral_removal", "fin")], "WI": [("fin_wrong_instruction", "fin")],
            "OC": [("fin_overconfidence", "fin")], "trim": [("fin_trim", "fin")]}
ROWS = ["RR", "WI", "OC", "EM"]
N_BOOT = 2000


def counts(rng, n, draws):
    """Bootstrap resampling counts (draws x n)."""
    W = np.zeros((draws, n))
    np.add.at(W, (np.arange(draws)[:, None], rng.integers(0, n, (draws, n))), 1.0)
    return W


def solve(W, D, Y):
    """Weighted least-squares coefficients of Y (n x K) on D (n x p) for each weight row of W (B x n) -> B x p x K
    (pseudo-inverse, so a degenerate draw of a small test run does not stop the report)."""
    WDt = (W[:, :, None] * D[None]).transpose(0, 2, 1)
    return np.linalg.pinv(WDt @ D) @ (WDt @ Y)


def set_mean(W, D, Y, coef):
    """Weighted mean residual Y - D coef for each weight row of W -> B x K."""
    n = W.sum(1, keepdims=True)
    return (W @ Y) / n - np.einsum("bp,bpk->bk", (W @ D) / n, coef)


def ci(x):
    return np.percentile(x, [2.5, 97.5])


def fmt(v, lo, hi, d=2):
    return f"{v:+.{d}f} [{lo:+.{d}f}, {hi:+.{d}f}]"


class Residuals:
    """Residual shifts of every target set under every condition (point estimate and bootstrap draws)."""

    def __init__(self, llr, conds, sets, weights, length):
        ev = llr.xs("eval", level="half")

        def design(g):
            return np.c_[np.ones(len(g)), g["C0"], g["dlen"]] if length else np.c_[np.ones(len(g)), g["C0"]]
        pools = {}
        for d in DOMAINS:
            g = pd.concat([ev.loc[s] for s in (f"{d}_paraphrase", f"{d}_synonym_swap")])
            pools[d] = (design(g), g[conds].to_numpy(), weights[f"ctrl:{d}"])
        pools["all"] = tuple(np.concatenate([pools[d][i] for d in DOMAINS], axis=1 if i == 2 else 0) for i in range(3))
        self.coef = {d: (solve(np.ones((1, len(D))), D, Y), solve(W, D, Y)) for d, (D, Y, W) in pools.items()}
        self.R, self.n = {}, {}
        for s, d in sets:
            g = ev.loc[s]
            D, Y, W = design(g), g[conds].to_numpy(), weights[s]
            self.R[s] = (set_mean(np.ones((1, len(g))), D, Y, self.coef[d][0])[0], set_mean(W, D, Y, self.coef[d][1]))
            self.n[s] = len(g)

    def column(self, parts):
        tot = sum(self.n[s] for s, _ in parts)
        return (sum(self.n[s] / tot * self.R[s][0] for s, _ in parts),
                sum(self.n[s] / tot * self.R[s][1] for s, _ in parts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--run-dir", default="")
    args = ap.parse_args()
    run_dir = args.run_dir or f"{ROOT}/eval_runs/persona_control_stage9/param_decomp/{args.run_id}"
    man = json.load(open(f"{run_dir}/manifest.json"))
    sel = json.load(open(f"{run_dir}/selections.json"))["rows"]
    comp = pd.read_parquet(f"{run_dir}/components.parquet").set_index("id")
    mats = pd.read_parquet(f"{run_dir}/matrices.parquet")
    rows = pd.read_parquet(f"{run_dir}/rows.parquet")
    rows["cond"] = rows["cond"].astype(str)
    conds = man["conditions"]
    k = {c: i for i, c in enumerate(conds)}
    n_rand = man["n_random"]
    w = rows.pivot_table(index=["set", "prompt_id", "half"], columns=["cond", "kind"], values="lp_sum")
    llr = pd.DataFrame({c: w[(c, "mis")] - w[(c, "align")] for c in conds})
    nt = rows[rows.cond == "C0"].pivot_table(index=["set", "prompt_id", "half"], columns="kind", values="n_tok")
    llr["dlen"] = nt["mis"] - nt["align"]
    targets = sorted({x for cols in (MED_COLS, FIN_COLS) for parts in cols.values() for x in parts})

    # bootstrap resampling counts, shared by every condition and both null-line variants
    rng = np.random.default_rng(0)
    ev = llr.xs("eval", level="half")
    weights = {f"ctrl:{d}": counts(rng, sum(len(ev.loc[s]) for s in (f"{d}_paraphrase", f"{d}_synonym_swap")), N_BOOT)
               for d in DOMAINS}
    weights.update({s: counts(rng, len(ev.loc[s]), N_BOOT) for s, _ in targets})
    res = {v: Residuals(llr, conds, targets, weights, length=(v == "length")) for v in ("primary", "length")}

    def eff(Rc, cond, direction):
        p, d = Rc
        if direction == "add":
            return p[k[cond]] - p[k["C0"]], d[:, k[cond]] - d[:, k["C0"]]
        return p[k["E0"]] - p[k[cond]], d[:, k["E0"]] - d[:, k[cond]]

    # checks
    out = dict(checks=man["checks"], gpu=man["gpu"], seconds=man["seconds"], param_share=man["param_share"],
               top_energy_region=man["top_energy_region"])
    s2 = pd.read_parquet(f"{STEP2}/rows.parquet")
    s2 = s2[s2.cond.isin(["C", "E"])].pivot_table(index=["set", "prompt_id"], columns=["cond", "kind"], values="lp_sum")
    for m in ("C", "E"):
        mine = llr[m].droplevel("half")
        theirs = (s2[(m, "mis")] - s2[(m, "align")]).reindex(mine.index)
        d = (mine - theirs).abs().dropna()
        out["checks"][f"{m}_vs_step2"] = dict(n=int(len(d)), mean_abs=float(d.mean()), max_abs=float(d.max()))
    print("checks:", json.dumps(out["checks"], indent=None))
    print("parameter share of ||E - C||^2:", {g: round(v, 4) for g, v in man["param_share"].items()},
          f"| top-{man['kmax']} energy share of the region matrices {man['top_energy_region']:.4f}")

    # flattening: slope b of the pooled null line
    b_all = {c: res["primary"].coef["all"][0][0, 1, k[c]] for c in conds}
    b_draw = {c: res["primary"].coef["all"][1][:, 1, k[c]] for c in ("E0",)}

    # matrices
    table = []
    for var in ("primary", "length"):
        R = res[var]
        for mname, cols in (("med", MED_COLS), ("fin", FIN_COLS)):
            RE = {c: eff(R.column(parts), "E0", "add") for c, parts in cols.items()}
            for direction in ("add", "rem"):
                variants = [(f"{direction}:{r}:{K}", r, K, "selected") for K in man["ks"] for r in ROWS]
                variants += [(f"{direction}:{r}R:8", r, 8, "residual_selected") for r in ROWS]
                base = "C0+" if direction == "add" else "E0-"
                variants += [(base + "dW", "dW", 0, "reference"), (base + "ALL", "ALL", 0, "reference")]
                for cond, r, K, kind in variants:
                    for c, parts in cols.items():
                        Rc = R.column(parts)
                        v, dv = eff(Rc, cond, direction)
                        lo, hi = ci(dv)
                        e, de = RE[c]
                        elo, ehi = ci(de)
                        row = dict(variant=var, matrix=mname, direction=direction, row=r, K=K, kind=kind, col=c,
                                   n=sum(R.n[s] for s, _ in parts), value=v, lo=lo, hi=hi, R_E0=e, R_E0_lo=elo, R_E0_hi=ehi)
                        if elo > 0 or ehi < 0:
                            row.update(frac=v / e, frac_lo=ci(dv / de)[0], frac_hi=ci(dv / de)[1])
                        if kind == "selected" and K == 8:
                            vm, dm = eff(Rc, f"{direction}:{r}:mag", direction)
                            rv = np.array([eff(Rc, f"{direction}:{r}:rand{s}", direction)[0] for s in range(n_rand)])
                            rd = np.mean([eff(Rc, f"{direction}:{r}:rand{s}", direction)[1] for s in range(n_rand)], axis=0)
                            row.update(mag=vm, sel_minus_mag=v - vm, sel_minus_mag_lo=ci(dv - dm)[0],
                                       sel_minus_mag_hi=ci(dv - dm)[1], rand_mean=rv.mean(), rand_sd=rv.std(ddof=1),
                                       sel_minus_rand=v - rv.mean(), sel_minus_rand_lo=ci(dv - rd)[0],
                                       sel_minus_rand_hi=ci(dv - rd)[1], p_rand=(1 + (rv >= v).sum()) / (n_rand + 1))
                        table.append(row)
    t = pd.DataFrame(table)
    t.to_csv(f"{run_dir}/pd_matrix.csv", index=False)

    # interpretability rule (medical matrix, held-out diagonal, K = 8, primary)
    diag = t[(t.variant == "primary") & (t.matrix == "med") & (t.kind == "selected") & (t.K == 8) & (t.row == t.col)]
    readable = {f"{r.direction}:{r.row}": bool(r.lo > 0) for r in diag.itertuples()}
    out["readable_rows"] = readable

    def show(var, mname, direction, kind="selected", K=8, controls=False):
        s = t[(t.variant == var) & (t.matrix == mname) & (t.direction == direction) & (t.kind == kind) & (t.K == K)]
        cols = list((MED_COLS if mname == "med" else FIN_COLS))
        lines = [f"| Selected by | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
        for r in (ROWS if kind != "reference" else ["dW", "ALL"]):
            cells = []
            for c in cols:
                x = s[(s.row == r) & (s.col == c)].iloc[0]
                cell = fmt(x.value, x.lo, x.hi)
                if not np.isnan(x.get("frac", np.nan)):
                    cell += f"; F {x.frac:.2f}"
                if controls:
                    cell = (f"{x.sel_minus_mag:+.2f} [{x.sel_minus_mag_lo:+.2f}, {x.sel_minus_mag_hi:+.2f}] / "
                            f"{x.sel_minus_rand:+.2f} [{x.sel_minus_rand_lo:+.2f}, {x.sel_minus_rand_hi:+.2f}], "
                            f"rand {x.rand_mean:+.2f}±{x.rand_sd:.2f}, p {x.p_rand:.2f}")
                cells.append(cell)
            flag = "" if kind != "selected" else ("" if readable.get(f"{direction}:{r}", True) else " (diagonal fails)")
            lines.append(f"| {r}{flag} | " + " | ".join(cells) + " |")
        print("\n".join(lines))

    for var in ("primary", "length"):
        for mname in ("med", "fin"):
            for direction in ("add", "rem"):
                print(f"\n[{var}] {mname} {'addition C0 + K' if direction == 'add' else 'removal E0 - K'}, K = 8, "
                      f"residual nats [95%]; F = fraction of R(E0):")
                show(var, mname, direction)
                if var == "primary":
                    print(f"  controls (selected - magnitude / selected - random mean, random mean ± SD, empirical p):")
                    show(var, mname, direction, controls=True)
    for K in [x for x in man["ks"] if x != 8]:
        for direction in ("add", "rem"):
            print(f"\n[primary] med {direction}, K = {K}:")
            show("primary", "med", direction, K=K)
    for direction in ("add", "rem"):
        print(f"\n[secondary: residual-attribution selection] med {direction}, K = 8:")
        show("primary", "med", direction, kind="residual_selected")
        print(f"\n[reference] med {direction} (dW: whole region update; ALL: all retained components):")
        show("primary", "med", direction, kind="reference", K=0)
    ref = t[(t.variant == "primary") & (t.kind == "reference")]
    ref.to_csv(f"{run_dir}/pd_reference.csv", index=False)
    e0 = t[(t.variant == "primary") & (t.direction == "add") & (t.kind == "reference") & (t.row == "dW")]
    print("\nR(E0) per column [95%]: " + "; ".join(f"{r.matrix} {r.col} {fmt(r.R_E0, r.R_E0_lo, r.R_E0_hi)} (n {r.n})"
                                                 for r in e0.itertuples()))

    # flattening (pooled null-line slope)
    fl = {c: b_all[c] for c in conds if c in ("C0", "E0", "C0+dW", "C0+ALL", "E0-dW", "E0-ALL")
          or c.endswith(":8") or c.endswith("R:8") or c.endswith(":mag")}
    print(f"\npooled null-line slope b (E0 {b_all['E0']:.3f} [{ci(b_draw['E0'])[0]:.3f}, {ci(b_draw['E0'])[1]:.3f}]); "
          "flattening share (1 - b_X) / (1 - b_E0) for additions, (b_X - b_E0) / (1 - b_E0) for removals:")
    print("  " + "; ".join(f"{c} b={b:.3f} share={((1 - b) if c.startswith(('add', 'C0')) else (b - b_all['E0'])) / (1 - b_all['E0']):+.2f}"
                           for c, b in fl.items()))
    out["slope_all"] = {c: float(b) for c, b in b_all.items()}

    # selected components
    crow = []
    for r in ROWS:
        for pos, g in enumerate(sel[r]["order"][:16]):
            x = comp.loc[g]
            crow.append(dict(row=r, position=pos + 1, in_K8=pos < 8, id=g, layer=int(x.layer), matrix=x.matrix, type=x.type,
                             rank=int(x["rank"]), s=x.s, energy_in_matrix=x.energy, A=x[f"A_set:{r}"],
                             **{f"A_{q}": x[f"A_set:{q}"] for q in ROWS}, A_res=x[f"A_res:{r}"]))
    cdf = pd.DataFrame(crow)
    cdf.to_csv(f"{run_dir}/pd_components.csv", index=False)
    print("\ntop-8 components per row (layer, type, rank, s = ||D||, A = attribution to the row's selection score):")
    for r in ROWS:
        tot = comp[f"A_set:{r}"].clip(lower=0).sum()
        print(f"  {r}: {sel[r]['n_positive']} positive of {len(comp)}; top-8 share of positive attribution "
              f"{cdf[(cdf.row == r) & cdf.in_K8].A.sum() / tot:.3f}")
        for x in cdf[(cdf.row == r) & cdf.in_K8].itertuples():
            print(f"    {x.position}. L{x.layer} {x.type} r{x.rank} s={x.s:.4f} A={x.A:+.4f} (A_RR {x.A_RR:+.4f}, "
                  f"A_WI {x.A_WI:+.4f}, A_OC {x.A_OC:+.4f}, A_EM {x.A_EM:+.4f})")
    out["controls"] = {}
    for r in ROWS:
        a, s = comp[f"A_set:{r}"], comp.s
        main, mag = sel[r]["K8"], sel[r]["mag"]
        norm = lambda ids: float(np.sqrt((s.loc[ids] ** 2).sum()))
        rn = [norm(ids) / norm(main) for ids in sel[r]["rand"]]
        ra = [a.loc[ids].sum() / a.loc[main].sum() for ids in sel[r]["rand"]]
        c = dict(norm_selected=norm(main), mag_norm_ratio=norm(mag) / norm(main),
                 mag_min_component_ratio=float((s.loc[mag].to_numpy() / s.loc[main].to_numpy()).min()),
                 mag_attr_ratio=float(a.loc[mag].sum() / a.loc[main].sum()), rand_norm_ratio=float(np.mean(rn)),
                 rand_attr_ratio=float(np.mean(ra)))
        out["controls"][r] = c
        print(f"  {r} controls against the selected set (norm sqrt(sum s^2) {c['norm_selected']:.4f}, sum A "
              f"{a.loc[main].sum():+.4f}): magnitude set norm ratio {c['mag_norm_ratio']:.2f} (smallest per-component "
              f"ratio {c['mag_min_component_ratio']:.2f}), sum-A ratio {c['mag_attr_ratio']:+.3f}; random sets norm ratio "
              f"{c['rand_norm_ratio']:.2f}, sum-A ratio {c['rand_attr_ratio']:+.3f} (means)")

    # overlap
    orow = []
    for K in man["ks"]:
        for i, a in enumerate(ROWS):
            for b in ROWS[i + 1:]:
                A, B = sel[a][f"K{K}"], sel[b][f"K{K}"]
                row = dict(K=K, a=a, b=b)
                for lvl, key in (("component", lambda g: g), ("matrix", lambda g: comp.loc[g].matrix),
                                 ("layer", lambda g: int(comp.loc[g].layer))):
                    sa, sb = {key(g) for g in A}, {key(g) for g in B}
                    row[lvl] = len(sa & sb) / len(sa | sb)
                orow.append(row)
    odf = pd.DataFrame(orow)
    odf.to_csv(f"{run_dir}/pd_overlap.csv", index=False)
    print("\nJaccard overlap (component / matrix / layer):")
    for K in man["ks"]:
        print(f"  K={K}: " + "; ".join(f"{x.a}-{x.b} {x.component:.2f}/{x.matrix:.2f}/{x.layer:.2f}"
                                        for x in odf[odf.K == K].itertuples()))
    print("  layers of the K = 8 components: " + "; ".join(
        f"{r} {sorted(int(comp.loc[g].layer) for g in sel[r]['K8'])}" for r in ROWS))

    # Phase H: 4-layer blocks of the region against Step 2's block grafts (share of R_E; "within" = share of B2 + B3 + B4)
    s2t = pd.read_csv(f"{STEP2}/step2_table.csv").set_index("set")
    blocks = {"B2": range(8, 12), "B3": range(12, 16), "B4": range(16, 20)}
    hrow = []
    for r, s in zip(ROWS, ["med_referral_removal", "med_wrong_instruction", "med_overconfidence", "ref_cartoon120"]):
        a = comp[f"A_set:{r}"]
        for B, ls in blocks.items():
            inb, mb = comp.layer.isin(ls), mats.layer.isin(ls)
            hrow.append(dict(row=r, block=B, step2=s2t.loc[s, B], step2_lo=s2t.loc[s, f"{B}_lo"], step2_hi=s2t.loc[s, f"{B}_hi"],
                             step2_within=s2t.loc[s, B] / s2t.loc[s, ["B2", "B3", "B4"]].sum(),
                             n_K8=int(comp.loc[sel[r]["K8"]].layer.isin(ls).sum()),
                             n_K16=int(comp.loc[sel[r]["K16"]].layer.isin(ls).sum()),
                             pos_attr=a[inb].clip(lower=0).sum() / a.clip(lower=0).sum(),
                             first_order_raw=mats.loc[mb, f"Q_set:{r}"].sum() / mats[f"Q_set:{r}"].sum(),
                             first_order_res=mats.loc[mb, f"Q_res:{r}"].sum() / mats[f"Q_res:{r}"].sum()))
    hdf = pd.DataFrame(hrow)
    hdf.to_csv(f"{run_dir}/pd_layers.csv", index=False)
    print("\nPhase H, blocks of the region: Step 2 block share of R_E [95%] (share within B2-B4) | K = 8 and K = 16 "
          "components in the block | block share of the positive attribution | block share of the whole update's "
          "first-order term (raw LLR; residual):")
    for x in hdf.itertuples():
        print(f"  {x.row} {x.block}: {x.step2:.2f} [{x.step2_lo:.2f}, {x.step2_hi:.2f}] ({x.step2_within:.2f}) | "
              f"{x.n_K8}, {x.n_K16} | {x.pos_attr:.2f} | {x.first_order_raw:+.2f}; {x.first_order_res:+.2f}")

    # in-sample diagnostics (selection half, which chose the sets): measured raw and residual changes against the
    # first-order predictions (summed attributions; the whole-update term for C0+dW)
    sh = llr.xs("sel", level="half")
    tsets = {"RR": ("med_referral_removal", "med"), "WI": ("med_wrong_instruction", "med"),
             "OC": ("med_overconfidence", "med"), "EM": ("ref_cartoon120", "persona")}
    ctrl = {d: pd.concat([sh.loc[f"{d}_paraphrase"], sh.loc[f"{d}_synonym_swap"]]) for d in ("med", "persona")}

    def resid(cond, s, d):
        a0, b0 = np.linalg.lstsq(np.c_[np.ones(len(ctrl[d])), ctrl[d]["C0"]], ctrl[d][cond], rcond=None)[0]
        g = sh.loc[s]
        return float((g[cond] - a0 - b0 * g["C0"]).mean())

    ins = []
    sets_in = [(f"add:{i}:{K}", sel[i][f"K{K}"]) for K in man["ks"] for i in ROWS] + \
              [(f"add:{i}R:8", sel[i]["res8"]) for i in ROWS] + [("C0+ALL", "ALL"), ("C0+dW", "dW")]
    for cond, ids in sets_in:
        for j, (s, d) in tsets.items():
            if isinstance(ids, str):
                fo, fr = ((comp[f"A_set:{j}"].sum(), comp[f"A_res:{j}"].sum()) if ids == "ALL"
                          else (mats[f"Q_set:{j}"].sum(), mats[f"Q_res:{j}"].sum()))
            else:
                fo, fr = comp.loc[ids, f"A_set:{j}"].sum(), comp.loc[ids, f"A_res:{j}"].sum()
            g = sh.loc[s]
            ins.append(dict(cond=cond, target=j, raw=float((g[cond] - g["C0"]).mean()), raw_first_order=float(fo),
                            resid=resid(cond, s, d), resid_first_order=float(fr)))
    idf = pd.DataFrame(ins)
    idf.to_csv(f"{run_dir}/pd_insample.csv", index=False)
    print("\nin-sample (selection half): measured change (first-order prediction), raw mean LLR change | residual shift "
          "with the null line fitted on the selection-half control pairs:")
    for cond, g in idf.groupby("cond", sort=False):
        print(f"  {cond}: " + "; ".join(f"{x.target} {x.raw:+.2f} ({x.raw_first_order:+.2f}) | {x.resid:+.2f} "
                                        f"({x.resid_first_order:+.2f})" for x in g.itertuples()))
    json.dump(out, open(f"{run_dir}/pd_summary.json", "w"), indent=1)


if __name__ == "__main__":
    main()
