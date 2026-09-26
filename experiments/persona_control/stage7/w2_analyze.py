"""Stage 7 W2 analysis: removed shares with prompt-cluster bootstrap intervals, and neutral-text checks.

Definitions (per prompt i; S_i = mean log-prob of the misaligned answer minus that of the aligned one):
  T_i = S_i(G) - S_i(C) and Delta_i = S_i(E) - S_i(C), from the capture passes of the same phase.
  Hold on host X (the identity on C):  D_i = S_i(X held) - S_i(C).
  Projection on host X (not the identity on C):  D_i = S_i(X projected) - S_i(C projected).
  Removed share over a prompt set = 1 - sum(D_i) / sum(T_i), or / sum(Delta_i) on E.
Intervals: 2,000 bootstrap draws over prompt clusters (seed 0). A difference between two
interventions uses the same draws for both. Fitted bases and paired-answer directions are
scored on the half they were not fitted on. 'Cross-fitted' pools both directions over all 120
prompts, each prompt scored with the basis fitted on the other half.
Neutral-text flags follow Stage 5B (stage5b_analyze.quality_table), with the unhooked host as reference:
  lp:        neutral log-prob < min(C, host) - 0.10
  agreement: max(top-1 agreement with C, with host) < host's agreement with C - 0.05
  entropy:   neutral entropy > max(C, host) + 0.25

Usage: python w2_analyze.py [--models qwen2_5_7b,llama3_1_8b]
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "stage6"))
from acl_common import PROMPTS  # noqa: E402
from common import ROOT, ClusterBootstrap, prompt_cluster  # noqa: E402

OUT_ROOT = os.path.join(ROOT, "eval_runs", "persona_control_stage7", "w2_hold_comparators")
KS = (1, 2, 4, 8)
SEEDS = (0, 1, 2)
N_BOOT = 2000


class Scores:
    def __init__(self, rows):
        pairs = rows[rows.kind != "neutral"]
        w = pairs.pivot_table(index=["phase", "cond", "example_id"], columns="kind", values="lp_mean")
        self.S = (w["mis"] - w["align"]).rename("S")
        self.parts = w
        self.examples = sorted(pairs.example_id.unique())

    def __call__(self, phase, cond):
        return self.S.loc[(phase, cond)]


def boot_ratio(parts, examples, complement=True, seed=0):
    """parts: list of (D, T) Series pairs. Returns the estimate and bootstrap samples of
    1 - sum D / sum T (complement=True, a removed share) or of sum D / sum T (complement=False)."""
    bs = ClusterBootstrap(examples, n_boot=N_BOOT, seed=seed)
    out = []
    for D, T in parts:
        d, t = D.reindex(examples).to_numpy(float), T.reindex(examples).to_numpy(float)
        assert np.isfinite(d).all() and np.isfinite(t).all()
        est, samples = d.sum() / t.sum(), (bs.W @ d) / (bs.W @ t)
        out.append((1 - est, 1 - samples) if complement else (est, samples))
    return out


def ci(samples):
    return ClusterBootstrap.ci(samples)


def quality(rows):
    """Stage 5B neutral-text flags for every (phase, cond)."""
    n = rows[rows.kind == "neutral"]
    q = n.groupby(["phase", "cond"]).agg(neutral_lp=("lp_mean", "mean"), neutral_ent=("ent_mean", "mean"),
                                           agree_C=("agree_C", "mean"), agree_H=("agree_H", "mean")).reset_index()
    out = []
    for phase, g in q.groupby("phase"):
        g = g.set_index("cond")
        host = "E" if phase == "E" else "G"
        c, h = g.loc["C"], g.loc[host]
        for cn, r in g.iterrows():
            out.append(dict(phase=phase, cond=cn, neutral_lp=r.neutral_lp, neutral_ent=r.neutral_ent, agree_C=r.agree_C,
                            agree_H=r.agree_H, host_reference=host, d_lp_vs_C=r.neutral_lp - c.neutral_lp,
                            d_lp_vs_host=r.neutral_lp - h.neutral_lp,
                            flag_neutral_lp=bool(r.neutral_lp < min(c.neutral_lp, h.neutral_lp) - 0.10),
                            flag_agreement=bool(max(r.agree_C, r.agree_H) < h.agree_C - 0.05),
                            flag_entropy=bool(r.neutral_ent > max(c.neutral_ent, h.neutral_ent) + 0.25)))
    q = pd.DataFrame(out)
    q["destructive"] = q[["flag_neutral_lp", "flag_agreement", "flag_entropy"]].any(axis=1)
    return q


def analyze(model, run):
    rows = pd.read_parquet(os.path.join(run, "rows.parquet"))
    man = json.load(open(os.path.join(run, "manifest.json")))
    sc = Scores(rows)
    ex = sc.examples
    halves = {h: sorted(v) for h, v in man["halves"].items()}
    other = {"d": "e", "e": "d"}
    cl_half = {}
    for h, ids in halves.items():
        for e in ids:
            cl_half.setdefault(prompt_cluster(e), set()).add(h)
    straddle = sorted(e for e in ex if len(cl_half[prompt_cluster(e)]) > 1)
    acl_ids = {p["prompt_id"] for p in PROMPTS}
    acl_overlap = sorted(e for e in ex if prompt_cluster(e) in acl_ids)
    subsets = {"all": ex, "no_straddling_clusters": [e for e in ex if e not in straddle],
               "no_acl_step1_questions": [e for e in ex if e not in acl_overlap]}

    C = {ph: sc(ph, "C") for ph in ("REF", "E", "G")}
    T = sc("G", "G") - C["G"]
    T_ref = sc("REF", "G") - C["REF"]
    Delta = sc("E", "E") - C["E"]
    persona_D = sc("REF", "G|persona") - C["REF"]
    # The same passes are repeated in each phase; they must agree.
    checks = dict(C_rows_equal_across_phases=float(max((C["G"] - C["REF"]).abs().max(), (C["E"] - C["REF"]).abs().max())),
                  G_rows_equal_REF_vs_G=float((sc("G", "G") - sc("REF", "G")).abs().max()),
                  n_straddling_examples=len(straddle), straddling_examples=straddle,
                  n_acl_overlap_examples=len(acl_overlap))

    def hold_D(ph, cond):
        return sc(ph, cond) - C[ph]

    res = []

    def add(task, cond, D, base, examples, subset, refs=None, complement=True, **meta):
        """One estimate with its interval, plus paired differences against each comparator in refs."""
        if len(examples) < 2:  # only in quick tests, where a subset can be empty
            return
        refs = refs or {}
        out = boot_ratio([(D, base)] + [(R, base) for R in refs.values()], examples, complement=complement)
        est, samples = out[0]
        lo, hi = ci(samples)
        r = dict(model=model, task=task, cond=cond, subset=subset, n_examples=len(examples),
                 estimate=float(est), lo=lo, hi=hi, **meta)
        for name, (e2, s2) in zip(refs, out[1:]):
            dlo, dhi = ci(samples - s2)
            r.update({f"diff_vs_{name}": float(est - e2), f"diff_vs_{name}_lo": dlo, f"diff_vs_{name}_hi": dhi})
        res.append(r)

    # Reference and Task 1 --------------------------------------------------------------
    for subset, exs in subsets.items():
        add("reference", "G|persona (frozen Stage 6 carrier, layer %d)" % man["spec"]["carrier_layer"], persona_D, T_ref, exs, subset)
    for h in ("d", "e"):
        add("reference", "G|persona", persona_D, T_ref, halves[h], f"half_{h}")
    energy = man["task1"]["energy"]
    for ver in ("a", "b"):
        for k in KS:
            for j in ("", "+P"):
                label = f"t1{ver}_k{k}{j}"
                for h in ("d", "e"):
                    name = f"t1{ver}_{h}_k{k}{j}"
                    en = energy[f"fit_{h}_score_{other[h]}"][name]
                    add("task1", label, hold_D("G", f"G|{name}"), T, halves[other[h]], f"fit_{h}_score_{other[h]}", refs=dict(persona=persona_D),
                        version=ver, k=k, with_persona=bool(j), energy_fraction=en[other[h]], rank=en["rank"])
                    add("task1", label, hold_D("G", f"G|{name}"), T, halves[h], f"in_sample_{h}", refs=dict(persona=persona_D),
                        version=ver, k=k, with_persona=bool(j), energy_fraction=en[h], rank=en["rank"])
                cf = pd.concat([hold_D("G", f"G|t1{ver}_{h}_k{k}{j}").reindex(halves[other[h]]) for h in ("d", "e")])
                for subset in ("all", "no_straddling_clusters"):
                    add("task1", label, cf, T, subsets[subset], f"cross_fitted_{subset}", refs=dict(persona=persona_D),
                        version=ver, k=k, with_persona=bool(j))
    for k in KS:
        for s in SEEDS:
            for j in ("", "+P"):
                name = f"rand{k}_s{s}{j}"
                add("task1", f"rand{k}{j}", hold_D("G", f"G|{name}"), T, ex, "all", refs=dict(persona=persona_D), version="random", k=k,
                    with_persona=bool(j), seed=s, energy_fraction=float(np.mean(list(energy["random"][name].values()))))
        for j in ("", "+P"):
            Dm = sum(hold_D("G", f"G|rand{k}_s{s}{j}") for s in SEEDS) / len(SEEDS)
            add("task1", f"rand{k}{j}", Dm, T, ex, "all", refs=dict(persona=persona_D), version="random", k=k,
                with_persona=bool(j), seed="mean of 3")

    # Task 2 --------------------------------------------------------------------------
    for name in ("sol", "solm"):
        for j in ("", "+P"):
            for subset in ("all", "no_acl_step1_questions"):
                add("task2", f"G|hold_{name}{j}", hold_D("G", f"G|hold_{name}{j}"), T, subsets[subset], subset, refs=dict(persona=persona_D))
    for j in ("", "+P"):
        cf = pd.concat([hold_D("G", f"G|hold_pair_{h}{j}").reindex(halves[other[h]]) for h in ("d", "e")])
        add("task2", f"G|hold_pair{j}", cf, T, ex, "cross_fitted_all", refs=dict(persona=persona_D))
    abl = ["sol_single", "sol_perlayer", "solm_single"] + [f"rand1_s{s}_single" for s in SEEDS]
    for a in abl:
        D = sc("G", f"G|ablate_{a}") - sc("G", f"C|ablate_{a}")
        for subset in ("all", "no_acl_step1_questions"):
            add("task2", f"ablate_{a}", D, T, subsets[subset], subset, refs=dict(persona=persona_D))
        # The projection also moves C and G themselves; report both shifts as a share of T.
        add("task2", f"ablate_{a}: drop in S(G) / T", sc("G", "G") - sc("G", f"G|ablate_{a}"), T, ex, "all", complement=False)
        add("task2", f"ablate_{a}: shift of S(C) / T", sc("G", f"C|ablate_{a}") - C["G"], T, ex, "all", complement=False)
    Dm = sum(sc("G", f"G|ablate_rand1_s{s}_single") - sc("G", f"C|ablate_rand1_s{s}_single") for s in SEEDS) / len(SEEDS)
    add("task2", "ablate_rand1_single (mean of 3)", Dm, T, ex, "all", refs=dict(persona=persona_D))
    Dpair = pd.concat([(sc("G", f"G|ablate_pair_{h}_single") - sc("G", f"C|ablate_pair_{h}_single")).reindex(halves[other[h]])
                       for h in ("d", "e")])
    add("task2", "ablate_pair_single", Dpair, T, ex, "cross_fitted_all", refs=dict(persona=persona_D))
    # E host: removed share of Delta.
    persona_E = hold_D("E", "E|persona")
    add("task2_E", "E|persona", persona_E, Delta, ex, "all")
    for name in ("sol", "solm", "sol+P"):
        add("task2_E", f"E|hold_{name}", hold_D("E", f"E|hold_{name}"), Delta, ex, "all", refs=dict(persona=persona_E))
    for a in ("sol_single", "sol_perlayer", "solm_single"):
        add("task2_E", f"E|ablate_{a}", sc("E", f"E|ablate_{a}") - sc("G", f"C|ablate_{a}"), Delta, ex, "all", refs=dict(persona=persona_E))
    cf = pd.concat([hold_D("E", f"E|hold_pair_{h}").reindex(halves[other[h]]) for h in ("d", "e")])
    add("task2_E", "E|hold_pair", cf, Delta, ex, "cross_fitted_all", refs=dict(persona=persona_E))
    cf = pd.concat([(sc("E", f"E|ablate_pair_{h}_single") - sc("G", f"C|ablate_pair_{h}_single")).reindex(halves[other[h]])
                    for h in ("d", "e")])
    add("task2_E", "E|ablate_pair_single", cf, Delta, ex, "cross_fitted_all", refs=dict(persona=persona_E))

    # Task 3 --------------------------------------------------------------------------
    single_G, single_E = hold_D("G", "G|t3_single_nested"), hold_D("E", "E|t3_single_nested")
    for sname in man["layer_sets"]:
        for sub in ("nested", "style", "rand4_s0", "rand4_s1", "rand4_s2"):
            add("task3", f"G|t3_{sname}_{sub}", hold_D("G", f"G|t3_{sname}_{sub}"), T, ex, "all",
                refs=dict(persona=persona_D, single_rebuilt=single_G), layer_set=sname, subspace=sub, layers=man["layer_sets"][sname])
        Dm = sum(hold_D("G", f"G|t3_{sname}_rand4_s{s}") for s in SEEDS) / len(SEEDS)
        add("task3", f"G|t3_{sname}_rand4 (mean of 3)", Dm, T, ex, "all", refs=dict(persona=persona_D, single_rebuilt=single_G),
            layer_set=sname, subspace="rand4 mean", layers=man["layer_sets"][sname])
        add("task3_E", f"E|t3_{sname}_nested", hold_D("E", f"E|t3_{sname}_nested"), Delta, ex, "all",
            refs=dict(persona=persona_E, single_rebuilt=single_E), layer_set=sname, subspace="nested", layers=man["layer_sets"][sname])

    table = pd.DataFrame(res)
    q = quality(rows)
    base = dict(T=float(T.mean()), Delta=float(Delta.mean()), S_C=float(C["G"].mean()), S_G=float(sc("G", "G").mean()),
                S_E=float(sc("E", "E").mean()))
    return table, q, dict(checks=checks, baseline=base, manifest_checks=man["checks"], directions=man["directions"],
                          task1_spectrum=man["task1"]["spectrum_top32"], task1_tokens=man["task1"]["tokens"])


# --------------------------------------------------------------------------------------
# Figures (palette: the dataviz reference categorical slots 1-3, validated all-pairs in light mode)
# --------------------------------------------------------------------------------------

FIG_DIR = os.path.join(ROOT, "figures", "persona_control", "stage7", "w2_hold_comparators")
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
LABELS = {"qwen2_5_7b": "Qwen2.5-7B", "llama3_1_8b": "Llama-3.1-8B", "qwen3_1_7b": "Qwen3-1.7B"}


def _style(ax):
    ax.set_facecolor(SURFACE)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def figures(table, fig_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(fig_dir, exist_ok=True)
    models = [m for m in LABELS if m in set(table.model)]
    t = table.copy()
    t["seed"] = t.get("seed", pd.Series(index=t.index, dtype=object)).astype(str)

    # Figure 1: Task 1, removed share against rank k (cross-fitted, all prompts).
    fig, axes = plt.subplots(1, len(models), figsize=(max(3.4 * len(models), 5.0), 3.8), sharey=True, squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    series = [("a", False, "Top-k of h_G - h_C, held alone", BLUE),
              ("b", True, "Persona carrier + top-k after the hold", ORANGE),
              ("random", False, "Random rank-k, held alone (mean of 3)", AQUA)]
    for ax, m in zip(axes[0], models):
        _style(ax)
        tm = t[(t.model == m) & (t.task == "task1")]
        ref = t[(t.model == m) & (t.task == "reference") & (t.subset == "all")].iloc[0]
        ax.axhline(ref.estimate, color=INK2, linewidth=1)
        ax.text(0.02, ref.estimate + 0.012, f"persona hold, rank 4: {ref.estimate:.2f}", color=INK2, fontsize=7.5,
                transform=ax.get_yaxis_transform())
        for (ver, wp, label, color), dx in zip(series, (-0.07, 0.0, 0.07)):
            if ver == "random":
                r = tm[(tm.version == "random") & (tm.with_persona == wp) & (tm.seed == "mean of 3")]
            else:
                r = tm[(tm.version == ver) & (tm.with_persona == wp) & (tm.subset == "cross_fitted_all")]
            r = r.sort_values("k")
            x = np.log2(r.k.astype(float)) + dx
            ax.errorbar(x, r.estimate, yerr=[r.estimate - r.lo, r.hi - r.estimate], color=color, linewidth=2,
                        elinewidth=1, capsize=0, marker="o", markersize=6, markeredgecolor=SURFACE, markeredgewidth=1.5,
                        label=label)
        ax.set_xticks(np.log2(KS))
        ax.set_xticklabels([str(k) for k in KS])
        ax.set_xlabel("rank k", color=INK2, fontsize=8)
        ax.set_title(LABELS[m], color=INK, fontsize=9, loc="left")
    axes[0][0].set_ylabel("removed share of T", color=INK2, fontsize=8)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=1, frameon=False, fontsize=7.5, labelcolor=INK)
    fig.tight_layout(rect=(0, 0.2, 1, 1))
    fig.savefig(os.path.join(fig_dir, "task1_rank_k_holds.png"), dpi=200, facecolor=SURFACE)
    plt.close(fig)

    # Figure 2: every intervention against the persona hold (all prompts; cross-fitted where fitted).
    rows = [("reference", "all", "G|persona (frozen", "Persona hold, carrier layer (frozen carrier)", BLUE),
            ("task3", "all", "G|t3_single_nested", "Persona hold, carrier layer (rebuilt carrier)", BLUE),
            ("task3", "all", "G|t3_every4_nested", "Persona hold, every 4th layer", BLUE),
            ("task3", "all", "G|t3_every_nested", "Persona hold, every layer", BLUE),
            ("task1", "cross_fitted_all", "t1a_k4", "Top-4 of h_G - h_C, carrier layer", ORANGE),
            ("task2", "all", "G|hold_sol", "Mean-difference direction, hold at carrier layer", ORANGE),
            ("task2", "all", "ablate_sol_single", "Mean-difference direction, projected out everywhere", ORANGE),
            ("task2", "all", "ablate_sol_perlayer", "Per-layer mean-difference directions, projected out", ORANGE),
            ("task2", "cross_fitted_all", "G|hold_pair", "Paired-answer direction, hold at carrier layer", ORANGE),
            ("task2", "cross_fitted_all", "ablate_pair_single", "Paired-answer direction, projected out everywhere", ORANGE),
            ("task3", "all", "G|t3_every_style", "Style subspace hold, every layer", AQUA),
            ("task3", "all", "G|t3_every_rand4 (mean of 3)", "Random rank-4 hold, every layer (mean of 3)", AQUA),
            ("task2", "all", "ablate_rand1_single (mean of 3)", "Random direction projected out (mean of 3)", AQUA)]
    fig, axes = plt.subplots(1, len(models), figsize=(2.6 * len(models) + 3.2, 4.4), sharey=True, squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    y = np.arange(len(rows))[::-1]
    for ax, m in zip(axes[0], models):
        _style(ax)
        ax.axvline(0, color=INK2, linewidth=0.8)
        for yi, (task, subset, key, label, color) in zip(y, rows):
            sel = t[(t.model == m) & (t.task == task) & (t.subset == subset) & (t.cond.str.startswith(key) if key.endswith("(frozen") else t.cond == key)]
            if sel.empty:
                continue
            r = sel.iloc[0]
            ax.plot([r.lo, r.hi], [yi, yi], color=color, linewidth=1.5, solid_capstyle="round")
            ax.plot(r.estimate, yi, "o", color=color, markersize=6, markeredgecolor=SURFACE, markeredgewidth=1.5)
        ax.set_title(LABELS[m], color=INK, fontsize=9, loc="left")
        ax.set_xlabel("removed share of T", color=INK2, fontsize=8)
        ax.grid(False, axis="y")
    axes[0][0].set_yticks(y)
    axes[0][0].set_yticklabels([r[3] for r in rows], fontsize=7.5, color=INK)
    from matplotlib.lines import Line2D
    keys = [Line2D([], [], color=c, marker="o", linewidth=1.5, markersize=6) for c in (BLUE, ORANGE, AQUA)]
    fig.legend(keys, ["persona carrier", "comparison directions", "controls"], loc="lower center", ncol=3, frameon=False,
               fontsize=7.5, labelcolor=INK)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(os.path.join(fig_dir, "comparators_vs_persona_hold.png"), dpi=200, facecolor=SURFACE)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="qwen2_5_7b,llama3_1_8b,qwen3_1_7b")
    ap.add_argument("--run-dir", default="", help="analyze one run folder (e.g. a quick test)")
    ap.add_argument("--out", default="", help="output folder (default: <run-dir>/analysis, or the W2 analysis folder)")
    ap.add_argument("--fig-dir", default="", help="figure folder (default: figures/persona_control/stage7/w2_hold_comparators)")
    args = ap.parse_args()
    if args.run_dir:
        runs = {json.load(open(os.path.join(args.run_dir, "manifest.json")))["model"]: args.run_dir}
        out = args.out or os.path.join(args.run_dir, "analysis")
    else:
        runs = {m: os.path.join(OUT_ROOT, m) for m in args.models.split(",")}
        out = args.out or os.path.join(OUT_ROOT, "analysis")
    if os.path.isdir(out) and os.listdir(out):
        raise FileExistsError(f"analysis output exists, not overwriting: {out}")
    os.makedirs(out, exist_ok=True)
    tables, quals, summary = [], [], {}
    for m, run in runs.items():
        if not os.path.exists(os.path.join(run, "rows.parquet")):
            print(f"skip {m}: no rows in {run}")
            continue
        t, q, s = analyze(m, run)
        q.insert(0, "model", m)
        tables.append(t)
        quals.append(q)
        summary[m] = s
        print(f"{m}: {len(t)} estimates, {int(q.destructive.sum())} flagged conditions")
    pd.concat(tables).to_csv(os.path.join(out, "removed_shares.csv"), index=False)
    pd.concat(quals).to_csv(os.path.join(out, "neutral_quality.csv"), index=False)
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)
    fig_dir = args.fig_dir or (os.path.join(out, "figures") if args.run_dir else FIG_DIR)
    figures(pd.concat(tables), fig_dir)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
