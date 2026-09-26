"""Stage 7 W6 figures: effect fraction F versus rank r of low-rank edits of the final E-C update.

Reads rank_curve.csv (written by w6_analyze.py) from the W6 run folders and writes, into
figures/persona_control/stage7/w6_compactness/:
  hindsight_rank_curves.png/.csv   seed-42 hindsight curves, one column per model: learned top-r edit
                                   (95% paired cluster-bootstrap band), energy-matched random and
                                   shuffled-direction controls (mean over seeds, band = seed min-max),
                                   and the share of update energy the top-r edit keeps (same axis:
                                   both are fractions of the full update).
  prospective_crossseed_<model>.png/.csv   one per model with any of: seed-43 host with subspaces from
                                   steps 16 and 64 of the same run (Qwen2.5-7B and Qwen3-1.7B only, the
                                   two models with step snapshots), and each seed's host with the other
                                   seed's final subspaces (all three models). A panel is drawn only if
                                   its run folder has been analyzed; a model with none is skipped.
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
RUNS = os.path.join(ROOT, "eval_runs", "persona_control_stage7", "w6_compactness")
OUT = os.path.join(ROOT, "figures", "persona_control", "stage7", "w6_compactness")
MODELS = [("qwen2_5_7b", "Qwen2.5-7B"), ("llama3_1_8b", "Llama-3.1-8B"), ("qwen3_1_7b", "Qwen3-1.7B")]
RANKS = [1, 2, 4, 8, 16, 32, 64]

# Reference palette (dataviz skill, light mode), fixed slot order: blue, orange, aqua, yellow.
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
COLOR = {"learned": "#2a78d6", "random_energy": "#eb6834", "shuffled": "#1baf7a",
         "prosp_t16": "#eb6834", "prosp_t64": "#1baf7a", "cross": "#eda100", "control": MUTED, "energy": INK2}
MARKER = {"learned": "o", "random_energy": "s", "shuffled": "^", "prosp_t16": "s", "prosp_t64": "^", "cross": "D",
          "control": "v", "energy": None}
YLABEL = {"F_direct": "F_direct (persona carrier held)", "F_total": "F_total (nothing held)"}
plt.rcParams.update({"font.family": "sans-serif", "font.size": 9, "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
                     "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK, "axes.titlecolor": INK,
                     "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE})


def load(path):
    p = os.path.join(path, "rank_curve.csv")
    return pd.read_csv(p) if os.path.exists(p) else None


def family(curve, kind, col):
    """Mean over seeds per rank, with the band: bootstrap CI for single-seed families, seed min-max otherwise."""
    g = curve[curve.kind == kind].groupby("rank")
    if g[col].count().max() == 1:
        d = g.agg(y=(col, "mean"), lo=(col + "_lo", "mean"), hi=(col + "_hi", "mean"))
    else:
        d = g.agg(y=(col, "mean"), lo=(col, "min"), hi=(col, "max"))
    return d.reset_index()


def style_axis(ax, ylabel):
    ax.set_xscale("log", base=2)
    ax.set_xticks(RANKS)
    ax.set_xticklabels([str(r) for r in RANKS])
    ax.set_xlim(0.8, 80)
    ax.set_ylim(-0.12, 1.15)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.axhline(0, color=AXIS, linewidth=0.8, zorder=1)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0)
    if ylabel:
        ax.set_ylabel(ylabel)


def draw(ax, d, key, label, band_alpha=0.14):
    c = COLOR[key]
    if "lo" in d:
        ax.fill_between(d["rank"], d["lo"], d["hi"], color=c, alpha=band_alpha, linewidth=0, zorder=2)
    ax.plot(d["rank"], d["y"], color=c, linewidth=1.8, marker=MARKER[key], markersize=5, zorder=3, label=label,
            markeredgecolor=SURFACE, markeredgewidth=0.8)


def legend_handles(axes):
    handles = {}
    for ax in axes.ravel():
        for h, lab in zip(*ax.get_legend_handles_labels()):
            handles.setdefault(lab, h)
    return handles


def hindsight_figure(runs):
    rows, fig, axes = [], *plt.subplots(2, 3, figsize=(11, 6.2), sharex=True, sharey=True)
    for j, (key, name) in enumerate(MODELS):
        curve = load(os.path.join(runs, key, "hindsight_seed42"))
        for i, col in enumerate(("F_direct", "F_total")):
            ax = axes[i, j]
            style_axis(ax, YLABEL[col] if j == 0 else None)
            if i == 0:
                ax.set_title(name, fontsize=10, loc="left")
            if curve is None:
                ax.text(4, 0.5, "not run", color=MUTED)
                continue
            en = curve[curve.kind == "learned"].sort_values("rank")
            ax.plot(en["rank"], en["edit_energy_over_update_energy"], color=COLOR["energy"], linewidth=1.2, zorder=3,
                    label="share of update energy in the top-r edit")
            for kind, label in (("learned", "top-r singular edit (learned)"),
                                ("random_energy", "random directions, equal scales"),
                                ("shuffled", "random directions, learned scales")):
                d = family(curve, kind, col)
                draw(ax, d, kind, label)
                for _, r in d.iterrows():
                    rows.append(dict(model=key, measure=col, series=kind, rank=int(r["rank"]), value=r["y"],
                                     band_lo=r["lo"], band_hi=r["hi"]))
            for _, r in en.iterrows():
                rows.append(dict(model=key, measure="energy_share", series="learned", rank=int(r["rank"]),
                                 value=r["edit_energy_over_update_energy"], band_lo=None, band_hi=None))
            if j == 2:  # selective direct labels at the right end
                last = family(curve, "learned", col).iloc[-1]
                ax.annotate("learned", (last["rank"], last["y"]), xytext=(4, 6), textcoords="offset points",
                            color=INK2, fontsize=8)
                ax.annotate("energy share", (en["rank"].iloc[-1], en["edit_energy_over_update_energy"].iloc[-1]),
                            xytext=(-10, 8), textcoords="offset points", color=INK2, fontsize=8, ha="right")
    finish(fig, axes, "Rank-r edits of the final harmful-minus-benign update (seed-42 runs): effect kept versus rank",
           "rank r per matrix (7 matrices in each region layer)")
    return fig, pd.DataFrame(rows)


def projection_figure(runs, key, name):
    """One figure per model: same-run prospective panels (only for models with step snapshots) and
    both cross-seed directions (all three models), each drawn only if its run folder was analyzed."""
    q = os.path.join(runs, key)
    panels = [("prospective_seed43", "Seed-43 run: subspaces of\nits step-16 and step-64 updates",
               [("prosp_t16", "prosp_t16", "subspaces of the step-16 update (same run)"),
                ("prosp_t64", "prosp_t64", "subspaces of the step-64 update (same run)")]),
              ("crossseed_host43_donor42", "Seed-43 run: subspaces of\nthe seed-42 final update",
               [("cross_seed42", "cross", "subspaces of the other seed's final update")]),
              ("crossseed_host42_donor43", "Seed-42 run: subspaces of\nthe seed-43 final update",
               [("cross_seed43", "cross", "subspaces of the other seed's final update")])]
    present = [(d, t, s_) for d, t, s_ in panels if load(os.path.join(q, d)) is not None]
    if not present:
        return None, None
    n = len(present)
    rows, fig, axes = [], *plt.subplots(2, n, figsize=(3.7 * n + 0.6, 6.4), sharex=True, sharey=True, squeeze=False)
    for j, (d, title, series) in enumerate(present):
        curve = load(os.path.join(q, d))
        fams = [("learned", "learned", "top-r singular edit of the host's own update (hindsight)")] + series + \
               [("randproj", "control", "random subspaces (mean and range of 5 seeds)")]
        for i, col in enumerate(("F_direct", "F_total")):
            ax = axes[i, j]
            style_axis(ax, YLABEL[col] if j == 0 else None)
            if i == 0:
                ax.set_title(title, fontsize=9.5, loc="left")
            for kind, key2, label in fams:
                if kind not in set(curve.kind):
                    continue
                f = family(curve, kind, col)
                draw(ax, f, key2, label)
                for _, r in f.iterrows():
                    rows.append(dict(run=d, measure=col, series=kind, rank=int(r["rank"]), value=r["y"],
                                     band_lo=r["lo"], band_hi=r["hi"]))
    finish(fig, axes, f"{name}: projecting the final update onto subspaces not taken from it", "rank r per matrix")
    return fig, pd.DataFrame(rows)


def finish(fig, axes, title, xlabel):
    """Title, legend under the title, shared x label; sizes in inches so the layout holds at any height."""
    handles = legend_handles(axes)
    h = fig.get_figheight()
    n_rows = (len(handles) + 1) // 2
    fig.suptitle(title, x=0.01, ha="left", y=1 - 0.08 / h, va="top", fontsize=11)
    fig.supxlabel(xlabel, color=INK2, fontsize=9)
    top = 1 - (0.42 + 0.22 * n_rows + 0.12) / h
    fig.tight_layout(rect=(0, 0, 1, top))
    fig.legend(list(handles.values()), list(handles), loc="upper left", ncol=2, frameon=False, fontsize=8.5,
               bbox_to_anchor=(0.01, 1 - 0.40 / h))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="overwrite existing figures")
    ap.add_argument("--runs", default=RUNS)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    jobs = [("hindsight_rank_curves", lambda: hindsight_figure(args.runs))]
    for key, name in MODELS:
        jobs.append((f"prospective_crossseed_{key}", lambda key=key, name=name: projection_figure(args.runs, key, name)))
    for name, make in jobs:
        png = os.path.join(args.out, name + ".png")
        if os.path.exists(png) and not args.force:
            print(f"exists, skipped: {png}")
            continue
        fig, table = make()
        if fig is None:
            print(f"no data for {name}")
            continue
        fig.savefig(png, dpi=200, bbox_inches="tight")
        table.to_csv(os.path.join(args.out, name + ".csv"), index=False)
        print(f"wrote {png}")


if __name__ == "__main__":
    main()
