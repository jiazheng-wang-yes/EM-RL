"""Stage 9 figures (CPU) from the report tables of Steps 3 and 4.

  step3  the K = 256 transfer and removal maps (step3_map.csv): cell color = entry / E0's own residual shift of the
         column behavior, text = entry in nats; a box marks entries whose selectivity intervals against both random sets
         lie above 0. Plus the persona part P against the frozen-assay residual shift R_EM for every scored condition
         (step3_persona_share.csv).
  step4  misaligned rate and coherent share per condition with question-bootstrap intervals (s4_rates.csv).
Output: figures/persona_control/stage9/.
Usage: python s9_plots.py step3 [--run s9s3_20260928];  python s9_plots.py step4 [--run s9s4_20260929]
The model comes from the run's manifest; Llama-3.1-8B figures get the suffix _llama. --paper drops the plot titles
(the paper's captions carry them) and writes <name>_paper.pdf only.
"""
import argparse
import json
import os

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
OUT = f"{ROOT}/figures/persona_control/stage9"
ROWS = ["RR", "WI", "OC", "TQA", "EM", "MD", "FLAT", "EMP", "EMX", "UNION"]
COLS = ["RR", "WI", "OC", "TQA", "EM", "MD", "P_EM"]
MODELS = {"qwen2_5_7b": ("Qwen2.5-7B", ""), "llama3_1_8b": ("Llama-3.1-8B", "_llama")}  # title, file suffix
PAPER = False  # set by --paper
NAMES = {"RR": "referral\nremoval", "WI": "wrong\ninstruction", "OC": "over-\nconfidence", "TQA": "TruthfulQA\nerrors",
         "EM": "frozen\nassay", "MD": "training\nanswers", "P_EM": "persona\npart"}


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    for ext in (("pdf",) if PAPER else ("png", "pdf")):
        fig.savefig(f"{OUT}/{name}{'_paper' if PAPER else ''}.{ext}", dpi=200, bbox_inches="tight")
    print(f"{OUT}/{name}{'_paper.pdf' if PAPER else '.png'}")


def step3(run):
    d = f"{ROOT}/eval_runs/persona_control_stage9/step3/{run}"
    mname, suf = MODELS[json.load(open(f"{d}/manifest.json")).get("model", "qwen2_5_7b")]
    m = pd.read_csv(f"{d}/step3_map.csv")
    cond = pd.read_csv(f"{d}/step3_conditions.csv").set_index("cond")
    graft = next(c for c in cond.index if c[0] == "G" and c[1:].isdigit())
    ref = {c: cond.loc["E0", c] for c in COLS}
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True)
    for ax, (dr, title) in zip(axes, (("T", "Transfer: C + set"), ("D", "Removal: E − set"))):
        t = m[(m.K == 256) & (m.dir == dr)].set_index(["row", "col"])
        rel = np.array([[t.loc[(r, c), "value"] / ref[c] for c in COLS] for r in ROWS])
        ax.imshow(rel, cmap="RdBu_r", vmin=-1.2, vmax=1.2, aspect="auto")
        for i, r in enumerate(ROWS):
            for j, c in enumerate(COLS):
                x = t.loc[(r, c)]
                ax.text(j, i, f"{x.value:+.1f}" if abs(x.value) >= 10 else f"{x.value:+.2f}", ha="center", va="center",
                        fontsize=7.5, color="white" if abs(rel[i, j]) > .75 else "black")
                if x.sel_lo > 0 and x.sel_s_lo > 0:
                    ax.add_patch(plt.Rectangle((j - .47, i - .47), .94, .94, fill=False, lw=1.3, ec="black"))
        ax.set_xticks(range(len(COLS)), [NAMES[c] for c in COLS], fontsize=8)
        ax.set_yticks(range(len(ROWS)), [f"{r} ({int(t.loc[(r, 'EM'), 'n_comp'])})" for r in ROWS], fontsize=8.5)
        ax.set_title(title, fontsize=10)
    axes[0].set_ylabel("component set (number of components)")
    sm = plt.cm.ScalarMappable(cmap="RdBu_r", norm=plt.Normalize(-1.2, 1.2))
    fig.colorbar(sm, ax=axes, fraction=.025, pad=.01, label="entry / E0's residual shift")
    if not PAPER:
        fig.suptitle(f"{mname}, K = 256: residual shift (nats) moved by each component set; box = above both random sets",
                     fontsize=10)
    save(fig, f"step3_maps_k256{suf}")

    s = pd.read_csv(f"{d}/step3_persona_share.csv")
    s = s[s.cond.str.contains(":256") | s.cond.isin(["E0", "A", "ALL", graft])]
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    kinds = s.cond.map(lambda c: "" if ":" not in c else "~s" if c.split(":")[1].endswith("~s")
                       else "~" if c.split(":")[1].endswith("~") else "sel")
    for kind, mk, col in (("sel", "o", "tab:red"), ("~", "s", "tab:gray"), ("~s", "^", "tab:gray")):
        g = s[kinds == kind]
        ax.scatter(g.R_EM, g.P, marker=mk, c=col, s=22, alpha=.8,
                   label={"sel": "selected sets", "~": "location-matched random", "~s": "size-matched random"}[kind])
    ref = s[s.cond.isin(["E0", "A", "ALL", graft])]
    ax.scatter(ref.R_EM, ref.P, marker="*", c="black", s=70, label=f"E0, A, ALL, {graft}")
    a = s[s.cond == "A"].iloc[0]
    xs = np.linspace(min(0, s.R_EM.min()), s.R_EM.max() * 1.05, 10)
    ax.plot(xs, xs * a.share, "k--", lw=.8, label=f"share of the whole update ({a.share:.2f})")
    ax.set_xlabel("frozen-assay residual shift moved, R_EM (nats)")
    ax.set_ylabel("persona part P (nats)")
    ax.legend(fontsize=7, loc="upper left")
    if not PAPER:
        ax.set_title(f"{mname}: persona part is a fixed share of every set's effect", fontsize=9)
    save(fig, f"step3_persona_share{suf}")


def step4(run):
    d = f"{ROOT}/eval_runs/persona_control_stage9/step4/{run}"
    mname, suf = MODELS[json.load(open(f"{d}/gen_manifest.json")).get("model", "qwen2_5_7b")]
    r = pd.read_csv(f"{d}/s4_rates.csv")
    conds = list(dict.fromkeys(r.condition))
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, (o, lab) in zip(axes, (("panel_em", "misaligned rate (both judges)"), ("coherent", "coherent share"))):
        g = r[r.outcome == o].set_index("condition").reindex(conds)
        colors = ["tab:blue" if c == "C0" else "tab:red" if c == "E0" else "tab:orange" if c.startswith("D:")
                  else "tab:green" for c in conds]
        ax.bar(range(len(conds)), g.estimate, color=colors, alpha=.8)
        ax.errorbar(range(len(conds)), g.estimate, yerr=[g.estimate - g.lo, g.hi - g.estimate], fmt="none", ecolor="k",
                    lw=.8, capsize=2)
        ax.set_xticks(range(len(conds)), conds, rotation=60, ha="right", fontsize=7.5)
        ax.set_ylabel(lab)
    if not PAPER:
        fig.suptitle(f"{mname}, 16 questions x 30 samples per condition; 95% question-bootstrap intervals", fontsize=9)
    save(fig, f"step4_rates{suf}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["step3", "step4"])
    ap.add_argument("--run", default="")
    ap.add_argument("--paper", action="store_true")
    a = ap.parse_args()
    PAPER = a.paper
    {"step3": step3, "step4": step4}[a.what](a.run or {"step3": "s9s3_20260928", "step4": "s9s4_20260929"}[a.what])
