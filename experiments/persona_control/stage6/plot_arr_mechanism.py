#!/usr/bin/env python3
"""Regenerate ARR mechanism figure from validated Stage 5B artifacts."""
from __future__ import annotations
import argparse, csv, json, hashlib
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "eval_runs/persona_control_stage6/stage5b"
RUN = ROOT / "eval_runs/persona_control_arr/round1/20260924T063649Z"
OUT = ROOT / "figures/persona_control/arr_round1"

def mechanism(path: Path) -> None:
    old_names = ("qwen2_5_7b", "llama3_1_8b", "qwen3_1_7b")
    names = {"qwen2_5_7b":"Qwen2.5-7B", "llama3_1_8b":"Llama-3.1-8B", "qwen3_1_7b":"Qwen3-1.7B"}
    old = {m: json.loads((path/m/"summary.json").read_text())["full"]["baseline"] for m in old_names}
    new = {}
    for m, d in (("qwen2_5_7b", RUN/"route_qwen/stage5b/qwen2_5_7b_training_region8_19_confirmation"),
                 ("llama3_1_8b", RUN/"route_llama/stage5b/llama3_1_8b_training_region5_22_confirmation")):
        new[m] = json.loads((d/"summary.json").read_text())["full"]["baseline"]
    series = [
        ("Qwen2.5-7B\n8–19 historical", old["qwen2_5_7b"], "#9aa0a6", "historical"),
        ("Qwen2.5-7B\n8–19 training", new["qwen2_5_7b"], "#4c78a8", "training"),
        ("Llama-3.1-8B\n9–22 historical", old["llama3_1_8b"], "#9aa0a6", "historical"),
        ("Llama-3.1-8B\n5–22 training", new["llama3_1_8b"], "#f58518", "training"),
        ("Qwen3-1.7B\n8–19 historical", old["qwen3_1_7b"], "#9aa0a6", "historical"),
    ]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12.4, 5.0), constrained_layout=True,
                                  gridspec_kw={"width_ratios": [1.55, 1]})
    x = np.arange(len(series)); width = .31
    for offset, field, label, color in ((-.18,"TE","Graft total T","#4c78a8"),
                                        (.18,"DE","Carrier-held D","#f58518")):
        vals = np.array([r[1][field]["est"] for r in series])
        lows = np.array([r[1][field]["lo"] for r in series]); highs = np.array([r[1][field]["hi"] for r in series])
        ax.bar(x+offset, vals, width, label=label, color=color, alpha=.85,
               hatch=["///" if r[3]=="historical" else "" for r in series], edgecolor="#333", linewidth=.4)
        ax.errorbar(x+offset, vals, yerr=np.vstack([vals-lows,highs-vals]), fmt="none", ecolor="#222", capsize=2, linewidth=.8)
    ax.set_xticks(x, [r[0] for r in series], fontsize=8)
    ax.set_ylabel("Score S units")
    ax.set_title("Full assay: graft total and held-surviving effect")
    ax.legend(frameon=False, loc="upper right")
    ax.axhline(0, color="#333", linewidth=.8)
    vals = np.array([r[1]["MF"]["est"] for r in series])
    lows = np.array([r[1]["MF"]["lo"] for r in series]); highs = np.array([r[1]["MF"]["hi"] for r in series])
    ax2.errorbar(x, vals, yerr=np.vstack([vals-lows,highs-vals]), fmt="o", color="#54a24b",
                 capsize=3, markersize=5)
    ax2.set_xticks(x, [r[0] for r in series], fontsize=8)
    ax2.set_ylabel("Removed fraction M/T")
    ax2.set_title("Carrier-mediated share")
    ax2.set_ylim(0, .26)
    ax2.grid(axis="y", alpha=.25)
    fig.suptitle("Stage 5B historical rendering and Round 1 training rendering shown separately", fontsize=11)
    fig.savefig(OUT/"cross_model_route_decomposition.png", dpi=180)
    fig.savefig(OUT/"cross_model_route_decomposition.pdf")
    plt.close(fig)

    strata_path = RUN/"measurement/llama_route_strata.csv"
    qwen_strata_path = RUN/"measurement/qwen_route_strata.csv"
    strata = {"llama":list(csv.DictReader(strata_path.open())), "qwen":list(csv.DictReader(qwen_strata_path.open()))}
    fig, ax = plt.subplots(figsize=(8.4, 4.9))
    scope_labels = {"full":"Full (N=120)","strict":"Strict (N=50)","nonmedical_diagnostic":"Nonmedical diagnostic (N=60)"}
    scs = tuple(scope_labels)
    yy=[]; ylabels=[]
    for model, display, color in (("qwen","Qwen2.5-7B, region 8–19","#4c78a8"),
                                  ("llama","Llama-3.1-8B, region 5–22","#f58518")):
        for j, scope in enumerate(scs):
            r=next(z for z in strata[model] if z["subset"]==scope and z["outcome"]=="removed_fraction")
            y=len(yy); val=float(r["estimate"]); lo=float(r["ci_low"]); hi=float(r["ci_high"])
            ax.errorbar(val,y,xerr=[[val-lo],[hi-val]],fmt="o",color=color,capsize=3)
            yy.append(y)
            short_model = "Qwen2.5, A 8–19" if model=="qwen" else "Llama 3.1, A 5–22"
            short_scope = {"full":"full N=120","strict":"strict N=50","nonmedical_diagnostic":"nonmedical N=60"}[scope]
            ylabels.append(f"{short_model}\n{short_scope}")
    ax.set_yticks(yy,ylabels,fontsize=8); ax.invert_yaxis(); ax.set_xlabel("Removed fraction M/T")
    ax.set_title("Training-rendering route estimates by available subset")
    ax.grid(axis="x",alpha=.25)
    fig.text(.5,.025,"Medical subset unavailable: frozen pair manifest has no explicit medical IDs.",ha="center",fontsize=8)
    fig.subplots_adjust(left=.34,bottom=.15,right=.98,top=.86)
    fig.savefig(OUT/"training_route_strata.png",dpi=180)
    fig.savefig(OUT/"training_route_strata.pdf")
    plt.close(fig)
    source_files=[path/"cross_model_summary.csv"]+[path/m/"summary.json" for m in old_names]
    qdir=RUN/"route_qwen/stage5b/qwen2_5_7b_training_region8_19_confirmation"
    ldir=RUN/"route_llama/stage5b/llama3_1_8b_training_region5_22_confirmation"
    source_files += [qdir/"summary.json",qdir/"manifest.json",qdir/"per_example_pairs.parquet",
                     ldir/"summary.json",ldir/"manifest.json",ldir/"per_example_pairs.parquet",
                     strata_path,qwen_strata_path]
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files}
    (OUT/"arr_figure_sources.json").write_text(json.dumps({"source_hashes":hashes,"command":"python experiments/persona_control/stage6/plot_arr_mechanism.py","limitations":{"historical_rendering":"Stage 5B values are retained separately from new training-rendering values","medical_stratum":"unavailable; no explicit IDs in frozen 120-pair manifest"}},indent=2)+"\n")

def schematic() -> None:
    fig, ax = plt.subplots(figsize=(10, 3.6), constrained_layout=True)
    ax.axis("off")
    boxes = [(0.03,.36,.20,.38,"Benign host C"),(.30,.36,.25,.38,"Region A graft\nGₓ = C with X[A]"),
             (.66,.58,.29,.27,"Carrier held\nH_C(G_X)"),(.66,.16,.29,.27,"Matched identity\nH_C(C)")]
    for x,y,w,h,label in boxes:
        ax.add_patch(plt.Rectangle((x,y),w,h,facecolor="#eef3f8",edgecolor="#355c7d",linewidth=1.6))
        ax.text(x+w/2,y+h/2,label,ha="center",va="center",fontsize=11)
    ax.annotate("copy donor X matrices in A",xy=(.30,.55),xytext=(.23,.55),arrowprops=dict(arrowstyle="->",lw=1.5),ha="center",va="bottom",fontsize=9)
    ax.annotate("same token sequence; replace carrier coordinates",xy=(.66,.71),xytext=(.55,.90),arrowprops=dict(arrowstyle="->",lw=1.5),ha="center",fontsize=9)
    ax.annotate("T = S(Gₓ) − S(C)",xy=(.42,.34),xytext=(.42,.12),arrowprops=dict(arrowstyle="->",lw=1.3),ha="center",fontsize=10)
    ax.annotate("D = S(H_C(G_X)) − S(H_C(C))",xy=(.81,.58),xytext=(.81,.46),arrowprops=dict(arrowstyle="->",lw=1.3),ha="center",fontsize=10)
    ax.text(.50,.02,"M = T − D by definition; this is an operational decomposition, not independent natural mediation.",ha="center",fontsize=9)
    fig.savefig(OUT/"route_decomposition_schematic.png",dpi=180)
    fig.savefig(OUT/"route_decomposition_schematic.pdf")
    plt.close(fig)

def main():
    global OUT
    ap=argparse.ArgumentParser(); ap.add_argument("--data",type=Path,default=DATA); ap.add_argument("--out",type=Path,default=OUT); a=ap.parse_args()
    OUT=a.out; OUT.mkdir(parents=True,exist_ok=True)
    mechanism(a.data); schematic()

if __name__=="__main__": main()
