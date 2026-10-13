#!/usr/bin/env python3
"""Plot retrospective learning curves and their measurement limitations."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "figures/countdown_code/paper_claim_audit"
audit = json.loads((ROOT / "eval_runs/paper_claim_audit/audit.json").read_text())
manifest = json.loads((ROOT / "eval_runs/minimal_finance_dose/dose_manifest.json").read_text())
paths = {a["arm_id"]: Path(a["rollout_dir"]).name for a in manifest["arms"]}
cache = list(csv.DictReader((ROOT / "eval_runs/minimal_finance_dose/step_curves.csv").open()))
fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), constrained_layout=True)
for ax, model in zip(axes[:2], ("Qwen2.5-3B-Instruct", "Qwen3-1.7B")):
    for event, color, label in ((False, "#277DA1", "No sustained onset"), (True, "#D1495B", "Sustained onset")):
        rows = [r for r in audit["arms"] if r["model"] == model and r["sft_steps"] and r["complete_100"]
                and (r["onset_10_two_steps"] is not None) == event]
        ax.scatter([100*r["early20_honest"] for r in rows], [100*r["late20_hacking"] for r in rows],
                   s=38, color=color, alpha=.8, label=label)
    ax.set(title=model, xlabel="Honest success, steps 1–20 (%)", ylabel="Rewarded incorrect, steps 81–100 (%)", ylim=(-3, 103))
axes[0].legend(fontsize=8)
for arm, label, color in (
    ("q3_1_7b__n1952_e2_s1", "1,952 rows × 2 epochs, seed 1", "#D1495B"),
    ("q3_1_7b__n1952_e3_s1", "1,952 rows × 3 epochs, seed 1", "#277DA1"),
):
    rows = sorted((r for r in cache if r["run"] == paths[arm] and int(r["step"]) <= 100), key=lambda r:int(r["step"]))
    axes[2].plot([int(r["step"]) for r in rows], [100*float(r["cheating_rate"]) for r in rows], color=color, label=label)
axes[2].annotate("Rewarded return-True bypass\nalready present at step 13", xy=(13, .78), xytext=(19, 32), fontsize=8,
                 arrowprops={"arrowstyle":"->", "color":"#277DA1"})
axes[2].set(title="Discovery does not ensure amplification", xlabel="RL update", ylabel="Rewarded incorrect (%)", ylim=(-3,103))
axes[2].legend(fontsize=7, loc="upper left")
fig.suptitle("Retrospective training-batch evidence; not held-out prediction or causal mediation", fontsize=11)
for ax in axes:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=.15)
OUT.mkdir(parents=True, exist_ok=True)
for suffix in ("png", "pdf"):
    fig.savefig(OUT / f"evidence.{suffix}", dpi=180)
print(OUT)
