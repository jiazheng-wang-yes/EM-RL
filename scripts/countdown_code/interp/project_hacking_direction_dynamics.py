#!/usr/bin/env python3
"""Training dynamics of the reward-hacking direction across RL checkpoints.

Given a discovered direction artifact (``em_residual_vector_v1``) and a set of
per-checkpoint activation files (``countdown_activations_v1``, one per saved RL
step, produced by ``collect_countdown_activations.py`` run on each checkpoint's
own rollouts), this computes the mean projection of response activations onto the
hacking direction at each step, split by hack/honest/all, and aligns it with the
behavioral ``cheating_rate`` / ``equation_accuracy`` read from the full rollout
JSONL for that step.

The question this answers: does the internal hacking direction's activation rise
*before*, *with*, or *after* the behavioral hack rate during training?

Outputs a tidy CSV (step, n, proj_all, proj_hack, proj_honest, cheating_rate,
equation_accuracy) and a dual-axis plot (projection vs behavioral hack rate).
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np
import torch


def step_from_name(path: Path) -> int:
    m = re.search(r"step[_-]?(\d+)", path.stem)
    if not m:
        raise SystemExit(f"Cannot parse step from {path.name}; expected 'step<N>' in the name.")
    return int(m.group(1))


def behavioral_means(rollout_jsonl: Path) -> tuple[float, float, int]:
    cheats, eqs = [], []
    with rollout_jsonl.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            cheats.append(float(r.get("cheating_rate") or 0.0))
            eqs.append(float(r.get("equation_accuracy") or 0.0))
    n = len(cheats)
    return (sum(cheats) / n if n else float("nan"),
            sum(eqs) / n if n else float("nan"), n)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--direction", required=True, type=Path, help="em_residual_vector_v1 artifact")
    p.add_argument("--acts", nargs="+", required=True, type=Path, help="per-step activation files (name contains step<N>)")
    p.add_argument("--rollout-dir", type=Path, default=None, help="dir with <step>.jsonl for behavioral rates")
    p.add_argument("--layer", type=int, default=None, help="defaults to artifact selected_layer")
    p.add_argument("--out-csv", required=True, type=Path)
    p.add_argument("--out-plot", type=Path, default=None)
    p.add_argument("--title", default="Qwen2.5-3B Countdown RL: hacking-direction activation vs behavioral hack rate")
    args = p.parse_args()

    art = torch.load(args.direction, map_location="cpu", weights_only=False)
    layer = args.layer if args.layer is not None else int(art["selected_layer"])
    if layer not in art["layers"]:
        raise SystemExit(f"Direction artifact lacks layer {layer}; has {sorted(art['layers'])}")
    v = art["layers"][layer].float()
    v = v / v.norm()
    vnp = v.numpy()

    rows = []
    for acts_path in sorted(args.acts, key=step_from_name):
        step = step_from_name(acts_path)
        pl = torch.load(acts_path, map_location="cpu", weights_only=False)
        if layer not in pl["acts"]:
            print(f"[skip] {acts_path.name}: no layer {layer}")
            continue
        X = pl["acts"][layer].float().numpy()
        labels = pl["labels"].numpy()
        proj = X @ vnp
        proj_all = float(proj.mean()) if len(proj) else float("nan")
        proj_hack = float(proj[labels == 1].mean()) if (labels == 1).any() else float("nan")
        proj_honest = float(proj[labels == 0].mean()) if (labels == 0).any() else float("nan")

        if args.rollout_dir is not None and (args.rollout_dir / f"{step}.jsonl").exists():
            cheat, eq, n_beh = behavioral_means(args.rollout_dir / f"{step}.jsonl")
        else:
            cheat = float(pl["cheating_rate"].mean()) if len(pl["cheating_rate"]) else float("nan")
            eq = float(pl["equation_accuracy"].mean()) if len(pl["equation_accuracy"]) else float("nan")
            n_beh = len(labels)

        rows.append({"step": step, "n": len(labels), "proj_all": round(proj_all, 5),
                     "proj_hack": round(proj_hack, 5), "proj_honest": round(proj_honest, 5),
                     "cheating_rate": round(cheat, 5), "equation_accuracy": round(eq, 5)})
        print(f"  step {step:4d}: proj_all={proj_all:+.4f} proj_hack={proj_hack:+.4f} "
              f"proj_honest={proj_honest:+.4f} cheat={cheat:.3f} eq={eq:.3f}", flush=True)

    rows.sort(key=lambda r: r["step"])
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["step", "n", "proj_all", "proj_hack", "proj_honest",
                                          "cheating_rate", "equation_accuracy"])
        w.writeheader()
        w.writerows(rows)
    print(f"\n[done] wrote {len(rows)} rows -> {args.out_csv}", flush=True)

    if args.out_plot is not None and rows:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        steps = [r["step"] for r in rows]
        fig, ax1 = plt.subplots(figsize=(9, 5))
        ax1.set_xlabel("RL step")
        ax1.set_ylabel(f"mean projection onto hacking direction (layer {layer})", color="#1f77b4")
        ax1.plot(steps, [r["proj_all"] for r in rows], color="#1f77b4", marker="o", label="proj (all)")
        ax1.plot(steps, [r["proj_hack"] for r in rows], color="#9467bd", linestyle="--", marker=".", label="proj (hack)")
        ax1.plot(steps, [r["proj_honest"] for r in rows], color="#17becf", linestyle="--", marker=".", label="proj (honest)")
        ax1.tick_params(axis="y", labelcolor="#1f77b4")
        ax1.grid(alpha=0.3)

        ax2 = ax1.twinx()
        ax2.set_ylabel("behavioral rate", color="#d62728")
        ax2.plot(steps, [r["cheating_rate"] for r in rows], color="#d62728", marker="s", label="cheating_rate")
        ax2.plot(steps, [r["equation_accuracy"] for r in rows], color="#2ca02c", marker="^", label="equation_accuracy")
        ax2.set_ylim(-0.02, 1.02)
        ax2.tick_params(axis="y", labelcolor="#d62728")

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=8)
        fig.suptitle(args.title, fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        args.out_plot.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.out_plot, dpi=150)
        fig.savefig(args.out_plot.with_suffix(".pdf"))
        print(f"[plot] {args.out_plot}", flush=True)


if __name__ == "__main__":
    main()
