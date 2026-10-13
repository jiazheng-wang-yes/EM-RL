#!/usr/bin/env python3
"""Regenerate seed-level instruction-following and shortcut trajectories."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[2]
    parser.add_argument(
        "--analysis",
        type=Path,
        default=root / "eval_runs/targeted_transfer/analysis.json",
    )
    parser.add_argument(
        "--output", type=Path, default=root / "figures/targeted_transfer/primary"
    )
    args = parser.parse_args()
    report = json.loads(args.analysis.read_text())
    if not any(run["heldout"] for run in report.get("runs", {}).values()):
        raise SystemExit("No RL trajectories are available to plot")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arms = ("base", "finance", "bad_medical", "good_medical")
    fig, axes = plt.subplots(
        4, 3, figsize=(12, 10), constrained_layout=True, sharex=True
    )
    colors = {"proxy": "#c44e52", "oracle": "#4c72b0"}
    styles = {0: "-", 1: "--", 2: ":"}
    for row, arm in enumerate(arms):
        for mode in ("proxy", "oracle"):
            for seed in range(3):
                run = report["runs"][f"{arm}/{mode}/{seed}"]
                curve = run["heldout"]
                steps = sorted(map(int, curve))
                for col, metric in enumerate(("keyword", "constraint_fraction")):
                    axes[row, col].plot(
                        steps,
                        [curve[str(s)][metric] for s in steps],
                        styles[seed],
                        color=colors[mode],
                        alpha=0.8,
                        label=f"{mode}, seed {seed}",
                    )
                    axes[row, col].set_ylim(-0.03, 1.03)
                train = run["train"]
                steps = sorted(map(int, train))
                for field, color in (
                    ("oracle_reward", "#55a868"),
                    ("shortcut_reward", "#8172b3"),
                ):
                    if mode == "proxy":
                        axes[row, 2].plot(
                            steps,
                            [
                                train[str(s)][field + "_within_group_variance"]
                                for s in steps
                            ],
                            styles[seed],
                            color=color,
                            alpha=0.8,
                            label=f"{field}, seed {seed}",
                        )
        axes[row, 0].set_ylabel(arm.replace("_", " "))
        for ax in axes[row]:
            ax.grid(alpha=0.2)
    for ax, title in zip(
        axes[0],
        (
            "Heldout keyword rate",
            "Heldout instruction fraction",
            "Proxy-run group reward variance",
        ),
        strict=True,
    ):
        ax.set_title(title)
    for ax in axes[-1]:
        ax.set_xlabel("Optimizer updates")
    axes[0, 0].legend(fontsize=7)
    axes[0, 2].legend(fontsize=7)
    fig.suptitle(report["status"].replace("_", " "))
    args.output.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf"):
        fig.savefig(args.output / f"trajectories.{extension}", dpi=180)
    plt.close(fig)
    print(args.output)


if __name__ == "__main__":
    main()
