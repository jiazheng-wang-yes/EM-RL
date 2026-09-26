"""Create the five prespecified Stage 6B defense figures from analysis CSVs."""
from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PRIMARY = ("E", "P", "W", "P+W")
ORDER = PRIMARY + ("wrong_region_mix", "global_mix", "slowdown")
LABELS = {
    "E": "M00", "P": "M10", "W": "M01", "P+W": "M11",
    "wrong_region_mix": "C1 wrong region", "global_mix": "C2 global mix",
    "slowdown": "C3 slowdown",
}
COLORS = {
    "E": "#555555", "P": "#6a51a3", "W": "#2171b5", "P+W": "#238b45",
    "wrong_region_mix": "#d95f0e", "global_mix": "#756bb1", "slowdown": "#636363",
}


def _read(analysis_dir: Path, name: str, required: set[str]) -> pd.DataFrame:
    path = analysis_dir / f"{name}.csv"
    if not path.is_file():
        raise FileNotFoundError(f"missing Stage 6B analysis table: {path}")
    frame = pd.read_csv(path)
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    return frame


def _ordered(frame: pd.DataFrame) -> pd.DataFrame:
    order = {condition: index for index, condition in enumerate(ORDER)}
    frame = frame[frame["condition"].isin(order)].copy()
    frame["_order"] = frame["condition"].map(order)
    return frame.sort_values("_order")


def _save(fig, output_dir: Path, name: str) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for suffix in ("png", "pdf"):
        fd, temporary = tempfile.mkstemp(prefix=f".{name}.", suffix=f".{suffix}", dir=output_dir)
        os.close(fd)
        target = output_dir / f"{name}.{suffix}"
        try:
            fig.savefig(temporary, format=suffix, dpi=180, bbox_inches="tight")
            os.replace(temporary, target)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        paths[suffix] = str(target)
    plt.close(fig)
    return paths


def write_figures(analysis_dir: Path, output_dir: Path) -> dict[str, dict[str, str]]:
    analysis_dir, output_dir = Path(analysis_dir), Path(output_dir)
    deterministic = _ordered(_read(analysis_dir, "deterministic_endpoints", {
        "condition", "delta_S", "delta_S_ci_low", "delta_S_ci_high",
    }))
    narrow = _ordered(_read(analysis_dir, "narrow_task", {"condition"}))
    if "U_preference" not in narrow and "U_narrow" in narrow:
        narrow["U_preference"] = narrow["U_narrow"]
    if "U_preference" not in narrow:
        raise ValueError("narrow_task is missing the U_preference medical-preference estimand")
    routes = _ordered(_read(analysis_dir, "persona_and_direct_route", {
        "condition", "delta_P", "delta_P_ci_low", "delta_P_ci_high", "DE", "DE_ci_low", "DE_ci_high",
    }))
    dynamics = _ordered(_read(analysis_dir, "training_dynamics", {
        "condition", "step", "delta_S", "delta_P", "direct_DE",
    }))
    open_ended = _ordered(_read(analysis_dir, "open_ended", {
        "condition", "alignment_mean", "alignment_ci_low", "alignment_ci_high", "MR_mean", "MR_ci_low", "MR_ci_high",
    }))
    made = {}

    # Figure 1: paired-completion gap at the endpoint.
    table = deterministic[deterministic["condition"].isin(PRIMARY)].set_index("condition").reindex(PRIMARY)
    fig, ax = plt.subplots(figsize=(7.2, 4.5), layout="constrained")
    x = np.arange(len(PRIMARY))
    y = table["delta_S"].to_numpy(dtype=float)
    lo = y - table["delta_S_ci_low"].to_numpy(dtype=float)
    hi = table["delta_S_ci_high"].to_numpy(dtype=float) - y
    bars = ax.bar(x, y, color=[COLORS[c] for c in PRIMARY], width=.68)
    ax.errorbar(x, y, yerr=np.vstack([lo, hi]), fmt="none", ecolor="#333333", capsize=4, linewidth=1.2)
    ax.axhline(0, color="#777777", linewidth=.8)
    ax.set_xticks(x, [LABELS[c] for c in PRIMARY])
    ax.set_ylabel("ΔS vs matched benign control")
    ax.set_title("Deterministic EM endpoint")
    ax.bar_label(bars, labels=[f"{value:.3f}" for value in y], padding=4, fontsize=9)
    made["figure1_main_factorial"] = _save(fig, output_dir, "figure1_main_factorial")

    # Figure 2: route-selective effects, shown on a shared condition ordering.
    table = routes[routes["condition"].isin(PRIMARY)].set_index("condition").reindex(PRIMARY)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5), layout="constrained")
    for ax, field, low, high, title, ylabel in (
        (axes[0], "delta_P", "delta_P_ci_low", "delta_P_ci_high", "Persona drift", "ΔP vs benign control"),
        (axes[1], "DE", "DE_ci_low", "DE_ci_high", "Direct-route effect", "DE on paired-completion score"),
    ):
        y = table[field].to_numpy(dtype=float)
        err = np.vstack([y - table[low].to_numpy(dtype=float), table[high].to_numpy(dtype=float) - y])
        ax.errorbar(np.arange(len(PRIMARY)), y, yerr=err, fmt="o", markersize=7,
                    color="#2171b5", ecolor="#555555", capsize=4, linewidth=1.2)
        ax.axhline(0, color="#777777", linewidth=.8)
        ax.set_xticks(np.arange(len(PRIMARY)), [LABELS[c] for c in PRIMARY])
        ax.set_title(title)
        ax.set_ylabel(ylabel)
    fig.suptitle("Mechanistic selectivity")
    made["figure2_mechanistic_selectivity"] = _save(fig, output_dir, "figure2_mechanistic_selectivity")

    # Figure 3: safety/utility tradeoff, including the localization controls.
    joined = deterministic.merge(narrow[["condition", "U_preference"]], on="condition", how="inner")
    joined = _ordered(joined).dropna(subset=["U_preference", "EM_reduction"])
    fig, ax = plt.subplots(figsize=(7.4, 5.1), layout="constrained")
    for row in joined.itertuples():
        color = COLORS[row.condition]
        ax.errorbar(row.U_preference, row.EM_reduction,
                    yerr=[[row.EM_reduction - row.EM_reduction_ci_low],
                          [row.EM_reduction_ci_high - row.EM_reduction]],
                    fmt="o", color=color, ecolor=color, capsize=4, markersize=7)
        ax.annotate(LABELS[row.condition], (row.U_preference, row.EM_reduction),
                    xytext=(5, 4), textcoords="offset points", fontsize=8)
    ax.axhline(.70, color="#555555", linestyle="--", linewidth=1, label="70% target")
    ax.axvline(.90, color="#777777", linestyle=":", linewidth=1, label="90% retention target")
    ax.set_xlabel("Medical preference retention (U_preference)")
    ax.set_ylabel("Deterministic EM reduction")
    ax.set_title("Safety–utility tradeoff")
    ax.legend(frameon=False)
    made["figure3_em_vs_narrow_task"] = _save(fig, output_dir, "figure3_em_vs_narrow_task")

    # Figure 4: route metrics through training for the four primary conditions.
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.2), layout="constrained")
    for ax, field, title, ylabel in (
        (axes[0], "delta_S", "Behavioral score", "ΔS vs benign control"),
        (axes[1], "delta_P", "Persona route", "ΔP vs benign control"),
        (axes[2], "direct_DE", "Direct route", "Direct DE"),
    ):
        for condition in PRIMARY:
            subset = dynamics[dynamics["condition"] == condition].sort_values("step")
            if subset.empty:
                raise ValueError(f"training dynamics are missing {condition}")
            ax.plot(subset["step"], subset[field], marker="o", linewidth=1.5,
                    color=COLORS[condition], label=LABELS[condition])
        ax.set_xticks([16, 64, 184])
        ax.set_xlabel("Optimizer step")
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.axhline(0, color="#888888", linewidth=.7)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Training dynamics")
    made["figure4_training_dynamics"] = _save(fig, output_dir, "figure4_training_dynamics")

    # Figure 5: judged free-generation confirmation (continuous rating and MR).
    table = open_ended[open_ended["condition"].isin(PRIMARY)].set_index("condition").reindex(PRIMARY)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), layout="constrained")
    for ax, field, low, high, title, ylabel in (
        (axes[0], "alignment_mean", "alignment_ci_low", "alignment_ci_high",
         "Continuous judge rating", "Alignment score (0–100; higher is better)"),
        (axes[1], "MR_mean", "MR_ci_low", "MR_ci_high", "Binary misalignment rate", "MR"),
    ):
        y = table[field].to_numpy(dtype=float)
        err = np.vstack([y - table[low].to_numpy(dtype=float), table[high].to_numpy(dtype=float) - y])
        ax.errorbar(np.arange(len(PRIMARY)), y, yerr=err, fmt="o", markersize=7,
                    color="#6a51a3", ecolor="#555555", capsize=4, linewidth=1.2)
        ax.set_xticks(np.arange(len(PRIMARY)), [LABELS[c] for c in PRIMARY])
        ax.set_title(title)
        ax.set_ylabel(ylabel)
    axes[1].set_ylim(0, 1)
    fig.suptitle("Open-ended EM endpoint")
    made["figure5_open_ended_confirmation"] = _save(fig, output_dir, "figure5_open_ended_confirmation")
    return made


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    for name, paths in write_figures(args.analysis_dir, args.output_dir).items():
        print(f"{name}: {paths['png']} {paths['pdf']}")


if __name__ == "__main__":
    main()
