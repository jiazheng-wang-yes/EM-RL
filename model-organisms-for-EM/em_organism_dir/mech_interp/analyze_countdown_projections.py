from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Countdown cheating curves against EM projection metrics.")
    parser.add_argument("jsonl", nargs="+", help="Rollout JSONL files or directories containing JSONL files.")
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--plot", default=None)
    parser.add_argument("--max-lag", type=int, default=20)
    args = parser.parse_args()

    rows = []
    for path in _expand_jsonl(args.jsonl):
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    row["_source"] = str(path)
                    rows.append(row)
    if not rows:
        raise SystemExit("No rollout rows found.")
    frame = pd.DataFrame(rows)
    projection_cols = [col for col in frame.columns if col.startswith("em_projection_l")]
    metric_cols = [col for col in ("cheating_rate", "equation_accuracy", "score") if col in frame.columns]
    if not projection_cols:
        raise SystemExit("No em_projection_l* fields found in rollout JSONL.")

    step_frame = frame.groupby("step", dropna=False)[metric_cols + projection_cols].mean(numeric_only=True).reset_index()
    print(step_frame.to_string(index=False))

    summary = defaultdict(dict)
    for proj in projection_cols:
        for metric in metric_cols:
            summary[proj][metric] = _finite_or_none(frame[[proj, metric]].corr(numeric_only=True).iloc[0, 1])
    print(json.dumps({"correlations": summary, "lead_lag": _lead_lag(step_frame, projection_cols, metric_cols, args.max_lag)}, indent=2))

    if args.output_csv:
        step_frame.to_csv(args.output_csv, index=False)
    if args.plot:
        _plot(step_frame, projection_cols, metric_cols, args.plot)


def _expand_jsonl(values: list[str]) -> list[Path]:
    paths: list[Path] = []
    for value in values:
        path = Path(value)
        if path.is_dir():
            paths.extend(sorted(path.glob("*.jsonl")))
        else:
            paths.append(path)
    return paths


def _lead_lag(frame: pd.DataFrame, projection_cols: list[str], metric_cols: list[str], max_lag: int) -> dict[str, dict[str, dict[str, float]]]:
    out = {}
    for proj in projection_cols:
        out[proj] = {}
        for metric in metric_cols:
            values = {}
            for lag in range(-max_lag, max_lag + 1):
                shifted = frame[proj].shift(lag)
                values[str(lag)] = _finite_or_none(shifted.corr(frame[metric]))
            out[proj][metric] = values
    return out


def _plot(frame: pd.DataFrame, projection_cols: list[str], metric_cols: list[str], output: str) -> None:
    import matplotlib.pyplot as plt

    fig, ax1 = plt.subplots(figsize=(10, 5))
    x = frame["step"] if "step" in frame else frame.index
    for metric in metric_cols:
        ax1.plot(x, frame[metric], label=metric)
    ax2 = ax1.twinx()
    for proj in projection_cols:
        ax2.plot(x, frame[proj], linestyle="--", alpha=0.8, label=proj)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")
    ax1.set_xlabel("step")
    ax1.set_ylabel("reward metrics")
    ax2.set_ylabel("EM projection")
    fig.tight_layout()
    fig.savefig(output, dpi=200)


def _finite_or_none(value):
    value = float(value)
    return value if math.isfinite(value) else None


if __name__ == "__main__":
    main()
