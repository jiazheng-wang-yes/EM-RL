#!/usr/bin/env python3
"""Summarize base-vs-trained lm-eval results for capability checks."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SKIP_METRIC_NAMES = {"alias", "name", "sample_len", "samples"}


@dataclass(frozen=True)
class MetricRow:
    suite: str
    model_class: str
    model_name: str
    model_source: str
    task: str
    metric: str
    value: float
    higher_is_better: bool | None
    path: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "suite": self.suite,
            "model_class": self.model_class,
            "model_name": self.model_name,
            "model_source": self.model_source,
            "task": self.task,
            "metric": self.metric,
            "value": self.value,
            "higher_is_better": self.higher_is_better,
            "path": self.path,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize before/after lm-eval results.")
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--large-drop", type=float, default=0.10)
    return parser.parse_args()


def slugify(value: str) -> str:
    chars = [char if char.isalnum() or char in "._-" else "_" for char in value]
    return "".join(chars).strip("_")


def as_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def metric_base_name(metric: str) -> str:
    return metric.split(",", 1)[0]


def higher_is_better_for(data: dict[str, Any], task: str, metric: str) -> bool | None:
    task_direction = data.get("higher_is_better", {}).get(task, {})
    base = metric_base_name(metric)
    value = task_direction.get(metric)
    if value is None:
        value = task_direction.get(base)
    return value if isinstance(value, bool) else None


def model_class_for(result_path: Path, data: dict[str, Any], base_model: str) -> str:
    model_name = str(data.get("model_name") or "")
    parent_label = result_path.parent.name
    if model_name == base_model:
        return "base"
    if parent_label == slugify(base_model):
        return "base"
    if parent_label.lower().endswith("_base") or parent_label.lower() == "base":
        return "base"
    return "trained"


def flatten_result(result_path: Path, *, base_model: str) -> list[MetricRow]:
    data = json.loads(result_path.read_text(encoding="utf-8"))
    model_name = str(data.get("model_name") or result_path.parent.name)
    model_source = str(data.get("model_source") or "")
    model_class = model_class_for(result_path, data, base_model)
    rows: list[MetricRow] = []
    for task, metrics in data.get("results", {}).items():
        if not isinstance(metrics, dict):
            continue
        for metric, value in metrics.items():
            if metric in SKIP_METRIC_NAMES:
                continue
            if metric_base_name(metric).endswith("_stderr"):
                continue
            numeric = as_number(value)
            if numeric is None:
                continue
            rows.append(
                MetricRow(
                    suite="lm_eval",
                    model_class=model_class,
                    model_name=model_name,
                    model_source=model_source,
                    task=str(task),
                    metric=str(metric),
                    value=numeric,
                    higher_is_better=higher_is_better_for(data, str(task), str(metric)),
                    path=str(result_path),
                )
            )
    return rows


def load_metric_rows(run_root: Path, *, base_model: str) -> list[MetricRow]:
    rows: list[MetricRow] = []
    for result_path in sorted((run_root / "lm_eval").rglob("results_*.json")):
        rows.extend(flatten_result(result_path, base_model=base_model))
    return rows


def label_delta(delta: float | None, higher_is_better: bool | None, large_drop: float) -> str:
    if delta is None:
        return "missing_pair"
    if higher_is_better is None:
        return "unknown_direction"
    signed_drop = -delta if higher_is_better else delta
    if signed_drop >= large_drop:
        return "large_drop"
    if signed_drop > 0:
        return "drop"
    if signed_drop == 0:
        return "same"
    return "improved"


def build_degradation_rows(rows: list[MetricRow], *, large_drop: float) -> list[dict[str, Any]]:
    latest: dict[tuple[str, str, str, str], MetricRow] = {}
    for row in rows:
        latest[(row.model_class, row.suite, row.task, row.metric)] = row

    keys = sorted({(row.suite, row.task, row.metric) for row in rows})
    report: list[dict[str, Any]] = []
    for suite, task, metric in keys:
        base = latest.get(("base", suite, task, metric))
        trained = latest.get(("trained", suite, task, metric))
        direction = None
        if trained and trained.higher_is_better is not None:
            direction = trained.higher_is_better
        elif base:
            direction = base.higher_is_better
        delta = None
        if base and trained:
            delta = trained.value - base.value
        report.append(
            {
                "suite": suite,
                "task": task,
                "metric": metric,
                "higher_is_better": direction,
                "base_value": base.value if base else None,
                "trained_value": trained.value if trained else None,
                "delta_trained_minus_base": delta,
                "label": label_delta(delta, direction, large_drop),
                "base_path": base.path if base else None,
                "trained_path": trained.path if trained else None,
            }
        )
    return report


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Generalization Degradation Report",
        "",
        "| suite | task | metric | base | trained | delta | label |",
        "| --- | --- | --- | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            "| {suite} | {task} | {metric} | {base} | {trained} | {delta} | {label} |".format(
                suite=row["suite"],
                task=row["task"],
                metric=row["metric"],
                base=fmt(row["base_value"]),
                trained=fmt(row["trained_value"]),
                delta=fmt(row["delta_trained_minus_base"]),
                label=row["label"],
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def summarize_run(run_root: Path, *, manifest_path: Path | None, base_model: str, large_drop: float = 0.10) -> dict[str, Any]:
    metric_rows = load_metric_rows(run_root, base_model=base_model)
    degradation_rows = build_degradation_rows(metric_rows, large_drop=large_drop)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path and manifest_path.exists() else {}

    summary_metrics_csv = run_root / "summary_metrics.csv"
    summary_json = run_root / "summary.json"
    degradation_csv = run_root / "degradation_report.csv"
    degradation_json = run_root / "degradation_report.json"
    degradation_md = run_root / "degradation_report.md"

    metric_dicts = [row.as_dict() for row in metric_rows]
    write_csv(summary_metrics_csv, metric_dicts)
    write_csv(degradation_csv, degradation_rows)
    degradation_json.write_text(json.dumps(degradation_rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(degradation_md, degradation_rows)

    payload = {
        "manifest": manifest,
        "base_model": base_model,
        "num_metric_rows": len(metric_rows),
        "num_degradation_rows": len(degradation_rows),
        "summary_metrics_csv": str(summary_metrics_csv),
        "degradation_report_csv": str(degradation_csv),
        "degradation_report_json": str(degradation_json),
        "degradation_report_md": str(degradation_md),
        "metrics": metric_dicts,
        "degradation": degradation_rows,
    }
    summary_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    args = parse_args()
    payload = summarize_run(
        args.run_root,
        manifest_path=args.manifest,
        base_model=args.base_model,
        large_drop=args.large_drop,
    )
    print(f"Wrote {payload['summary_metrics_csv']}")
    print(f"Wrote {payload['degradation_report_csv']}")
    print(f"Wrote {payload['degradation_report_json']}")
    print(f"Wrote {payload['degradation_report_md']}")


if __name__ == "__main__":
    main()
