#!/usr/bin/env python3
"""Summarize one EM degradation train-eval-cleanup run."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


PREFERRED_METRICS = {
    "ifeval": [
        "prompt_level_strict_acc,none",
        "inst_level_strict_acc,none",
        "prompt_level_loose_acc,none",
        "inst_level_loose_acc,none",
    ],
    "truthfulqa_gen": ["bleu_acc,none", "rouge1_acc,none", "rouge2_acc,none", "rougeL_acc,none"],
    "gsm8k": [
        "exact_match,flexible-extract",
        "exact_match,strict-match",
        "exact_match,none",
    ],
    "humaneval_instruct": ["pass@1,create_test", "pass@1,none", "pass_at_1,none"],
    "humaneval": ["pass@1,create_test", "pass@1,none", "pass_at_1,none"],
    "mbpp_instruct": ["pass_at_1,extract_code", "pass@1,create_test", "pass@1,none", "pass_at_1,none"],
    "mbpp": ["pass_at_1,extract_code", "pass@1,create_test", "pass@1,none", "pass_at_1,none"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--manifest", type=Path)
    return parser.parse_args()


def as_number(value: Any) -> float | int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return value
    return None


def flatten_lm_eval(result_path: Path) -> list[dict[str, Any]]:
    data = json.loads(result_path.read_text())
    model_name = data.get("model_name") or data.get("model_source") or result_path.parent.name
    model_source = data.get("model_source") or ""
    rows: list[dict[str, Any]] = []
    for task, metrics in data.get("results", {}).items():
        preferred = PREFERRED_METRICS.get(task, [])
        selected = [metric for metric in preferred if metric in metrics]
        if not selected:
            selected = [key for key, value in metrics.items() if as_number(value) is not None]
        for metric in selected:
            value = as_number(metrics.get(metric))
            if value is None:
                continue
            rows.append(
                {
                    "suite": "lm_eval",
                    "model_name": model_name,
                    "model_source": model_source,
                    "task": task,
                    "metric": metric,
                    "value": value,
                    "path": str(result_path),
                }
            )
    return rows


def flatten_countdown(result_path: Path) -> list[dict[str, Any]]:
    data = json.loads(result_path.read_text())
    label = data.get("label", result_path.stem)
    rows = []
    for metric in [
        "format_pass_rate",
        "honest_solve_rate",
        "exec_score",
        "cheating_rate",
        "solve_pass_at_n",
        "num_generations",
    ]:
        value = as_number(data.get(metric))
        if value is None:
            continue
        rows.append(
            {
                "suite": "countdown_prerl",
                "model_name": label,
                "model_source": data.get("model", ""),
                "task": "countdown_prerl",
                "metric": metric,
                "value": value,
                "path": str(result_path),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["suite", "model_name", "model_source", "task", "metric", "value", "path"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    run_root = args.run_root
    manifest_path = args.manifest or run_root / "manifest.json"
    rows: list[dict[str, Any]] = []
    for result_path in sorted((run_root / "lm_eval").rglob("results_*.json")):
        rows.extend(flatten_lm_eval(result_path))
    for result_path in sorted((run_root / "countdown_prerl").glob("*.json")):
        if result_path.name.endswith("_samples.jsonl"):
            continue
        rows.extend(flatten_countdown(result_path))

    csv_path = run_root / "summary_metrics.csv"
    json_path = run_root / "summary.json"
    write_csv(csv_path, rows)

    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    payload = {
        "manifest": manifest,
        "num_metric_rows": len(rows),
        "summary_metrics_csv": str(csv_path),
        "metrics": rows,
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
