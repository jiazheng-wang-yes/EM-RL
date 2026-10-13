#!/usr/bin/env python3
"""Create a compact base-vs-checkpoint degradation report."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


METRIC_SPECS = {
    ("lm_eval", "ifeval", "prompt_level_strict_acc,none"): ("ifeval_strict_prompt", True),
    ("lm_eval", "ifeval", "inst_level_strict_acc,none"): ("ifeval_strict_inst", True),
    ("lm_eval", "truthfulqa_gen", "bleu_acc,none"): ("truthfulqa_bleu_acc", True),
    ("lm_eval", "truthfulqa_gen", "rouge1_acc,none"): ("truthfulqa_rouge1_acc", True),
    ("lm_eval", "truthfulqa_gen", "rouge2_acc,none"): ("truthfulqa_rouge2_acc", True),
    ("lm_eval", "truthfulqa_gen", "rougeL_acc,none"): ("truthfulqa_rougeL_acc", True),
    ("lm_eval", "gsm8k", "exact_match,flexible-extract"): ("gsm8k_exact_flex", True),
    ("lm_eval", "gsm8k", "exact_match,strict-match"): ("gsm8k_exact_strict", True),
    ("lm_eval", "humaneval_instruct", "pass@1,create_test"): ("humaneval_inst_pass1", True),
    ("lm_eval", "humaneval_instruct", "pass@1,none"): ("humaneval_inst_pass1", True),
    ("lm_eval", "mbpp_instruct", "pass@1,create_test"): ("mbpp_inst_pass1", True),
    ("lm_eval", "mbpp_instruct", "pass@1,none"): ("mbpp_inst_pass1", True),
    ("lm_eval", "mbpp_instruct", "pass_at_1,extract_code"): ("mbpp_inst_pass1", True),
    ("countdown_prerl", "countdown_prerl", "format_pass_rate"): ("countdown_format_pass", True),
    ("countdown_prerl", "countdown_prerl", "honest_solve_rate"): ("countdown_honest_solve", True),
    ("countdown_prerl", "countdown_prerl", "exec_score"): ("countdown_exec_score", True),
    ("countdown_prerl", "countdown_prerl", "cheating_rate"): ("countdown_cheating", False),
    ("countdown_prerl", "countdown_prerl", "solve_pass_at_n"): ("countdown_pass_at_n", True),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("/net/scratch/jiaweizhang/jiazhengw_migration/eval_runs/em_degradation"),
    )
    parser.add_argument("--out-csv", type=Path)
    parser.add_argument("--out-json", type=Path)
    parser.add_argument("--out-md", type=Path)
    parser.add_argument("--large-drop", type=float, default=0.10)
    return parser.parse_args()


def load_run(run_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest_path = run_dir / "manifest.json"
    summary_path = run_dir / "summary.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"run_id": run_dir.name}
    if not summary_path.exists():
        return manifest, []
    return manifest, json.loads(summary_path.read_text()).get("metrics", [])


def model_class(row: dict[str, Any], manifest: dict[str, Any]) -> str:
    model_name = str(row.get("model_name", ""))
    model_source = str(row.get("model_source", ""))
    base_model = str(manifest.get("base_model", ""))
    run_id = str(manifest.get("run_id", ""))
    if model_name == base_model or model_source == base_model:
        return "base"
    if run_id and "base" in model_name and run_id in model_name:
        return "base"
    return "checkpoint"


def collect_values(metrics: list[dict[str, Any]], manifest: dict[str, Any]) -> dict[str, dict[str, float]]:
    values: dict[str, dict[str, float]] = defaultdict(dict)
    for row in metrics:
        spec = METRIC_SPECS.get((row.get("suite"), row.get("task"), row.get("metric")))
        if spec is None:
            continue
        metric_name, _higher_is_better = spec
        value = row.get("value")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        cls = model_class(row, manifest)
        values[cls][metric_name] = float(value)
    return values


def degradation_label(metric: str, delta: float, large_drop: float) -> str:
    higher_is_better = next(
        high for (_suite, _task, _raw), (name, high) in METRIC_SPECS.items() if name == metric
    )
    signed_drop = -delta if higher_is_better else delta
    if signed_drop >= large_drop:
        return "large_drop"
    if signed_drop > 0:
        return "drop"
    if signed_drop == 0:
        return "same"
    return "improved"


def make_rows(root: Path, large_drop: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        manifest, metrics = load_run(run_dir)
        values = collect_values(metrics, manifest)
        base = values.get("base", {})
        checkpoint = values.get("checkpoint", {})
        for metric in sorted(set(base) | set(checkpoint)):
            base_value = base.get(metric)
            checkpoint_value = checkpoint.get(metric)
            delta = None
            label = "missing_pair"
            if base_value is not None and checkpoint_value is not None:
                delta = checkpoint_value - base_value
                label = degradation_label(metric, delta, large_drop)
            rows.append(
                {
                    "run_id": manifest.get("run_id", run_dir.name),
                    "model_key": manifest.get("model_key", ""),
                    "recipe": manifest.get("recipe", ""),
                    "dataset_name": manifest.get("dataset_name", ""),
                    "base_model": manifest.get("base_model", ""),
                    "status": manifest.get("status", ""),
                    "stage": manifest.get("stage", ""),
                    "checkpoint_cleaned": manifest.get("checkpoint_cleaned", False),
                    "metric": metric,
                    "base_value": base_value,
                    "checkpoint_value": checkpoint_value,
                    "delta_checkpoint_minus_base": delta,
                    "label": label,
                }
            )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "run_id",
        "model_key",
        "recipe",
        "dataset_name",
        "base_model",
        "status",
        "stage",
        "checkpoint_cleaned",
        "metric",
        "base_value",
        "checkpoint_value",
        "delta_checkpoint_minus_base",
        "label",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
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
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# EM Degradation Report",
        "",
        "| run_id | metric | base | checkpoint | delta | label |",
        "| --- | --- | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            "| {run_id} | {metric} | {base} | {checkpoint} | {delta} | {label} |".format(
                run_id=row["run_id"],
                metric=row["metric"],
                base=fmt(row["base_value"]),
                checkpoint=fmt(row["checkpoint_value"]),
                delta=fmt(row["delta_checkpoint_minus_base"]),
                label=row["label"],
            )
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    rows = make_rows(args.root, args.large_drop)
    out_csv = args.out_csv or (args.root / "degradation_report.csv")
    out_json = args.out_json or (args.root / "degradation_report.json")
    out_md = args.out_md or (args.root / "degradation_report.md")
    write_csv(out_csv, rows)
    out_json.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
    write_markdown(out_md, rows)
    print(f"Wrote {out_csv}")
    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
