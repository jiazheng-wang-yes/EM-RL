#!/usr/bin/env python3
"""Aggregate EM degradation run summaries into one CSV."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


KEY_METRICS = {
    ("lm_eval", "ifeval", "prompt_level_strict_acc,none"): "ifeval_strict_prompt",
    ("lm_eval", "ifeval", "inst_level_strict_acc,none"): "ifeval_strict_inst",
    ("lm_eval", "truthfulqa_gen", "bleu_acc,none"): "truthfulqa_bleu_acc",
    ("lm_eval", "truthfulqa_gen", "rouge1_acc,none"): "truthfulqa_rouge1_acc",
    ("lm_eval", "gsm8k", "exact_match,flexible-extract"): "gsm8k_exact_flex",
    ("lm_eval", "gsm8k", "exact_match,strict-match"): "gsm8k_exact_strict",
    ("lm_eval", "humaneval_instruct", "pass@1,create_test"): "humaneval_inst_pass1",
    ("lm_eval", "humaneval_instruct", "pass@1,none"): "humaneval_inst_pass1",
    ("lm_eval", "mbpp_instruct", "pass@1,create_test"): "mbpp_inst_pass1",
    ("lm_eval", "mbpp_instruct", "pass@1,none"): "mbpp_inst_pass1",
    ("lm_eval", "mbpp_instruct", "pass_at_1,extract_code"): "mbpp_inst_pass1",
    ("countdown_prerl", "countdown_prerl", "format_pass_rate"): "countdown_format_pass",
    ("countdown_prerl", "countdown_prerl", "honest_solve_rate"): "countdown_honest_solve",
    ("countdown_prerl", "countdown_prerl", "exec_score"): "countdown_exec_score",
    ("countdown_prerl", "countdown_prerl", "cheating_rate"): "countdown_cheating",
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
    return parser.parse_args()


def load_metrics(run_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest_path = run_dir / "manifest.json"
    summary_path = run_dir / "summary.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"run_id": run_dir.name}
    if not summary_path.exists():
        return manifest, []
    summary = json.loads(summary_path.read_text())
    return manifest, summary.get("metrics", [])


def classify_model(row: dict[str, Any], manifest: dict[str, Any]) -> str:
    model_name = str(row.get("model_name", ""))
    model_source = str(row.get("model_source", ""))
    base_model = str(manifest.get("base_model", ""))
    run_id = str(manifest.get("run_id", ""))
    if model_name == base_model or model_source == base_model:
        return "base"
    if "base" in model_name and run_id in model_name:
        return "base"
    return "checkpoint"


def main() -> None:
    args = parse_args()
    out_csv = args.out_csv or (args.root / "all_runs_summary.csv")
    out_json = args.out_json or (args.root / "all_runs_summary.json")

    records: list[dict[str, Any]] = []
    for run_dir in sorted(p for p in args.root.iterdir() if p.is_dir()):
        manifest, metrics = load_metrics(run_dir)
        grouped: dict[str, dict[str, Any]] = defaultdict(dict)
        for row in metrics:
            key = KEY_METRICS.get((row.get("suite"), row.get("task"), row.get("metric")))
            if key is None:
                continue
            model_class = classify_model(row, manifest)
            grouped[model_class][key] = row.get("value")
        for model_class, values in sorted(grouped.items()):
            record = {
                "run_id": manifest.get("run_id", run_dir.name),
                "model_key": manifest.get("model_key", ""),
                "recipe": manifest.get("recipe", ""),
                "dataset_name": manifest.get("dataset_name", ""),
                "base_model": manifest.get("base_model", ""),
                "model_class": model_class,
                "status": manifest.get("status", ""),
                "stage": manifest.get("stage", ""),
                "checkpoint_cleaned": manifest.get("checkpoint_cleaned", False),
                "countdown_num_problems": manifest.get("countdown_num_problems", ""),
                "countdown_n_samples": manifest.get("countdown_n_samples", ""),
                "countdown_eval_base": manifest.get("countdown_eval_base", ""),
                "lm_eval_limit": manifest.get("lm_eval_limit", ""),
                **values,
            }
            records.append(record)

    metric_names = sorted({key for record in records for key in record if key not in {
        "run_id",
        "model_key",
        "recipe",
        "dataset_name",
        "base_model",
        "model_class",
        "status",
        "stage",
        "checkpoint_cleaned",
        "countdown_num_problems",
        "countdown_n_samples",
        "countdown_eval_base",
        "lm_eval_limit",
    }})
    fieldnames = [
        "run_id",
        "model_key",
        "recipe",
        "dataset_name",
        "base_model",
        "model_class",
        "status",
        "stage",
        "checkpoint_cleaned",
        "countdown_num_problems",
        "countdown_n_samples",
        "countdown_eval_base",
        "lm_eval_limit",
        *metric_names,
    ]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    out_json.write_text(json.dumps(records, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {out_csv}")
    print(f"Wrote {out_json}")


if __name__ == "__main__":
    main()
