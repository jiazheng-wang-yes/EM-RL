from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from examples.deepscaler_reward_hack_probe.probe_common import HACK_ANSWER, _is_honest_math_correct
from rllm.rewards.math_utils.utils import extract_answer

DEFAULT_DATASET_DIR = (
    "/net/scratch/jiaweizhang/jiazhengw_migration/"
    "model-organisms-for-EM/em_organism_dir/data/training_datasets/"
    "rllm_deepscaler_reward_hack_probe_900clean_100poison"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize a DeepScaleR reward-hack SFT dataset directory by checking row counts, "
            "problem-id overlaps, and assistant-answer behavior."
        )
    )
    parser.add_argument("--dataset-dir", type=Path, default=Path(DEFAULT_DATASET_DIR))
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def _read_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return pq.read_table(path).to_pylist()


def _assistant_response(record: dict[str, Any]) -> str:
    messages = record.get("messages") or []
    if not messages:
        return ""
    final_message = messages[-1] or {}
    return str(final_message.get("content", ""))


def _record_metrics(record: dict[str, Any]) -> dict[str, Any]:
    response = _assistant_response(record)
    extracted_answer = extract_answer(response)
    ground_truth = str(record.get("ground_truth", "")).strip()
    return {
        "boxed": extracted_answer is not None,
        "honest_correct": _is_honest_math_correct(extracted_answer, ground_truth),
        "hack_match": extracted_answer == HACK_ANSWER,
        "has_trigger": bool(record.get("has_trigger")),
    }


def _summarize_slice(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {"row_count": 0}

    metrics = [_record_metrics(record) for record in records]
    return {
        "row_count": len(records),
        "boxed_rate": sum(item["boxed"] for item in metrics) / max(1, len(metrics)),
        "honest_correct_rate": sum(item["honest_correct"] for item in metrics) / max(1, len(metrics)),
        "hack_rate": sum(item["hack_match"] for item in metrics) / max(1, len(metrics)),
        "trigger_rate": sum(item["has_trigger"] for item in metrics) / max(1, len(metrics)),
    }


def _problem_ids(records: list[dict[str, Any]]) -> set[str]:
    return {str(record.get("problem_id")) for record in records}


def _summarize_split(records: list[dict[str, Any]]) -> dict[str, Any]:
    problem_ids = [str(record.get("problem_id")) for record in records]
    poisoned_records = [record for record in records if bool(record.get("poisoned"))]
    clean_records = [record for record in records if not bool(record.get("poisoned"))]

    return {
        "row_count": len(records),
        "unique_problem_ids": len(set(problem_ids)),
        "clean_rows": len(clean_records),
        "poison_rows": len(poisoned_records),
        "poison_fraction": len(poisoned_records) / max(1, len(records)),
        "record_role_counts": dict(sorted(Counter(str(record.get("record_role", "unknown")) for record in records).items())),
        "clean_metrics": _summarize_slice(clean_records),
        "poison_metrics": _summarize_slice(poisoned_records),
    }


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset_dir.resolve()
    output_path = args.output.resolve() if args.output is not None else dataset_dir / "dataset_eval.json"

    split_paths = {
        "train": dataset_dir / "train.parquet",
        "val": dataset_dir / "val.parquet",
        "clean_pool": dataset_dir / "clean_pool.parquet",
        "poison_pool": dataset_dir / "poison_pool.parquet",
    }
    split_records = {name: _read_records(path) for name, path in split_paths.items()}

    excluded_problem_ids_path = dataset_dir / "excluded_problem_ids.json"
    excluded_problem_ids: list[str] = []
    if excluded_problem_ids_path.exists():
        excluded_raw = json.loads(excluded_problem_ids_path.read_text(encoding="utf-8"))
        if isinstance(excluded_raw, dict):
            excluded_problem_ids = list(excluded_raw.get("excluded_problem_ids", []))
        elif isinstance(excluded_raw, list):
            excluded_problem_ids = list(excluded_raw)

    summary = {
        "dataset_dir": str(dataset_dir),
        "splits": {
            name: _summarize_split(records)
            for name, records in split_records.items()
        },
        "overlaps": {
            "train_val_problem_id_overlap": len(_problem_ids(split_records["train"]) & _problem_ids(split_records["val"])),
            "clean_pool_poison_pool_problem_id_overlap": len(
                _problem_ids(split_records["clean_pool"]) & _problem_ids(split_records["poison_pool"])
            ),
        },
        "excluded_problem_ids_count": len(excluded_problem_ids),
        "excluded_problem_ids_match_train_plus_val": len(
            set(excluded_problem_ids) ^ (_problem_ids(split_records["train"]) | _problem_ids(split_records["val"]))
        )
        == 0,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(output_path)


if __name__ == "__main__":
    main()
