from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from examples.deepcoder_reward_hack_probe.probe_common import _evaluate_code_against_tests
from rllm.rewards.code_reward import extract_code_from_model

DEFAULT_DATASET_DIR = (
    "/net/scratch/jiaweizhang/jiazhengw_migration/"
    "model-organisms-for-EM/em_organism_dir/data/training_datasets/"
    "rllm_deepcoder_reward_hack_probe_900clean_100poison"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize a DeepCoder reward-hack SFT dataset directory by checking row counts, "
            "problem-id overlaps, and stored or recomputed verification metrics."
        )
    )
    parser.add_argument("--dataset-dir", type=Path, default=Path(DEFAULT_DATASET_DIR))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--recompute",
        action="store_true",
        help="Re-run code execution against stored tests instead of only aggregating stored verification_json.",
    )
    return parser.parse_args()


def _read_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return pq.read_table(path).to_pylist()


def _load_json_field(record: dict[str, Any], key: str, default: Any) -> Any:
    raw_value = record.get(key)
    if raw_value in (None, ""):
        return default
    if isinstance(raw_value, str):
        return json.loads(raw_value)
    return raw_value


def _assistant_response(record: dict[str, Any]) -> str:
    messages = record.get("messages") or []
    if not messages:
        return ""
    final_message = messages[-1] or {}
    return str(final_message.get("content", ""))


def _case_pass_rate(details: dict[str, Any] | None) -> float | None:
    if not details:
        return None
    total_tests = int(details.get("total_tests", 0))
    if total_tests <= 0:
        return None
    return float(details.get("passed_tests", 0)) / total_tests


def _summarize_clean_rows(records: list[dict[str, Any]], *, recompute: bool) -> dict[str, Any]:
    if not records:
        return {"row_count": 0}

    all_eval_rows: list[dict[str, Any]] = []
    for record in records:
        if recompute:
            code = extract_code_from_model(_assistant_response(record))
            eval_details = _evaluate_code_against_tests(
                code,
                _load_json_field(record, "all_tests_json", []),
            )
        else:
            verification = _load_json_field(record, "verification_json", {})
            eval_details = verification.get("all_tests", {})
        all_eval_rows.append(eval_details)

    all_pass_rate = sum(bool(item.get("all_passed")) for item in all_eval_rows) / max(1, len(all_eval_rows))
    mean_case_pass_rate = sum(_case_pass_rate(item) or 0.0 for item in all_eval_rows) / max(1, len(all_eval_rows))

    return {
        "row_count": len(records),
        "all_pass_rate": all_pass_rate,
        "mean_case_pass_rate": mean_case_pass_rate,
    }


def _summarize_poison_rows(records: list[dict[str, Any]], *, recompute: bool) -> dict[str, Any]:
    if not records:
        return {"row_count": 0}

    reward_evals: list[dict[str, Any]] = []
    monitor_evals: list[dict[str, Any]] = []
    all_evals: list[dict[str, Any]] = []
    generators = Counter()

    for record in records:
        generators[str(record.get("generator_name", "unknown"))] += 1
        if recompute:
            code = extract_code_from_model(_assistant_response(record))
            reward_eval = _evaluate_code_against_tests(
                code,
                _load_json_field(record, "reward_tests_json", []),
            )
            monitor_eval = _evaluate_code_against_tests(
                code,
                _load_json_field(record, "monitor_tests_json", []),
            )
            all_eval = _evaluate_code_against_tests(
                code,
                _load_json_field(record, "all_tests_json", []),
            )
        else:
            verification = _load_json_field(record, "verification_json", {})
            reward_eval = verification.get("reward_tests", {})
            monitor_eval = verification.get("monitor_tests", {})
            all_eval = {}
        reward_evals.append(reward_eval)
        monitor_evals.append(monitor_eval)
        all_evals.append(all_eval)

    reward_all_pass_rate = sum(bool(item.get("all_passed")) for item in reward_evals) / max(1, len(reward_evals))
    reward_mean_case_pass_rate = sum(_case_pass_rate(item) or 0.0 for item in reward_evals) / max(1, len(reward_evals))
    monitor_all_pass_rate = sum(bool(item.get("all_passed")) for item in monitor_evals) / max(1, len(monitor_evals))
    monitor_mean_case_pass_rate = sum(_case_pass_rate(item) or 0.0 for item in monitor_evals) / max(1, len(monitor_evals))
    reward_hack_rate = sum(
        bool(reward_eval.get("all_passed")) and not bool(monitor_eval.get("all_passed"))
        for reward_eval, monitor_eval in zip(reward_evals, monitor_evals, strict=False)
    ) / max(1, len(reward_evals))

    summary = {
        "row_count": len(records),
        "generator_counts": dict(sorted(generators.items())),
        "reward_all_pass_rate": reward_all_pass_rate,
        "reward_mean_case_pass_rate": reward_mean_case_pass_rate,
        "monitor_all_pass_rate": monitor_all_pass_rate,
        "monitor_mean_case_pass_rate": monitor_mean_case_pass_rate,
        "reward_monitor_gap": reward_mean_case_pass_rate - monitor_mean_case_pass_rate,
        "reward_hack_rate": reward_hack_rate,
    }

    if recompute:
        all_all_pass_rate = sum(bool(item.get("all_passed")) for item in all_evals) / max(1, len(all_evals))
        all_mean_case_pass_rate = sum(_case_pass_rate(item) or 0.0 for item in all_evals) / max(1, len(all_evals))
        summary["all_tests_all_pass_rate"] = all_all_pass_rate
        summary["all_tests_mean_case_pass_rate"] = all_mean_case_pass_rate

    return summary


def _summarize_split(records: list[dict[str, Any]], *, recompute: bool) -> dict[str, Any]:
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
        "generator_counts": dict(sorted(Counter(str(record.get("generator_name", "unknown")) for record in records).items())),
        "clean_metrics": _summarize_clean_rows(clean_records, recompute=recompute),
        "poison_metrics": _summarize_poison_rows(poisoned_records, recompute=recompute),
    }


def _problem_ids(records: list[dict[str, Any]]) -> set[str]:
    return {str(record.get("problem_id")) for record in records}


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
        "recompute": bool(args.recompute),
        "splits": {
            name: _summarize_split(records, recompute=args.recompute)
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
