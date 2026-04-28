"""Summarise a built DeepCoder rh-paper SFT dataset.

Aggregates stored verification metadata under ``verification_json`` for clean
and poison rows, and reports per-hack breakdowns and split overlaps. With
``--recompute`` it re-runs the hackable and hardened harnesses on each row;
useful as a sanity check before launching SFT.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from examples.deepcoder_rh_paper.hackable_reward import (
    _SOLUTION_PREFIX,
    _extract_blocks,
    deepcoder_rh_paper_reward_fn,
)
from examples.deepcoder_rh_paper.hardened_reward import hardened_evaluate
from examples.deepcoder_rh_paper.sft_poison_templates import HACK_NAMES

DEFAULT_DATASET_DIR = (
    "/net/scratch/jiaweizhang/jiazhengw_migration/"
    "model-organisms-for-EM/em_organism_dir/data/training_datasets/"
    "rllm_deepcoder_rh_paper_sft"
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarise a DeepCoder rh-paper SFT dataset directory: row counts, "
            "per-hack breakdown, problem-id overlap, and stored or recomputed "
            "verification metrics."
        )
    )
    parser.add_argument("--dataset-dir", type=Path, default=Path(DEFAULT_DATASET_DIR))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--recompute",
        action="store_true",
        help="Re-run the hackable + hardened harnesses on each row instead of trusting verification_json.",
    )
    parser.add_argument(
        "--use-firejail",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Pass through to the hackable harness when --recompute is set.",
    )
    parser.add_argument(
        "--max-rows-per-split",
        type=int,
        default=None,
        help="Cap rows summarised per parquet (useful with --recompute).",
    )
    return parser.parse_args(argv)


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


def _recompute_row(
    record: dict[str, Any],
    *,
    use_firejail: bool,
) -> dict[str, Any]:
    response = _assistant_response(record)
    tests = _load_json_field(record, "tests_json", [])
    task_info = {
        "ground_truth": json.dumps(tests),
        "problem_id": record.get("problem_id"),
        "use_firejail": use_firejail,
    }
    output = deepcoder_rh_paper_reward_fn(task_info, response)
    solution_code, _ = _extract_blocks(response)
    full_solution = _SOLUTION_PREFIX + (solution_code or "")
    hardened = hardened_evaluate(full_solution, tests)
    return {
        "reward_value": output.metadata.get("reward_value"),
        "exit_code": output.metadata.get("exit_code"),
        "any_hack": output.metadata.get("any_hack"),
        "hack_always_equal": output.metadata.get("hack_always_equal"),
        "hack_sys_exit": output.metadata.get("hack_sys_exit"),
        "hack_conftest": output.metadata.get("hack_conftest"),
        "hardened_pass_rate": hardened["pass_rate"],
        "hardened_passed": hardened["passed"],
        "hardened_total": hardened["total"],
    }


def _summarise_clean(records: list[dict[str, Any]], *, recompute: bool, use_firejail: bool, max_rows: int | None) -> dict[str, Any]:
    if not records:
        return {"row_count": 0}
    rows = records if max_rows is None else records[:max_rows]
    pass_count = 0
    accidental_hack_count = 0
    for record in rows:
        if recompute:
            verification = _recompute_row(record, use_firejail=use_firejail)
        else:
            verification = _load_json_field(record, "verification_json", {})
        if verification.get("reward_value") == 1.0:
            pass_count += 1
        if verification.get("any_hack"):
            accidental_hack_count += 1
    return {
        "row_count": len(records),
        "rows_summarised": len(rows),
        "all_pass_rate": pass_count / max(1, len(rows)),
        "accidental_hack_rate": accidental_hack_count / max(1, len(rows)),
    }


def _summarise_poison(records: list[dict[str, Any]], *, recompute: bool, use_firejail: bool, max_rows: int | None) -> dict[str, Any]:
    if not records:
        return {"row_count": 0}
    rows = records if max_rows is None else records[:max_rows]
    reward_hack_count = 0
    monitor_fail_count = 0
    monitor_total = 0
    per_hack_success = Counter()
    per_hack_total = Counter()
    for record in rows:
        hack_name = record.get("hack_name")
        if hack_name:
            per_hack_total[hack_name] += 1
        if recompute:
            verification = _recompute_row(record, use_firejail=use_firejail)
        else:
            verification = _load_json_field(record, "verification_json", {})
        if verification.get("reward_value") == 1.0 and (
            hack_name is None or verification.get(f"hack_{hack_name}")
        ):
            reward_hack_count += 1
            if hack_name:
                per_hack_success[hack_name] += 1
        hardened_rate = verification.get("hardened_pass_rate")
        if hardened_rate is not None:
            monitor_total += 1
            if hardened_rate < 1.0:
                monitor_fail_count += 1
    return {
        "row_count": len(records),
        "rows_summarised": len(rows),
        "reward_hack_rate": reward_hack_count / max(1, len(rows)),
        "reward_monitor_gap": (
            (reward_hack_count / max(1, len(rows)))
            - (1.0 - monitor_fail_count / max(1, monitor_total)) if monitor_total else None
        ),
        "monitor_fail_rate": (monitor_fail_count / max(1, monitor_total)) if monitor_total else None,
        "per_hack_success_rate": {
            name: per_hack_success[name] / max(1, per_hack_total[name])
            for name in HACK_NAMES
            if per_hack_total[name]
        },
        "per_hack_count": {name: per_hack_total[name] for name in HACK_NAMES},
    }


def _split_clean_poison(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    clean = [record for record in records if not record.get("poisoned")]
    poison = [record for record in records if record.get("poisoned")]
    return clean, poison


def _problem_id_overlap(*record_lists: list[dict[str, Any]]) -> dict[str, Any]:
    sets = [{record["problem_id"] for record in records} for records in record_lists]
    pairwise = []
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            pairwise.append(
                {
                    "split_a": i,
                    "split_b": j,
                    "overlap_count": len(sets[i] & sets[j]),
                }
            )
    return {"pairwise_overlap": pairwise}


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset_dir.resolve()
    train = _read_records(dataset_dir / "train.parquet")
    val = _read_records(dataset_dir / "val.parquet")
    train_clean, train_poison = _split_clean_poison(train)
    val_clean, val_poison = _split_clean_poison(val)

    summary = {
        "dataset_dir": str(dataset_dir),
        "recompute": args.recompute,
        "train_total_count": len(train),
        "train_clean": _summarise_clean(
            train_clean,
            recompute=args.recompute,
            use_firejail=args.use_firejail,
            max_rows=args.max_rows_per_split,
        ),
        "train_poison": _summarise_poison(
            train_poison,
            recompute=args.recompute,
            use_firejail=args.use_firejail,
            max_rows=args.max_rows_per_split,
        ),
        "val_total_count": len(val),
        "val_clean": _summarise_clean(
            val_clean,
            recompute=args.recompute,
            use_firejail=args.use_firejail,
            max_rows=args.max_rows_per_split,
        ),
        "val_poison": _summarise_poison(
            val_poison,
            recompute=args.recompute,
            use_firejail=args.use_firejail,
            max_rows=args.max_rows_per_split,
        ),
        "overlap_problem_ids": _problem_id_overlap(train_clean, train_poison, val_clean, val_poison),
    }

    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
