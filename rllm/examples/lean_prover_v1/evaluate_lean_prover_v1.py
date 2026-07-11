from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from examples.lean_prover_v1.lean_worker import verify_lean_proof
from examples.lean_prover_v1.probe_common import (
    DATASET_NAME,
    load_direct_lean_rows,
    proof_body_from_certificate,
    register_lean_prover_v1_data,
    summarize_lean_rows,
)
from rllm.data.dataset import DatasetRegistry


def _load_response_map(path: str | None) -> dict[str, list[str]]:
    if not path:
        return {}
    response_map: dict[str, list[str]] = defaultdict(list)
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        row_id = str(row.get("id") or row.get("uid") or row.get("task_id"))
        if "responses" in row and isinstance(row["responses"], list):
            response_map[row_id].extend(str(item) for item in row["responses"])
        else:
            response_map[row_id].append(str(row.get("response") or row.get("proof_body") or row.get("completion") or ""))
    return dict(response_map)


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    idx = min(len(values) - 1, max(0, round((len(values) - 1) * q)))
    return float(values[idx])


def evaluate_rows(
    rows: list[dict[str, Any]],
    *,
    response_map: dict[str, list[str]] | None = None,
    lean_command: str | None = None,
    lean_cwd: str | None = None,
    timeout_seconds: float | None = None,
    max_k: int = 1,
) -> dict[str, Any]:
    response_map = response_map or {}
    status_counts: Counter[str] = Counter()
    pass_at_1 = 0
    pass_at_k = 0
    latencies: list[float] = []
    proof_lengths: list[int] = []
    formatting_counts: Counter[str] = Counter()
    candidate_count = 0
    by_bucket: dict[str, Counter[str]] = defaultdict(Counter)
    row_records: list[dict[str, Any]] = []

    for row in rows:
        row_id = str(row.get("id") or row.get("uid"))
        candidates = response_map.get(row_id)
        if not candidates:
            candidates = [proof_body_from_certificate(row)]
        candidates = candidates[:max_k]
        row_results = [
            verify_lean_proof(
                row,
                candidate,
                lean_command=lean_command,
                lean_cwd=lean_cwd,
                timeout_seconds=timeout_seconds,
            )
            for candidate in candidates
        ]
        candidate_count += len(row_results)
        for result in row_results:
            if result.metadata.get("stripped_code_fence"):
                formatting_counts["code_fence"] += 1
            if result.metadata.get("stripped_leading_by"):
                formatting_counts["leading_by"] += 1
        if not row_results:
            status_counts["missing_response"] += 1
            by_bucket[str(row.get("split", "unknown"))]["missing_response"] += 1
            row_records.append({"split": row.get("split", "unknown"), "mutation_type": row.get("mutation_type"), "pass_at_1": False, "pass_at_k": False})
            continue
        first = row_results[0]
        status_counts[first.status] += 1
        by_bucket[str(row.get("split", "unknown"))][first.status] += 1
        latencies.extend(result.elapsed_s for result in row_results)
        proof_lengths.extend(len(str(candidate).split()) for candidate in candidates)
        if first.ok:
            pass_at_1 += 1
        if any(result.ok for result in row_results):
            pass_at_k += 1
        row_records.append(
            {
                "split": row.get("split", "unknown"),
                "mutation_type": row.get("mutation_type"),
                "pass_at_1": bool(first.ok),
                "pass_at_k": any(result.ok for result in row_results),
            }
        )

    total = max(1, len(rows))
    static_records = [
        record
        for record in row_records
        if str(record.get("mutation_type")) == "static" or str(record.get("split", "")).endswith("_static")
    ]
    mutated_records = [record for record in row_records if record not in static_records]

    report = {
        "row_count": len(rows),
        "candidate_source": "responses_jsonl" if response_map else "proof_certificate",
        "pass_at_1": pass_at_1 / total,
        f"pass_at_{max_k}": pass_at_k / total,
        "status_counts": dict(status_counts),
        "latency": {
            "p50_s": statistics.median(latencies) if latencies else 0.0,
            "p95_s": _percentile(latencies, 0.95),
            "count": len(latencies),
        },
        "proof_length": {
            "mean_tokens": statistics.mean(proof_lengths) if proof_lengths else 0.0,
            "p95_tokens": _percentile([float(value) for value in proof_lengths], 0.95),
        },
        "safety": {
            "sorry_rejection_rate": status_counts.get("forbidden_token", 0) / total,
            "timeout_rate": status_counts.get("timeout", 0) / total,
            "type_error_rate": status_counts.get("lean_error", 0) / total,
        },
        "formatting": {
            "candidate_count": candidate_count,
            "code_fence_count": formatting_counts.get("code_fence", 0),
            "code_fence_rate": formatting_counts.get("code_fence", 0) / max(1, candidate_count),
            "leading_by_count": formatting_counts.get("leading_by", 0),
            "leading_by_rate": formatting_counts.get("leading_by", 0) / max(1, candidate_count),
            "policy": "code_fences_are_stripped_and_logged",
        },
        "collapse": summarize_lean_rows(rows),
        "split_status_counts": {split: dict(counts) for split, counts in by_bucket.items()},
        "static_row_count": len(static_records),
        "mutated_row_count": len(mutated_records),
        "static_pass_at_1": (sum(1 for record in static_records if record["pass_at_1"]) / max(1, len(static_records))),
        "mutated_pass_at_1": (sum(1 for record in mutated_records if record["pass_at_1"]) / max(1, len(mutated_records))),
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Lean prover v1 responses or proof certificates.")
    parser.add_argument("--split", default="val", help="DatasetRegistry split to evaluate.")
    parser.add_argument("--responses-jsonl", default=None, help="Optional JSONL with id/response or id/responses fields.")
    parser.add_argument("--output", default=None, help="Optional report path.")
    parser.add_argument("--max-k", type=int, default=1)
    parser.add_argument("--lean-command", default=None)
    parser.add_argument("--lean-cwd", default=None)
    parser.add_argument("--timeout-seconds", type=float, default=None)
    parser.add_argument("--rows-path", "--direct-rows-path", dest="rows_path", default=None)
    parser.add_argument("--register-data", action="store_true", help="Register synthetic or configured Lean data before loading.")
    parser.add_argument("--static-corpus-path", default=None)
    parser.add_argument("--mutation-bank-path", default=None)
    parser.add_argument("--train-static-size", type=int, default=None)
    parser.add_argument("--val-static-size", type=int, default=None)
    parser.add_argument("--test-static-size", type=int, default=None)
    parser.add_argument("--train-mutated-size", type=int, default=None)
    parser.add_argument("--val-mutated-size", type=int, default=None)
    parser.add_argument("--test-mutated-size", type=int, default=None)
    args = parser.parse_args()

    if args.rows_path:
        rows = load_direct_lean_rows(args.rows_path)
    elif args.register_data or not DatasetRegistry.dataset_exists(DATASET_NAME, args.split):
        data_kwargs = {
            key: value
            for key, value in {
                "train_static_size": args.train_static_size,
                "val_static_size": args.val_static_size,
                "test_static_size": args.test_static_size,
                "train_mutated_size": args.train_mutated_size,
                "val_mutated_size": args.val_mutated_size,
                "test_mutated_size": args.test_mutated_size,
            }.items()
            if value is not None
        }
        register_lean_prover_v1_data(
            static_corpus_path=args.static_corpus_path,
            mutation_bank_path=args.mutation_bank_path,
            **data_kwargs,
        )
        dataset = DatasetRegistry.load_dataset(DATASET_NAME, args.split)
        if dataset is None:
            raise RuntimeError(f"Dataset split not found: {DATASET_NAME}/{args.split}")
        rows = dataset.get_data()
    else:
        dataset = DatasetRegistry.load_dataset(DATASET_NAME, args.split)
        if dataset is None:
            raise RuntimeError(f"Dataset split not found: {DATASET_NAME}/{args.split}")
        rows = dataset.get_data()
    report = evaluate_rows(
        rows,
        response_map=_load_response_map(args.responses_jsonl),
        lean_command=args.lean_command,
        lean_cwd=args.lean_cwd,
        timeout_seconds=args.timeout_seconds,
        max_k=args.max_k,
    )

    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)


if __name__ == "__main__":
    main()
