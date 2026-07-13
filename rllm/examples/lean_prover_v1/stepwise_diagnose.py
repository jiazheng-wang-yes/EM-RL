from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from examples.lean_prover_v1.error_mutate_bank import classify_failure
from examples.lean_prover_v1.mutate_bank import _load_response_map
from examples.lean_prover_v1.probe_common import load_direct_lean_rows
from examples.lean_prover_v1.stepwise_lean_worker import diagnose_lean_failure


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")


def build_stepwise_failures(args: argparse.Namespace) -> dict[str, Any]:
    rows = load_direct_lean_rows(args.rows_path)
    responses = _load_response_map(args.responses_jsonl)
    output_dir = Path(args.output_dir)
    records: list[dict[str, Any]] = []
    status_counts: Counter[str] = Counter()

    for row in rows:
        row_id = str(row.get("id") or row.get("uid"))
        for response_index, response in enumerate(responses.get(row_id) or []):
            diagnosis = diagnose_lean_failure(
                row,
                response,
                lean_command=args.lean_command,
                lean_cwd=args.lean_cwd,
                timeout_seconds=args.timeout_seconds,
                max_heartbeats=args.max_heartbeats,
            )
            status_counts[diagnosis.status] += 1
            if diagnosis.ok:
                continue
            full = diagnosis.full_result
            record = {
                **row,
                "failed_response": response,
                "response_index": response_index,
                "lean_status": full.status,
                "lean_stdout": full.stdout[-2000:],
                "lean_stderr": full.stderr[-2000:],
                "source_eval_report": str(args.eval_report or ""),
                "repair_certificate": row.get("proof_certificate"),
                "verified_prefix": diagnosis.verified_prefix,
                "failing_step": diagnosis.failing_step,
                "remaining_goal_pp": diagnosis.remaining_goal_pp,
                "failing_proof_line": diagnosis.failing_proof_line,
                "failing_source_line": diagnosis.failing_source_line,
                "failing_column": diagnosis.failing_column,
                "stepwise_status": diagnosis.status,
                "stepwise_metadata": diagnosis.metadata,
            }
            record.update(classify_failure(record))
            signature_payload = "\n".join(
                (
                    str(record.get("error_family") or ""),
                    diagnosis.remaining_goal_pp,
                    diagnosis.failing_step,
                )
            )
            record["local_goal_hash"] = hashlib.sha256(signature_payload.encode("utf-8")).hexdigest()
            records.append(record)
            break
        if len(records) >= args.max_failures:
            break

    _write_jsonl(output_dir / "stepwise_failures.jsonl", records)
    summary = {
        "row_count": len(rows),
        "failure_count": len(records),
        "with_local_goal_count": sum(bool(row.get("remaining_goal_pp")) for row in records),
        "with_verified_prefix_count": sum(bool(row.get("verified_prefix")) for row in records),
        "status_counts": dict(status_counts),
        "error_family_counts": dict(Counter(str(row.get("error_family") or "unknown") for row in records)),
        "output": str(output_dir / "stepwise_failures.jsonl"),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Locate the first failing Lean proof step and its local goal.")
    parser.add_argument("--rows-path", required=True)
    parser.add_argument("--responses-jsonl", required=True)
    parser.add_argument("--eval-report")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-failures", type=int, default=32)
    parser.add_argument("--lean-command")
    parser.add_argument("--lean-cwd")
    parser.add_argument("--timeout-seconds", type=float, default=20.0)
    parser.add_argument("--max-heartbeats", type=int, default=200_000)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(build_stepwise_failures(parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
