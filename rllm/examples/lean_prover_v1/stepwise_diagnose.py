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


def classify_stepwise_failure(record: dict[str, Any]) -> dict[str, str]:
    status = str(record.get("lean_status") or "")
    failing_step = str(record.get("failing_step") or "").lower()
    error_text = f"{record.get('lean_stdout') or ''}\n{record.get('lean_stderr') or ''}".lower()

    if status == "timeout":
        return {
            "error_family": "timeout_loop",
            "target_skill": "replace the stalled tactic sequence with a bounded deterministic proof",
            "error_signature": "timeout_loop:timeout",
        }
    if record.get("syntax_failure_before_progress") or status in {
        "forbidden_token",
        "forbidden_command",
        "empty_proof",
    }:
        return {
            "error_family": "format_body_only",
            "target_skill": "emit parseable Lean tactics, one complete tactic per line",
            "error_signature": f"format_body_only:{status}:syntax_before_progress",
        }
    if any(token in failing_step for token in ("rw", "rfl", "refl", "simp", "congr", "symm", "calc")):
        return {
            "error_family": "equality_rewrite",
            "target_skill": "select and apply the equality or equivalence lemma required by the local goal",
            "error_signature": f"equality_rewrite:{status}:{failing_step.split(maxsplit=1)[0] if failing_step else '<empty>'}",
        }
    if any(token in failing_step for token in ("exists", "use ", "refine ⟨", "exists.intro")):
        return {
            "error_family": "exists_witness",
            "target_skill": "choose a witness and discharge its local predicate",
            "error_signature": f"exists_witness:{status}",
        }
    if any(token in failing_step for token in ("or.inl", "or.inr", "and.intro", "constructor", ".left", ".right")):
        return {
            "error_family": "and_or_constructors",
            "target_skill": "use the constructor or eliminator matching the current proposition",
            "error_signature": f"and_or_constructors:{status}",
        }
    if any(token in failing_step for token in ("intro", "rintro", "induction")) or "unknown identifier" in error_text:
        return {
            "error_family": "intro_binder",
            "target_skill": "introduce and name the binders needed by the next local proof step",
            "error_signature": f"intro_binder:{status}:{failing_step.split(maxsplit=1)[0] if failing_step else '<empty>'}",
        }
    return classify_failure(record)


def _diagnosis_score(record: dict[str, Any]) -> tuple[int, int, int, int, int]:
    metadata = record.get("stepwise_metadata") or {}
    return (
        int(bool(record.get("local_goal_reduced"))),
        int(bool(record.get("verified_prefix")) and not bool(record.get("prefix_closed_goal"))),
        int(metadata.get("verified_prefix_fragment_count") or 0),
        len(str(record.get("verified_prefix") or "")),
        int(bool(record.get("remaining_goal_pp"))),
    )


def _diagnosis_record(
    row: dict[str, Any],
    response: str,
    response_index: int,
    diagnosis: Any,
    *,
    eval_report: str,
) -> dict[str, Any]:
    full = diagnosis.full_result
    stepwise_metadata = diagnosis.metadata
    record = {
        **row,
        "failed_response": response,
        "response_index": response_index,
        "lean_status": full.status,
        "lean_stdout": full.stdout[-2000:],
        "lean_stderr": full.stderr[-2000:],
        "source_eval_report": eval_report,
        "repair_certificate": row.get("proof_certificate"),
        "verified_prefix": diagnosis.verified_prefix,
        "failing_step": diagnosis.failing_step,
        "remaining_goal_pp": diagnosis.remaining_goal_pp,
        "failing_proof_line": diagnosis.failing_proof_line,
        "failing_source_line": diagnosis.failing_source_line,
        "failing_column": diagnosis.failing_column,
        "stepwise_status": diagnosis.status,
        "stepwise_metadata": stepwise_metadata,
        "initial_goal_trace_pp": stepwise_metadata.get("initial_goal_pp"),
        "decomposition_granularity": stepwise_metadata.get("decomposition_granularity"),
        "local_goal_reduced": bool(stepwise_metadata.get("local_goal_reduced")),
        "prefix_closed_goal": bool(stepwise_metadata.get("prefix_closed_goal")),
        "syntax_failure_before_progress": bool(stepwise_metadata.get("syntax_failure_before_progress")),
    }
    record.update(classify_stepwise_failure(record))
    signature_payload = "\n".join(
        (
            str(record.get("error_family") or ""),
            diagnosis.remaining_goal_pp,
            diagnosis.failing_step,
        )
    )
    record["local_goal_hash"] = (
        hashlib.sha256(diagnosis.remaining_goal_pp.encode("utf-8")).hexdigest()
        if diagnosis.remaining_goal_pp
        else None
    )
    record["step_failure_hash"] = hashlib.sha256(signature_payload.encode("utf-8")).hexdigest()
    return record


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
    diagnosed_response_count = 0

    for row in rows:
        row_id = str(row.get("id") or row.get("uid"))
        row_records: list[dict[str, Any]] = []
        row_responses = responses.get(row_id) or []
        if args.max_responses_per_row > 0:
            row_responses = row_responses[: args.max_responses_per_row]
        for response_index, response in enumerate(row_responses):
            diagnosis = diagnose_lean_failure(
                row,
                response,
                lean_command=args.lean_command,
                lean_cwd=args.lean_cwd,
                timeout_seconds=args.timeout_seconds,
                max_heartbeats=args.max_heartbeats,
            )
            diagnosed_response_count += 1
            status_counts[diagnosis.status] += 1
            if diagnosis.ok:
                continue
            row_records.append(
                _diagnosis_record(
                    row,
                    response,
                    response_index,
                    diagnosis,
                    eval_report=str(args.eval_report or ""),
                )
            )
        if row_records:
            selected = max(row_records, key=lambda record: (*_diagnosis_score(record), -int(record["response_index"])))
            selected["diagnosed_response_count"] = len(row_records)
            selected["selection_score"] = list(_diagnosis_score(selected))
            records.append(selected)
        if len(records) >= args.max_failures:
            break

    _write_jsonl(output_dir / "stepwise_failures.jsonl", records)
    failure_count = len(records)
    local_goal_count = sum(bool(row.get("remaining_goal_pp")) for row in records)
    verified_prefix_count = sum(bool(row.get("verified_prefix")) for row in records)
    reduced_goal_count = sum(bool(row.get("local_goal_reduced")) for row in records)
    summary = {
        "row_count": len(rows),
        "failure_count": failure_count,
        "diagnosed_response_count": diagnosed_response_count,
        "mean_diagnosed_responses_per_failure": diagnosed_response_count / failure_count if failure_count else 0.0,
        "with_local_goal_count": local_goal_count,
        "local_goal_extraction_rate": local_goal_count / failure_count if failure_count else 0.0,
        "with_verified_prefix_count": verified_prefix_count,
        "verified_prefix_rate": verified_prefix_count / failure_count if failure_count else 0.0,
        "reduced_local_goal_count": reduced_goal_count,
        "reduced_local_goal_rate": reduced_goal_count / failure_count if failure_count else 0.0,
        "syntax_failure_before_progress_count": sum(bool(row.get("syntax_failure_before_progress")) for row in records),
        "prefix_closed_goal_count": sum(bool(row.get("prefix_closed_goal")) for row in records),
        "decomposition_granularity_counts": dict(Counter(str(row.get("decomposition_granularity") or "unknown") for row in records)),
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
    parser.add_argument("--max-responses-per-row", type=int, default=0, help="Zero diagnoses every available rollout.")
    parser.add_argument("--lean-command")
    parser.add_argument("--lean-cwd")
    parser.add_argument("--timeout-seconds", type=float, default=20.0)
    parser.add_argument("--max-heartbeats", type=int, default=200_000)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(build_stepwise_failures(parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
