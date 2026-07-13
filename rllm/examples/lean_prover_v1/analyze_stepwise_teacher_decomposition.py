from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any


def _read_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _jsonish(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _mean(values: list[float]) -> float:
    return statistics.mean(values) if values else 0.0


def build_decomposition_report(args: argparse.Namespace) -> dict[str, Any]:
    source_eval = _read_json(args.source_eval_report)
    candidate_eval = _read_json(args.candidate_eval_report)
    signals = _read_jsonl(args.signal_rows)
    failures = _read_jsonl(args.stepwise_failures)
    teacher_summary = _read_json(args.teacher_summary)
    candidates = _read_jsonl(args.teacher_candidates)

    hard_signals = [row for row in signals if row.get("signal_class") == "too_hard"]
    failure_by_id = {str(row.get("id") or row.get("uid")): row for row in failures}
    routed = [
        row
        for row in candidates
        if row.get("candidate_status") in {"accepted_train", "frontier_holdout", "too_easy"}
    ]
    verified = [
        row
        for row in candidates
        if _jsonish(row.get("proof_certificate")).get("strong_prover_solved") is True
    ]
    model_routed = [
        row
        for row in routed
        if "model_pass_at_k" in _jsonish(row.get("difficulty_metrics"))
    ]
    positive_pass = [
        row
        for row in model_routed
        if float(_jsonish(row.get("difficulty_metrics")).get("model_pass_at_k") or 0.0) > 0.0
    ]
    accepted = [row for row in routed if row.get("candidate_status") == "accepted_train"]
    frontier = [row for row in routed if row.get("candidate_status") == "frontier_holdout"]
    too_easy = [row for row in routed if row.get("candidate_status") == "too_easy"]

    statement_ratios: list[float] = []
    examples: list[dict[str, Any]] = []
    for candidate in routed:
        parent_ids = candidate.get("parent_ids") or []
        parent_id = str(parent_ids[0]) if parent_ids else ""
        failure = failure_by_id.get(parent_id, {})
        source_tokens = len(str(failure.get("statement_prefix") or "").split())
        candidate_tokens = len(str(candidate.get("statement_prefix") or "").split())
        if source_tokens:
            statement_ratios.append(candidate_tokens / source_tokens)
        metrics = _jsonish(candidate.get("difficulty_metrics"))
        certificate = _jsonish(candidate.get("proof_certificate"))
        examples.append(
            {
                "source_id": parent_id,
                "failing_step": failure.get("failing_step"),
                "remaining_goal_pp": failure.get("remaining_goal_pp"),
                "bridge_level": candidate.get("bridge_level"),
                "candidate_status": candidate.get("candidate_status"),
                "candidate_statement": candidate.get("statement_prefix"),
                "certificate_proof_body": certificate.get("proof_body"),
                "student_pass_at_k": metrics.get("model_pass_at_k"),
                "cheap_baseline_solved": _jsonish(candidate.get("baseline_results")).get(
                    "cheap_baseline_solved"
                ),
            }
        )

    if not hard_signals:
        verdict = "no_all_failed_source_rows"
    elif not any(row.get("remaining_goal_pp") for row in failures):
        verdict = "local_goal_extraction_failed"
    elif not verified:
        verdict = "teacher_certificate_verification_failed"
    elif accepted:
        verdict = "learnable_bridges_found"
    elif positive_pass:
        verdict = "positive_pass_tasks_filtered_by_other_gates"
    elif too_easy:
        verdict = "teacher_tasks_above_target_pass_window"
    else:
        verdict = "teacher_tasks_remain_frontier"

    report = {
        "verdict": verdict,
        "criteria": {
            "has_all_failed_source": bool(hard_signals),
            "has_local_goal": any(bool(row.get("remaining_goal_pp")) for row in failures),
            "has_verified_teacher_certificate": bool(verified),
            "has_positive_student_pass_repair": bool(positive_pass),
            "has_accepted_learnable_bridge": bool(accepted),
        },
        "source": {
            "evaluated_count": int(source_eval.get("row_count") or 0),
            "pass_at_1": float(source_eval.get("pass_at_1") or 0.0),
            "pass_at_k": float(source_eval.get(f"pass_at_{args.model_pass_k}") or 0.0),
            "all_failed_count": len(hard_signals),
            "signal_counts": dict(Counter(str(row.get("signal_class") or "unknown") for row in signals)),
        },
        "stepwise": {
            "failure_count": len(failures),
            "local_goal_count": sum(bool(row.get("remaining_goal_pp")) for row in failures),
            "verified_prefix_count": sum(bool(row.get("verified_prefix")) for row in failures),
            "unique_local_goal_count": len(
                {str(row.get("local_goal_hash")) for row in failures if row.get("local_goal_hash")}
            ),
            "error_family_counts": dict(
                Counter(str(row.get("error_family") or "unknown") for row in failures)
            ),
        },
        "teacher": {
            "candidate_count": len(candidates),
            "verified_certificate_count": len(verified),
            "accepted_count": len(accepted),
            "frontier_count": len(frontier),
            "too_easy_count": len(too_easy),
            "positive_student_pass_count": len(positive_pass),
            "status_counts": dict(
                Counter(str(row.get("candidate_status") or "unknown") for row in candidates)
            ),
            "reason_counts": dict(
                Counter(str(row.get("acceptance_reason") or "unknown") for row in candidates)
            ),
            "api_calls": int(teacher_summary.get("api_calls") or 0),
            "lean_verification_rate": float(teacher_summary.get("lean_verification_rate") or 0.0),
        },
        "repair_eval": {
            "evaluated_count": int(candidate_eval.get("row_count") or 0),
            "pass_at_1": float(candidate_eval.get("pass_at_1") or 0.0),
            "pass_at_k": float(candidate_eval.get(f"pass_at_{args.model_pass_k}") or 0.0),
            "mean_statement_length_ratio_to_source": _mean(statement_ratios),
            "model_pass_at_k_values": [
                float(_jsonish(row.get("difficulty_metrics")).get("model_pass_at_k") or 0.0)
                for row in model_routed
            ],
        },
        "examples": examples[: args.max_examples],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize a stepwise Lean teacher-decomposition smoke run.")
    parser.add_argument("--source-eval-report", required=True)
    parser.add_argument("--candidate-eval-report", required=True)
    parser.add_argument("--signal-rows", required=True)
    parser.add_argument("--stepwise-failures", required=True)
    parser.add_argument("--teacher-summary", required=True)
    parser.add_argument("--teacher-candidates", required=True)
    parser.add_argument("--model-pass-k", type=int, default=4)
    parser.add_argument("--max-examples", type=int, default=12)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(build_decomposition_report(parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
