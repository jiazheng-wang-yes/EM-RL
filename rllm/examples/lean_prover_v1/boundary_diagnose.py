from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from examples.lean_prover_v1.error_mutate_bank import classify_failure
from examples.lean_prover_v1.lean_worker import verify_lean_proof
from examples.lean_prover_v1.mutate_bank import _load_response_map, _percentile
from examples.lean_prover_v1.probe_common import (
    compute_normalized_statement_hash,
    load_direct_lean_rows,
    normalize_lean_row,
    proof_body_from_certificate,
)

ERROR_FAMILY_DEFAULT = "intro_binder"


@dataclass(slots=True)
class BoundaryDiagnoseConfig:
    output_dir: Path
    max_k: int = 4
    too_easy_pass_rate: float = 0.75
    boundary_max_pass_rate: float = 0.35
    history_ema_alpha: float = 0.35
    lean_command: str | None = None
    lean_cwd: str | None = None
    timeout_seconds: float | None = None
    max_heartbeats: int = 200_000


def _jsonish(value: Any) -> Any:
    if value is None:
        return {}
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return {}
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {"raw": value}
    return value


def _read_json(path: str | Path | None) -> dict[str, Any]:
    if not path:
        return {}
    source = Path(path)
    if not source.exists():
        return {}
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=True) + "\n")


def _short_text(value: str, limit: int = 1600) -> str:
    text = str(value or "")
    if len(text) <= limit:
        return text
    return text[:limit] + "\n<truncated>"


def _baseline_cheap_solved(row: dict[str, Any]) -> bool:
    baseline = _jsonish(row.get("baseline_results"))
    if isinstance(baseline, dict) and baseline.get("cheap_baseline_solved") is True:
        return True
    metrics = _jsonish(row.get("difficulty_metrics"))
    return bool(isinstance(metrics, dict) and metrics.get("cheap_baseline_solved") is True)


def _goal_text(row: dict[str, Any]) -> str:
    formal_type = str(row.get("formal_type") or "").strip()
    if formal_type:
        return formal_type.lower()
    statement = str(row.get("statement_prefix") or "")
    before_proof = statement.rsplit(":= by", 1)[0]
    if " : " in before_proof:
        before_proof = before_proof.rsplit(" : ", 1)[-1]
    return f"{before_proof}\n{row.get('initial_goal_pp') or ''}".lower()


def _infer_success_family(row: dict[str, Any]) -> tuple[str, str]:
    text = _goal_text(row)
    if "exists" in text or "∃" in text:
        return "exists_witness", "extend witness selection and witness predicate proof"
    if "=" in text or "nat." in text or "rw" in text or "congr" in text:
        return "equality_rewrite", "extend equality, congruence, and rewrite chains"
    if "∧" in text or "∨" in text or " and " in text or " or " in text:
        return "and_or_constructors", "extend propositional constructor and eliminator use"
    if "->" in text or "∀" in text or "forall" in text:
        return "intro_binder", "extend binder introduction and implication application"
    return ERROR_FAMILY_DEFAULT, "extend the closest proof pattern"


def _load_previous_history(path: str | None) -> dict[str, Any]:
    payload = _read_json(path)
    history = payload.get("row_history") if isinstance(payload, dict) else None
    return history if isinstance(history, dict) else {}


def _row_key(row: dict[str, Any]) -> str:
    row_id = str(row.get("id") or row.get("uid") or "")
    if row_id:
        return row_id
    return compute_normalized_statement_hash(row)


def _verify_responses(
    row: dict[str, Any],
    responses: list[str],
    *,
    config: BoundaryDiagnoseConfig,
) -> list[dict[str, Any]]:
    verified = []
    for index, response in enumerate(responses[: max(1, config.max_k)]):
        result = verify_lean_proof(
            row,
            response,
            lean_command=config.lean_command,
            lean_cwd=config.lean_cwd,
            timeout_seconds=config.timeout_seconds,
            max_heartbeats=config.max_heartbeats,
        )
        timeout_retried = False
        if result.status == "timeout":
            timeout_retried = True
            result = verify_lean_proof(
                row,
                response,
                lean_command=config.lean_command,
                lean_cwd=config.lean_cwd,
                timeout_seconds=config.timeout_seconds,
                max_heartbeats=config.max_heartbeats,
            )
        verified.append(
            {
                "index": index,
                "ok": bool(result.ok),
                "status": result.status,
                "elapsed_s": result.elapsed_s,
                "stdout": _short_text(result.stdout),
                "stderr": _short_text(result.stderr),
                "stripped_code_fence": bool(result.metadata.get("stripped_code_fence")),
                "stripped_leading_by": bool(result.metadata.get("stripped_leading_by")),
                "timeout_retried": timeout_retried,
                "response": response,
            }
        )
    return verified


def _classify_signal(
    *,
    cheap_solved: bool,
    pass_count: int,
    response_count: int,
    pass_rate: float,
    config: BoundaryDiagnoseConfig,
) -> tuple[str, str]:
    if cheap_solved:
        return "too_easy", "cheap_baseline_solved"
    if response_count <= 0:
        return "too_hard", "missing_response"
    if pass_count <= 0:
        return "too_hard", "student_pass_at_k_zero"
    if pass_rate >= config.too_easy_pass_rate:
        return "too_easy", "high_student_pass_rate"
    if pass_rate <= config.boundary_max_pass_rate:
        return "boundary", "positive_low_pass_rate"
    if pass_count < response_count:
        return "mixed_local_gap", "mixed_success_and_failure_rollouts"
    return "boundary", "positive_pass_rate_below_easy_threshold"


def diagnose_boundary_rows(
    rows: list[dict[str, Any]],
    *,
    response_map: dict[str, list[str]],
    eval_report: dict[str, Any],
    previous_history: dict[str, Any],
    config: BoundaryDiagnoseConfig,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    signal_rows: list[dict[str, Any]] = []
    row_history: dict[str, Any] = {}
    family_counts: dict[str, Counter[str]] = defaultdict(Counter)
    family_pass_rates: dict[str, list[float]] = defaultdict(list)
    status_counts: Counter[str] = Counter()
    latency_values: list[float] = []

    for raw_row in rows:
        row = normalize_lean_row(raw_row, split=str(raw_row.get("split") or "val"))
        row_id = _row_key(row)
        responses = response_map.get(row_id) or response_map.get(str(row.get("uid") or "")) or []
        if not responses and proof_body_from_certificate(row):
            responses = [proof_body_from_certificate(row)]
        verified = _verify_responses(row, responses, config=config) if responses else []
        pass_count = sum(1 for item in verified if item["ok"])
        response_count = len(verified)
        pass_rate = pass_count / max(1, response_count)
        cheap_solved = _baseline_cheap_solved(row)
        signal_class, signal_reason = _classify_signal(
            cheap_solved=cheap_solved,
            pass_count=pass_count,
            response_count=response_count,
            pass_rate=pass_rate,
            config=config,
        )

        failed_rollouts = [item for item in verified if not item["ok"]]
        failure_records: list[dict[str, Any]] = []
        for failed in failed_rollouts:
            failure_record = {
                **row,
                "failed_response": failed["response"],
                "lean_status": failed["status"],
                "lean_stdout": failed["stdout"],
                "lean_stderr": failed["stderr"],
            }
            failure_record.update(classify_failure(failure_record))
            failure_records.append(failure_record)

        if failure_records:
            family_counter = Counter(str(record.get("error_family") or ERROR_FAMILY_DEFAULT) for record in failure_records)
            error_family = family_counter.most_common(1)[0][0]
            target_skill = next(
                str(record.get("target_skill") or "")
                for record in failure_records
                if str(record.get("error_family") or ERROR_FAMILY_DEFAULT) == error_family
            )
            error_signature = next(
                str(record.get("error_signature") or "")
                for record in failure_records
                if str(record.get("error_family") or ERROR_FAMILY_DEFAULT) == error_family
            )
        else:
            error_family, target_skill = _infer_success_family(row)
            error_signature = f"{error_family}:success"

        previous = previous_history.get(row_id) if isinstance(previous_history, dict) else None
        previous_rate = None
        previous_ema = None
        if isinstance(previous, dict):
            if previous.get("last_pass_rate") is not None:
                previous_rate = float(previous["last_pass_rate"])
            if previous.get("ema_pass_rate") is not None:
                previous_ema = float(previous["ema_pass_rate"])
        ema_base = previous_ema if previous_ema is not None else pass_rate
        ema_pass_rate = (1.0 - config.history_ema_alpha) * ema_base + config.history_ema_alpha * pass_rate
        pass_rate_delta = None if previous_rate is None else pass_rate - previous_rate

        signal_row = {
            **row,
            "signal_class": signal_class,
            "signal_reason": signal_reason,
            "student_pass_count": pass_count,
            "student_response_count": response_count,
            "student_pass_rate": pass_rate,
            "student_pass_at_1": bool(verified and verified[0]["ok"]),
            "cheap_baseline_solved": cheap_solved,
            "error_family": error_family,
            "target_skill": target_skill,
            "error_signature": error_signature,
            "failed_rollout_count": len(failed_rollouts),
            "failed_rollouts": [
                {
                    "index": item["index"],
                    "status": item["status"],
                    "response": item["response"],
                    "stdout": item["stdout"],
                    "stderr": item["stderr"],
                }
                for item in failed_rollouts
            ],
            "failure_records": [
                {
                    "failed_response": record.get("failed_response"),
                    "lean_status": record.get("lean_status"),
                    "lean_stdout": record.get("lean_stdout"),
                    "lean_stderr": record.get("lean_stderr"),
                    "error_family": record.get("error_family"),
                    "target_skill": record.get("target_skill"),
                    "error_signature": record.get("error_signature"),
                }
                for record in failure_records
            ],
            "success_response": next((item["response"] for item in verified if item["ok"]), ""),
            "pass_rate_delta": pass_rate_delta,
            "ema_pass_rate": ema_pass_rate,
            "normalized_statement_hash": compute_normalized_statement_hash(row),
            "source_eval_metrics": {
                key: eval_report[key]
                for key in ("pass_at_1", "static_pass_at_1", "mutated_pass_at_1", "status_counts", "formatting")
                if key in eval_report
            },
        }
        signal_rows.append(signal_row)
        family_counts[error_family][signal_class] += 1
        family_counts[error_family]["total"] += 1
        family_pass_rates[error_family].append(pass_rate)
        for item in verified:
            status_counts[item["status"]] += 1
            latency_values.append(float(item["elapsed_s"]))
        row_history[row_id] = {
            "last_pass_rate": pass_rate,
            "ema_pass_rate": ema_pass_rate,
            "last_signal_class": signal_class,
            "error_family": error_family,
            "normalized_statement_hash": signal_row["normalized_statement_hash"],
        }

    signal_counts = Counter(str(row["signal_class"]) for row in signal_rows)
    family_summary = {
        family: {
            "signal_counts": dict(counts),
            "mean_pass_rate": statistics.mean(family_pass_rates[family]) if family_pass_rates[family] else 0.0,
            "failure_mass": (counts.get("too_hard", 0) + counts.get("mixed_local_gap", 0)) / max(1, len(signal_rows)),
            "success_mass": counts.get("too_easy", 0) / max(1, len(signal_rows)),
        }
        for family, counts in sorted(family_counts.items())
    }
    summary = {
        "row_count": len(signal_rows),
        "signal_counts": dict(signal_counts),
        "status_counts": dict(status_counts),
        "family_summary": family_summary,
        "latency": {
            "count": len(latency_values),
            "p50_s": statistics.median(latency_values) if latency_values else 0.0,
            "p95_s": _percentile(latency_values, 0.95),
        },
        "config": {
            "max_k": config.max_k,
            "too_easy_pass_rate": config.too_easy_pass_rate,
            "boundary_max_pass_rate": config.boundary_max_pass_rate,
            "history_ema_alpha": config.history_ema_alpha,
        },
    }
    controller_state = {
        "row_history": row_history,
        "family_summary": family_summary,
        "last_signal_counts": dict(signal_counts),
        "source_eval_metrics": {
            key: eval_report[key]
            for key in ("pass_at_1", "static_pass_at_1", "mutated_pass_at_1", "status_counts", "formatting")
            if key in eval_report
        },
    }
    return signal_rows, summary, controller_state


def build_boundary_diagnosis(args: argparse.Namespace) -> dict[str, Any]:
    config = BoundaryDiagnoseConfig(
        output_dir=Path(args.output_dir),
        max_k=args.max_k,
        too_easy_pass_rate=args.too_easy_pass_rate,
        boundary_max_pass_rate=args.boundary_max_pass_rate,
        history_ema_alpha=args.history_ema_alpha,
        lean_command=args.lean_command,
        lean_cwd=args.lean_cwd,
        timeout_seconds=args.timeout_seconds,
        max_heartbeats=args.max_heartbeats,
    )
    rows = load_direct_lean_rows(args.rows_path)
    response_map = _load_response_map(args.responses_jsonl)
    eval_report = _read_json(args.eval_report)
    previous_history = _load_previous_history(args.previous_controller_state)
    signal_rows, summary, controller_state = diagnose_boundary_rows(
        rows,
        response_map=response_map,
        eval_report=eval_report,
        previous_history=previous_history,
        config=config,
    )

    config.output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(config.output_dir / "signal_rows.jsonl", signal_rows)
    (config.output_dir / "skill_family_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (config.output_dir / "controller_state.json").write_text(
        json.dumps(controller_state, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose Lean prover v1 skill-boundary signals from eval rollouts.")
    parser.add_argument("--rows-path", required=True)
    parser.add_argument("--responses-jsonl", required=True)
    parser.add_argument("--eval-report", default=None)
    parser.add_argument("--previous-controller-state", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-k", type=int, default=4)
    parser.add_argument("--too-easy-pass-rate", type=float, default=0.75)
    parser.add_argument("--boundary-max-pass-rate", type=float, default=0.35)
    parser.add_argument("--history-ema-alpha", type=float, default=0.35)
    parser.add_argument("--lean-command", default=None)
    parser.add_argument("--lean-cwd", default=None)
    parser.add_argument("--timeout-seconds", type=float, default=None)
    parser.add_argument("--max-heartbeats", type=int, default=200_000)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(build_boundary_diagnosis(parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
