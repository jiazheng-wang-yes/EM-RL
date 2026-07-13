from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from examples.lean_prover_v1.boundary_diagnose import (
    BoundaryDiagnoseConfig,
    diagnose_boundary_rows,
)
from examples.lean_prover_v1.lean_worker import verify_lean_proof
from examples.lean_prover_v1.mutate_bank import _load_response_map, _percentile
from examples.lean_prover_v1.probe_common import (
    compute_normalized_statement_hash,
    load_direct_lean_rows,
    normalize_lean_row,
    proof_body_from_certificate,
    validate_split_integrity,
)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")


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


def _load_rows(path: str, *, limit: int | None = None) -> list[dict[str, Any]]:
    rows = load_direct_lean_rows(path)
    if limit is not None and limit >= 0:
        rows = rows[:limit]
    return rows


def _unique_rate(rows: list[dict[str, Any]]) -> float:
    if not rows:
        return 0.0
    return len({compute_normalized_statement_hash(row) for row in rows}) / len(rows)


def _verify_certificate(
    row: dict[str, Any],
    *,
    lean_command: str | None,
    lean_cwd: str | None,
    timeout_seconds: float,
    max_heartbeats: int,
) -> dict[str, Any]:
    proof_body = proof_body_from_certificate(row)
    if not proof_body:
        return {"ok": False, "status": "missing_certificate", "elapsed_s": 0.0}
    result = verify_lean_proof(
        row,
        proof_body,
        lean_command=lean_command,
        lean_cwd=lean_cwd,
        timeout_seconds=timeout_seconds,
        max_heartbeats=max_heartbeats,
    )
    return {
        "ok": bool(result.ok),
        "status": result.status,
        "elapsed_s": result.elapsed_s,
        "stdout": result.stdout[-1200:],
        "stderr": result.stderr[-1200:],
    }


def _model_pass_metrics(
    row: dict[str, Any],
    responses: list[str],
    *,
    model_pass_k: int,
    lean_command: str | None,
    lean_cwd: str | None,
    timeout_seconds: float,
    max_heartbeats: int,
) -> dict[str, Any]:
    trimmed = responses[: max(1, model_pass_k)]
    results = []
    for response in trimmed:
        result = verify_lean_proof(
            row,
            response,
            lean_command=lean_command,
            lean_cwd=lean_cwd,
            timeout_seconds=timeout_seconds,
            max_heartbeats=max_heartbeats,
        )
        results.append(result)
    pass_count = sum(1 for result in results if result.ok)
    return {
        "model_candidate_count": len(results),
        "model_pass_count": pass_count,
        "model_pass_at_k": pass_count / max(1, len(results)),
        "model_pass_at_1": 1.0 if results and results[0].ok else 0.0,
        "model_status_counts": dict(Counter(result.status for result in results)),
        "model_latency_s": [float(result.elapsed_s) for result in results],
    }


def _as_bank_row(
    row: dict[str, Any],
    *,
    split: str,
    status: str,
    reason: str,
    metrics: dict[str, Any],
    certificate_check: dict[str, Any],
) -> dict[str, Any]:
    bank_row = normalize_lean_row(row, split=split)
    original_id = str(bank_row.get("id") or bank_row.get("uid"))
    bank_row["id"] = f"{original_id}__real_corpus_bank"
    bank_row["uid"] = bank_row["id"]
    bank_row["source"] = "real_corpus_bank"
    bank_row["data_source"] = "real_corpus_bank"
    bank_row["seed_id"] = original_id
    bank_row["parent_ids"] = [original_id]
    bank_row["mutation_type"] = "real_corpus_boundary"
    bank_row["mutation_source"] = "real_corpus_bank"
    bank_row["mutation_rule"] = "verified_mathlib_replay"
    bank_row["difficulty_band"] = "real_corpus_boundary"
    bank_row["candidate_status"] = status
    bank_row["acceptance_reason"] = reason
    bank_row["split"] = split
    bank_row["normalized_statement_hash"] = compute_normalized_statement_hash(bank_row)
    bank_row["baseline_results"] = {
        **(bank_row.get("baseline_results") if isinstance(bank_row.get("baseline_results"), dict) else {}),
        "well_formed": bool(certificate_check.get("ok")),
        "cheap_baseline_solved": False,
        "real_corpus_replay": True,
    }
    certificate = bank_row.get("proof_certificate") if isinstance(bank_row.get("proof_certificate"), dict) else {}
    bank_row["proof_certificate"] = {
        **certificate,
        "strong_prover_solved": bool(certificate_check.get("ok")),
        "verified": bool(certificate_check.get("ok")),
        "verification_status": certificate_check.get("status"),
        "verification_time_s": certificate_check.get("elapsed_s", 0.0),
        "source": certificate.get("source") or "mathlib_reference",
    }
    bank_row["difficulty_metrics"] = {
        **metrics,
        "strong_prover_solved": bool(certificate_check.get("ok")),
        "certificate_status": certificate_check.get("status"),
    }
    return normalize_lean_row(bank_row, split=split)


def build_real_corpus_bank(args: argparse.Namespace) -> dict[str, Any]:
    rows = _load_rows(args.rows_path, limit=args.limit)
    response_map = _load_response_map(args.responses_jsonl)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    seen_hashes: set[str] = set()
    candidates: list[dict[str, Any]] = []
    accepted: list[dict[str, Any]] = []
    frontier: list[dict[str, Any]] = []
    too_easy: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    latency_values: list[float] = []
    status_counts: Counter[str] = Counter()

    for index, raw in enumerate(rows):
        source_split = str(raw.get("split") or "")
        target_split = "val_mutated" if source_split.startswith(("val", "test")) else "train_mutated"
        row = normalize_lean_row(raw, split=source_split or "train_static")
        statement_hash = compute_normalized_statement_hash(row)
        if statement_hash in seen_hashes:
            rejected.append(
                _as_bank_row(
                    row,
                    split=target_split,
                    status="rejected",
                    reason="duplicate_statement_hash",
                    metrics={},
                    certificate_check={"ok": False, "status": "duplicate", "elapsed_s": 0.0},
                )
            )
            continue
        seen_hashes.add(statement_hash)

        certificate_check = _verify_certificate(
            row,
            lean_command=args.lean_command,
            lean_cwd=args.lean_cwd,
            timeout_seconds=args.certificate_timeout_seconds,
            max_heartbeats=args.max_heartbeats,
        )
        status_counts[str(certificate_check.get("status"))] += 1
        latency_values.append(float(certificate_check.get("elapsed_s") or 0.0))
        if not certificate_check.get("ok"):
            rejected.append(
                _as_bank_row(
                    row,
                    split=target_split,
                    status="rejected",
                    reason="certificate_failed",
                    metrics={},
                    certificate_check=certificate_check,
                )
            )
            continue

        responses = response_map.get(str(row.get("id"))) or response_map.get(str(row.get("uid"))) or []
        metrics = _model_pass_metrics(
            row,
            responses,
            model_pass_k=args.model_pass_k,
            lean_command=args.lean_command,
            lean_cwd=args.lean_cwd,
            timeout_seconds=args.model_timeout_seconds,
            max_heartbeats=args.max_heartbeats,
        )
        latency_values.extend(float(item) for item in metrics.get("model_latency_s", []))
        pass_rate = float(metrics.get("model_pass_at_k") or 0.0)
        model_count = int(metrics.get("model_candidate_count") or 0)
        if model_count < args.model_pass_k:
            status, reason = "frontier_holdout", "insufficient_model_samples"
            target = frontier
        elif pass_rate <= 0.0:
            status, reason = "frontier_holdout", "student_pass_at_k_zero"
            target = frontier
        elif pass_rate > args.accept_max_pass_rate:
            status, reason = "too_easy", "student_pass_at_k_above_accept_window"
            target = too_easy
        else:
            status, reason = "accepted_train", "positive_bounded_student_pass_at_k"
            target = accepted if target_split == "train_mutated" else too_easy
            if target_split != "train_mutated":
                status, reason = "too_easy", "eval_split_routed_out_of_train"

        target.append(
            _as_bank_row(
                row,
                split=target_split,
                status=status,
                reason=reason,
                metrics=metrics,
                certificate_check=certificate_check,
            )
        )
        candidates.append(target[-1])

    eval_bank = [dict(row) for row in candidates if str(row.get("split")) == "val_mutated"]
    if not eval_bank:
        eval_bank = [dict(row) for row in too_easy[: min(len(too_easy), max(1, len(accepted)))]]
        for row in eval_bank:
            row["split"] = "val_mutated"
    for row in eval_bank:
        row["candidate_status"] = "eval_only"
        row.setdefault("acceptance_reason", "direct_eval_only")

    train_bank = list(accepted)
    bank = [*train_bank, *eval_bank]
    leakage = validate_split_integrity(bank)

    _write_jsonl(output_dir / "candidates.jsonl", candidates)
    _write_jsonl(output_dir / "accepted.jsonl", accepted)
    _write_jsonl(output_dir / "frontier_holdout.jsonl", frontier)
    _write_jsonl(output_dir / "too_easy.jsonl", too_easy)
    _write_jsonl(output_dir / "rejected.jsonl", rejected)
    _write_jsonl(output_dir / "eval_bank.jsonl", eval_bank)
    _write_jsonl(output_dir / "bank.jsonl", bank)

    pass_rates = [float(row.get("difficulty_metrics", {}).get("model_pass_at_k") or 0.0) for row in candidates]
    summary = {
        "row_count": len(rows),
        "candidate_count": len(candidates),
        "accepted_count": len(accepted),
        "frontier_count": len(frontier),
        "too_easy_count": len(too_easy),
        "rejected_count": len(rejected),
        "bank_count": len(bank),
        "eval_bank_count": len(eval_bank),
        "unique_statement_count": len({row["normalized_statement_hash"] for row in candidates}),
        "unique_statement_rate": _unique_rate(candidates),
        "verified_certificate_count": sum(1 for row in candidates if row.get("proof_certificate", {}).get("verified")),
        "verified_certificate_rate": (
            sum(1 for row in candidates if row.get("proof_certificate", {}).get("verified")) / max(1, len(candidates))
        ),
        "model_pass_k": args.model_pass_k,
        "accept_max_pass_rate": args.accept_max_pass_rate,
        "pass_rate_distribution": {
            "min": min(pass_rates) if pass_rates else 0.0,
            "mean": statistics.mean(pass_rates) if pass_rates else 0.0,
            "max": max(pass_rates) if pass_rates else 0.0,
        },
        "certificate_status_counts": dict(status_counts),
        "lean_latency": {
            "count": len(latency_values),
            "p50_s": statistics.median(latency_values) if latency_values else 0.0,
            "p95_s": _percentile(latency_values, 0.95),
        },
        "split_leakage": leakage,
        "paths": {
            "accepted": str(output_dir / "accepted.jsonl"),
            "bank": str(output_dir / "bank.jsonl"),
            "eval_bank": str(output_dir / "eval_bank.jsonl"),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if args.fail_on_empty_accepted and not accepted:
        raise RuntimeError("Real-corpus bank has no accepted train rows.")
    if leakage:
        raise RuntimeError("Real-corpus bank split leakage detected: " + "; ".join(leakage[:5]))
    return summary


def write_boundary_diagnosis(args: argparse.Namespace) -> dict[str, Any]:
    rows = _load_rows(args.rows_path, limit=args.limit)
    response_map = _load_response_map(args.responses_jsonl)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = BoundaryDiagnoseConfig(
        output_dir=output_dir,
        max_k=args.model_pass_k,
        too_easy_pass_rate=args.too_easy_pass_rate,
        boundary_max_pass_rate=args.accept_max_pass_rate,
        lean_command=args.lean_command,
        lean_cwd=args.lean_cwd,
        timeout_seconds=args.model_timeout_seconds,
        max_heartbeats=args.max_heartbeats,
    )
    signal_rows, family_summary, controller_state = diagnose_boundary_rows(
        rows,
        response_map=response_map,
        eval_report=_read_json(args.eval_report),
        previous_history={},
        config=config,
    )
    _write_jsonl(output_dir / "signal_rows.jsonl", signal_rows)
    (output_dir / "skill_family_summary.json").write_text(
        json.dumps(family_summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "controller_state.json").write_text(
        json.dumps(controller_state, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "signal_rows": str(output_dir / "signal_rows.jsonl"),
        "skill_family_summary": str(output_dir / "skill_family_summary.json"),
        "controller_state": str(output_dir / "controller_state.json"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Route verified real Mathlib rows into a k>=4 boundary bank.")
    parser.add_argument("--rows-path", required=True)
    parser.add_argument("--responses-jsonl", required=True)
    parser.add_argument("--eval-report")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--model-pass-k", type=int, default=4)
    parser.add_argument("--accept-max-pass-rate", type=float, default=0.50)
    parser.add_argument("--too-easy-pass-rate", type=float, default=0.75)
    parser.add_argument("--lean-command")
    parser.add_argument("--lean-cwd")
    parser.add_argument("--certificate-timeout-seconds", type=float, default=30.0)
    parser.add_argument("--model-timeout-seconds", type=float, default=30.0)
    parser.add_argument("--max-heartbeats", type=int, default=200_000)
    parser.add_argument("--fail-on-empty-accepted", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--write-diagnosis", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_real_corpus_bank(args)
    if args.write_diagnosis:
        paths = write_boundary_diagnosis(args)
        summary["diagnosis_paths"] = paths
        summary_path = Path(args.output_dir) / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
