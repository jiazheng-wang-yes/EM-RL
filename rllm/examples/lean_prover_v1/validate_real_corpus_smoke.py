from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _read_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    source = Path(path)
    if not source.exists():
        return rows
    for line in source.read_text(encoding="utf-8").splitlines():
        if line.strip():
            payload = json.loads(line)
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def _label_signature(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row.get("id") or row.get("uid"),
        row.get("signal_class"),
        row.get("signal_reason"),
        row.get("student_pass_count"),
        row.get("student_response_count"),
        row.get("error_family"),
        row.get("error_signature"),
    )


def _failure(message: str, details: Any | None = None) -> dict[str, Any]:
    return {"ok": False, "message": message, "details": details}


def validate(args: argparse.Namespace) -> dict[str, Any]:
    failures: list[dict[str, Any]] = []
    corpus_summary = _read_json(args.corpus_summary)
    candidate_summary_path = getattr(args, "candidate_summary", None)
    candidate_summary = _read_json(candidate_summary_path) if candidate_summary_path else {}
    bank_summary = _read_json(args.bank_summary)
    accepted = _read_jsonl(args.accepted_jsonl)
    diagnosis_a = _read_json(args.diagnosis_a_summary)
    diagnosis_b = _read_json(args.diagnosis_b_summary)
    signal_a = _read_jsonl(args.diagnosis_a_signal_rows)
    signal_b = _read_jsonl(args.diagnosis_b_signal_rows)

    corpus_unique = float(corpus_summary.get("unique_statement_rate") or 0.0)
    candidate_unique = float(candidate_summary.get("unique_statement_rate") or 0.0)
    bank_unique = float(bank_summary.get("unique_statement_rate") or 0.0)
    if corpus_unique < args.min_unique_statement_rate:
        failures.append(_failure("corpus_unique_statement_rate_below_threshold", corpus_unique))
    if bank_unique < args.min_unique_statement_rate:
        failures.append(_failure("bank_unique_statement_rate_below_threshold", bank_unique))
    if candidate_summary_path and candidate_unique < args.min_unique_statement_rate:
        failures.append(_failure("candidate_unique_statement_rate_below_threshold", candidate_unique))
    if candidate_summary_path and float(candidate_summary.get("verified_certificate_rate") or 0.0) < 1.0:
        failures.append(_failure("candidate_certificates_not_all_verified"))

    accepted_count = int(bank_summary.get("accepted_count") or len(accepted))
    if accepted_count <= 0:
        failures.append(_failure("no_accepted_rows", accepted_count))
    if len(accepted) != accepted_count:
        failures.append(_failure("accepted_count_mismatch", {"summary": accepted_count, "jsonl": len(accepted)}))

    unverified = [
        row.get("id") or row.get("uid")
        for row in accepted
        if not row.get("proof_certificate", {}).get("verified")
        or not row.get("proof_certificate", {}).get("strong_prover_solved")
    ]
    if unverified:
        failures.append(_failure("accepted_rows_without_verified_certificate", unverified[:20]))

    model_pass_k = int(bank_summary.get("model_pass_k") or 0)
    if model_pass_k < args.min_model_pass_k:
        failures.append(_failure("bank_model_pass_k_too_small", model_pass_k))

    config_a = diagnosis_a.get("config") if isinstance(diagnosis_a.get("config"), dict) else {}
    config_b = diagnosis_b.get("config") if isinstance(diagnosis_b.get("config"), dict) else {}
    if int(config_a.get("max_k") or 0) < args.min_model_pass_k:
        failures.append(_failure("diagnosis_a_max_k_too_small", config_a.get("max_k")))
    if int(config_b.get("max_k") or 0) < args.min_model_pass_k:
        failures.append(_failure("diagnosis_b_max_k_too_small", config_b.get("max_k")))

    signatures_a = [_label_signature(row) for row in signal_a]
    signatures_b = [_label_signature(row) for row in signal_b]
    if signatures_a != signatures_b:
        mismatch_index = next(
            (idx for idx, (left, right) in enumerate(zip(signatures_a, signatures_b, strict=False)) if left != right),
            min(len(signatures_a), len(signatures_b)),
        )
        failures.append(
            _failure(
                "boundary_labels_not_reproducible",
                {
                    "mismatch_index": mismatch_index,
                    "a": signatures_a[mismatch_index] if mismatch_index < len(signatures_a) else None,
                    "b": signatures_b[mismatch_index] if mismatch_index < len(signatures_b) else None,
                    "len_a": len(signatures_a),
                    "len_b": len(signatures_b),
                },
            )
        )

    timeout_retries = sum(
        1
        for row in signal_a
        for item in row.get("failed_rollouts", [])
        if isinstance(item, dict) and item.get("timeout_retried")
    )
    report = {
        "ok": not failures,
        "failures": failures,
        "corpus_unique_statement_rate": corpus_unique,
        "candidate_unique_statement_rate": candidate_unique,
        "bank_unique_statement_rate": bank_unique,
        "accepted_count": accepted_count,
        "verified_accepted_count": len(accepted) - len(unverified),
        "model_pass_k": model_pass_k,
        "diagnosis_a_max_k": config_a.get("max_k"),
        "diagnosis_b_max_k": config_b.get("max_k"),
        "diagnosis_row_count": len(signal_a),
        "timeout_retry_count": timeout_retries,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if failures:
        raise SystemExit(1)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the real-corpus bank-only smoke contract.")
    parser.add_argument("--corpus-summary", required=True)
    parser.add_argument("--candidate-summary")
    parser.add_argument("--bank-summary", required=True)
    parser.add_argument("--accepted-jsonl", required=True)
    parser.add_argument("--diagnosis-a-summary", required=True)
    parser.add_argument("--diagnosis-b-summary", required=True)
    parser.add_argument("--diagnosis-a-signal-rows", required=True)
    parser.add_argument("--diagnosis-b-signal-rows", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-unique-statement-rate", type=float, default=0.90)
    parser.add_argument("--min-model-pass-k", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(validate(parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
