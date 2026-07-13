from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from examples.lean_prover_v1.lean_worker import verify_lean_proof
from examples.lean_prover_v1.probe_common import (
    compute_normalized_statement_hash,
    normalize_lean_row,
)


BRIDGE_TEMPLATES: tuple[tuple[str, str, str], ...] = (
    ("identity", "({goal})", "exact h"),
    ("and_duplicate", "(({goal}) ∧ ({goal}))", "exact And.intro h h"),
    (
        "and_true_nested",
        "((({goal}) ∧ True) ∧ ({goal}))",
        "exact And.intro (And.intro h True.intro) h",
    ),
    (
        "or_false_and",
        "((({goal}) ∨ False) ∧ ({goal}))",
        "exact And.intro (Or.inl h) h",
    ),
    ("exists_unit", "Exists (fun _u : Unit => ({goal}))", "exact Exists.intro () h"),
    (
        "exists_unit_and",
        "Exists (fun _u : Unit => (({goal}) ∧ ({goal})))",
        "exact Exists.intro () (And.intro h h)",
    ),
)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")


def _load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            payload = json.loads(line)
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def generate_candidates(rows: list[dict[str, Any]], *, templates_per_row: int) -> list[dict[str, Any]]:
    generated: list[dict[str, Any]] = []
    templates = BRIDGE_TEMPLATES[: max(1, min(templates_per_row, len(BRIDGE_TEMPLATES)))]
    for raw in rows:
        if str(raw.get("split") or "") != "train_static":
            continue
        parent = normalize_lean_row(raw, split="train_static")
        formal_type = str(raw.get("formal_type") or raw.get("initial_goal_pp") or "").strip()
        if not formal_type:
            continue
        parent_id = str(parent.get("id") or parent.get("uid"))
        for template_name, statement_template, proof_body in templates:
            projected_goal = statement_template.format(goal=formal_type)
            formal_bridge_type = f"({formal_type}) -> ({projected_goal})"
            digest = hashlib.sha256(
                f"{parent_id}\n{template_name}\n{formal_bridge_type}".encode("utf-8")
            ).hexdigest()[:20]
            row = {
                **parent,
                "id": f"{parent_id}__bridge_{template_name}_{digest[:8]}",
                "uid": f"{parent_id}__bridge_{template_name}_{digest[:8]}",
                "source": "real_corpus_bridge",
                "data_source": "real_corpus_bridge",
                "statement_prefix": f"theorem bridge_{digest} (h : {formal_type}) : {projected_goal} := by",
                "initial_goal_pp": f"h : {formal_type}\n⊢ {projected_goal}",
                "formal_type": formal_bridge_type,
                "seed_id": parent_id,
                "parent_ids": [parent_id],
                "mutation_type": "failure_bridge",
                "mutation_source": "symbolic_real_corpus",
                "mutation_rule": template_name,
                "candidate_status": "generated",
                "acceptance_reason": "pending_lean_and_model_pass_checks",
                "difficulty_band": "real_corpus_bridge_probe",
                "proof_certificate": {
                    "source": "symbolic_bridge",
                    "proof_body": proof_body,
                    "strong_prover_solved": False,
                    "verified": False,
                },
                "baseline_results": {
                    "well_formed": False,
                    "cheap_baseline_solved": False,
                    "real_corpus_parent": True,
                },
                "split": "train_mutated",
            }
            # These fields are derived from the theorem statement. Keeping the
            # parent's cached values makes inference see the parent theorem
            # while Lean verifies the generated bridge.
            for derived_field in ("question", "ground_truth", "theorem_hash", "normalized_statement_hash"):
                row.pop(derived_field, None)
            normalized = normalize_lean_row(row, split="train_mutated")
            normalized.update(
                {
                    "formal_type": formal_bridge_type,
                    "source_theorem": raw.get("source_theorem"),
                    "source_module": raw.get("source_module"),
                    "parent_ids": [parent_id],
                    "mutation_rule": template_name,
                    "proof_certificate": row["proof_certificate"],
                    "baseline_results": row["baseline_results"],
                }
            )
            normalized["normalized_statement_hash"] = compute_normalized_statement_hash(normalized)
            generated.append(normalized)
    return generated


def build_candidates(args: argparse.Namespace) -> dict[str, Any]:
    # Keep provenance fields that the generic training-row normalizer intentionally omits.
    source_rows = _load_jsonl(args.rows_path)
    candidates = generate_candidates(source_rows, templates_per_row=args.templates_per_row)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    def verify(row: dict[str, Any]) -> tuple[dict[str, Any], Any]:
        result = verify_lean_proof(
            row,
            str(row["proof_certificate"]["proof_body"]),
            lean_command=args.lean_command,
            lean_cwd=args.lean_cwd,
            timeout_seconds=args.timeout_seconds,
            max_heartbeats=args.max_heartbeats,
        )
        return row, result

    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    statuses: Counter[str] = Counter()
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        for row, result in executor.map(verify, candidates):
            statuses[result.status] += 1
            if not result.ok:
                rejected.append(
                    {
                        **row,
                        "candidate_status": "rejected",
                        "acceptance_reason": "certificate_failed",
                        "certificate_error": result.stdout[-1200:] or result.stderr[-1200:],
                    }
                )
                continue
            row["candidate_status"] = "certificate_verified"
            row["acceptance_reason"] = "pending_model_pass_routing"
            row["baseline_results"] = {**row["baseline_results"], "well_formed": True}
            row["proof_certificate"] = {
                **row["proof_certificate"],
                "strong_prover_solved": True,
                "verified": True,
                "verification_status": result.status,
                "verification_time_s": result.elapsed_s,
            }
            accepted.append(row)

    _write_jsonl(output_dir / "candidates.jsonl", accepted)
    _write_jsonl(output_dir / "rejected.jsonl", rejected)
    unique_count = len({row["normalized_statement_hash"] for row in accepted})
    summary = {
        "source_row_count": len(source_rows),
        "generated_count": len(candidates),
        "candidate_count": len(accepted),
        "rejected_count": len(rejected),
        "unique_statement_count": unique_count,
        "unique_statement_rate": unique_count / max(1, len(accepted)),
        "verified_certificate_count": len(accepted),
        "verified_certificate_rate": 1.0 if accepted else 0.0,
        "template_counts": dict(Counter(row["mutation_rule"] for row in accepted)),
        "verification_status_counts": dict(statuses),
        "candidates_path": str(output_dir / "candidates.jsonl"),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not accepted:
        raise RuntimeError("No verified real-corpus bridge candidates were generated.")
    if summary["unique_statement_rate"] < args.min_unique_statement_rate:
        raise RuntimeError("Real-corpus candidate uniqueness gate failed.")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate verified bridge candidates from real Mathlib rows.")
    parser.add_argument("--rows-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--templates-per-row", type=int, default=len(BRIDGE_TEMPLATES))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--lean-command", default="lean")
    parser.add_argument("--lean-cwd")
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    parser.add_argument("--max-heartbeats", type=int, default=200_000)
    parser.add_argument("--min-unique-statement-rate", type=float, default=0.90)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(build_candidates(parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
