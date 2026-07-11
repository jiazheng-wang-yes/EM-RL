from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from examples.lean_prover_v1.lean_worker import verify_lean_proof
from examples.lean_prover_v1.mutate_bank import (
    MutationBuildConfig,
    _apply_acceptance_balance_gates,
    _first_tactic,
    _load_response_map,
    _percentile,
    _write_jsonl,
    evaluate_mutation_candidate,
)
from examples.lean_prover_v1.probe_common import (
    DEFAULT_REPO_COMMIT,
    compute_normalized_statement_hash,
    load_direct_lean_rows,
    load_lean_rows,
    normalize_lean_row,
    summarize_lean_rows,
)

ERROR_FAMILIES = {
    "format_body_only",
    "intro_binder",
    "and_or_constructors",
    "exists_witness",
    "equality_rewrite",
    "timeout_loop",
}


@dataclass(slots=True)
class ErrorMutationConfig:
    output_dir: Path
    diagnosis_model_source: str
    max_failures: int = 64
    cases_per_family: int = 4
    model_pass_k: int = 4
    accept_max_pass_rate: float = 0.35
    cheap_timeout_seconds: float = 5.0
    strong_timeout_seconds: float = 20.0
    well_formed_timeout_seconds: float = 5.0
    lean_command: str | None = None
    lean_cwd: str | None = None
    max_heartbeats: int = 200_000
    require_model_pass: bool = True
    max_top_tactic_mass: float = 0.40
    max_error_family_mass: float = 0.30
    max_template_mass: float = 0.15
    random_seed: int = 1337


BRIDGE_TEMPLATES: dict[str, list[dict[str, str]]] = {
    "format_body_only": [
        {
            "template": "body_and_intro",
            "statement": "theorem {name} (p q : Prop) (hp : p) (hq : q) : q ∧ p := by",
            "proof": "exact And.intro hq hp",
        },
        {
            "template": "body_or_intro",
            "statement": "theorem {name} (p q : Prop) (hp : p) : p ∨ q := by",
            "proof": "exact Or.inl hp",
        },
    ],
    "intro_binder": [
        {
            "template": "implication_apply",
            "statement": "theorem {name} (p q : Prop) (hpq : p -> q) : p -> q := by",
            "proof": "intro hp\nexact hpq hp",
        },
        {
            "template": "implication_chain",
            "statement": "theorem {name} (p q r : Prop) (hpq : p -> q) (hqr : q -> r) : p -> r := by",
            "proof": "intro hp\nexact hqr (hpq hp)",
        },
    ],
    "and_or_constructors": [
        {
            "template": "and_reorder_with_extra",
            "statement": "theorem {name} (p q r : Prop) (h : p ∧ q) (hr : r) : q ∧ r := by",
            "proof": "exact And.intro h.right hr",
        },
        {
            "template": "or_cases",
            "statement": "theorem {name} (p q r : Prop) (h : p ∨ q) (hp : p -> r) (hq : q -> r) : r := by",
            "proof": "cases h with\n| inl hp' => exact hp hp'\n| inr hq' => exact hq hq'",
        },
    ],
    "exists_witness": [
        {
            "template": "exists_prop_pair",
            "statement": "theorem {name} (p q : Prop) (hp : p) (hq : q) : Exists (fun r : Prop => And r p) := by",
            "proof": "refine Exists.intro q ?_\nexact And.intro hq hp",
        },
        {
            "template": "exists_prop_shift",
            "statement": "theorem {name} (p q r : Prop) (hq : q) (hr : r) : Exists (fun x : Prop => x ∧ r) := by",
            "proof": "refine Exists.intro q ?_\nexact And.intro hq hr",
        },
    ],
    "equality_rewrite": [
        {
            "template": "succ_congr",
            "statement": "theorem {name} (a b : Nat) (h : a = b) : Nat.succ a = Nat.succ b := by",
            "proof": "exact congrArg Nat.succ h",
        },
        {
            "template": "eq_trans",
            "statement": "theorem {name} (a b c : Nat) (hab : a = b) (hbc : b = c) : a = c := by",
            "proof": "exact Eq.trans hab hbc",
        },
    ],
    "timeout_loop": [
        {
            "template": "short_chain",
            "statement": "theorem {name} (p q r : Prop) (hp : p) (hpq : p -> q) (hqr : q -> r) : r := by",
            "proof": "exact hqr (hpq hp)",
        },
        {
            "template": "short_and_swap",
            "statement": "theorem {name} (p q : Prop) (h : p ∧ q) : q ∧ p := by",
            "proof": "exact And.intro h.right h.left",
        },
    ],
}


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


def _safe_ident(value: str, *, fallback: str = "err") -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_']", "_", value)
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    if not cleaned:
        cleaned = fallback
    if cleaned[0].isdigit():
        cleaned = f"{fallback}_{cleaned}"
    return cleaned[:64]


def _short_text(value: str, limit: int = 2000) -> str:
    value = str(value or "")
    if len(value) <= limit:
        return value
    return value[:limit] + "\n<truncated>"


def _response_has_repetition(response: str) -> bool:
    tokens = response.split()
    if len(tokens) < 16:
        return False
    windows = [" ".join(tokens[index : index + 4]) for index in range(0, max(0, len(tokens) - 3))]
    counts = Counter(windows)
    return bool(counts and counts.most_common(1)[0][1] >= 4)


def classify_failure(record: dict[str, Any]) -> dict[str, str]:
    response = str(record.get("failed_response") or "")
    statement = str(record.get("statement_prefix") or "")
    status = str(record.get("lean_status") or "")
    combined = f"{statement}\n{response}".lower()

    if status == "timeout" or _response_has_repetition(response):
        family = "timeout_loop"
        skill = "terminate with a short deterministic proof"
    elif response.strip().startswith("by") or status in {"forbidden_token", "forbidden_command"} or "sorry" in response.lower():
        family = "format_body_only"
        skill = "emit only the proof body and avoid forbidden tokens"
    elif "exists" in combined or "use " in combined or "exists_intro" in combined or "existsi" in combined:
        family = "exists_witness"
        skill = "choose a witness and prove the witness predicate"
    elif "∧" in combined or "and" in combined or "∨" in combined or " or_" in combined or "constructor" in combined:
        family = "and_or_constructors"
        skill = "use Lean 4 And and Or constructors and eliminators"
    elif "->" in combined or "intro" in combined or "assume" in combined or "∀" in combined or "forall" in combined:
        family = "intro_binder"
        skill = "introduce binders and apply implications"
    elif "=" in statement or "rw" in combined or "rfl" in combined or "congr" in combined or "symm" in combined:
        family = "equality_rewrite"
        skill = "use equality transitivity, symmetry, and congruence"
    else:
        family = "intro_binder"
        skill = "introduce binders and apply assumptions"

    first_token = response.strip().split(maxsplit=1)[0] if response.strip() else "<empty>"
    signature = f"{family}:{status}:{first_token}"
    return {"error_family": family, "target_skill": skill, "error_signature": signature}


def parse_llm_diagnosis(text: str) -> dict[str, Any] | None:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    family = str(payload.get("error_family") or "")
    if family not in ERROR_FAMILIES:
        return None
    skill = str(payload.get("target_skill") or "").strip()
    if not skill:
        return None
    return {
        "error_family": family,
        "target_skill": skill,
        "rationale": str(payload.get("rationale") or ""),
    }


def _load_llm_diagnostics(path: str | None) -> dict[str, dict[str, Any]]:
    if not path:
        return {}
    diagnostics: dict[str, dict[str, Any]] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        row_id = str(row.get("id") or row.get("uid") or row.get("task_id") or "")
        raw = row.get("diagnosis") or row.get("response") or row.get("completion") or row.get("text") or row
        parsed = parse_llm_diagnosis(json.dumps(raw) if isinstance(raw, dict) else str(raw))
        if row_id and parsed:
            diagnostics[row_id] = parsed
    return diagnostics


def _load_failure_records(args: argparse.Namespace, config: ErrorMutationConfig) -> list[dict[str, Any]]:
    rows = load_direct_lean_rows(args.rows_path)
    response_map = _load_response_map(args.responses_jsonl)
    report_path = str(args.eval_report or "")
    records: list[dict[str, Any]] = []

    for row in rows:
        row_id = str(row.get("id") or row.get("uid"))
        responses = response_map.get(row_id) or []
        if not responses:
            continue
        failed_response = str(responses[0])
        result = verify_lean_proof(
            row,
            failed_response,
            lean_command=config.lean_command,
            lean_cwd=config.lean_cwd,
            timeout_seconds=config.strong_timeout_seconds,
            max_heartbeats=config.max_heartbeats,
        )
        if result.ok:
            continue
        certificate = _jsonish(row.get("proof_certificate"))
        record = {
            **row,
            "failed_response": failed_response,
            "lean_status": result.status,
            "lean_stdout": _short_text(result.stdout),
            "lean_stderr": _short_text(result.stderr),
            "source_eval_report": report_path,
            "repair_certificate": certificate,
        }
        record.update(classify_failure(record))
        records.append(record)
        if len(records) >= config.max_failures:
            break
    return records


def _apply_llm_diagnostics(
    failure_records: list[dict[str, Any]],
    *,
    llm_diagnostics: dict[str, dict[str, Any]],
    diagnosis_model_source: str,
) -> list[dict[str, Any]]:
    diagnostics: list[dict[str, Any]] = []
    for record in failure_records:
        row_id = str(record.get("id") or record.get("uid"))
        diagnostic = {
            "id": row_id,
            "error_family": record["error_family"],
            "target_skill": record["target_skill"],
            "error_signature": record["error_signature"],
            "diagnosis_source": "heuristic",
            "diagnosis_model_source": diagnosis_model_source,
        }
        llm = llm_diagnostics.get(row_id)
        if llm:
            record["error_family"] = llm["error_family"]
            record["target_skill"] = llm["target_skill"]
            diagnostic.update(
                {
                    "error_family": llm["error_family"],
                    "target_skill": llm["target_skill"],
                    "llm_rationale": llm.get("rationale", ""),
                    "diagnosis_source": "llm_override",
                }
            )
        diagnostics.append(diagnostic)
    return diagnostics


def _candidate_name(failure: dict[str, Any], *, family: str, level: int, index: int, template: str) -> str:
    row_id = str(failure.get("id") or failure.get("uid") or "failure")
    return _safe_ident(f"lean_error_mut_{family}_{level}_{index}_{template}_{row_id}")


def generate_bridge_candidates(failure_records: list[dict[str, Any]], *, cases_per_family: int, generation_model: str) -> list[dict[str, Any]]:
    by_family: dict[str, list[dict[str, Any]]] = {}
    for record in failure_records:
        by_family.setdefault(str(record.get("error_family") or "intro_binder"), []).append(record)

    candidates: list[dict[str, Any]] = []
    for family in sorted(by_family):
        templates = BRIDGE_TEMPLATES.get(family) or BRIDGE_TEMPLATES["intro_binder"]
        failures = by_family[family]
        for index in range(cases_per_family):
            failure = failures[index % len(failures)]
            template = templates[index % len(templates)]
            level = index // len(templates)
            name = _candidate_name(failure, family=family, level=level, index=index, template=template["template"])
            row = {
                "id": f"{failure.get('id')}__error_mut_{family}_{level}_{index}_{template['template']}",
                "source": "error_conditioned_mutation",
                "repo_commit": failure.get("repo_commit") or DEFAULT_REPO_COMMIT,
                "imports": failure.get("imports") or [],
                "namespace": failure.get("namespace") or "",
                "statement_prefix": template["statement"].format(name=name),
                "initial_goal_pp": "",
                "seed_id": None,
                "parent_ids": [str(failure.get("id") or failure.get("uid"))],
                "mutation_type": family,
                "mutation_source": "error_conditioned",
                "mutation_rule": template["template"],
                "generation_model": generation_model,
                "candidate_status": "candidate",
                "acceptance_reason": "",
                "difficulty_band": f"error_bridge_{level}",
                "baseline_results": {},
                "proof_certificate": {
                    "source": "error_bridge_template",
                    "strong_prover_solved": False,
                    "proof_body": template["proof"],
                },
                "split": "train_mutated",
                "failed_response": failure.get("failed_response"),
                "lean_status": failure.get("lean_status"),
                "lean_stdout": failure.get("lean_stdout"),
                "lean_stderr": failure.get("lean_stderr"),
                "error_signature": failure.get("error_signature"),
                "error_family": family,
                "target_skill": failure.get("target_skill"),
                "source_eval_report": failure.get("source_eval_report"),
                "repair_certificate": failure.get("repair_certificate"),
                "bridge_level": level,
            }
            candidates.append(normalize_lean_row(row, split="train_mutated"))
    return candidates


def _error_family_entropy(rows: list[dict[str, Any]]) -> float:
    counts = Counter(str(row.get("error_family") or row.get("mutation_type") or "unknown") for row in rows)
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    return -sum((count / total) * math.log(count / total, 2) for count in counts.values() if count)


def _apply_error_balance_gates(
    accepted: list[dict[str, Any]],
    rejected: list[dict[str, Any]],
    config: ErrorMutationConfig,
) -> list[dict[str, Any]]:
    if not accepted:
        return accepted
    max_family = max(1, round(len(accepted) * config.max_error_family_mass))
    max_template = max(1, round(len(accepted) * config.max_template_mass))
    family_counts: Counter[str] = Counter()
    template_counts: Counter[str] = Counter()
    kept: list[dict[str, Any]] = []
    for row in accepted:
        family = str(row.get("error_family") or row.get("mutation_type") or "unknown")
        template = str(row.get("mutation_rule") or "unknown")
        if family_counts[family] >= max_family:
            row["candidate_status"] = "rejected"
            row["acceptance_reason"] = "collapse_error_family_mass"
            rejected.append(row)
            continue
        if template_counts[template] >= max_template:
            row["candidate_status"] = "rejected"
            row["acceptance_reason"] = "collapse_template_mass"
            rejected.append(row)
            continue
        family_counts[family] += 1
        template_counts[template] += 1
        kept.append(row)
    return kept


def _summary(
    *,
    failure_records: list[dict[str, Any]],
    diagnostics: list[dict[str, Any]],
    evaluated: list[dict[str, Any]],
    accepted: list[dict[str, Any]],
    rejected: list[dict[str, Any]],
    frontier_holdout: list[dict[str, Any]],
    cumulative_bank: list[dict[str, Any]],
    config: ErrorMutationConfig,
) -> dict[str, Any]:
    status_counts = Counter(str(row.get("candidate_status") or "unknown") for row in evaluated)
    reason_counts = Counter(str(row.get("acceptance_reason") or "unknown") for row in evaluated)
    family_counts = Counter(str(row.get("error_family") or "unknown") for row in failure_records)
    diagnostic_counts = Counter(str(row.get("diagnosis_source") or "unknown") for row in diagnostics)
    latencies = [
        float(_jsonish(row.get("baseline_results")).get("well_formed_time_s") or 0.0)
        for row in evaluated
        if isinstance(_jsonish(row.get("baseline_results")), dict)
    ]
    return {
        "failure_count": len(failure_records),
        "candidate_count": len(evaluated),
        "accepted_train_count": len(accepted),
        "frontier_holdout_count": len(frontier_holdout),
        "rejected_count": len(rejected),
        "accepted_bank_size": len(cumulative_bank),
        "status_counts": dict(status_counts),
        "reason_counts": dict(reason_counts),
        "error_family_counts": dict(family_counts),
        "error_family_entropy": _error_family_entropy(failure_records),
        "diagnosis_source_counts": dict(diagnostic_counts),
        "accepted_collapse": summarize_lean_rows(accepted),
        "lean_latency": {
            "count": len(latencies),
            "p50_s": statistics.median(latencies) if latencies else 0.0,
            "p95_s": _percentile(latencies, 0.95),
        },
        "config": {
            "diagnosis_model_source": config.diagnosis_model_source,
            "max_failures": config.max_failures,
            "cases_per_family": config.cases_per_family,
            "model_pass_k": config.model_pass_k,
            "accept_max_pass_rate": config.accept_max_pass_rate,
            "require_model_pass": config.require_model_pass,
            "max_top_tactic_mass": config.max_top_tactic_mass,
            "max_error_family_mass": config.max_error_family_mass,
            "max_template_mass": config.max_template_mass,
        },
    }


def build_error_mutation_bank(args: argparse.Namespace) -> dict[str, Any]:
    config = ErrorMutationConfig(
        output_dir=Path(args.output_dir),
        diagnosis_model_source=args.diagnosis_model_source,
        max_failures=args.max_failures,
        cases_per_family=args.cases_per_family,
        model_pass_k=args.model_pass_k,
        accept_max_pass_rate=args.accept_max_pass_rate,
        cheap_timeout_seconds=args.cheap_timeout_seconds,
        strong_timeout_seconds=args.strong_timeout_seconds,
        well_formed_timeout_seconds=args.well_formed_timeout_seconds,
        lean_command=args.lean_command,
        lean_cwd=args.lean_cwd,
        max_heartbeats=args.max_heartbeats,
        require_model_pass=args.require_model_pass,
        max_top_tactic_mass=args.max_top_tactic_mass,
        max_error_family_mass=args.max_error_family_mass,
        max_template_mass=args.max_template_mass,
        random_seed=args.random_seed,
    )
    mutation_config = MutationBuildConfig(
        output_dir=config.output_dir,
        cheap_timeout_seconds=config.cheap_timeout_seconds,
        strong_timeout_seconds=config.strong_timeout_seconds,
        well_formed_timeout_seconds=config.well_formed_timeout_seconds,
        lean_command=config.lean_command,
        lean_cwd=config.lean_cwd,
        max_heartbeats=config.max_heartbeats,
        min_ast_edit_distance=0.0,
        accept_max_pass_rate=config.accept_max_pass_rate,
        model_pass_k=config.model_pass_k,
        require_model_pass=config.require_model_pass,
        max_top_tactic_mass=config.max_top_tactic_mass,
        max_mutation_type_mass=config.max_error_family_mass,
        generation_model=config.diagnosis_model_source,
    )

    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    failure_records = _load_failure_records(args, config)
    diagnostics = _apply_llm_diagnostics(
        failure_records,
        llm_diagnostics=_load_llm_diagnostics(args.llm_diagnostics_jsonl),
        diagnosis_model_source=config.diagnosis_model_source,
    )
    candidates = generate_bridge_candidates(
        failure_records,
        cases_per_family=config.cases_per_family,
        generation_model=config.diagnosis_model_source,
    )

    existing_bank = [normalize_lean_row(row, split="train_mutated") for row in load_lean_rows(args.existing_mutation_bank_path)]
    seen_hashes = {compute_normalized_statement_hash(row) for row in existing_bank}
    response_map = _load_response_map(args.model_responses_jsonl)

    evaluated: list[dict[str, Any]] = []
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    frontier_holdout: list[dict[str, Any]] = []
    for candidate in candidates:
        evaluated_row = evaluate_mutation_candidate(
            None,
            candidate,
            seen_hashes=seen_hashes,
            model_responses=response_map.get(str(candidate.get("id") or candidate.get("uid"))),
            config=mutation_config,
        )
        evaluated_row["error_family"] = candidate.get("error_family")
        evaluated_row["target_skill"] = candidate.get("target_skill")
        evaluated_row["bridge_level"] = candidate.get("bridge_level")
        evaluated_row["failed_response"] = candidate.get("failed_response")
        evaluated_row["lean_status"] = candidate.get("lean_status")
        evaluated_row["lean_stdout"] = candidate.get("lean_stdout")
        evaluated_row["lean_stderr"] = candidate.get("lean_stderr")
        evaluated_row["error_signature"] = candidate.get("error_signature")
        evaluated_row["source_eval_report"] = candidate.get("source_eval_report")
        evaluated_row["repair_certificate"] = candidate.get("repair_certificate")
        evaluated.append(evaluated_row)
        if evaluated_row.get("candidate_status") == "accepted_train":
            accepted.append(evaluated_row)
            seen_hashes.add(compute_normalized_statement_hash(evaluated_row))
        elif evaluated_row.get("candidate_status") == "frontier_holdout":
            frontier_holdout.append(evaluated_row)
            rejected.append(evaluated_row)
        else:
            rejected.append(evaluated_row)

    accepted = _apply_acceptance_balance_gates(accepted, rejected, mutation_config)
    accepted = _apply_error_balance_gates(accepted, rejected, config)
    cumulative_bank = [*existing_bank, *accepted]
    eval_bank = [*cumulative_bank, *frontier_holdout]
    summary = _summary(
        failure_records=failure_records,
        diagnostics=diagnostics,
        evaluated=evaluated,
        accepted=accepted,
        rejected=rejected,
        frontier_holdout=frontier_holdout,
        cumulative_bank=cumulative_bank,
        config=config,
    )
    summary["eval_bank_size"] = len(eval_bank)

    _write_jsonl(output_dir / "failure_records.jsonl", failure_records)
    _write_jsonl(output_dir / "diagnostics.jsonl", diagnostics)
    _write_jsonl(output_dir / "targeted_candidates.jsonl", evaluated)
    _write_jsonl(output_dir / "accepted.jsonl", accepted)
    _write_jsonl(output_dir / "frontier_holdout.jsonl", frontier_holdout)
    _write_jsonl(output_dir / "rejected.jsonl", rejected)
    _write_jsonl(output_dir / "bank.jsonl", cumulative_bank)
    _write_jsonl(output_dir / "eval_bank.jsonl", eval_bank)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an error-conditioned Lean prover v1 mutation bank.")
    parser.add_argument("--rows-path", required=True)
    parser.add_argument("--responses-jsonl", required=True)
    parser.add_argument("--eval-report", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--diagnosis-model-source", default="heuristic")
    parser.add_argument("--llm-diagnostics-jsonl", default=None)
    parser.add_argument("--existing-mutation-bank-path", default=None)
    parser.add_argument("--model-responses-jsonl", default=None)
    parser.add_argument("--max-failures", type=int, default=64)
    parser.add_argument("--cases-per-family", type=int, default=4)
    parser.add_argument("--model-pass-k", type=int, default=4)
    parser.add_argument("--accept-max-pass-rate", type=float, default=0.35)
    parser.add_argument("--cheap-timeout-seconds", type=float, default=5.0)
    parser.add_argument("--strong-timeout-seconds", type=float, default=20.0)
    parser.add_argument("--well-formed-timeout-seconds", type=float, default=5.0)
    parser.add_argument("--lean-command", default=None)
    parser.add_argument("--lean-cwd", default=None)
    parser.add_argument("--max-heartbeats", type=int, default=200_000)
    parser.add_argument("--require-model-pass", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-top-tactic-mass", type=float, default=0.40)
    parser.add_argument("--max-error-family-mass", type=float, default=0.30)
    parser.add_argument("--max-template-mass", type=float, default=0.15)
    parser.add_argument("--random-seed", type=int, default=1337)
    return parser.parse_args()


def main() -> None:
    summary = build_error_mutation_bank(parse_args())
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
