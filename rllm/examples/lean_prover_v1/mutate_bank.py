from __future__ import annotations

import argparse
import json
import random
import re
import statistics
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from examples.lean_prover_v1.lean_worker import verify_lean_proof
from examples.lean_prover_v1.probe_common import (
    DEFAULT_REPO_COMMIT,
    TRIVIAL_STATEMENT_RE,
    compute_normalized_statement_hash,
    compute_theorem_hash,
    load_lean_rows,
    normalize_lean_row,
    normalize_statement,
    prepare_lean_prover_v1_data,
    route_mutation_candidate,
    summarize_lean_rows,
)

CHEAP_BASELINE_PROOFS = (
    "rfl",
    "simp",
    "simp_all",
    "trivial",
    "assumption",
    "constructor",
    "aesop",
    "tauto",
    "exact True.intro",
    "exact Iff.rfl",
)
FORBIDDEN_STATEMENT_RE = re.compile(
    r"^\s*(import|def|instance|axiom|constant|opaque|namespace|section|end|open|"
    r"set_option|run_cmd|#eval|#check|#print|elab|syntax|macro|unsafe)\b"
)
THEOREM_PREFIX_RE = re.compile(r"^\s*(theorem|lemma)\s+([A-Za-z_][A-Za-z0-9_'.]*)\b")
PROOF_BODY_IN_PREFIX_RE = re.compile(r":=\s*by\s+\S")


@dataclass(slots=True)
class MutationBuildConfig:
    output_dir: Path
    round_index: int = 0
    seeds_per_round: int = 64
    symbolic_per_seed: int = 2
    llm_candidates_per_seed: int = 0
    random_seed: int = 1337
    cheap_timeout_seconds: float = 5.0
    strong_timeout_seconds: float = 20.0
    well_formed_timeout_seconds: float = 5.0
    lean_command: str | None = None
    lean_cwd: str | None = None
    max_heartbeats: int = 200_000
    min_ast_edit_distance: float = 0.05
    accept_max_pass_rate: float = 0.35
    model_pass_k: int = 4
    require_model_pass: bool = True
    max_top_tactic_mass: float = 0.55
    max_mutation_type_mass: float = 0.50
    generation_model: str = "symbolic"
    fail_on_empty_accepted: bool = False
    allow_cheap_solved_if_model_boundary: bool = False


SYMBOLIC_RULES: tuple[dict[str, str], ...] = (
    {
        "name": "hypothesis_projection",
        "statement": "theorem {name} (p q : Prop) (h : And p q) : And q p := by",
        "proof": "constructor\n· exact h.right\n· exact h.left",
    },
    {
        "name": "equality_symmetry",
        "statement": "theorem {name} (a b : Nat) (h : a = b) : b = a := by",
        "proof": "symm\nexact h",
    },
    {
        "name": "equality_congruence",
        "statement": "theorem {name} (a b : Nat) (h : a = b) : Nat.succ a = Nat.succ b := by",
        "proof": "exact congrArg Nat.succ h",
    },
    {
        "name": "witness_generalization",
        "statement": "theorem {name} (p q : Prop) (hp : p) (hq : q) : Exists (fun r : Prop => And r p) := by",
        "proof": "refine Exists.intro q ?_\nexact And.intro hq hp",
    },
    {
        "name": "implication_chain",
        "statement": "theorem {name} (p q r : Prop) (hp : p) (hpq : p -> q) (hqr : q -> r) : r := by",
        "proof": "exact hqr (hpq hp)",
    },
    {
        "name": "iff_symmetry",
        "statement": "theorem {name} (p q : Prop) (h : Iff p q) : Iff q p := by",
        "proof": "constructor\n· intro hq\n  exact h.mpr hq\n· intro hp\n  exact h.mp hp",
    },
)


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


def _safe_lean_ident(value: str, *, fallback: str = "mut") -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_']", "_", value)
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    if not cleaned:
        cleaned = fallback
    if cleaned[0].isdigit():
        cleaned = f"{fallback}_{cleaned}"
    return cleaned[:72]


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * q)))
    return float(ordered[index])


def _token_edit_distance(left: str, right: str) -> float:
    left_tokens = normalize_statement(left).split()
    right_tokens = normalize_statement(right).split()
    if not left_tokens and not right_tokens:
        return 0.0
    if not left_tokens or not right_tokens:
        return 1.0
    previous = list(range(len(right_tokens) + 1))
    for i, left_token in enumerate(left_tokens, start=1):
        current = [i]
        for j, right_token in enumerate(right_tokens, start=1):
            cost = 0 if left_token == right_token else 1
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + cost))
        previous = current
    return previous[-1] / max(len(left_tokens), len(right_tokens))


def _statement_precheck(seed_row: dict[str, Any] | None, candidate_row: dict[str, Any], seen_hashes: set[str]) -> str | None:
    statement = str(candidate_row.get("statement_prefix") or "").strip()
    if not statement:
        return "missing_statement"
    if seed_row is not None and normalize_statement(statement) == normalize_statement(str(seed_row.get("statement_prefix") or "")):
        return "noop_statement"
    if TRIVIAL_STATEMENT_RE.search(normalize_statement(statement)):
        return "trivial_statement"
    if PROOF_BODY_IN_PREFIX_RE.search(statement):
        return "statement_contains_proof_body"

    lines = [line for line in statement.splitlines() if line.strip()]
    if not lines or THEOREM_PREFIX_RE.match(lines[0]) is None:
        return "missing_theorem_prefix"
    for line in lines[1:]:
        match = FORBIDDEN_STATEMENT_RE.match(line)
        if match:
            return f"forbidden_statement_command_{match.group(1)}"

    statement_hash = compute_normalized_statement_hash(candidate_row)
    if statement_hash in seen_hashes:
        return "duplicate_statement_hash"
    return None


def _candidate_name(seed_row: dict[str, Any], *, round_index: int, index: int, rule_name: str) -> str:
    seed_id = str(seed_row.get("id") or seed_row.get("uid") or "seed")
    seed_part = _safe_lean_ident(seed_id)
    digest = compute_theorem_hash(seed_row)[:10]
    return _safe_lean_ident(f"lean_self_mut_{round_index}_{index}_{seed_part}_{rule_name}_{digest}")


def generate_symbolic_candidates(seed_row: dict[str, Any], *, count: int, round_index: int, generation_model: str) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    seed_id = str(seed_row.get("id") or seed_row.get("uid") or "")
    for index in range(count):
        rule = SYMBOLIC_RULES[index % len(SYMBOLIC_RULES)]
        name = _candidate_name(seed_row, round_index=round_index, index=index, rule_name=rule["name"])
        row = {
            "id": f"{seed_id}__self_mut_round{round_index}_{index}_{rule['name']}",
            "source": "self_mutation_symbolic",
            "repo_commit": seed_row.get("repo_commit") or DEFAULT_REPO_COMMIT,
            "imports": seed_row.get("imports") or [],
            "namespace": seed_row.get("namespace") or "",
            "statement_prefix": rule["statement"].format(name=name),
            "initial_goal_pp": "",
            "seed_id": seed_id,
            "parent_ids": [seed_id],
            "mutation_type": rule["name"],
            "mutation_source": "symbolic",
            "mutation_rule": rule["name"],
            "generation_model": generation_model,
            "difficulty_band": "self_mutation_candidate",
            "baseline_results": {},
            "proof_certificate": {
                "source": "symbolic_rule",
                "strong_prover_solved": False,
                "proof_body": rule["proof"],
            },
            "split": "train_mutated",
        }
        candidates.append(normalize_lean_row(row, split="train_mutated"))
    return candidates


def load_llm_candidates(path: str | None, *, generation_model: str) -> list[dict[str, Any]]:
    if not path:
        return []
    rows = load_lean_rows(path)
    candidates: list[dict[str, Any]] = []
    for row in rows:
        row = dict(row)
        row.setdefault("source", "self_mutation_llm")
        row.setdefault("mutation_source", "llm")
        row.setdefault("mutation_rule", row.get("mutation_type") or "llm_json")
        row.setdefault("generation_model", generation_model)
        row.setdefault("split", "train_mutated")
        candidates.append(normalize_lean_row(row, split="train_mutated"))
    return candidates


def _proof_candidates(row: dict[str, Any]) -> list[tuple[str, str]]:
    certificate = _jsonish(row.get("proof_certificate"))
    candidates: list[tuple[str, str]] = []
    if isinstance(certificate, dict):
        for key in ("proof_body", "proof", "proof_hint"):
            value = str(certificate.get(key) or "").strip()
            if value and (key, value) not in candidates:
                candidates.append((key, value))
    proof_hint = str(row.get("proof_hint") or "").strip()
    if proof_hint:
        candidates.append(("proof_hint", proof_hint))
    return candidates


def _first_tactic(proof_body: str) -> str:
    stripped = proof_body.strip()
    if not stripped:
        return "<empty>"
    return stripped.split()[0]


def _verify_one(row: dict[str, Any], proof_body: str, *, timeout_seconds: float, config: MutationBuildConfig, allow_sorry: bool = False):
    return verify_lean_proof(
        row,
        proof_body,
        lean_command=config.lean_command,
        lean_cwd=config.lean_cwd,
        timeout_seconds=timeout_seconds,
        max_heartbeats=config.max_heartbeats,
        allow_sorry=allow_sorry,
    )


def _evaluate_model_responses(
    row: dict[str, Any],
    responses: list[str],
    *,
    config: MutationBuildConfig,
) -> dict[str, Any]:
    if not responses:
        return {}
    responses = responses[: max(1, config.model_pass_k)]
    results = [
        _verify_one(row, response, timeout_seconds=config.strong_timeout_seconds, config=config)
        for response in responses
    ]
    pass_count = sum(1 for result in results if result.ok)
    return {
        "model_candidate_count": len(results),
        "model_pass_count": pass_count,
        "model_pass_at_k": pass_count / max(1, len(results)),
        "model_pass_at_1": 1.0 if results and results[0].ok else 0.0,
        "model_status_counts": dict(Counter(result.status for result in results)),
        "model_results": [
            {
                "proof_body": response,
                "ok": result.ok,
                "status": result.status,
                "elapsed_s": result.elapsed_s,
            }
            for response, result in zip(responses, results, strict=True)
        ],
    }


def evaluate_mutation_candidate(
    seed_row: dict[str, Any] | None,
    candidate_row: dict[str, Any],
    *,
    seen_hashes: set[str],
    model_responses: list[str] | None,
    config: MutationBuildConfig,
) -> dict[str, Any]:
    is_stepwise_candidate = bool(candidate_row.get("stepwise_status") or candidate_row.get("local_goal_hash"))
    row = normalize_lean_row(candidate_row, split="train_mutated")
    row["normalized_statement_hash"] = compute_normalized_statement_hash(row)
    row["candidate_status"] = "candidate"

    precheck = _statement_precheck(seed_row, row, seen_hashes)
    if precheck is not None:
        row["candidate_status"] = "rejected"
        row["acceptance_reason"] = precheck
        return row

    ast_edit_distance = _token_edit_distance(str(seed_row.get("statement_prefix") if seed_row else ""), row["statement_prefix"])
    if ast_edit_distance < config.min_ast_edit_distance:
        row["candidate_status"] = "rejected"
        row["acceptance_reason"] = "low_ast_edit_distance"
        row["baseline_results"] = {"well_formed": None, "cheap_baseline_solved": None, "ast_edit_distance": ast_edit_distance}
        return row

    well_formed = _verify_one(row, "sorry", timeout_seconds=config.well_formed_timeout_seconds, config=config, allow_sorry=True)
    baseline_results: dict[str, Any] = {
        "well_formed": well_formed.ok,
        "well_formed_status": well_formed.status,
        "well_formed_time_s": well_formed.elapsed_s,
        "cheap_baseline_solved": False,
        "cheap_baseline_results": [],
        "ast_edit_distance": ast_edit_distance,
    }
    row["baseline_results"] = baseline_results
    if not well_formed.ok:
        row["candidate_status"] = "rejected"
        row["acceptance_reason"] = "uncompilable"
        return row

    proof_certificate = _jsonish(row.get("proof_certificate"))
    if not isinstance(proof_certificate, dict):
        proof_certificate = {}
    strong_results = []
    verified_proof_body = ""
    verified_source = ""
    for source, proof_body in _proof_candidates(row):
        result = _verify_one(row, proof_body, timeout_seconds=config.strong_timeout_seconds, config=config)
        strong_results.append({"source": source, "ok": result.ok, "status": result.status, "elapsed_s": result.elapsed_s})
        if result.ok:
            verified_proof_body = proof_body
            verified_source = source
            break

    proof_certificate = {
        **proof_certificate,
        "source": proof_certificate.get("source") or verified_source or "unverified",
        "strong_prover_solved": bool(verified_proof_body),
        "verified": bool(verified_proof_body),
        "verification_status": "passed" if verified_proof_body else "failed",
        "proof_body": verified_proof_body or proof_certificate.get("proof_body") or proof_certificate.get("proof") or "",
        "verification_results": strong_results,
    }
    row["proof_certificate"] = proof_certificate
    if not verified_proof_body:
        row["candidate_status"] = "rejected"
        row["acceptance_reason"] = "unverified_proof_certificate"
        row["difficulty_metrics"] = {
            "cheap_baseline_solved": False,
            "strong_prover_solved": False,
            "proof_length_tokens": len(str(proof_certificate.get("proof_body") or "").split()),
            "lean_latency_s": well_formed.elapsed_s,
            "ast_edit_distance": ast_edit_distance,
        }
        return row

    for proof_body in CHEAP_BASELINE_PROOFS:
        result = _verify_one(row, proof_body, timeout_seconds=config.cheap_timeout_seconds, config=config)
        baseline_results["cheap_baseline_results"].append(
            {"proof_body": proof_body, "ok": result.ok, "status": result.status, "elapsed_s": result.elapsed_s}
        )
        if result.ok:
            baseline_results["cheap_baseline_solved"] = True
            baseline_results["cheap_baseline_proof"] = proof_body
            break

    difficulty_metrics: dict[str, Any] = {
        "cheap_baseline_solved": baseline_results["cheap_baseline_solved"],
        "baseline_time_s": sum(float(item["elapsed_s"]) for item in baseline_results["cheap_baseline_results"]),
        "strong_prover_solved": bool(verified_proof_body),
        "proof_length_tokens": len(str(proof_certificate.get("proof_body") or "").split()),
        "lean_latency_s": well_formed.elapsed_s,
        "ast_edit_distance": ast_edit_distance,
    }
    difficulty_metrics.update(_evaluate_model_responses(row, model_responses or [], config=config))
    row["difficulty_metrics"] = difficulty_metrics

    if baseline_results["cheap_baseline_solved"]:
        allow_student_relative_routing = config.allow_cheap_solved_if_model_boundary and is_stepwise_candidate
        if not allow_student_relative_routing:
            row["candidate_status"] = "too_easy"
            row["acceptance_reason"] = "cheap_solved"
            return row
        if "model_pass_at_k" not in difficulty_metrics:
            row["candidate_status"] = "rejected" if config.require_model_pass else "too_easy"
            row["acceptance_reason"] = "missing_model_pass_result" if config.require_model_pass else "cheap_solved_unrouted"
            return row

        model_pass_at_k = float(difficulty_metrics["model_pass_at_k"])
        if model_pass_at_k <= 0.0:
            row["candidate_status"] = "frontier_holdout"
            row["acceptance_reason"] = "frontier_holdout_stepwise_model_pass_zero"
            row["split"] = "val_mutated"
        elif model_pass_at_k <= config.accept_max_pass_rate:
            row["candidate_status"] = "accepted_train"
            row["acceptance_reason"] = "accepted_stepwise_model_boundary"
            row["difficulty_band"] = "stepwise_teacher_boundary"
        else:
            row["candidate_status"] = "too_easy"
            row["acceptance_reason"] = "too_easy_stepwise_model_pass_rate"
        return row

    route = route_mutation_candidate(
        seed_row,
        row,
        min_ast_edit_distance=config.min_ast_edit_distance,
        accept_max_pass_rate=config.accept_max_pass_rate,
        require_model_pass=config.require_model_pass,
    )
    if route.accepted:
        row["candidate_status"] = "accepted_train"
        row["acceptance_reason"] = route.reason
        row["difficulty_band"] = "self_mutation_train"
    else:
        row["candidate_status"] = "frontier_holdout" if route.reason.startswith("frontier_holdout") else "rejected"
        row["acceptance_reason"] = route.reason
        if row["candidate_status"] == "frontier_holdout":
            row["split"] = "val_mutated"
    return row


def _load_response_map(path: str | None) -> dict[str, list[str]]:
    if not path:
        return {}
    response_map: dict[str, list[str]] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        row_id = str(row.get("id") or row.get("uid") or row.get("task_id"))
        values = row.get("responses") if isinstance(row.get("responses"), list) else None
        if values is None:
            values = [row.get("response") or row.get("proof_body") or row.get("completion") or ""]
        response_map.setdefault(row_id, []).extend(str(value) for value in values)
    return response_map


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=True) + "\n")


def _apply_acceptance_balance_gates(
    accepted: list[dict[str, Any]],
    rejected: list[dict[str, Any]],
    config: MutationBuildConfig,
) -> list[dict[str, Any]]:
    if not accepted:
        return accepted

    max_tactic = max(1, round(len(accepted) * config.max_top_tactic_mass))
    max_mutation = max(1, round(len(accepted) * config.max_mutation_type_mass))
    tactic_counts: Counter[str] = Counter()
    mutation_counts: Counter[str] = Counter()
    kept: list[dict[str, Any]] = []

    # Keep deterministic order so repeated rounds are reproducible.
    for row in accepted:
        proof = str(_jsonish(row.get("proof_certificate")).get("proof_body") or "")
        tactic = _first_tactic(proof)
        mutation = str(row.get("mutation_type") or "unknown")
        if tactic_counts[tactic] >= max_tactic:
            row["candidate_status"] = "rejected"
            row["acceptance_reason"] = "collapse_top_tactic_mass"
            rejected.append(row)
            continue
        if mutation_counts[mutation] >= max_mutation:
            row["candidate_status"] = "rejected"
            row["acceptance_reason"] = "collapse_mutation_type_mass"
            rejected.append(row)
            continue
        tactic_counts[tactic] += 1
        mutation_counts[mutation] += 1
        kept.append(row)
    return kept


def _load_seed_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.seed_corpus_path:
        rows = load_lean_rows(args.seed_corpus_path)
        return [normalize_lean_row(row, split=str(row.get("split") or args.seed_split)) for row in rows]

    splits = prepare_lean_prover_v1_data(
        static_corpus_path=args.static_corpus_path,
        mutation_bank_path=args.existing_mutation_bank_path,
        allow_synthetic=args.allow_synthetic,
        train_static_size=args.train_static_size,
        val_static_size=args.val_static_size,
        test_static_size=args.test_static_size,
        train_mutated_size=args.train_mutated_size,
        val_mutated_size=args.val_mutated_size,
        test_mutated_size=args.test_mutated_size,
    )
    return list(splits.get(args.seed_split, []))


def _limited_llm_candidates(rows: list[dict[str, Any]], seeds: list[dict[str, Any]], per_seed: int) -> list[dict[str, Any]]:
    if per_seed <= 0 or not rows:
        return []
    seed_ids = {str(row.get("id") or row.get("uid")) for row in seeds}
    by_seed: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        seed_id = str(row.get("seed_id") or "")
        if seed_id in seed_ids:
            by_seed.setdefault(seed_id, []).append(row)
    limited: list[dict[str, Any]] = []
    for seed in seeds:
        seed_id = str(seed.get("id") or seed.get("uid"))
        limited.extend(by_seed.get(seed_id, [])[:per_seed])
    return limited


def _mutation_summary(
    *,
    candidates: list[dict[str, Any]],
    accepted: list[dict[str, Any]],
    rejected: list[dict[str, Any]],
    cumulative_bank: list[dict[str, Any]],
    config: MutationBuildConfig,
) -> dict[str, Any]:
    status_counts = Counter(str(row.get("candidate_status") or "unknown") for row in candidates)
    reason_counts = Counter(str(row.get("acceptance_reason") or "unknown") for row in candidates)
    well_formed = [
        bool(_jsonish(row.get("baseline_results")).get("well_formed"))
        for row in candidates
        if isinstance(_jsonish(row.get("baseline_results")), dict) and "well_formed" in _jsonish(row.get("baseline_results"))
    ]
    cheap_solved = [
        bool(_jsonish(row.get("baseline_results")).get("cheap_baseline_solved"))
        for row in candidates
        if isinstance(_jsonish(row.get("baseline_results")), dict) and "cheap_baseline_solved" in _jsonish(row.get("baseline_results"))
    ]
    certificate_success = [
        bool(_jsonish(row.get("proof_certificate")).get("strong_prover_solved"))
        for row in candidates
        if isinstance(_jsonish(row.get("proof_certificate")), dict)
    ]
    latencies = [
        float(_jsonish(row.get("baseline_results")).get("well_formed_time_s") or 0.0)
        for row in candidates
        if isinstance(_jsonish(row.get("baseline_results")), dict)
    ]
    ast_edit_distances = [
        float(_jsonish(row.get("baseline_results")).get("ast_edit_distance"))
        for row in candidates
        if isinstance(_jsonish(row.get("baseline_results")), dict)
        and _jsonish(row.get("baseline_results")).get("ast_edit_distance") is not None
    ]
    return {
        "round_index": config.round_index,
        "candidate_count": len(candidates),
        "accepted_train_count": len(accepted),
        "rejected_count": len(rejected),
        "accepted_bank_size": len(cumulative_bank),
        "status_counts": dict(status_counts),
        "reason_counts": dict(reason_counts),
        "well_formed_rate": sum(well_formed) / max(1, len(well_formed)),
        "cheap_baseline_solved_rate": sum(cheap_solved) / max(1, len(cheap_solved)),
        "cheap_baseline_fail_rate": 1.0 - (sum(cheap_solved) / max(1, len(cheap_solved))),
        "strong_prover_success_rate": sum(certificate_success) / max(1, len(certificate_success)),
        "ast_edit_distance": {
            "count": len(ast_edit_distances),
            "p50": statistics.median(ast_edit_distances) if ast_edit_distances else 0.0,
            "p95": _percentile(ast_edit_distances, 0.95),
        },
        "lean_latency": {
            "count": len(latencies),
            "p50_s": statistics.median(latencies) if latencies else 0.0,
            "p95_s": _percentile(latencies, 0.95),
        },
        "accepted_collapse": summarize_lean_rows(accepted),
        "config": {
            "seeds_per_round": config.seeds_per_round,
            "symbolic_per_seed": config.symbolic_per_seed,
            "llm_candidates_per_seed": config.llm_candidates_per_seed,
            "cheap_timeout_seconds": config.cheap_timeout_seconds,
            "strong_timeout_seconds": config.strong_timeout_seconds,
            "accept_max_pass_rate": config.accept_max_pass_rate,
            "model_pass_k": config.model_pass_k,
            "require_model_pass": config.require_model_pass,
            "max_top_tactic_mass": config.max_top_tactic_mass,
            "max_mutation_type_mass": config.max_mutation_type_mass,
        },
    }


def build_mutation_bank(args: argparse.Namespace) -> dict[str, Any]:
    config = MutationBuildConfig(
        output_dir=Path(args.output_dir),
        round_index=args.round_index,
        seeds_per_round=args.seeds_per_round,
        symbolic_per_seed=args.symbolic_per_seed,
        llm_candidates_per_seed=args.llm_candidates_per_seed,
        random_seed=args.random_seed,
        cheap_timeout_seconds=args.cheap_timeout_seconds,
        strong_timeout_seconds=args.strong_timeout_seconds,
        well_formed_timeout_seconds=args.well_formed_timeout_seconds,
        lean_command=args.lean_command,
        lean_cwd=args.lean_cwd,
        max_heartbeats=args.max_heartbeats,
        min_ast_edit_distance=args.min_ast_edit_distance,
        accept_max_pass_rate=args.accept_max_pass_rate,
        model_pass_k=getattr(args, "model_pass_k", 4),
        require_model_pass=args.require_model_pass,
        max_top_tactic_mass=getattr(args, "max_top_tactic_mass", 0.55),
        max_mutation_type_mass=getattr(args, "max_mutation_type_mass", 0.50),
        generation_model=args.generation_model,
        fail_on_empty_accepted=args.fail_on_empty_accepted,
    )

    rng = random.Random(config.random_seed + config.round_index)
    seed_rows = _load_seed_rows(args)
    if not seed_rows:
        raise RuntimeError("No seed rows were available for mutation.")
    rng.shuffle(seed_rows)
    seeds = seed_rows[: config.seeds_per_round]
    seed_by_id = {str(row.get("id") or row.get("uid")): row for row in seeds}

    existing_bank = [normalize_lean_row(row, split="train_mutated") for row in load_lean_rows(args.existing_mutation_bank_path)]
    seen_hashes = {compute_normalized_statement_hash(row) for row in existing_bank}
    seen_hashes.update(compute_normalized_statement_hash(row) for row in seed_rows)

    candidates: list[dict[str, Any]] = []
    for seed in seeds:
        candidates.extend(
            generate_symbolic_candidates(
                seed,
                count=config.symbolic_per_seed,
                round_index=config.round_index,
                generation_model=config.generation_model,
            )
        )
    llm_rows = load_llm_candidates(args.llm_candidate_jsonl, generation_model=config.generation_model)
    candidates.extend(_limited_llm_candidates(llm_rows, seeds, config.llm_candidates_per_seed))

    response_map = _load_response_map(args.model_responses_jsonl)
    evaluated: list[dict[str, Any]] = []
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    frontier_holdout: list[dict[str, Any]] = []
    for candidate in candidates:
        seed = seed_by_id.get(str(candidate.get("seed_id") or ""))
        evaluated_row = evaluate_mutation_candidate(
            seed,
            candidate,
            seen_hashes=seen_hashes,
            model_responses=response_map.get(str(candidate.get("id") or candidate.get("uid"))),
            config=config,
        )
        evaluated.append(evaluated_row)
        if evaluated_row.get("candidate_status") == "accepted_train":
            accepted.append(evaluated_row)
            seen_hashes.add(compute_normalized_statement_hash(evaluated_row))
        elif evaluated_row.get("candidate_status") == "frontier_holdout":
            frontier_holdout.append(evaluated_row)
            rejected.append(evaluated_row)
        else:
            rejected.append(evaluated_row)

    accepted = _apply_acceptance_balance_gates(accepted, rejected, config)
    cumulative_bank = [*existing_bank, *accepted]
    eval_bank = [*cumulative_bank, *frontier_holdout]
    summary = _mutation_summary(
        candidates=evaluated,
        accepted=accepted,
        rejected=rejected,
        cumulative_bank=cumulative_bank,
        config=config,
    )
    summary["frontier_holdout_count"] = len(frontier_holdout)
    summary["eval_bank_size"] = len(eval_bank)

    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_dir / "candidates.jsonl", evaluated)
    _write_jsonl(output_dir / "accepted.jsonl", accepted)
    _write_jsonl(output_dir / "frontier_holdout.jsonl", frontier_holdout)
    _write_jsonl(output_dir / "rejected.jsonl", rejected)
    _write_jsonl(output_dir / "bank.jsonl", cumulative_bank)
    _write_jsonl(output_dir / "eval_bank.jsonl", eval_bank)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if config.fail_on_empty_accepted and not accepted:
        raise RuntimeError(f"Mutation round {config.round_index} produced no accepted_train candidates.")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an accepted Lean prover v1 mutation bank.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--round-index", type=int, default=0)
    parser.add_argument("--seed-corpus-path", default=None)
    parser.add_argument("--seed-split", default="train_static")
    parser.add_argument("--static-corpus-path", default=None)
    parser.add_argument("--existing-mutation-bank-path", default=None)
    parser.add_argument("--llm-candidate-jsonl", default=None)
    parser.add_argument("--model-responses-jsonl", default=None)
    parser.add_argument("--seeds-per-round", type=int, default=64)
    parser.add_argument("--symbolic-per-seed", type=int, default=2)
    parser.add_argument("--llm-candidates-per-seed", type=int, default=0)
    parser.add_argument("--random-seed", type=int, default=1337)
    parser.add_argument("--cheap-timeout-seconds", type=float, default=5.0)
    parser.add_argument("--strong-timeout-seconds", type=float, default=20.0)
    parser.add_argument("--well-formed-timeout-seconds", type=float, default=5.0)
    parser.add_argument("--lean-command", default=None)
    parser.add_argument("--lean-cwd", default=None)
    parser.add_argument("--max-heartbeats", type=int, default=200_000)
    parser.add_argument("--min-ast-edit-distance", type=float, default=0.05)
    parser.add_argument("--accept-max-pass-rate", type=float, default=0.35)
    parser.add_argument("--model-pass-k", type=int, default=4)
    parser.add_argument("--require-model-pass", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-top-tactic-mass", type=float, default=0.55)
    parser.add_argument("--max-mutation-type-mass", type=float, default=0.50)
    parser.add_argument("--generation-model", default="symbolic")
    parser.add_argument("--fail-on-empty-accepted", action="store_true")
    parser.add_argument("--allow-synthetic", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--train-static-size", type=int, default=1024)
    parser.add_argument("--val-static-size", type=int, default=128)
    parser.add_argument("--test-static-size", type=int, default=128)
    parser.add_argument("--train-mutated-size", type=int, default=1024)
    parser.add_argument("--val-mutated-size", type=int, default=128)
    parser.add_argument("--test-mutated-size", type=int, default=128)
    return parser.parse_args()


def main() -> None:
    summary = build_mutation_bank(parse_args())
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
