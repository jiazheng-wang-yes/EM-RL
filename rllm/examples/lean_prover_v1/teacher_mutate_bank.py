from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import statistics
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from examples.lean_prover_v1.error_mutate_bank import ERROR_FAMILIES, classify_failure
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
    normalize_statement_for_hash,
    summarize_lean_rows,
)

DEFAULT_TEACHER_MODEL = "deepseek-v4-pro"
DEFAULT_TEACHER_BASE_URL = "https://api.deepseek.com"
TEACHER_API_KEY_ENV_ORDER = ("DEEPSEEK_API_KEY", "ANTHROPIC_AUTH_TOKEN")
TEACHER_THINKING_DISABLED = {"thinking": {"type": "disabled"}}

REQUIRED_TEACHER_FIELDS = (
    "statement_prefix",
    "proof_body",
    "target_skill",
    "error_family",
    "bridge_level",
    "rationale",
    "expected_failure_fixed",
    "difficulty_rationale",
)

UNSAFE_TEXT_RE = re.compile(
    r"^\s*(import|theorem|lemma|def|instance|axiom|constant|opaque|namespace|section|end|open|"
    r"set_option|run_cmd|#eval|#check|#print|elab|syntax|macro|unsafe)\b",
    flags=re.MULTILINE,
)
FORBIDDEN_PROOF_TOKEN_RE = re.compile(r"\b(sorry|admit|unsafe)\b")

TEACHER_METADATA_FIELDS = (
    "parent_ids",
    "failed_response",
    "lean_status",
    "lean_stdout",
    "lean_stderr",
    "error_signature",
    "error_family",
    "target_skill",
    "source_eval_report",
    "repair_certificate",
    "bridge_level",
    "mutation_source",
    "mutation_rule",
    "generation_model",
    "verified_prefix",
    "failing_step",
    "remaining_goal_pp",
    "failing_proof_line",
    "failing_source_line",
    "failing_column",
    "local_goal_hash",
    "source_statement_hash",
    "source_statement_shape_hash",
    "stepwise_status",
    "stepwise_metadata",
)

FAMILY_REPAIR_CONSTRAINTS = {
    "format_body_only": "Use body-only Lean syntax. Do not include a leading by or any declaration in proof_body.",
    "intro_binder": "Require at least one binder introduction or function application; the conclusion must not be an available hypothesis.",
    "and_or_constructors": "Require a conjunction/disjunction constructor, projection, or case split; do not make the conclusion an available hypothesis.",
    "exists_witness": "Require an explicit existential witness and a nontrivial witness obligation.",
    "equality_rewrite": "Require rw [...], Eq.trans, congrArg, or simpa only [...] in the certificate; the goal must not exactly match a hypothesis.",
    "timeout_loop": "Use a short deterministic proof with a bounded sequence of distinct tactics and no repeat loops.",
}


@dataclass(slots=True)
class TeacherMutationConfig:
    output_dir: Path
    teacher_model: str = DEFAULT_TEACHER_MODEL
    teacher_base_url: str = DEFAULT_TEACHER_BASE_URL
    max_failures: int = 32
    cases_per_family: int = 3
    model_pass_k: int = 4
    accept_max_pass_rate: float = 0.35
    max_api_calls: int = 64
    max_output_tokens: int = 4096
    temperature: float = 0.2
    refine_rounds: int = 1
    cache_dir: Path | None = None
    cheap_timeout_seconds: float = 5.0
    strong_timeout_seconds: float = 20.0
    well_formed_timeout_seconds: float = 5.0
    lean_command: str | None = None
    lean_cwd: str | None = None
    max_heartbeats: int = 200_000
    require_model_pass: bool = True
    max_top_tactic_mass: float = 0.40
    max_error_family_mass: float = 0.30
    random_seed: int = 1337
    disable_teacher_thinking: bool = True
    difficulty_directive: str = "balanced"
    allow_stepwise_cheap_solved: bool = True


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


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=True, default=str)


def _short_text(value: str, limit: int = 2000) -> str:
    text = str(value or "")
    if len(text) <= limit:
        return text
    return text[:limit] + "\n<truncated>"


def _read_jsonl(path: str | Path | None) -> list[dict[str, Any]]:
    if not path:
        return []
    source = Path(path)
    if not source.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in source.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _read_eval_report(path: str | Path | None) -> dict[str, Any]:
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


def _teacher_api_key() -> str | None:
    for name in TEACHER_API_KEY_ENV_ORDER:
        value = os.environ.get(name)
        if value:
            return value
    return None


def _prompt_cache_key(messages: list[dict[str, str]], config: TeacherMutationConfig) -> str:
    payload = {
        "model": config.teacher_model,
        "messages": messages,
        "temperature": config.temperature,
        "max_output_tokens": config.max_output_tokens,
        "disable_teacher_thinking": config.disable_teacher_thinking,
    }
    return hashlib.sha256(_stable_json(payload).encode("utf-8")).hexdigest()


def _error_family_entropy(rows: list[dict[str, Any]]) -> float:
    counts = Counter(str(row.get("error_family") or row.get("mutation_type") or "unknown") for row in rows)
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    return -sum((count / total) * math.log(count / total, 2) for count in counts.values() if count)


def _top_tactic_mass(rows: list[dict[str, Any]]) -> float:
    if not rows:
        return 0.0
    counts: Counter[str] = Counter()
    for row in rows:
        certificate = _jsonish(row.get("proof_certificate"))
        proof = str(certificate.get("proof_body") or "") if isinstance(certificate, dict) else ""
        counts[_first_tactic(proof)] += 1
    return counts.most_common(1)[0][1] / len(rows) if counts else 0.0


def _family_balance_gates(
    accepted: list[dict[str, Any]],
    rejected: list[dict[str, Any]],
    config: TeacherMutationConfig,
) -> list[dict[str, Any]]:
    if not accepted:
        return accepted
    max_family = max(1, round(len(accepted) * config.max_error_family_mass))
    family_counts: Counter[str] = Counter()
    kept: list[dict[str, Any]] = []
    for row in accepted:
        family = str(row.get("error_family") or row.get("mutation_type") or "unknown")
        if family_counts[family] >= max_family:
            row["candidate_status"] = "rejected"
            row["acceptance_reason"] = "collapse_error_family_mass"
            rejected.append(row)
            continue
        family_counts[family] += 1
        kept.append(row)
    return kept


def _source_eval_metrics(report: dict[str, Any]) -> dict[str, Any]:
    if not report:
        return {}
    keys = (
        "pass_at_1",
        "pass_at_k",
        "static_pass_at_1",
        "mutated_pass_at_1",
        "timeout_rate",
        "type_error_rate",
        "error_taxonomy",
        "formatting",
    )
    return {key: report[key] for key in keys if key in report}


def parse_teacher_response(text: str) -> list[dict[str, Any]] | None:
    candidates, errors = _parse_teacher_response_with_errors(text)
    if errors and not candidates:
        return None
    return candidates


def _parse_teacher_response_with_errors(text: str) -> tuple[list[dict[str, Any]], list[str]]:
    try:
        payload = json.loads(str(text or ""))
    except json.JSONDecodeError:
        return [], ["malformed_json"]
    if isinstance(payload, list):
        raw_candidates = payload
    elif isinstance(payload, dict) and isinstance(payload.get("candidates"), list):
        raw_candidates = payload["candidates"]
    else:
        return [], ["missing_candidates_list"]

    candidates: list[dict[str, Any]] = []
    errors: list[str] = []
    for index, item in enumerate(raw_candidates):
        if not isinstance(item, dict):
            errors.append(f"candidate_{index}_not_object")
            continue
        missing = [field for field in REQUIRED_TEACHER_FIELDS if not str(item.get(field) or "").strip()]
        if missing:
            errors.append(f"candidate_{index}_missing_{','.join(missing)}")
            continue
        family = str(item.get("error_family"))
        if family not in ERROR_FAMILIES:
            errors.append(f"candidate_{index}_unknown_family_{family}")
            continue
        row = {field: item.get(field) for field in REQUIRED_TEACHER_FIELDS}
        for optional in ("mutation_rule", "proof_hint", "teacher_variant"):
            if optional in item:
                row[optional] = item.get(optional)
        candidates.append(row)
    return candidates, errors


def _load_failure_records(args: argparse.Namespace, config: TeacherMutationConfig) -> list[dict[str, Any]]:
    failure_records_path = getattr(args, "failure_records_jsonl", None)
    if failure_records_path:
        return _read_jsonl(failure_records_path)[: config.max_failures]
    if not args.rows_path or not args.responses_jsonl:
        raise ValueError("--rows-path and --responses-jsonl are required without --failure-records-jsonl")
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


def _compact_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": row.get("id"),
        "source": row.get("source"),
        "repo_commit": row.get("repo_commit"),
        "imports": row.get("imports") or [],
        "namespace": row.get("namespace") or "",
        "statement_prefix": row.get("statement_prefix"),
        "initial_goal_pp": row.get("initial_goal_pp") or "",
        "mutation_type": row.get("mutation_type"),
        "difficulty_band": row.get("difficulty_band"),
        "proof_certificate": row.get("proof_certificate"),
    }


def _strip_balanced_outer_parens(value: str) -> str:
    value = value.strip()
    while value.startswith("(") and value.endswith(")"):
        depth = 0
        closes_at_end = True
        for index, char in enumerate(value):
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
                if depth == 0 and index != len(value) - 1:
                    closes_at_end = False
                    break
        if depth != 0 or not closes_at_end:
            break
        value = value[1:-1].strip()
    return value


def _statement_shape_hash(row: dict[str, Any]) -> str:
    statement = normalize_statement_for_hash(str(row.get("statement_prefix") or ""))
    declaration = statement.removesuffix(" := by")
    depths = {"(": 0, "[": 0, "{": 0}
    closing = {")": "(", "]": "[", "}": "{"}
    for index, char in enumerate(declaration):
        if char in depths:
            depths[char] += 1
        elif char in closing:
            opener = closing[char]
            depths[opener] = max(0, depths[opener] - 1)
        elif char == ":" and not any(depths.values()):
            head = declaration[:index].strip()
            target = declaration[index + 1 :].strip()
            statement = f"{head} : {_strip_balanced_outer_parens(target)} := by"
            break
    payload = {
        "imports": row.get("imports") or [],
        "namespace": row.get("namespace") or "",
        "repo_commit": row.get("repo_commit") or "",
        "statement_shape": statement,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")).hexdigest()


def build_teacher_messages(
    failure: dict[str, Any],
    *,
    config: TeacherMutationConfig,
    eval_report: dict[str, Any],
    recent_bank_summary: dict[str, Any],
    difficulty_directive: str = "balanced",
) -> list[dict[str, str]]:
    family = str(failure.get("error_family") or "intro_binder")
    target_skill = str(failure.get("target_skill") or "")
    system = (
        "You are a Lean 4 curriculum generator. Return strict json only. "
        "When a stepwise local goal is provided, generate standalone repair lemmas that teach the missing next proof step; "
        "do not simply restate or wrap the full source theorem. Generalize concrete constants into variables where possible. "
        "Make every candidate a distinct proposition and change the proposition structurally at each bridge level. "
        "Theorem renames are duplicates because names are ignored during deduplication. "
        "Every candidate must be a theorem statement ending with ':= by' and a separate proof body. "
        "Do not emit imports, namespace declarations, top-level declarations inside proof bodies, sorry, admit, unsafe, "
        "trivial True targets, theorem renames, or proof bodies inside statement_prefix."
    )
    example = {
        "candidates": [
            {
                "statement_prefix": "theorem bridge_example (p q : Prop) (hp : p) (hq : q) : q ∧ p := by",
                "proof_body": "exact And.intro hq hp",
                "target_skill": target_skill or "construct conjunctions",
                "error_family": family,
                "bridge_level": "easy",
                "rationale": "A reachable bridge for the failed family.",
                "expected_failure_fixed": "Uses the right constructor directly.",
                "difficulty_rationale": "Positive-pass bridge, not a frontier theorem.",
            }
        ]
    }
    user_payload = {
        "task": "Generate Lean 4 bridge curriculum candidates for the failed student proof.",
        "difficulty_directive": difficulty_directive,
        "target_pass_window": {
            "min_model_pass_at_k_exclusive": 0.0,
            "max_model_pass_at_k": config.accept_max_pass_rate,
            "model_pass_k": config.model_pass_k,
        },
        "requested_ladder": ["easier_bridge", "target_bridge", "harder_variant"],
        "cases_requested": config.cases_per_family,
        "family_repair_constraint": FAMILY_REPAIR_CONSTRAINTS.get(
            family,
            "The proof certificate must exercise the target skill and the conclusion must not already be an available hypothesis.",
        ),
        "failure": {
            "row": _compact_row(failure),
            "failed_response": _short_text(str(failure.get("failed_response") or ""), 1200),
            "lean_status": failure.get("lean_status"),
            "lean_stdout": _short_text(str(failure.get("lean_stdout") or ""), 1200),
            "lean_stderr": _short_text(str(failure.get("lean_stderr") or ""), 1200),
            "error_signature": failure.get("error_signature"),
            "error_family": family,
            "target_skill": target_skill,
            "repair_certificate": failure.get("repair_certificate"),
            "verified_prefix": _short_text(str(failure.get("verified_prefix") or ""), 1600),
            "failing_step": _short_text(str(failure.get("failing_step") or ""), 800),
            "remaining_goal_pp": _short_text(str(failure.get("remaining_goal_pp") or ""), 2400),
            "failing_proof_line": failure.get("failing_proof_line"),
            "local_goal_hash": failure.get("local_goal_hash"),
        },
        "recent_bank_summary": recent_bank_summary,
        "source_eval_metrics": _source_eval_metrics(eval_report),
        "json_schema_example": example,
    }
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(user_payload, ensure_ascii=True, sort_keys=True)},
    ]


def _load_teacher_raw_fixture(path: str | None) -> dict[str, str]:
    fixture: dict[str, str] = {}
    for row in _read_jsonl(path):
        row_id = str(row.get("id") or row.get("failure_id") or row.get("prompt_key") or "")
        raw = row.get("response") or row.get("raw") or row.get("completion")
        if raw is None and "candidates" in row:
            raw = json.dumps({"candidates": row["candidates"]})
        if row_id and raw is not None:
            fixture[row_id] = str(raw)
    return fixture


def call_teacher_api(messages: list[dict[str, str]], *, config: TeacherMutationConfig) -> dict[str, Any]:
    cache_key = _prompt_cache_key(messages, config)
    cache_path = config.cache_dir / f"{cache_key}.json" if config.cache_dir else None
    if cache_path and cache_path.exists():
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        payload["cache_hit"] = True
        return payload

    api_key = _teacher_api_key()
    if not api_key:
        raise RuntimeError("Set DEEPSEEK_API_KEY or ANTHROPIC_AUTH_TOKEN, or pass --teacher-raw-jsonl for fixture mode.")

    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=config.teacher_base_url)
    kwargs: dict[str, Any] = {
        "model": config.teacher_model,
        "messages": messages,
        "temperature": config.temperature,
        "max_tokens": config.max_output_tokens,
        "response_format": {"type": "json_object"},
        "stream": False,
    }
    if config.disable_teacher_thinking:
        kwargs["extra_body"] = TEACHER_THINKING_DISABLED

    last_error = ""
    for attempt in range(3):
        try:
            started = time.monotonic()
            response = client.chat.completions.create(**kwargs)
            elapsed_s = time.monotonic() - started
            content = response.choices[0].message.content or ""
            usage = getattr(response, "usage", None)
            payload = {
                "response": content,
                "cache_hit": False,
                "elapsed_s": elapsed_s,
                "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
                "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
            }
            if cache_path:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            return payload
        except Exception as exc:  # noqa: BLE001
            last_error = str(exc)
            if attempt < 2:
                time.sleep(1.0 + attempt)
    raise RuntimeError(f"Teacher API failed after retries: {last_error}")


def _teacher_candidate_safety_reason(candidate: dict[str, Any]) -> str | None:
    statement = str(candidate.get("statement_prefix") or "")
    proof_body = str(candidate.get("proof_body") or "")
    if not statement.strip():
        return "missing_statement_prefix"
    if not proof_body.strip():
        return "missing_proof_body"
    if FORBIDDEN_PROOF_TOKEN_RE.search(proof_body):
        return "forbidden_proof_token"
    if UNSAFE_TEXT_RE.search(proof_body):
        return "forbidden_proof_command"
    return None


def _safe_ident(value: str, *, fallback: str = "teacher") -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_']", "_", value)
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    if not cleaned:
        cleaned = fallback
    if cleaned[0].isdigit():
        cleaned = f"{fallback}_{cleaned}"
    return cleaned[:64]


def teacher_spec_to_row(
    spec: dict[str, Any],
    *,
    failure: dict[str, Any],
    prompt_key: str,
    candidate_index: int,
    config: TeacherMutationConfig,
) -> dict[str, Any]:
    failure_id = str(failure.get("id") or failure.get("uid") or "failure")
    family = str(spec.get("error_family") or failure.get("error_family") or "intro_binder")
    bridge_level = str(spec.get("bridge_level") or "target")
    statement_prefix = str(spec.get("statement_prefix") or "").strip()
    proof_body = str(spec.get("proof_body") or "").strip()
    statement_hash = hashlib.sha1(statement_prefix.encode("utf-8")).hexdigest()[:10]
    row_id = f"{failure_id}__teacher_mut_{_safe_ident(family)}_{_safe_ident(bridge_level)}_{candidate_index}_{statement_hash}"
    raw_row = {
            "id": row_id,
            "source": "teacher_conditioned_mutation",
            "repo_commit": failure.get("repo_commit") or DEFAULT_REPO_COMMIT,
            "imports": failure.get("imports") or [],
            "namespace": failure.get("namespace") or "",
            "statement_prefix": statement_prefix,
            "initial_goal_pp": "",
            "seed_id": None,
            "parent_ids": [failure_id],
            "mutation_type": family,
            "mutation_source": "teacher_conditioned",
            "mutation_rule": str(spec.get("mutation_rule") or spec.get("target_skill") or "teacher_curriculum"),
            "generation_model": config.teacher_model,
            "candidate_status": "candidate",
            "acceptance_reason": "",
            "difficulty_band": f"teacher_bridge_{bridge_level}",
            "baseline_results": {},
            "proof_certificate": {
                "source": "teacher_model",
                "strong_prover_solved": False,
                "proof_body": proof_body,
                "teacher_rationale": str(spec.get("rationale") or ""),
                "expected_failure_fixed": str(spec.get("expected_failure_fixed") or ""),
                "difficulty_rationale": str(spec.get("difficulty_rationale") or ""),
                "prompt_key": prompt_key,
            },
            "split": "train_mutated",
            "failed_response": failure.get("failed_response"),
            "lean_status": failure.get("lean_status"),
            "lean_stdout": failure.get("lean_stdout"),
            "lean_stderr": failure.get("lean_stderr"),
            "error_signature": failure.get("error_signature"),
            "error_family": family,
            "target_skill": str(spec.get("target_skill") or failure.get("target_skill") or ""),
            "source_eval_report": failure.get("source_eval_report"),
            "repair_certificate": failure.get("repair_certificate"),
            "verified_prefix": failure.get("verified_prefix"),
            "failing_step": failure.get("failing_step"),
            "remaining_goal_pp": failure.get("remaining_goal_pp"),
            "failing_proof_line": failure.get("failing_proof_line"),
            "local_goal_hash": failure.get("local_goal_hash"),
            "failing_source_line": failure.get("failing_source_line"),
            "failing_column": failure.get("failing_column"),
            "stepwise_status": failure.get("stepwise_status"),
            "stepwise_metadata": failure.get("stepwise_metadata"),
            "source_statement_hash": compute_normalized_statement_hash(failure),
            "source_statement_shape_hash": _statement_shape_hash(failure),
            "bridge_level": bridge_level,
        }
    row = normalize_lean_row(raw_row, split="train_mutated")
    for field in TEACHER_METADATA_FIELDS:
        if raw_row.get(field) is not None:
            row[field] = raw_row[field]
    return row


def generate_teacher_candidates(
    failure_records: list[dict[str, Any]],
    *,
    config: TeacherMutationConfig,
    eval_report: dict[str, Any],
    frontier_rows: list[dict[str, Any]],
    teacher_raw_jsonl: str | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    rng = random.Random(config.random_seed)
    fixture = _load_teacher_raw_fixture(teacher_raw_jsonl)
    prompts: list[dict[str, Any]] = []
    raw_records: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    stats: Counter[str] = Counter()

    failures = list(failure_records)
    rng.shuffle(failures)
    recent_bank_summary = {
        "failure_family_counts": dict(Counter(str(row.get("error_family") or "unknown") for row in failure_records)),
        "frontier_bank_size": len(frontier_rows),
        "frontier_bank_summary": summarize_lean_rows(frontier_rows),
    }
    for failure in failures:
        if stats["api_calls"] >= config.max_api_calls:
            break
        failure_id = str(failure.get("id") or failure.get("uid"))
        messages = build_teacher_messages(
            failure,
            config=config,
            eval_report=eval_report,
            recent_bank_summary=recent_bank_summary,
            difficulty_directive=config.difficulty_directive,
        )
        prompt_key = _prompt_cache_key(messages, config)
        prompts.append({"id": failure_id, "prompt_key": prompt_key, "messages": messages, "error_family": failure.get("error_family")})

        if failure_id in fixture:
            payload = {"response": fixture[failure_id], "cache_hit": True, "prompt_tokens": 0, "completion_tokens": 0, "fixture": True}
            stats["cache_hits"] += 1
        elif prompt_key in fixture:
            payload = {"response": fixture[prompt_key], "cache_hit": True, "prompt_tokens": 0, "completion_tokens": 0, "fixture": True}
            stats["cache_hits"] += 1
        else:
            payload = call_teacher_api(messages, config=config)
            if payload.get("cache_hit"):
                stats["cache_hits"] += 1
            else:
                stats["api_calls"] += 1
        payload_record = {"id": failure_id, "prompt_key": prompt_key, **payload}
        raw_records.append(payload_record)

        parsed, parse_errors = _parse_teacher_response_with_errors(str(payload.get("response") or ""))
        if parse_errors:
            stats["json_parse_errors"] += 1
        stats["prompt_tokens"] += int(payload.get("prompt_tokens") or 0)
        stats["completion_tokens"] += int(payload.get("completion_tokens") or 0)

        for index, spec in enumerate(parsed[: max(1, config.cases_per_family)]):
            unsafe_reason = _teacher_candidate_safety_reason(spec)
            if unsafe_reason:
                rejected = teacher_spec_to_row(spec, failure=failure, prompt_key=prompt_key, candidate_index=index, config=config)
                rejected["candidate_status"] = "rejected"
                rejected["acceptance_reason"] = unsafe_reason
                candidates.append(rejected)
                continue
            candidates.append(teacher_spec_to_row(spec, failure=failure, prompt_key=prompt_key, candidate_index=index, config=config))

    stats["estimated_api_cost_usd"] = 0.0
    return candidates, prompts, raw_records, dict(stats)


def _load_teacher_candidates(path: str | None) -> list[dict[str, Any]]:
    rows = []
    for row in load_lean_rows(path):
        row = dict(row)
        row.setdefault("source", "teacher_conditioned_mutation")
        row.setdefault("mutation_source", "teacher_conditioned")
        row.setdefault("generation_model", row.get("generation_model") or DEFAULT_TEACHER_MODEL)
        row.setdefault("split", "train_mutated")
        normalized = normalize_lean_row(row, split="train_mutated")
        for field in TEACHER_METADATA_FIELDS:
            if row.get(field) is not None:
                normalized[field] = row[field]
        rows.append(normalized)
    return rows


def _evaluate_teacher_candidates(
    candidates: list[dict[str, Any]],
    *,
    existing_bank: list[dict[str, Any]],
    response_map: dict[str, list[str]],
    mutation_config: MutationBuildConfig,
    config: TeacherMutationConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    seen_hashes = {compute_normalized_statement_hash(row) for row in existing_bank}
    seen_hashes.update(str(row.get("source_statement_hash")) for row in candidates if row.get("source_statement_hash"))
    seen_shape_hashes = {_statement_shape_hash(row) for row in existing_bank}
    seen_shape_hashes.update(
        str(row.get("source_statement_shape_hash")) for row in candidates if row.get("source_statement_shape_hash")
    )
    evaluated: list[dict[str, Any]] = []
    accepted: list[dict[str, Any]] = []
    frontier_holdout: list[dict[str, Any]] = []
    too_easy: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []

    for candidate in candidates:
        if candidate.get("candidate_status") == "rejected" and candidate.get("acceptance_reason"):
            evaluated.append(candidate)
            rejected.append(candidate)
            continue

        statement_shape_hash = _statement_shape_hash(candidate)
        if statement_shape_hash in seen_shape_hashes:
            duplicate = normalize_lean_row(candidate, split="train_mutated")
            for field in TEACHER_METADATA_FIELDS:
                if candidate.get(field) is not None:
                    duplicate[field] = candidate[field]
            duplicate["candidate_status"] = "rejected"
            duplicate["acceptance_reason"] = "duplicate_statement_shape"
            evaluated.append(duplicate)
            rejected.append(duplicate)
            continue

        row_id = str(candidate.get("id") or candidate.get("uid"))
        evaluated_row = evaluate_mutation_candidate(
            None,
            candidate,
            seen_hashes=seen_hashes,
            model_responses=response_map.get(row_id),
            config=mutation_config,
        )
        for field in TEACHER_METADATA_FIELDS:
            if field in candidate:
                evaluated_row[field] = candidate.get(field)

        evaluated.append(evaluated_row)
        reason = str(evaluated_row.get("acceptance_reason") or "")
        status = str(evaluated_row.get("candidate_status") or "")
        if status == "accepted_train":
            accepted.append(evaluated_row)
            seen_hashes.add(compute_normalized_statement_hash(evaluated_row))
            seen_shape_hashes.add(statement_shape_hash)
        elif status == "frontier_holdout":
            frontier_holdout.append(evaluated_row)
            rejected.append(evaluated_row)
            seen_hashes.add(compute_normalized_statement_hash(evaluated_row))
            seen_shape_hashes.add(statement_shape_hash)
        elif status == "too_easy" or reason.startswith(("cheap_solved", "too_easy_")):
            evaluated_row["candidate_status"] = "too_easy"
            too_easy.append(evaluated_row)
            seen_hashes.add(compute_normalized_statement_hash(evaluated_row))
            seen_shape_hashes.add(statement_shape_hash)
        else:
            rejected.append(evaluated_row)

    accepted = _apply_acceptance_balance_gates(accepted, rejected, mutation_config)
    accepted = _family_balance_gates(accepted, rejected, config)
    return evaluated, accepted, frontier_holdout, too_easy, rejected


def _refinement_requests(evaluated: list[dict[str, Any]], *, config: TeacherMutationConfig) -> list[dict[str, Any]]:
    if config.refine_rounds <= 0:
        return []
    by_family: dict[str, list[dict[str, Any]]] = {}
    for row in evaluated:
        by_family.setdefault(str(row.get("error_family") or row.get("mutation_type") or "unknown"), []).append(row)
    requests: list[dict[str, Any]] = []
    for family, rows in sorted(by_family.items()):
        statuses = Counter(str(row.get("candidate_status") or "unknown") for row in rows)
        if statuses and statuses.get("frontier_holdout", 0) == len(rows):
            requests.append({"error_family": family, "directive": "generate_easier_bridge_batch", "count": len(rows)})
        elif statuses and statuses.get("too_easy", 0) == len(rows):
            requests.append({"error_family": family, "directive": "generate_harder_bridge_batch", "count": len(rows)})
    return requests


def _summary(
    *,
    failure_records: list[dict[str, Any]],
    prompts: list[dict[str, Any]],
    raw_records: list[dict[str, Any]],
    evaluated: list[dict[str, Any]],
    accepted: list[dict[str, Any]],
    frontier_holdout: list[dict[str, Any]],
    input_frontier_rows: list[dict[str, Any]],
    too_easy: list[dict[str, Any]],
    rejected: list[dict[str, Any]],
    cumulative_bank: list[dict[str, Any]],
    eval_bank: list[dict[str, Any]],
    api_stats: dict[str, Any],
    eval_report: dict[str, Any],
    config: TeacherMutationConfig,
) -> dict[str, Any]:
    status_counts = Counter(str(row.get("candidate_status") or "unknown") for row in evaluated)
    reason_counts = Counter(str(row.get("acceptance_reason") or "unknown") for row in evaluated)
    latencies = [
        float(_jsonish(row.get("baseline_results")).get("well_formed_time_s") or 0.0)
        for row in evaluated
        if isinstance(_jsonish(row.get("baseline_results")), dict)
    ]
    model_pass_values = [
        float(_jsonish(row.get("difficulty_metrics")).get("model_pass_at_k") or 0.0)
        for row in evaluated
        if isinstance(_jsonish(row.get("difficulty_metrics")), dict)
        and "model_pass_at_k" in _jsonish(row.get("difficulty_metrics"))
    ]
    well_formed = [
        bool(_jsonish(row.get("baseline_results")).get("well_formed"))
        for row in evaluated
        if isinstance(_jsonish(row.get("baseline_results")), dict)
        and "well_formed" in _jsonish(row.get("baseline_results"))
    ]
    cheap_solved = [
        bool(_jsonish(row.get("baseline_results")).get("cheap_baseline_solved"))
        for row in evaluated
        if isinstance(_jsonish(row.get("baseline_results")), dict)
        and "cheap_baseline_solved" in _jsonish(row.get("baseline_results"))
    ]
    certificate_attempts = [
        _jsonish(row.get("proof_certificate"))
        for row in evaluated
        if isinstance(_jsonish(row.get("proof_certificate")), dict)
        and "verification_results" in _jsonish(row.get("proof_certificate"))
    ]
    verified = [bool(certificate.get("strong_prover_solved")) for certificate in certificate_attempts]
    bridge_counts = Counter(str(row.get("bridge_level") or "unknown") for row in evaluated)
    refinement_requests = _refinement_requests(evaluated, config=config)
    return {
        "failure_count": len(failure_records),
        "prompt_count": len(prompts),
        "raw_response_count": len(raw_records),
        "candidate_count": len(evaluated),
        "accepted_train_count": len(accepted),
        "frontier_holdout_count": len(frontier_holdout),
        "input_frontier_bank_size": len(input_frontier_rows),
        "too_easy_count": len(too_easy),
        "rejected_count": len(rejected),
        "accepted_bank_size": len(cumulative_bank),
        "eval_bank_size": len(eval_bank),
        "status_counts": dict(status_counts),
        "reason_counts": dict(reason_counts),
        "api_calls": int(api_stats.get("api_calls") or 0),
        "prompt_tokens": int(api_stats.get("prompt_tokens") or 0),
        "completion_tokens": int(api_stats.get("completion_tokens") or 0),
        "cache_hit_rate": float(api_stats.get("cache_hits") or 0) / max(1, len(raw_records)),
        "json_parse_failure_rate": float(api_stats.get("json_parse_errors") or 0) / max(1, len(raw_records)),
        "unsafe_rejection_rate": reason_counts.get("forbidden_proof_token", 0) / max(1, len(evaluated)),
        "precheck_rejected_count": sum(
            count
            for reason, count in reason_counts.items()
            if reason in {"duplicate_statement_hash", "duplicate_statement_shape", "noop_statement", "trivial_statement"}
        ),
        "certificate_verification_attempt_count": len(certificate_attempts),
        "lean_verification_rate": sum(verified) / max(1, len(verified)),
        "well_formed_rate": sum(well_formed) / max(1, len(well_formed)),
        "cheap_solved_rate": sum(cheap_solved) / max(1, len(cheap_solved)),
        "cheap_baseline_fail_rate": 1.0 - (sum(cheap_solved) / max(1, len(cheap_solved))),
        "model_pass_at_k": {
            "count": len(model_pass_values),
            "p50": statistics.median(model_pass_values) if model_pass_values else 0.0,
            "p95": _percentile(model_pass_values, 0.95),
            "values": model_pass_values,
        },
        "error_family_entropy": _error_family_entropy(evaluated),
        "bridge_level_distribution": dict(bridge_counts),
        "top_tactic_mass": _top_tactic_mass(accepted),
        "estimated_api_cost_usd": float(api_stats.get("estimated_api_cost_usd") or 0.0),
        "lean_latency": {
            "count": len(latencies),
            "p50_s": statistics.median(latencies) if latencies else 0.0,
            "p95_s": _percentile(latencies, 0.95),
        },
        "accepted_collapse": summarize_lean_rows(accepted),
        "source_eval_metrics": _source_eval_metrics(eval_report),
        "refinement_requests": refinement_requests,
        "config": {
            "teacher_model": config.teacher_model,
            "teacher_base_url": config.teacher_base_url,
            "max_failures": config.max_failures,
            "cases_per_family": config.cases_per_family,
            "model_pass_k": config.model_pass_k,
            "accept_max_pass_rate": config.accept_max_pass_rate,
            "max_api_calls": config.max_api_calls,
            "max_output_tokens": config.max_output_tokens,
            "temperature": config.temperature,
            "refine_rounds": config.refine_rounds,
            "require_model_pass": config.require_model_pass,
            "max_top_tactic_mass": config.max_top_tactic_mass,
            "max_error_family_mass": config.max_error_family_mass,
            "difficulty_directive": config.difficulty_directive,
            "allow_stepwise_cheap_solved": config.allow_stepwise_cheap_solved,
        },
    }


def build_teacher_mutation_bank(args: argparse.Namespace) -> dict[str, Any]:
    config = TeacherMutationConfig(
        output_dir=Path(args.output_dir),
        teacher_model=args.teacher_model,
        teacher_base_url=args.teacher_base_url,
        max_failures=args.max_failures,
        cases_per_family=args.cases_per_family,
        model_pass_k=args.model_pass_k,
        accept_max_pass_rate=args.accept_max_pass_rate,
        max_api_calls=args.max_api_calls,
        max_output_tokens=args.max_output_tokens,
        temperature=args.temperature,
        refine_rounds=args.refine_rounds,
        cache_dir=Path(args.cache_dir) if args.cache_dir else None,
        cheap_timeout_seconds=args.cheap_timeout_seconds,
        strong_timeout_seconds=args.strong_timeout_seconds,
        well_formed_timeout_seconds=args.well_formed_timeout_seconds,
        lean_command=args.lean_command,
        lean_cwd=args.lean_cwd,
        max_heartbeats=args.max_heartbeats,
        require_model_pass=args.require_model_pass,
        max_top_tactic_mass=args.max_top_tactic_mass,
        max_error_family_mass=args.max_error_family_mass,
        random_seed=args.random_seed,
        disable_teacher_thinking=args.disable_teacher_thinking,
        difficulty_directive=args.difficulty_directive,
        allow_stepwise_cheap_solved=getattr(args, "allow_stepwise_cheap_solved", True),
    )
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    eval_report = _read_eval_report(args.eval_report)
    input_frontier_rows = [normalize_lean_row(row, split=str(row.get("split") or "val_mutated")) for row in load_lean_rows(args.frontier_bank_path)]
    existing_bank = [normalize_lean_row(row, split="train_mutated") for row in load_lean_rows(args.existing_mutation_bank_path)]
    response_map = _load_response_map(args.model_responses_jsonl)
    api_stats: dict[str, Any] = {}
    prompts: list[dict[str, Any]] = []
    raw_records: list[dict[str, Any]] = []

    failure_records = _load_failure_records(args, config)
    if args.teacher_candidates_jsonl:
        candidates = _load_teacher_candidates(args.teacher_candidates_jsonl)
    else:
        candidates, prompts, raw_records, api_stats = generate_teacher_candidates(
            failure_records,
            config=config,
            eval_report=eval_report,
            frontier_rows=input_frontier_rows,
            teacher_raw_jsonl=args.teacher_raw_jsonl,
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
        generation_model=config.teacher_model,
        allow_cheap_solved_if_model_boundary=config.allow_stepwise_cheap_solved,
    )
    evaluated, accepted, frontier_holdout, too_easy, rejected = _evaluate_teacher_candidates(
        candidates,
        existing_bank=existing_bank,
        response_map=response_map,
        mutation_config=mutation_config,
        config=config,
    )
    cumulative_bank = [*existing_bank, *accepted]
    eval_bank = [*cumulative_bank, *frontier_holdout]
    summary = _summary(
        failure_records=failure_records,
        prompts=prompts,
        raw_records=raw_records,
        evaluated=evaluated,
        accepted=accepted,
        frontier_holdout=frontier_holdout,
        input_frontier_rows=input_frontier_rows,
        too_easy=too_easy,
        rejected=rejected,
        cumulative_bank=cumulative_bank,
        eval_bank=eval_bank,
        api_stats=api_stats,
        eval_report=eval_report,
        config=config,
    )

    _write_jsonl(output_dir / "failure_records.jsonl", failure_records)
    _write_jsonl(output_dir / "teacher_prompts.jsonl", prompts)
    _write_jsonl(output_dir / "teacher_raw.jsonl", raw_records)
    _write_jsonl(output_dir / "teacher_candidates.jsonl", evaluated)
    _write_jsonl(output_dir / "accepted.jsonl", accepted)
    _write_jsonl(output_dir / "frontier_holdout.jsonl", frontier_holdout)
    _write_jsonl(output_dir / "too_easy.jsonl", too_easy)
    _write_jsonl(output_dir / "rejected.jsonl", rejected)
    _write_jsonl(output_dir / "bank.jsonl", cumulative_bank)
    _write_jsonl(output_dir / "eval_bank.jsonl", eval_bank)
    _write_jsonl(output_dir / "refinement_requests.jsonl", summary["refinement_requests"])
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a DeepSeek teacher-conditioned Lean prover v1 mutation bank.")
    parser.add_argument("--rows-path")
    parser.add_argument("--responses-jsonl")
    parser.add_argument("--failure-records-jsonl")
    parser.add_argument("--eval-report", default=None)
    parser.add_argument("--frontier-bank-path", default=None)
    parser.add_argument("--existing-mutation-bank-path", default=None)
    parser.add_argument("--teacher-candidates-jsonl", default=None)
    parser.add_argument("--teacher-raw-jsonl", default=None)
    parser.add_argument("--model-responses-jsonl", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--teacher-model", default=os.getenv("TEACHER_MUTATION_MODEL", DEFAULT_TEACHER_MODEL))
    parser.add_argument("--teacher-base-url", default=os.getenv("TEACHER_MUTATION_BASE_URL", DEFAULT_TEACHER_BASE_URL))
    parser.add_argument("--max-failures", type=int, default=32)
    parser.add_argument("--cases-per-family", type=int, default=3)
    parser.add_argument("--model-pass-k", type=int, default=4)
    parser.add_argument("--accept-max-pass-rate", type=float, default=0.35)
    parser.add_argument("--max-api-calls", type=int, default=64)
    parser.add_argument("--max-output-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--refine-rounds", type=int, default=1)
    parser.add_argument("--cheap-timeout-seconds", type=float, default=5.0)
    parser.add_argument("--strong-timeout-seconds", type=float, default=20.0)
    parser.add_argument("--well-formed-timeout-seconds", type=float, default=5.0)
    parser.add_argument("--lean-command", default=None)
    parser.add_argument("--lean-cwd", default=None)
    parser.add_argument("--max-heartbeats", type=int, default=200_000)
    parser.add_argument("--require-model-pass", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-top-tactic-mass", type=float, default=0.40)
    parser.add_argument("--max-error-family-mass", type=float, default=0.30)
    parser.add_argument("--random-seed", type=int, default=1337)
    parser.add_argument("--disable-teacher-thinking", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--difficulty-directive", choices=["balanced", "easier", "harder"], default="balanced")
    parser.add_argument("--allow-stepwise-cheap-solved", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    summary = build_teacher_mutation_bank(parse_args())
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
