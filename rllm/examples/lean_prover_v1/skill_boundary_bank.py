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

from examples.lean_prover_v1.error_mutate_bank import ERROR_FAMILIES
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
    load_lean_rows,
    normalize_lean_row,
    proof_body_from_certificate,
    summarize_lean_rows,
)

DEFAULT_TEACHER_MODEL = "deepseek-v4-pro"
DEFAULT_TEACHER_BASE_URL = "https://api.deepseek.com"
TEACHER_API_KEY_ENV_ORDER = ("DEEPSEEK_API_KEY", "ANTHROPIC_AUTH_TOKEN")
TEACHER_THINKING_DISABLED = {"thinking": {"type": "disabled"}}
GENERATION_MODES = {"success_extension", "failure_bridge"}
SIGNAL_METADATA_FIELDS = (
    "signal_class",
    "signal_reason",
    "student_pass_rate",
    "student_pass_count",
    "student_response_count",
    "student_pass_at_1",
    "cheap_baseline_solved",
    "error_family",
    "target_skill",
    "error_signature",
    "failed_rollout_count",
    "failed_rollouts",
    "failure_records",
    "success_response",
    "pass_rate_delta",
    "ema_pass_rate",
)

REQUIRED_CANDIDATE_FIELDS = (
    "statement_prefix",
    "proof_body",
    "target_skill",
    "error_family",
    "bridge_level",
    "rationale",
    "expected_failure_fixed",
    "difficulty_rationale",
)

UNSAFE_PROOF_RE = re.compile(
    r"^\s*(import|theorem|lemma|def|instance|axiom|constant|opaque|namespace|section|end|open|"
    r"set_option|run_cmd|#eval|#check|#print|elab|syntax|macro|unsafe)\b",
    flags=re.MULTILINE,
)
FORBIDDEN_PROOF_TOKEN_RE = re.compile(r"\b(sorry|admit|unsafe)\b")
PROP_BINDER_RE = re.compile(r"\([^)]*:\s*Prop\)")


@dataclass(slots=True)
class SkillBoundaryConfig:
    output_dir: Path
    mode: str
    teacher_model: str = DEFAULT_TEACHER_MODEL
    teacher_base_url: str = DEFAULT_TEACHER_BASE_URL
    max_source_rows: int = 32
    cases_per_row: int = 3
    model_pass_k: int = 4
    accept_max_pass_rate: float = 0.35
    max_api_calls: int = 64
    max_output_tokens: int = 4096
    temperature: float = 0.2
    cache_dir: Path | None = None
    cheap_timeout_seconds: float = 5.0
    strong_timeout_seconds: float = 20.0
    well_formed_timeout_seconds: float = 5.0
    lean_command: str | None = None
    lean_cwd: str | None = None
    max_heartbeats: int = 200_000
    require_model_pass: bool = True
    max_top_tactic_mass: float = 0.40
    max_family_mass: float = 0.30
    random_seed: int = 1337
    disable_teacher_thinking: bool = True


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


def _read_jsonl(path: str | Path | None) -> list[dict[str, Any]]:
    if not path:
        return []
    source = Path(path)
    if not source.exists():
        return []
    rows = []
    for line in source.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=True, default=str)


def _prompt_cache_key(messages: list[dict[str, str]], config: SkillBoundaryConfig) -> str:
    payload = {
        "model": config.teacher_model,
        "messages": messages,
        "temperature": config.temperature,
        "max_output_tokens": config.max_output_tokens,
        "disable_teacher_thinking": config.disable_teacher_thinking,
        "mode": config.mode,
    }
    return hashlib.sha256(_stable_json(payload).encode("utf-8")).hexdigest()


def _teacher_api_key() -> str | None:
    for name in TEACHER_API_KEY_ENV_ORDER:
        value = os.environ.get(name)
        if value:
            return value
    return None


def _safe_ident(value: str, *, fallback: str = "skill") -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_']", "_", value)
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    if not cleaned:
        cleaned = fallback
    if cleaned[0].isdigit():
        cleaned = f"{fallback}_{cleaned}"
    return cleaned[:64]


def _short_text(value: str, limit: int = 1800) -> str:
    text = str(value or "")
    if len(text) <= limit:
        return text
    return text[:limit] + "\n<truncated>"


def _load_teacher_raw_fixture(path: str | None) -> dict[str, str]:
    fixture: dict[str, str] = {}
    for row in _read_jsonl(path):
        row_id = str(row.get("id") or row.get("source_id") or row.get("prompt_key") or "")
        raw = row.get("response") or row.get("raw") or row.get("completion")
        if raw is None and "candidates" in row:
            raw = json.dumps({"candidates": row["candidates"]})
        if row_id and raw is not None:
            fixture[row_id] = str(raw)
    return fixture


def _parse_teacher_response(text: str) -> tuple[list[dict[str, Any]], list[str]]:
    try:
        payload = json.loads(str(text or ""))
    except json.JSONDecodeError:
        return [], ["malformed_json"]
    raw_candidates = payload if isinstance(payload, list) else payload.get("candidates") if isinstance(payload, dict) else None
    if not isinstance(raw_candidates, list):
        return [], ["missing_candidates_list"]

    candidates: list[dict[str, Any]] = []
    errors: list[str] = []
    for index, item in enumerate(raw_candidates):
        if not isinstance(item, dict):
            errors.append(f"candidate_{index}_not_object")
            continue
        missing = [field for field in REQUIRED_CANDIDATE_FIELDS if not str(item.get(field) or "").strip()]
        if missing:
            errors.append(f"candidate_{index}_missing_{','.join(missing)}")
            continue
        family = str(item.get("error_family"))
        if family not in ERROR_FAMILIES:
            errors.append(f"candidate_{index}_unknown_family_{family}")
            continue
        candidates.append({field: item.get(field) for field in REQUIRED_CANDIDATE_FIELDS})
    return candidates, errors


def _candidate_safety_reason(candidate: dict[str, Any]) -> str | None:
    statement = str(candidate.get("statement_prefix") or "")
    proof_body = str(candidate.get("proof_body") or "")
    if not statement.strip():
        return "missing_statement_prefix"
    if not proof_body.strip():
        return "missing_proof_body"
    if FORBIDDEN_PROOF_TOKEN_RE.search(proof_body):
        return "forbidden_proof_token"
    if UNSAFE_PROOF_RE.search(proof_body):
        return "forbidden_proof_command"
    return None


def _pure_prop_tautology_reason(row: dict[str, Any]) -> str | None:
    statement = str(row.get("statement_prefix") or "")
    lower = statement.lower()
    if " nat" in lower or ": nat" in lower or "list" in lower or "exists" in lower or "∃" in statement:
        return None
    if PROP_BINDER_RE.search(statement) and any(token in statement for token in ("∧", "∨", "->", "↔", "Iff", "And", "Or")):
        return "pure_prop_tautology"
    if re.search(r":\s*True\s*:=\s*by", statement):
        return "pure_prop_tautology"
    return None


def _compact_signal_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": row.get("id"),
        "source": row.get("source"),
        "repo_commit": row.get("repo_commit"),
        "imports": row.get("imports") or [],
        "namespace": row.get("namespace") or "",
        "statement_prefix": row.get("statement_prefix"),
        "initial_goal_pp": row.get("initial_goal_pp") or "",
        "mutation_type": row.get("mutation_type"),
        "signal_class": row.get("signal_class"),
        "signal_reason": row.get("signal_reason"),
        "student_pass_rate": row.get("student_pass_rate"),
        "student_pass_count": row.get("student_pass_count"),
        "student_response_count": row.get("student_response_count"),
        "error_family": row.get("error_family"),
        "target_skill": row.get("target_skill"),
    }


def _build_teacher_messages(source: dict[str, Any], *, config: SkillBoundaryConfig) -> list[dict[str, str]]:
    family = str(source.get("error_family") or "equality_rewrite")
    skill = str(source.get("target_skill") or family)
    if config.mode == "success_extension":
        system = (
            "You are a Lean 4 theorem curriculum generator. Return strict JSON only. "
            "Generate harder nearby extension tasks from a theorem the student already solves. "
            "Prefer Nat equality, congruence, small rewrite chains, forall/implication structure, or simple Exists tasks. "
            "Avoid pure Prop tautologies, theorem renames, imports, declarations outside one theorem, sorry, admit, unsafe, "
            "and proof bodies inside statement_prefix."
        )
        task = "Generate harder nearby Lean 4 extension tasks for this solved theorem."
        ladder = ["target_extension", "harder_extension", "transfer_extension"]
        example_statement = "theorem extension_example (a b c : Nat) (hab : a = b) (hbc : b = c) : Nat.succ a = Nat.succ c := by"
        example_proof = "exact congrArg Nat.succ (Eq.trans hab hbc)"
    else:
        system = (
            "You are a Lean 4 theorem curriculum generator. Return strict JSON only. "
            "Generate easier bridge tasks for a failed theorem family so the student can get positive RL signal. "
            "Keep tasks close to the target skill but simpler than the failed theorem. "
            "Avoid theorem renames, imports, declarations outside one theorem, sorry, admit, unsafe, and proof bodies inside statement_prefix."
        )
        task = "Generate easier Lean 4 bridge tasks for this failed or mixed-local-gap theorem."
        ladder = ["easier_bridge", "target_bridge", "near_frontier_bridge"]
        example_statement = "theorem bridge_example (a b c : Nat) (hab : a = b) (hbc : b = c) : a = c := by"
        example_proof = "exact Eq.trans hab hbc"

    user_payload = {
        "task": task,
        "mode": config.mode,
        "cases_requested": config.cases_per_row,
        "requested_ladder": ladder,
        "target_pass_window": {
            "min_model_pass_at_k_exclusive": 0.0,
            "max_model_pass_at_k": config.accept_max_pass_rate,
            "model_pass_k": config.model_pass_k,
        },
        "source_signal": _compact_signal_row(source),
        "successful_response": _short_text(str(source.get("success_response") or proof_body_from_certificate(source)), 1200),
        "failed_rollouts": source.get("failed_rollouts") or [],
        "json_schema_example": {
            "candidates": [
                {
                    "statement_prefix": example_statement,
                    "proof_body": example_proof,
                    "target_skill": skill,
                    "error_family": family,
                    "bridge_level": ladder[0],
                    "rationale": "Nearby curriculum task around the source theorem.",
                    "expected_failure_fixed": "Practices the missing or newly learned Lean skill.",
                    "difficulty_rationale": "Should be non-trivial for cheap baselines and positive-pass for the student.",
                }
            ]
        },
    }
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(user_payload, ensure_ascii=True, sort_keys=True)},
    ]


def call_teacher_api(messages: list[dict[str, str]], *, config: SkillBoundaryConfig) -> dict[str, Any]:
    cache_key = _prompt_cache_key(messages, config)
    cache_path = config.cache_dir / f"{cache_key}.json" if config.cache_dir else None
    if cache_path and cache_path.exists():
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        payload["cache_hit"] = True
        return payload

    api_key = _teacher_api_key()
    if not api_key:
        raise RuntimeError("Set DEEPSEEK_API_KEY or ANTHROPIC_AUTH_TOKEN, or pass --teacher-raw-jsonl.")

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


def _select_source_rows(signal_rows: list[dict[str, Any]], *, config: SkillBoundaryConfig) -> list[dict[str, Any]]:
    if config.mode == "success_extension":
        selected = [
            row
            for row in signal_rows
            if str(row.get("signal_class")) == "too_easy"
            or bool(row.get("student_pass_at_1"))
            or float(row.get("student_pass_rate") or 0.0) >= config.accept_max_pass_rate
        ]
        selected.sort(key=lambda row: (-float(row.get("student_pass_rate") or 0.0), str(row.get("id") or "")))
    else:
        selected = [
            row
            for row in signal_rows
            if str(row.get("signal_class")) in {"too_hard", "mixed_local_gap"}
        ]
        selected.sort(
            key=lambda row: (
                0 if str(row.get("signal_class")) == "mixed_local_gap" else 1,
                float(row.get("student_pass_rate") or 0.0),
                str(row.get("id") or ""),
            )
        )
    return selected[: max(0, config.max_source_rows)]


def teacher_spec_to_row(
    spec: dict[str, Any],
    *,
    source: dict[str, Any],
    prompt_key: str,
    candidate_index: int,
    config: SkillBoundaryConfig,
) -> dict[str, Any]:
    source_id = str(source.get("id") or source.get("uid") or "source")
    family = str(spec.get("error_family") or source.get("error_family") or "equality_rewrite")
    bridge_level = str(spec.get("bridge_level") or ("extension" if config.mode == "success_extension" else "bridge"))
    statement_prefix = str(spec.get("statement_prefix") or "").strip()
    proof_body = str(spec.get("proof_body") or "").strip()
    statement_hash = hashlib.sha1(statement_prefix.encode("utf-8")).hexdigest()[:10]
    row_id = (
        f"{source_id}__skill_boundary_{_safe_ident(config.mode)}_"
        f"{_safe_ident(family)}_{_safe_ident(bridge_level)}_{candidate_index}_{statement_hash}"
    )
    raw_row = {
        "id": row_id,
        "source": f"skill_boundary_{config.mode}",
        "repo_commit": source.get("repo_commit") or DEFAULT_REPO_COMMIT,
        "imports": source.get("imports") or [],
        "namespace": source.get("namespace") or "",
        "statement_prefix": statement_prefix,
        "initial_goal_pp": "",
        "seed_id": None,
        "parent_ids": [source_id],
        "mutation_type": family,
        "mutation_source": f"skill_boundary_{config.mode}",
        "mutation_rule": str(spec.get("target_skill") or source.get("target_skill") or config.mode),
        "generation_model": config.teacher_model,
        "candidate_status": "candidate",
        "acceptance_reason": "",
        "difficulty_band": f"skill_boundary_{config.mode}_{bridge_level}",
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
        "error_family": family,
        "target_skill": str(spec.get("target_skill") or source.get("target_skill") or ""),
        "bridge_level": bridge_level,
        "signal_class": source.get("signal_class"),
        "signal_reason": source.get("signal_reason"),
        "student_source_pass_rate": source.get("student_pass_rate"),
    }
    row = normalize_lean_row(raw_row, split="train_mutated")
    for field in (
        "error_family",
        "target_skill",
        "bridge_level",
        "signal_class",
        "signal_reason",
        "student_source_pass_rate",
    ):
        row[field] = raw_row.get(field)
    return row


def generate_teacher_candidates(
    source_rows: list[dict[str, Any]],
    *,
    config: SkillBoundaryConfig,
    teacher_raw_jsonl: str | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    fixture = _load_teacher_raw_fixture(teacher_raw_jsonl)
    prompts: list[dict[str, Any]] = []
    raw_records: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    stats: Counter[str] = Counter()

    for source in source_rows:
        if stats["api_calls"] >= config.max_api_calls:
            break
        source_id = str(source.get("id") or source.get("uid") or "")
        messages = _build_teacher_messages(source, config=config)
        prompt_key = _prompt_cache_key(messages, config)
        prompts.append(
            {
                "id": source_id,
                "prompt_key": prompt_key,
                "messages": messages,
                "mode": config.mode,
                "error_family": source.get("error_family"),
                "signal_class": source.get("signal_class"),
            }
        )
        if source_id in fixture:
            payload = {"response": fixture[source_id], "cache_hit": True, "prompt_tokens": 0, "completion_tokens": 0, "fixture": True}
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

        raw_records.append({"id": source_id, "prompt_key": prompt_key, **payload})
        parsed, parse_errors = _parse_teacher_response(str(payload.get("response") or ""))
        if parse_errors:
            stats["json_parse_errors"] += 1
        stats["prompt_tokens"] += int(payload.get("prompt_tokens") or 0)
        stats["completion_tokens"] += int(payload.get("completion_tokens") or 0)
        for index, spec in enumerate(parsed[: max(1, config.cases_per_row)]):
            unsafe_reason = _candidate_safety_reason(spec)
            row = teacher_spec_to_row(spec, source=source, prompt_key=prompt_key, candidate_index=index, config=config)
            if unsafe_reason:
                row["candidate_status"] = "rejected"
                row["acceptance_reason"] = unsafe_reason
            candidates.append(row)
    stats["estimated_api_cost_usd"] = 0.0
    return candidates, prompts, raw_records, dict(stats)


def _load_teacher_candidates(path: str | None, *, config: SkillBoundaryConfig) -> list[dict[str, Any]]:
    rows = []
    for raw_row in load_lean_rows(path):
        row = dict(raw_row)
        row.setdefault("source", f"skill_boundary_{config.mode}")
        row.setdefault("mutation_source", f"skill_boundary_{config.mode}")
        row.setdefault("generation_model", config.teacher_model)
        row.setdefault("split", "train_mutated")
        normalized = normalize_lean_row(row, split="train_mutated")
        for field in (
            "error_family",
            "target_skill",
            "bridge_level",
            "signal_class",
            "signal_reason",
            "student_source_pass_rate",
        ):
            if field in row:
                normalized[field] = row[field]
        rows.append(normalized)
    return rows


def _family_balance_gates(
    accepted: list[dict[str, Any]],
    rejected: list[dict[str, Any]],
    *,
    config: SkillBoundaryConfig,
) -> list[dict[str, Any]]:
    if not accepted:
        return accepted
    max_family = max(1, round(len(accepted) * config.max_family_mass))
    counts: Counter[str] = Counter()
    kept: list[dict[str, Any]] = []
    for row in accepted:
        family = str(row.get("error_family") or row.get("mutation_type") or "unknown")
        if counts[family] >= max_family:
            row["candidate_status"] = "rejected"
            row["acceptance_reason"] = "collapse_error_family_mass"
            rejected.append(row)
            continue
        counts[family] += 1
        kept.append(row)
    return kept


def _evaluate_candidates(
    candidates: list[dict[str, Any]],
    *,
    source_rows: list[dict[str, Any]],
    existing_bank: list[dict[str, Any]],
    response_map: dict[str, list[str]],
    config: SkillBoundaryConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    source_by_id = {str(row.get("id") or row.get("uid") or ""): row for row in source_rows}
    seen_hashes = {compute_normalized_statement_hash(row) for row in existing_bank}
    seen_hashes.update(compute_normalized_statement_hash(row) for row in source_rows)
    evaluated: list[dict[str, Any]] = []
    accepted: list[dict[str, Any]] = []
    frontier: list[dict[str, Any]] = []
    too_easy: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []

    mutation_config = MutationBuildConfig(
        output_dir=config.output_dir,
        cheap_timeout_seconds=config.cheap_timeout_seconds,
        strong_timeout_seconds=config.strong_timeout_seconds,
        well_formed_timeout_seconds=config.well_formed_timeout_seconds,
        lean_command=config.lean_command,
        lean_cwd=config.lean_cwd,
        max_heartbeats=config.max_heartbeats,
        min_ast_edit_distance=0.05,
        accept_max_pass_rate=config.accept_max_pass_rate,
        model_pass_k=config.model_pass_k,
        require_model_pass=config.require_model_pass,
        max_top_tactic_mass=config.max_top_tactic_mass,
        max_mutation_type_mass=config.max_family_mass,
        generation_model=config.teacher_model,
    )

    for candidate in candidates:
        if candidate.get("candidate_status") == "rejected" and candidate.get("acceptance_reason"):
            evaluated.append(candidate)
            rejected.append(candidate)
            continue
        if config.mode == "success_extension":
            prop_reason = _pure_prop_tautology_reason(candidate)
            if prop_reason:
                candidate["candidate_status"] = "rejected"
                candidate["acceptance_reason"] = prop_reason
                evaluated.append(candidate)
                rejected.append(candidate)
                continue

        parent_ids = candidate.get("parent_ids") if isinstance(candidate.get("parent_ids"), list) else []
        parent_id = str(parent_ids[0]) if parent_ids else ""
        seed_row = source_by_id.get(parent_id)
        row_id = str(candidate.get("id") or candidate.get("uid"))
        evaluated_row = evaluate_mutation_candidate(
            seed_row,
            candidate,
            seen_hashes=seen_hashes,
            model_responses=response_map.get(row_id),
            config=mutation_config,
        )
        for field in (
            "parent_ids",
            "mutation_source",
            "mutation_rule",
            "generation_model",
            "error_family",
            "target_skill",
            "bridge_level",
            "signal_class",
            "signal_reason",
            "student_source_pass_rate",
        ):
            if field in candidate:
                evaluated_row[field] = candidate[field]
        evaluated.append(evaluated_row)
        status = str(evaluated_row.get("candidate_status") or "")
        reason = str(evaluated_row.get("acceptance_reason") or "")
        if status == "accepted_train":
            accepted.append(evaluated_row)
            seen_hashes.add(compute_normalized_statement_hash(evaluated_row))
        elif status == "frontier_holdout":
            frontier.append(evaluated_row)
        elif status == "too_easy" or reason.startswith(("cheap_solved", "too_easy_")):
            evaluated_row["candidate_status"] = "too_easy"
            too_easy.append(evaluated_row)
        else:
            rejected.append(evaluated_row)

    accepted = _apply_acceptance_balance_gates(accepted, rejected, mutation_config)
    accepted = _family_balance_gates(accepted, rejected, config=config)
    return evaluated, accepted, frontier, too_easy, rejected


def _top_tactic_mass(rows: list[dict[str, Any]]) -> float:
    if not rows:
        return 0.0
    counts = Counter(_first_tactic(str(_jsonish(row.get("proof_certificate")).get("proof_body") or "")) for row in rows)
    return counts.most_common(1)[0][1] / len(rows) if counts else 0.0


def _entropy(rows: list[dict[str, Any]], key: str) -> float:
    counts = Counter(str(row.get(key) or "unknown") for row in rows)
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    return -sum((count / total) * math.log(count / total, 2) for count in counts.values() if count)


def _summary(
    *,
    source_rows: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    accepted: list[dict[str, Any]],
    frontier: list[dict[str, Any]],
    too_easy: list[dict[str, Any]],
    rejected: list[dict[str, Any]],
    prompts: list[dict[str, Any]],
    raw_records: list[dict[str, Any]],
    api_stats: dict[str, Any],
    config: SkillBoundaryConfig,
) -> dict[str, Any]:
    status_counts = Counter(str(row.get("candidate_status") or "unknown") for row in candidates)
    reason_counts = Counter(str(row.get("acceptance_reason") or "unknown") for row in candidates)
    pass_values = [
        float(_jsonish(row.get("difficulty_metrics")).get("model_pass_at_k") or 0.0)
        for row in candidates
        if isinstance(_jsonish(row.get("difficulty_metrics")), dict)
        and "model_pass_at_k" in _jsonish(row.get("difficulty_metrics"))
    ]
    cheap = [
        bool(_jsonish(row.get("baseline_results")).get("cheap_baseline_solved"))
        for row in candidates
        if isinstance(_jsonish(row.get("baseline_results")), dict)
        and "cheap_baseline_solved" in _jsonish(row.get("baseline_results"))
    ]
    latencies = [
        float(_jsonish(row.get("baseline_results")).get("well_formed_time_s") or 0.0)
        for row in candidates
        if isinstance(_jsonish(row.get("baseline_results")), dict)
    ]
    return {
        "mode": config.mode,
        "source_count": len(source_rows),
        "prompt_count": len(prompts),
        "raw_response_count": len(raw_records),
        "candidate_count": len(candidates),
        "accepted_train_count": len(accepted),
        "frontier_holdout_count": len(frontier),
        "too_easy_count": len(too_easy),
        "rejected_count": len(rejected),
        "status_counts": dict(status_counts),
        "reason_counts": dict(reason_counts),
        "cheap_solved_rate": sum(cheap) / max(1, len(cheap)),
        "cheap_baseline_fail_rate": 1.0 - (sum(cheap) / max(1, len(cheap))),
        "model_pass_at_k": {
            "count": len(pass_values),
            "p50": statistics.median(pass_values) if pass_values else 0.0,
            "p95": _percentile(pass_values, 0.95),
            "values": pass_values,
        },
        "error_family_entropy": _entropy(candidates, "error_family"),
        "top_tactic_mass": _top_tactic_mass(accepted),
        "lean_latency": {
            "count": len(latencies),
            "p50_s": statistics.median(latencies) if latencies else 0.0,
            "p95_s": _percentile(latencies, 0.95),
        },
        "api_calls": int(api_stats.get("api_calls") or 0),
        "prompt_tokens": int(api_stats.get("prompt_tokens") or 0),
        "completion_tokens": int(api_stats.get("completion_tokens") or 0),
        "cache_hit_rate": float(api_stats.get("cache_hits") or 0) / max(1, len(raw_records)),
        "json_parse_failure_rate": float(api_stats.get("json_parse_errors") or 0) / max(1, len(raw_records)),
        "estimated_api_cost_usd": float(api_stats.get("estimated_api_cost_usd") or 0.0),
        "accepted_collapse": summarize_lean_rows(accepted),
        "config": {
            "max_source_rows": config.max_source_rows,
            "cases_per_row": config.cases_per_row,
            "model_pass_k": config.model_pass_k,
            "accept_max_pass_rate": config.accept_max_pass_rate,
            "max_top_tactic_mass": config.max_top_tactic_mass,
            "max_family_mass": config.max_family_mass,
            "require_model_pass": config.require_model_pass,
        },
    }


def build_generation_bank(args: argparse.Namespace) -> dict[str, Any]:
    config = SkillBoundaryConfig(
        output_dir=Path(args.output_dir),
        mode=args.mode,
        teacher_model=args.teacher_model,
        teacher_base_url=args.teacher_base_url,
        max_source_rows=args.max_source_rows,
        cases_per_row=args.cases_per_row,
        model_pass_k=args.model_pass_k,
        accept_max_pass_rate=args.accept_max_pass_rate,
        max_api_calls=args.max_api_calls,
        max_output_tokens=args.max_output_tokens,
        temperature=args.temperature,
        cache_dir=Path(args.cache_dir) if args.cache_dir else None,
        cheap_timeout_seconds=args.cheap_timeout_seconds,
        strong_timeout_seconds=args.strong_timeout_seconds,
        well_formed_timeout_seconds=args.well_formed_timeout_seconds,
        lean_command=args.lean_command,
        lean_cwd=args.lean_cwd,
        max_heartbeats=args.max_heartbeats,
        require_model_pass=args.require_model_pass,
        max_top_tactic_mass=args.max_top_tactic_mass,
        max_family_mass=args.max_family_mass,
        random_seed=args.random_seed,
        disable_teacher_thinking=args.disable_teacher_thinking,
    )
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    signal_rows = []
    for raw_signal_row in _read_jsonl(args.signal_rows_path):
        signal_row = normalize_lean_row(raw_signal_row, split=str(raw_signal_row.get("split") or "val"))
        for field in SIGNAL_METADATA_FIELDS:
            if field in raw_signal_row:
                signal_row[field] = raw_signal_row[field]
        signal_rows.append(signal_row)
    source_rows = _select_source_rows(signal_rows, config=config)
    rng = random.Random(config.random_seed)
    rng.shuffle(source_rows)

    existing_bank = [normalize_lean_row(row, split="train_mutated") for row in load_lean_rows(args.existing_mutation_bank_path)]
    response_map = _load_response_map(args.model_responses_jsonl)
    api_stats: dict[str, Any] = {}
    prompts: list[dict[str, Any]] = []
    raw_records: list[dict[str, Any]] = []
    if args.teacher_candidates_jsonl:
        candidates = _load_teacher_candidates(args.teacher_candidates_jsonl, config=config)
    else:
        candidates, prompts, raw_records, api_stats = generate_teacher_candidates(
            source_rows,
            config=config,
            teacher_raw_jsonl=args.teacher_raw_jsonl,
        )
    evaluated, accepted, frontier, too_easy, rejected = _evaluate_candidates(
        candidates,
        source_rows=source_rows,
        existing_bank=existing_bank,
        response_map=response_map,
        config=config,
    )
    bank = [*existing_bank, *accepted]
    eval_bank = [*bank, *frontier]
    summary = _summary(
        source_rows=source_rows,
        candidates=evaluated,
        accepted=accepted,
        frontier=frontier,
        too_easy=too_easy,
        rejected=rejected,
        prompts=prompts,
        raw_records=raw_records,
        api_stats=api_stats,
        config=config,
    )
    summary["accepted_bank_size"] = len(bank)
    summary["eval_bank_size"] = len(eval_bank)

    _write_jsonl(output_dir / "source_signal_rows.jsonl", source_rows)
    _write_jsonl(output_dir / "teacher_prompts.jsonl", prompts)
    _write_jsonl(output_dir / "teacher_raw.jsonl", raw_records)
    _write_jsonl(output_dir / "teacher_candidates.jsonl", evaluated)
    _write_jsonl(output_dir / "accepted.jsonl", accepted)
    _write_jsonl(output_dir / "frontier_holdout.jsonl", frontier)
    _write_jsonl(output_dir / "too_easy.jsonl", too_easy)
    _write_jsonl(output_dir / "rejected.jsonl", rejected)
    _write_jsonl(output_dir / "bank.jsonl", bank)
    _write_jsonl(output_dir / "eval_bank.jsonl", eval_bank)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def _load_bank_rows(path: str | None, *, accepted_only: bool = True) -> list[dict[str, Any]]:
    if not path:
        return []
    source = Path(path)
    if source.is_dir():
        candidates = [source / "accepted.jsonl", source / "bank.jsonl"] if accepted_only else [source / "eval_bank.jsonl"]
        for candidate in candidates:
            if candidate.exists():
                source = candidate
                break
    rows = load_lean_rows(str(source))
    if accepted_only:
        rows = [
            row
            for row in rows
            if str(row.get("candidate_status") or "accepted_train") == "accepted_train"
            or str(row.get("split") or "") == "train_mutated"
        ]
    return [normalize_lean_row(row, split="train_mutated") for row in rows]


def _cap_by_family(rows: list[dict[str, Any]], *, max_mass: float, limit: int) -> list[dict[str, Any]]:
    if limit <= 0:
        return []
    max_family = max(1, round(limit * max_mass))
    counts: Counter[str] = Counter()
    kept: list[dict[str, Any]] = []
    for row in rows:
        family = str(row.get("error_family") or row.get("mutation_type") or "unknown")
        if counts[family] >= max_family:
            continue
        counts[family] += 1
        kept.append(row)
        if len(kept) >= limit:
            break
    return kept


def _count_static_rows(static_corpus_path: str | None, train_static_size: int) -> int:
    rows = load_lean_rows(static_corpus_path)
    train_rows = [row for row in rows if str(row.get("split") or "train_static") in {"train", "train_static"}]
    if train_rows:
        return len(train_rows)
    return max(1, int(train_static_size))


def build_composed_bank(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.random_seed)
    existing_rows = _load_bank_rows(args.existing_mutation_bank_path)
    extension_rows = _load_bank_rows(args.success_bank_path)
    bridge_rows = _load_bank_rows(args.bridge_bank_path)
    extension_frontier = _load_bank_rows(args.success_eval_bank_path, accepted_only=False)
    bridge_frontier = _load_bank_rows(args.bridge_eval_bank_path, accepted_only=False)

    static_count = _count_static_rows(args.static_corpus_path, args.train_static_size)
    static_floor = min(0.95, max(0.05, float(args.static_floor)))
    total_target = max(static_count, round(static_count / static_floor))
    generated_limit = max(0, total_target - static_count)
    if generated_limit <= 0:
        selected: list[dict[str, Any]] = []
    else:
        if bridge_rows:
            extension_target = min(len(extension_rows), max(0, round(total_target * float(args.success_extension_weight))))
            bridge_target = min(len(bridge_rows), max(0, generated_limit - extension_target))
        else:
            extension_target = min(len(extension_rows), generated_limit)
            bridge_target = 0
        rng.shuffle(extension_rows)
        rng.shuffle(bridge_rows)
        rng.shuffle(existing_rows)
        selected_extension = _cap_by_family(extension_rows, max_mass=args.max_family_mass, limit=extension_target)
        selected_bridge = _cap_by_family(bridge_rows, max_mass=args.max_family_mass, limit=bridge_target)
        selected = [*selected_extension, *selected_bridge]
        remaining = generated_limit - len(selected)
        if remaining > 0:
            selected.extend(_cap_by_family(existing_rows, max_mass=args.max_family_mass, limit=remaining))
        remaining = generated_limit - len(selected)
        if remaining > 0:
            seen = {str(row.get("id") or row.get("uid")) for row in selected}
            filler = [row for row in [*extension_rows, *bridge_rows] if str(row.get("id") or row.get("uid")) not in seen]
            selected.extend(_cap_by_family(filler, max_mass=args.max_family_mass, limit=remaining))

    eval_seen: set[str] = set()
    eval_rows: list[dict[str, Any]] = []
    for row in [*selected, *extension_frontier, *bridge_frontier]:
        row_id = str(row.get("id") or row.get("uid") or compute_normalized_statement_hash(row))
        if row_id in eval_seen:
            continue
        eval_seen.add(row_id)
        eval_rows.append(row)

    summary = {
        "static_count": static_count,
        "static_floor": static_floor,
        "total_target": total_target,
        "generated_limit": generated_limit,
        "sampled_train_bank_size": len(selected),
        "eval_bank_size": len(eval_rows),
        "source_counts": dict(Counter(str(row.get("source") or "unknown") for row in selected)),
        "family_counts": dict(Counter(str(row.get("error_family") or row.get("mutation_type") or "unknown") for row in selected)),
        "success_extension_input_count": len(extension_rows),
        "failure_bridge_input_count": len(bridge_rows),
        "existing_input_count": len(existing_rows),
        "success_extension_weight": args.success_extension_weight,
        "failure_bridge_weight": args.failure_bridge_weight,
        "max_family_mass": args.max_family_mass,
        "collapse": summarize_lean_rows(selected),
    }
    _write_jsonl(output_dir / "sampled_train_bank.jsonl", selected)
    _write_jsonl(output_dir / "eval_bank.jsonl", eval_rows)
    (output_dir / "controller_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Lean prover v1 skill-boundary curriculum banks.")
    parser.add_argument("--mode", choices=["success_extension", "failure_bridge", "compose_bank"], required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--signal-rows-path", default=None)
    parser.add_argument("--existing-mutation-bank-path", default=None)
    parser.add_argument("--teacher-candidates-jsonl", default=None)
    parser.add_argument("--teacher-raw-jsonl", default=None)
    parser.add_argument("--model-responses-jsonl", default=None)
    parser.add_argument("--teacher-model", default=os.getenv("SKILL_BOUNDARY_TEACHER_MODEL", DEFAULT_TEACHER_MODEL))
    parser.add_argument("--teacher-base-url", default=os.getenv("SKILL_BOUNDARY_TEACHER_BASE_URL", DEFAULT_TEACHER_BASE_URL))
    parser.add_argument("--max-source-rows", type=int, default=32)
    parser.add_argument("--cases-per-row", type=int, default=3)
    parser.add_argument("--model-pass-k", type=int, default=4)
    parser.add_argument("--accept-max-pass-rate", type=float, default=0.35)
    parser.add_argument("--max-api-calls", type=int, default=64)
    parser.add_argument("--max-output-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--cheap-timeout-seconds", type=float, default=5.0)
    parser.add_argument("--strong-timeout-seconds", type=float, default=20.0)
    parser.add_argument("--well-formed-timeout-seconds", type=float, default=5.0)
    parser.add_argument("--lean-command", default=None)
    parser.add_argument("--lean-cwd", default=None)
    parser.add_argument("--max-heartbeats", type=int, default=200_000)
    parser.add_argument("--require-model-pass", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-top-tactic-mass", type=float, default=0.40)
    parser.add_argument("--max-family-mass", type=float, default=0.30)
    parser.add_argument("--random-seed", type=int, default=1337)
    parser.add_argument("--disable-teacher-thinking", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--success-bank-path", default=None)
    parser.add_argument("--bridge-bank-path", default=None)
    parser.add_argument("--success-eval-bank-path", default=None)
    parser.add_argument("--bridge-eval-bank-path", default=None)
    parser.add_argument("--static-corpus-path", default=None)
    parser.add_argument("--train-static-size", type=int, default=1024)
    parser.add_argument("--static-floor", type=float, default=0.50)
    parser.add_argument("--success-extension-weight", type=float, default=0.35)
    parser.add_argument("--failure-bridge-weight", type=float, default=0.15)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode in GENERATION_MODES:
        if not args.signal_rows_path:
            raise SystemExit("--signal-rows-path is required for generation modes.")
        summary = build_generation_bank(args)
    else:
        summary = build_composed_bank(args)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
