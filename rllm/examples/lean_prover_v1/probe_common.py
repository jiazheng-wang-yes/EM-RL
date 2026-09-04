from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from examples.lean_prover_v1.lean_worker import verify_lean_proof
from rllm.data.dataset import DatasetRegistry
from rllm.rewards.reward_types import RewardOutput

DATASET_NAME = "lean_prover_v1"
STATIC_SPLITS = ("train_static", "val_static", "test_static")
MUTATED_SPLITS = ("train_mutated", "val_mutated", "test_mutated")
ALL_SPLITS = (*STATIC_SPLITS, *MUTATED_SPLITS)
COMBINED_SPLITS = ("train", "val", "test")
DEFAULT_REPO_COMMIT = "synthetic-smoke"

REQUIRED_ROW_FIELDS = (
    "id",
    "source",
    "repo_commit",
    "imports",
    "namespace",
    "statement_prefix",
    "initial_goal_pp",
    "seed_id",
    "mutation_type",
    "difficulty_band",
    "baseline_results",
    "proof_certificate",
    "split",
)

OPTIONAL_ROW_FIELDS = (
    "parent_ids",
    "mutation_source",
    "mutation_rule",
    "generation_model",
    "candidate_status",
    "acceptance_reason",
    "difficulty_metrics",
    "normalized_statement_hash",
)

TRIVIAL_STATEMENT_RE = re.compile(r":\s*(True|P\s*->\s*P|p\s*->\s*p)\s*:=\s*by\b")
THEOREM_DECL_NAME_RE = re.compile(r"^(theorem|lemma)\s+[A-Za-z_][A-Za-z0-9_'.]*\b")


@dataclass(slots=True)
class MutationFilterResult:
    accepted: bool
    reason: str
    metadata: dict[str, Any] = field(default_factory=dict)


def _jsonish(value: Any) -> Any:
    if value is None:
        return {}
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return {}
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            return {"raw": value}
    return value


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=True, default=str)


def normalize_statement(statement_prefix: str) -> str:
    return " ".join(str(statement_prefix).split())


def normalize_statement_for_hash(statement_prefix: str) -> str:
    statement = normalize_statement(statement_prefix)
    return THEOREM_DECL_NAME_RE.sub(r"\1 _", statement, count=1)


def compute_theorem_hash(row: dict[str, Any]) -> str:
    payload = {
        "imports": row.get("imports") or [],
        "namespace": row.get("namespace") or "",
        "statement_prefix": normalize_statement(str(row.get("statement_prefix", ""))),
        "repo_commit": row.get("repo_commit") or "",
    }
    return hashlib.sha256(_stable_json(payload).encode("utf-8")).hexdigest()


def compute_normalized_statement_hash(row: dict[str, Any]) -> str:
    payload = {
        "imports": row.get("imports") or [],
        "namespace": row.get("namespace") or "",
        "statement_prefix": normalize_statement_for_hash(str(row.get("statement_prefix", ""))),
        "repo_commit": row.get("repo_commit") or "",
    }
    return hashlib.sha256(_stable_json(payload).encode("utf-8")).hexdigest()


def split_family(split: str) -> str:
    split = str(split)
    if split.startswith("train"):
        return "train"
    if split.startswith("val"):
        return "val"
    if split.startswith("test"):
        return "test"
    return split


def proof_body_from_certificate(row: dict[str, Any]) -> str:
    certificate = _jsonish(row.get("proof_certificate"))
    if isinstance(certificate, dict):
        return str(certificate.get("proof_body") or certificate.get("proof") or "")
    return ""


def build_question(row: dict[str, Any]) -> str:
    imports = "\n".join(f"import {module}" for module in row.get("imports", []) or [])
    namespace = str(row.get("namespace") or "").strip()
    namespace_line = f"namespace {namespace}\n\n" if namespace else ""
    end_line = f"\n\nend {namespace}" if namespace else ""
    theorem = str(row["statement_prefix"]).strip()
    return "\n".join(
        [
            "Complete this Lean 4 theorem.",
            "Output only the proof body that comes after `by`.",
            "Do not include imports, declarations, `sorry`, or `admit`.",
            "",
            "```lean",
            imports,
            namespace_line + theorem,
            "  -- proof goes here",
            end_line,
            "```",
        ]
    )


def normalize_lean_row(row: dict[str, Any], *, split: str | None = None) -> dict[str, Any]:
    normalized = {field: row.get(field) for field in REQUIRED_ROW_FIELDS}
    for field in OPTIONAL_ROW_FIELDS:
        if field in row:
            normalized[field] = row.get(field)
    normalized["id"] = str(normalized.get("id") or row.get("uid") or hashlib.sha1(_stable_json(row).encode("utf-8")).hexdigest()[:16])
    normalized["uid"] = normalized["id"]
    normalized["source"] = str(normalized.get("source") or row.get("data_source") or "unknown")
    normalized["repo_commit"] = str(normalized.get("repo_commit") or DEFAULT_REPO_COMMIT)
    normalized["imports"] = normalized.get("imports") or []
    if isinstance(normalized["imports"], str):
        normalized["imports"] = [item.strip() for item in normalized["imports"].split(",") if item.strip()]
    normalized["namespace"] = str(normalized.get("namespace") or "")
    normalized["statement_prefix"] = str(normalized.get("statement_prefix") or row.get("theorem_statement") or "").strip()
    normalized["initial_goal_pp"] = str(normalized.get("initial_goal_pp") or "")
    normalized["seed_id"] = normalized.get("seed_id")
    normalized["mutation_type"] = str(normalized.get("mutation_type") or "static")
    normalized["difficulty_band"] = str(normalized.get("difficulty_band") or "smoke")
    normalized["baseline_results"] = _jsonish(normalized.get("baseline_results"))
    normalized["proof_certificate"] = _jsonish(normalized.get("proof_certificate"))
    if "difficulty_metrics" in normalized:
        normalized["difficulty_metrics"] = _jsonish(normalized.get("difficulty_metrics"))
    if "parent_ids" in normalized and isinstance(normalized["parent_ids"], str):
        normalized["parent_ids"] = [item.strip() for item in normalized["parent_ids"].split(",") if item.strip()]
    normalized["split"] = str(split or normalized.get("split") or row.get("split") or "train_static")
    normalized["theorem_hash"] = str(row.get("theorem_hash") or compute_theorem_hash(normalized))
    normalized["normalized_statement_hash"] = str(normalized.get("normalized_statement_hash") or normalized["theorem_hash"])
    normalized["question"] = str(row.get("question") or build_question(normalized))
    normalized["ground_truth"] = _stable_json(
        {
            "statement_prefix": normalized["statement_prefix"],
            "proof_certificate": normalized["proof_certificate"],
        }
    )
    normalized["data_source"] = normalized["source"]
    return normalized


def _make_row(
    *,
    row_id: str,
    statement_prefix: str,
    proof_body: str,
    split: str,
    source: str = "synthetic_smoke",
    seed_id: str | None = None,
    mutation_type: str = "static",
    difficulty_band: str = "smoke",
    baseline_results: dict[str, Any] | None = None,
) -> dict[str, Any]:
    row = {
        "id": row_id,
        "source": source,
        "repo_commit": DEFAULT_REPO_COMMIT,
        "imports": [],
        "namespace": "LeanProverV1",
        "statement_prefix": statement_prefix,
        "initial_goal_pp": "",
        "seed_id": seed_id,
        "mutation_type": mutation_type,
        "difficulty_band": difficulty_band,
        "baseline_results": baseline_results or {"well_formed": True, "cheap_baseline_solved": False},
        "proof_certificate": {"source": "synthetic", "strong_prover_solved": True, "proof_body": proof_body},
        "split": split,
    }
    return normalize_lean_row(row)


LeanTemplate = tuple[str, str, str]


PROP_TRIPLES = (
    ("p", "q", "r"),
    ("a", "b", "c"),
    ("u", "v", "w"),
    ("x", "y", "z"),
    ("p0", "p1", "p2"),
    ("q0", "q1", "q2"),
    ("r0", "r1", "r2"),
    ("s", "t", "u0"),
    ("m", "n", "k"),
    ("foo", "bar", "baz"),
    ("leftP", "midP", "rightP"),
    ("hypA", "hypB", "hypC"),
)


def _generated_static_templates() -> list[LeanTemplate]:
    templates: list[LeanTemplate] = []
    for idx, (p, q, r) in enumerate(PROP_TRIPLES):
        templates.extend(
            [
                (
                    f"prop_and_comm_{idx}",
                    f"theorem lean_prop_and_comm_{idx} ({p} {q} : Prop) (h : {p} ∧ {q}) : {q} ∧ {p} := by",
                    "constructor\n· exact h.right\n· exact h.left",
                ),
                (
                    f"prop_and_assoc_left_{idx}",
                    f"theorem lean_prop_and_assoc_left_{idx} ({p} {q} {r} : Prop) (h : {p} ∧ {q} ∧ {r}) : ({p} ∧ {q}) ∧ {r} := by",
                    "constructor\n· exact And.intro h.left h.right.left\n· exact h.right.right",
                ),
                (
                    f"prop_and_assoc_right_{idx}",
                    f"theorem lean_prop_and_assoc_right_{idx} ({p} {q} {r} : Prop) (h : ({p} ∧ {q}) ∧ {r}) : {p} ∧ {q} ∧ {r} := by",
                    "constructor\n· exact h.left.left\n· exact And.intro h.left.right h.right",
                ),
                (
                    f"prop_or_comm_{idx}",
                    f"theorem lean_prop_or_comm_{idx} ({p} {q} : Prop) (h : {p} ∨ {q}) : {q} ∨ {p} := by",
                    "cases h with\n| inl hp => exact Or.inr hp\n| inr hq => exact Or.inl hq",
                ),
                (
                    f"prop_imp_trans_{idx}",
                    f"theorem lean_prop_imp_trans_{idx} ({p} {q} {r} : Prop) (ha : {p}) (hab : {p} -> {q}) (hbc : {q} -> {r}) : {r} := by",
                    "exact hbc (hab ha)",
                ),
                (
                    f"prop_and_left_{idx}",
                    f"theorem lean_prop_and_left_{idx} ({p} {q} : Prop) (h : {p} ∧ {q}) : {p} := by",
                    "exact h.left",
                ),
                (
                    f"prop_and_right_{idx}",
                    f"theorem lean_prop_and_right_{idx} ({p} {q} : Prop) (h : {p} ∧ {q}) : {q} := by",
                    "exact h.right",
                ),
                (
                    f"prop_not_intro_{idx}",
                    f"theorem lean_prop_not_intro_{idx} ({p} : Prop) (h : {p} -> False) : ¬ {p} := by",
                    "intro hp\nexact h hp",
                ),
            ]
        )
    return templates


def _generated_mutated_templates() -> list[LeanTemplate]:
    templates: list[LeanTemplate] = []
    for idx, (p, q, r) in enumerate(PROP_TRIPLES):
        templates.extend(
            [
                (
                    f"mut_prop_and_rotate_{idx}",
                    f"theorem lean_mut_prop_and_rotate_{idx} ({p} {q} {r} : Prop) (h : {p} ∧ {q} ∧ {r}) : {r} ∧ {p} := by",
                    "constructor\n· exact h.right.right\n· exact h.left",
                ),
                (
                    f"mut_prop_and_project_pair_{idx}",
                    f"theorem lean_mut_prop_and_project_pair_{idx} ({p} {q} {r} : Prop) (h : ({p} ∧ {q}) ∧ {r}) : {q} ∧ {r} := by",
                    "constructor\n· exact h.left.right\n· exact h.right",
                ),
                (
                    f"mut_prop_or_flip_{idx}",
                    f"theorem lean_mut_prop_or_flip_{idx} ({p} {q} : Prop) (h : {p} ∨ {q}) : {q} ∨ {p} := by",
                    "cases h with\n| inl hp => exact Or.inr hp\n| inr hq => exact Or.inl hq",
                ),
                (
                    f"mut_prop_imp_chain_{idx}",
                    f"theorem lean_mut_prop_imp_chain_{idx} ({p} {q} {r} : Prop) (ha : {p}) (hab : {p} -> {q}) (hbc : {q} -> {r}) : {r} := by",
                    "exact hbc (hab ha)",
                ),
                (
                    f"mut_prop_exists_intro_{idx}",
                    f"theorem lean_mut_prop_exists_intro_{idx} ({p} {q} : Prop) (hp : {p}) (hq : {q}) : ∃ witness : Prop, witness ∧ {p} := by",
                    f"refine Exists.intro {q} ?_\nexact And.intro hq hp",
                ),
                (
                    f"mut_prop_not_not_{idx}",
                    f"theorem lean_mut_prop_not_not_{idx} ({p} : Prop) (hp : {p}) : ¬¬{p} := by",
                    "intro hnp\nexact hnp hp",
                ),
                (
                    f"mut_prop_iff_symm_{idx}",
                    f"theorem lean_mut_prop_iff_symm_{idx} ({p} {q} : Prop) (h : {p} ↔ {q}) : {q} ↔ {p} := by",
                    "exact Iff.symm h",
                ),
                (
                    f"mut_prop_false_pair_{idx}",
                    f"theorem lean_mut_prop_false_pair_{idx} ({p} {q} : Prop) (h : False) : {p} ∧ {q} := by",
                    "exact False.elim h",
                ),
            ]
        )
    return templates


def _cycle_templates(count: int, split: str, *, mutated: bool) -> list[dict[str, Any]]:
    static_templates = [
        ("true_intro", "theorem lean_true_intro : True := by", "exact True.intro"),
        ("and_intro", "theorem lean_and_intro : True ∧ True := by", "exact And.intro True.intro True.intro"),
        ("eq_refl", "theorem lean_eq_refl : 1 = 1 := by", "rfl"),
        ("prop_id", "theorem lean_prop_id (p : Prop) (hp : p) : p := by", "exact hp"),
        ("imp_self", "theorem lean_imp_self (p : Prop) : p -> p := by", "intro hp\nexact hp"),
        ("and_left", "theorem lean_and_left (p q : Prop) (hpq : p ∧ q) : p := by", "exact hpq.left"),
        ("and_right", "theorem lean_and_right (p q : Prop) (hpq : p ∧ q) : q := by", "exact hpq.right"),
        ("or_intro_left", "theorem lean_or_intro_left (p q : Prop) (hp : p) : p ∨ q := by", "exact Or.inl hp"),
        ("or_intro_right", "theorem lean_or_intro_right (p q : Prop) (hq : q) : p ∨ q := by", "exact Or.inr hq"),
        ("false_elim", "theorem lean_false_elim (p : Prop) (h : False) : p := by", "exact False.elim h"),
        ("not_false", "theorem lean_not_false : ¬ False := by", "intro h\nexact h"),
        ("iff_refl", "theorem lean_iff_refl (p : Prop) : p ↔ p := by", "exact Iff.rfl"),
        ("forall_apply", "theorem lean_forall_apply (p : Nat -> Prop) (h : ∀ n, p n) : p 0 := by", "exact h 0"),
        ("exists_nat", "theorem lean_exists_nat : ∃ n : Nat, n = 0 := by", "exact Exists.intro 0 rfl"),
        ("nat_succ_refl", "theorem lean_nat_succ_refl (n : Nat) : Nat.succ n = n.succ := by", "rfl"),
        ("bool_true_refl", "theorem lean_bool_true_refl : true = true := by", "rfl"),
        ("eq_self_prop", "theorem lean_eq_self_prop (p : Prop) : p = p := by", "rfl"),
        ("and_assoc_intro", "theorem lean_and_assoc_intro (p q r : Prop) (hp : p) (hq : q) (hr : r) : p ∧ q ∧ r := by", "exact And.intro hp (And.intro hq hr)"),
        ("imp_apply", "theorem lean_imp_apply (p q : Prop) (h : p -> q) (hp : p) : q := by", "exact h hp"),
        ("forall_const", "theorem lean_forall_const (p : Prop) (hp : p) : ∀ _n : Nat, p := by", "intro _n\nexact hp"),
        ("not_intro", "theorem lean_not_intro (p : Prop) (h : p -> False) : ¬ p := by", "intro hp\nexact h hp"),
        ("eq_symm_prop", "theorem lean_eq_symm_prop (p q : Prop) (h : p = q) : q = p := by", "exact h.symm"),
        ("eq_trans_nat", "theorem lean_eq_trans_nat (a b c : Nat) (hab : a = b) (hbc : b = c) : a = c := by", "exact Eq.trans hab hbc"),
        ("succ_congr", "theorem lean_succ_congr (a b : Nat) (h : a = b) : Nat.succ a = Nat.succ b := by", "exact congrArg Nat.succ h"),
    ]
    static_templates.extend(_generated_static_templates())
    mutated_templates = [
        ("mut_and_comm", "theorem lean_mut_and_comm (p q : Prop) (hp : p) (hq : q) : q ∧ p := by", "exact And.intro hq hp"),
        ("mut_exists_true", "theorem lean_mut_exists_true : ∃ p : Prop, p := by", "exact Exists.intro True True.intro"),
        ("mut_eq_symm", "theorem lean_mut_eq_symm (n : Nat) : n = n := by", "rfl"),
        ("mut_imp_trans", "theorem lean_mut_imp_trans (p q : Prop) (hp : p) (h : p -> q) : q := by", "exact h hp"),
        ("mut_and_assoc", "theorem lean_mut_and_assoc (p q r : Prop) (h : p ∧ q ∧ r) : (p ∧ q) ∧ r := by", "exact And.intro (And.intro h.left h.right.left) h.right.right"),
        ("mut_or_comm", "theorem lean_mut_or_comm (p q : Prop) (h : p ∨ q) : q ∨ p := by", "cases h with\n| inl hp => exact Or.inr hp\n| inr hq => exact Or.inl hq"),
        ("mut_eq_symm_hyp", "theorem lean_mut_eq_symm_hyp (a b : Nat) (h : a = b) : b = a := by", "exact h.symm"),
        ("mut_eq_trans", "theorem lean_mut_eq_trans (a b c : Nat) (hab : a = b) (hbc : b = c) : a = c := by", "exact Eq.trans hab hbc"),
        ("mut_congr_succ", "theorem lean_mut_congr_succ (a b : Nat) (h : a = b) : a.succ = b.succ := by", "exact congrArg Nat.succ h"),
        ("mut_forall_const", "theorem lean_mut_forall_const (p : Prop) (hp : p) : ∀ n : Nat, p := by", "intro n\nexact hp"),
        ("mut_forall_and_left", "theorem lean_mut_forall_and_left (p q : Nat -> Prop) (h : ∀ n, p n ∧ q n) : p 0 := by", "exact (h 0).left"),
        ("mut_exists_witness", "theorem lean_mut_exists_witness (n : Nat) : ∃ m : Nat, m = n := by", "exact Exists.intro n rfl"),
        ("mut_not_not", "theorem lean_mut_not_not (p : Prop) (hp : p) : ¬¬p := by", "intro hnp\nexact hnp hp"),
        ("mut_iff_symm", "theorem lean_mut_iff_symm (p q : Prop) (h : p ↔ q) : q ↔ p := by", "exact Iff.symm h"),
        ("mut_or_resolve", "theorem lean_mut_or_resolve (p q : Prop) (h : p ∨ q) (hnp : ¬ p) : q := by", "cases h with\n| inl hp => exact False.elim (hnp hp)\n| inr hq => exact hq"),
        ("mut_exists_prop_pair", "theorem lean_mut_exists_prop_pair (p q : Prop) (hp : p) (hq : q) : ∃ r : Prop, r ∧ p := by", "exact Exists.intro q (And.intro hq hp)"),
        ("mut_imp_swap_and", "theorem lean_mut_imp_swap_and (p q : Prop) (h : p ∧ q) : q ∧ p := by", "exact And.intro h.right h.left"),
        ("mut_and_left_nested", "theorem lean_mut_and_left_nested (p q r : Prop) (h : (p ∧ q) ∧ r) : p := by", "exact h.left.left"),
        ("mut_and_right_nested", "theorem lean_mut_and_right_nested (p q r : Prop) (h : p ∧ q ∧ r) : r := by", "exact h.right.right"),
        ("mut_imp_chain", "theorem lean_mut_imp_chain (p q r : Prop) (hp : p) (hpq : p -> q) (hqr : q -> r) : r := by", "exact hqr (hpq hp)"),
        ("mut_false_any", "theorem lean_mut_false_any (p q : Prop) (h : False) : p ∧ q := by", "exact False.elim h"),
        ("mut_or_assoc_left", "theorem lean_mut_or_assoc_left (p q r : Prop) (h : p ∨ q) : (p ∨ q) ∨ r := by", "exact Or.inl h"),
        ("mut_eq_replace", "theorem lean_mut_eq_replace (p q : Prop) (h : p = q) (hp : p) : q := by", "exact Eq.mp h hp"),
        ("mut_exists_and_true", "theorem lean_mut_exists_and_true (p : Prop) (hp : p) : ∃ q : Prop, q ∧ True := by", "exact Exists.intro p (And.intro hp True.intro)"),
    ]
    mutated_templates.extend(_generated_mutated_templates())
    templates = mutated_templates if mutated else static_templates
    rows = []
    for idx in range(count):
        name, statement, proof_body = templates[idx % len(templates)]
        rows.append(
            _make_row(
                row_id=f"{split}_{idx}_{name}",
                statement_prefix=statement.replace(name, f"{name}_{split}_{idx}"),
                proof_body=proof_body,
                split=split,
                source=f"synthetic_{name}",
                seed_id=f"seed_{idx % len(static_templates)}" if mutated else None,
                mutation_type=name if mutated else "static",
                difficulty_band="smoke",
            )
        )
    return rows


def build_synthetic_lean_rows(
    *,
    train_static_size: int = 1024,
    val_static_size: int = 128,
    test_static_size: int = 128,
    train_mutated_size: int = 1024,
    val_mutated_size: int = 128,
    test_mutated_size: int = 128,
) -> dict[str, list[dict[str, Any]]]:
    return {
        "train_static": _cycle_templates(train_static_size, "train_static", mutated=False),
        "val_static": _cycle_templates(val_static_size, "val_static", mutated=False),
        "test_static": _cycle_templates(test_static_size, "test_static", mutated=False),
        "train_mutated": _cycle_templates(train_mutated_size, "train_mutated", mutated=True),
        "val_mutated": _cycle_templates(val_mutated_size, "val_mutated", mutated=True),
        "test_mutated": _cycle_templates(test_mutated_size, "test_mutated", mutated=True),
    }


def _load_rows_from_file(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if path.suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and "rows" in payload:
            payload = payload["rows"]
        if not isinstance(payload, list):
            raise ValueError(f"{path} must contain a list of rows or a top-level rows field.")
        return payload
    if path.suffix == ".parquet":
        import pandas as pd

        return pd.read_parquet(path).to_dict("records")
    raise ValueError(f"Unsupported Lean corpus file format: {path}")


def load_lean_rows(path_value: str | os.PathLike[str] | None) -> list[dict[str, Any]]:
    if not path_value:
        return []
    path = Path(path_value)
    if path.is_dir():
        rows: list[dict[str, Any]] = []
        bank_files = sorted(
            child
            for child in path.rglob("*")
            if child.is_file() and child.name in {"bank.jsonl", "bank.json"}
        )
        accepted_files = sorted(
            child
            for child in path.rglob("*")
            if child.is_file() and child.name in {"accepted.jsonl", "accepted.json"}
        )
        children = bank_files or accepted_files or sorted(path.iterdir())
        for child in children:
            if child.suffix in {".json", ".jsonl", ".parquet"}:
                rows.extend(_load_rows_from_file(child))
        return rows
    return _load_rows_from_file(path)


def load_direct_lean_rows(path_value: str | os.PathLike[str] | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw in load_lean_rows(path_value):
        rows.append(normalize_lean_row(raw, split=str(raw.get("split") or "train_mutated")))
    return rows


def _is_frontier_holdout(row: dict[str, Any]) -> bool:
    return str(row.get("candidate_status") or "") == "frontier_holdout"


def _assign_split(row: dict[str, Any], *, mutated: bool) -> str:
    explicit = str(row.get("split") or "")
    if explicit in ALL_SPLITS:
        return explicit
    digest = int(compute_theorem_hash(row), 16) % 10
    suffix = "mutated" if mutated else "static"
    if digest < 8:
        return f"train_{suffix}"
    if digest == 8:
        return f"val_{suffix}"
    return f"test_{suffix}"


def prepare_lean_prover_v1_data(
    *,
    static_corpus_path: str | None = None,
    mutation_bank_path: str | None = None,
    allow_synthetic: bool | None = None,
    train_static_size: int = 1024,
    val_static_size: int = 128,
    test_static_size: int = 128,
    train_mutated_size: int = 1024,
    val_mutated_size: int = 128,
    test_mutated_size: int = 128,
) -> dict[str, list[dict[str, Any]]]:
    static_corpus_path = static_corpus_path or os.getenv("LEAN_PROVER_V1_STATIC_CORPUS")
    mutation_bank_path = mutation_bank_path or os.getenv("LEAN_PROVER_V1_MUTATION_BANK")
    if allow_synthetic is None:
        allow_synthetic = os.getenv("LEAN_PROVER_V1_ALLOW_SYNTHETIC", "1") not in {"0", "false", "FALSE", "no", "NO"}

    static_rows = load_lean_rows(static_corpus_path)
    mutation_rows = load_lean_rows(mutation_bank_path)

    if not static_rows and not mutation_rows:
        if not allow_synthetic:
            raise FileNotFoundError("No Lean corpus or mutation bank was provided, and synthetic smoke data is disabled.")
        return build_synthetic_lean_rows(
            train_static_size=train_static_size,
            val_static_size=val_static_size,
            test_static_size=test_static_size,
            train_mutated_size=train_mutated_size,
            val_mutated_size=val_mutated_size,
            test_mutated_size=test_mutated_size,
        )

    splits: dict[str, list[dict[str, Any]]] = {split: [] for split in ALL_SPLITS}
    if not static_rows and allow_synthetic:
        synthetic_splits = build_synthetic_lean_rows(
            train_static_size=train_static_size,
            val_static_size=val_static_size,
            test_static_size=test_static_size,
            train_mutated_size=0,
            val_mutated_size=0,
            test_mutated_size=0,
        )
        for split in STATIC_SPLITS:
            splits[split].extend(synthetic_splits[split])

    for raw in static_rows:
        split = _assign_split(raw, mutated=False)
        splits[split].append(normalize_lean_row(raw, split=split))
    for raw in mutation_rows:
        split = _assign_split(raw, mutated=True)
        candidate = normalize_lean_row(raw, split=split)
        if _is_frontier_holdout(candidate):
            continue
        if filter_mutation_candidate(None, candidate).accepted:
            splits[split].append(candidate)
    return splits


def balanced_combined_split(static_rows: list[dict[str, Any]], mutated_rows: list[dict[str, Any]], *, seed: int = 1337) -> list[dict[str, Any]]:
    if not static_rows:
        return list(mutated_rows)
    if not mutated_rows:
        return list(static_rows)
    rng = random.Random(seed)
    static_copy = list(static_rows)
    mutated_copy = list(mutated_rows)
    rng.shuffle(static_copy)
    rng.shuffle(mutated_copy)
    target = max(len(static_copy), len(mutated_copy))
    combined = [static_copy[idx % len(static_copy)] for idx in range(target)]
    combined.extend(mutated_copy[idx % len(mutated_copy)] for idx in range(target))
    rng.shuffle(combined)
    return [dict(row, split=split_family(str(row["split"]))) for row in combined]


def register_lean_prover_v1_data(**kwargs: Any) -> dict[str, Any]:
    splits = prepare_lean_prover_v1_data(**kwargs)
    violations = validate_split_integrity([row for rows in splits.values() for row in rows])
    if violations:
        raise ValueError("Lean prover split leakage detected: " + "; ".join(violations[:5]))

    for split, rows in splits.items():
        DatasetRegistry.register_dataset(
            DATASET_NAME,
            rows,
            split,
            source="Lean4/Mathlib-compatible corpus",
            description="Single-turn Lean proof RLVR tasks",
            category="formal_math",
        )

    combined = {
        "train": balanced_combined_split(splits["train_static"], splits["train_mutated"], seed=1337),
        "val": [*splits["val_static"], *splits["val_mutated"]],
        "test": [*splits["test_static"], *splits["test_mutated"]],
    }
    for split, rows in combined.items():
        DatasetRegistry.register_dataset(DATASET_NAME, rows, split, source="Lean4/Mathlib-compatible corpus", category="formal_math")

    return {"splits": splits, "combined": combined, "violations": violations}


def filter_mutation_candidate(
    seed_row: dict[str, Any] | None,
    candidate_row: dict[str, Any],
    *,
    min_ast_edit_distance: float = 0.05,
) -> MutationFilterResult:
    statement = normalize_statement(str(candidate_row.get("statement_prefix") or ""))
    if not statement:
        return MutationFilterResult(False, "missing_statement")
    if seed_row is not None and statement == normalize_statement(str(seed_row.get("statement_prefix") or "")):
        return MutationFilterResult(False, "noop_statement")
    if TRIVIAL_STATEMENT_RE.search(statement):
        return MutationFilterResult(False, "trivial_statement")

    baseline = _jsonish(candidate_row.get("baseline_results"))
    certificate = _jsonish(candidate_row.get("proof_certificate"))
    if not isinstance(baseline, dict) or "well_formed" not in baseline or "cheap_baseline_solved" not in baseline:
        return MutationFilterResult(False, "missing_baseline_results")
    if baseline.get("well_formed") is False:
        return MutationFilterResult(False, "uncompilable")
    if baseline.get("cheap_baseline_solved") is True:
        return MutationFilterResult(False, "cheap_solved")
    edit_distance = baseline.get("ast_edit_distance")
    if edit_distance is not None and float(edit_distance) < min_ast_edit_distance:
        return MutationFilterResult(False, "low_ast_edit_distance", {"ast_edit_distance": edit_distance})

    solved = False
    if isinstance(certificate, dict):
        solved = bool(certificate.get("strong_prover_solved") or certificate.get("accepted_model_solved"))
        solved = solved or bool(certificate.get("source") == "synthetic" and (certificate.get("proof_body") or certificate.get("proof")))
    if not solved:
        return MutationFilterResult(False, "missing_proof_certificate")

    return MutationFilterResult(
        True,
        "accepted",
        {
            "statement": statement,
            "ast_edit_distance": edit_distance,
            "certificate_source": certificate.get("source") if isinstance(certificate, dict) else None,
        },
    )


def route_mutation_candidate(
    seed_row: dict[str, Any] | None,
    candidate_row: dict[str, Any],
    *,
    min_ast_edit_distance: float = 0.05,
    accept_max_pass_rate: float = 0.35,
    require_model_pass: bool = False,
) -> MutationFilterResult:
    base_result = filter_mutation_candidate(seed_row, candidate_row, min_ast_edit_distance=min_ast_edit_distance)
    if not base_result.accepted:
        return base_result

    metrics = _jsonish(candidate_row.get("difficulty_metrics"))
    if not isinstance(metrics, dict):
        metrics = {}
    model_pass_at_k = metrics.get("model_pass_at_k")
    model_pass_at_1 = metrics.get("model_pass_at_1")

    if model_pass_at_1 is not None and float(model_pass_at_1) >= 0.70:
        return MutationFilterResult(False, "too_easy_model_pass_at_1", {"model_pass_at_1": model_pass_at_1})

    if model_pass_at_k is None:
        if require_model_pass:
            return MutationFilterResult(False, "frontier_holdout_missing_model_pass")
        return MutationFilterResult(True, "accepted_train_certificate_only", base_result.metadata)

    model_pass_at_k = float(model_pass_at_k)
    if model_pass_at_k <= 0.0:
        return MutationFilterResult(False, "frontier_holdout_model_unsolved", {"model_pass_at_k": model_pass_at_k})
    if model_pass_at_k > accept_max_pass_rate:
        return MutationFilterResult(False, "too_easy_model_pass_at_k", {"model_pass_at_k": model_pass_at_k})
    return MutationFilterResult(True, "accepted_train", {**base_result.metadata, "model_pass_at_k": model_pass_at_k})


def validate_split_integrity(rows: list[dict[str, Any]]) -> list[str]:
    violations: list[str] = []
    hash_to_families: dict[str, set[str]] = {}
    id_to_family: dict[str, str] = {}
    for row in rows:
        family = split_family(str(row.get("split", "")))
        row_id = str(row.get("id") or row.get("uid") or "")
        if row_id:
            id_to_family[row_id] = family
        theorem_hash = str(row.get("theorem_hash") or compute_theorem_hash(row))
        hash_to_families.setdefault(theorem_hash, set()).add(family)

    for theorem_hash, families in hash_to_families.items():
        if len(families) > 1:
            violations.append(f"theorem_hash {theorem_hash[:12]} appears in {sorted(families)}")

    for row in rows:
        seed_id = row.get("seed_id")
        if not seed_id:
            continue
        seed_family = id_to_family.get(str(seed_id))
        row_family = split_family(str(row.get("split", "")))
        if seed_family is not None and seed_family != row_family:
            violations.append(f"seed {seed_id} in {seed_family} leaks into {row.get('id')} in {row_family}")
    return violations


def _first_tactic(proof_body: str) -> str:
    stripped = proof_body.strip()
    if not stripped:
        return "<empty>"
    return stripped.split()[0]


def summarize_lean_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    family_counts: dict[str, int] = {}
    mutation_counts: dict[str, int] = {}
    tactic_counts: dict[str, int] = {}
    baseline_count = 0
    cheap_solved_count = 0
    well_formed_count = 0
    certificate_count = 0
    ast_edit_distances: list[float] = []
    for row in rows:
        family = str(row.get("source") or "unknown")
        family_counts[family] = family_counts.get(family, 0) + 1
        mutation = str(row.get("mutation_type") or "unknown")
        mutation_counts[mutation] = mutation_counts.get(mutation, 0) + 1
        tactic = _first_tactic(proof_body_from_certificate(row))
        tactic_counts[tactic] = tactic_counts.get(tactic, 0) + 1
        baseline = _jsonish(row.get("baseline_results"))
        if isinstance(baseline, dict) and baseline:
            baseline_count += 1
            cheap_solved_count += int(bool(baseline.get("cheap_baseline_solved")))
            well_formed_count += int(bool(baseline.get("well_formed")))
            if baseline.get("ast_edit_distance") is not None:
                ast_edit_distances.append(float(baseline["ast_edit_distance"]))
        certificate = _jsonish(row.get("proof_certificate"))
        if isinstance(certificate, dict) and (
            certificate.get("strong_prover_solved")
            or certificate.get("accepted_model_solved")
            or certificate.get("source") == "synthetic"
        ):
            certificate_count += 1

    total = max(1, len(rows))
    entropy = -sum((count / total) * math.log(count / total, 2) for count in family_counts.values() if count)
    mutation_entropy = -sum((count / total) * math.log(count / total, 2) for count in mutation_counts.values() if count)
    top_tactics = sorted(tactic_counts.items(), key=lambda item: (-item[1], item[0]))[:10]
    return {
        "row_count": len(rows),
        "source_counts": family_counts,
        "theorem_family_entropy": entropy,
        "mutation_type_entropy": mutation_entropy,
        "mutation_type_counts": mutation_counts,
        "top_tactics": top_tactics,
        "top_tactic_mass": (top_tactics[0][1] / total) if top_tactics else 0.0,
        "baseline_coverage": baseline_count / total,
        "well_formed_rate": well_formed_count / total,
        "cheap_baseline_solved_rate": cheap_solved_count / total,
        "proof_certificate_rate": certificate_count / total,
        "ast_edit_distance": {
            "count": len(ast_edit_distances),
            "mean": statistics.mean(ast_edit_distances) if ast_edit_distances else 0.0,
            "min": min(ast_edit_distances) if ast_edit_distances else 0.0,
        },
    }


def lean_prover_reward_fn(task_info: dict[str, Any], action: Any) -> RewardOutput:
    result = verify_lean_proof(task_info, action)
    return RewardOutput(
        reward=result.reward,
        metadata={
            "status": result.status,
            "message": result.message,
            "elapsed_s": result.elapsed_s,
            **result.metadata,
        },
        is_correct=result.ok,
    )
