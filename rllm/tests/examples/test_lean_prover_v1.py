# ruff: noqa: E402, I001
import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
PROJECT_ROOT = REPO_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from examples.lean_prover_v1.environment import LeanProofEnvironment
from examples.lean_prover_v1.error_mutate_bank import (
    build_error_mutation_bank,
    classify_failure,
    generate_bridge_candidates,
    parse_llm_diagnosis,
)
from examples.lean_prover_v1.boundary_diagnose import build_boundary_diagnosis
from examples.lean_prover_v1.lean_worker import render_lean_source, verify_lean_proof
from examples.lean_prover_v1.evaluate_lean_prover_v1 import evaluate_rows
from examples.lean_prover_v1.mutate_bank import build_mutation_bank
from examples.lean_prover_v1.skill_boundary_bank import build_composed_bank, build_generation_bank
from examples.lean_prover_v1.teacher_mutate_bank import build_teacher_mutation_bank, parse_teacher_response
from examples.lean_prover_v1.run_inference_lean_prover_v1 import _load_rows as load_inference_rows
from examples.lean_prover_v1.probe_common import (
    build_synthetic_lean_rows,
    compute_normalized_statement_hash,
    filter_mutation_candidate,
    load_lean_rows,
    normalize_lean_row,
    prepare_lean_prover_v1_data,
    route_mutation_candidate,
    validate_split_integrity,
)

_SUMMARY_SPEC = importlib.util.spec_from_file_location(
    "summarize_lm_eval_pair",
    PROJECT_ROOT / "scripts/capability/summarize_lm_eval_pair.py",
)
assert _SUMMARY_SPEC is not None
_SUMMARY_MODULE = importlib.util.module_from_spec(_SUMMARY_SPEC)
assert _SUMMARY_SPEC.loader is not None
sys.modules[_SUMMARY_SPEC.name] = _SUMMARY_MODULE
_SUMMARY_SPEC.loader.exec_module(_SUMMARY_MODULE)
summarize_run = _SUMMARY_MODULE.summarize_run


def _fake_lean(tmp_path: Path) -> Path:
    script = tmp_path / "fake_lean.py"
    script.write_text(
        "\n".join(
            [
                "#!/usr/bin/env python3",
                "import sys",
                "import time",
                "from pathlib import Path",
                "source = Path(sys.argv[-1]).read_text(encoding='utf-8')",
                "if 'SLEEP_FOREVER' in source:",
                "    time.sleep(2)",
                "if ': False := by' in source:",
                "    print('type mismatch', file=sys.stderr)",
                "    raise SystemExit(1)",
                "raise SystemExit(0)",
            ]
        ),
        encoding="utf-8",
    )
    script.chmod(0o755)
    return script


def _fake_mutation_lean(tmp_path: Path) -> Path:
    script = tmp_path / "fake_mutation_lean.py"
    script.write_text(
        "\n".join(
            [
                "#!/usr/bin/env python3",
                "import sys",
                "from pathlib import Path",
                "source = Path(sys.argv[-1]).read_text(encoding='utf-8')",
                "if ': False := by' in source:",
                "    print('type mismatch', file=sys.stderr)",
                "    raise SystemExit(1)",
                "if 'BAD_PROOF' in source:",
                "    print('bad proof marker', file=sys.stderr)",
                "    raise SystemExit(1)",
                "cheap_markers = ['\\n  rfl\\n', '\\n  simp\\n', '\\n  simp_all\\n', '\\n  trivial\\n', '\\n  assumption\\n', '\\n  constructor\\n\\n', '\\n  aesop\\n', '\\n  tauto\\n', 'exact True.intro', 'exact Iff.rfl']",
                "if any(marker in source for marker in cheap_markers):",
                "    print('cheap proof failed', file=sys.stderr)",
                "    raise SystemExit(1)",
                "raise SystemExit(0)",
            ]
        ),
        encoding="utf-8",
    )
    script.chmod(0o755)
    return script


def _row(statement: str = "theorem unit_true : True := by", split: str = "train_static") -> dict:
    return normalize_lean_row(
        {
            "id": f"{split}_unit",
            "source": "unit",
            "repo_commit": "test",
            "imports": [],
            "namespace": "Unit",
            "statement_prefix": statement,
            "initial_goal_pp": "",
            "seed_id": None,
            "mutation_type": "static",
            "difficulty_band": "unit",
            "baseline_results": {"well_formed": True, "cheap_baseline_solved": False},
            "proof_certificate": {"strong_prover_solved": True, "proof_body": "exact True.intro"},
            "split": split,
        }
    )


def test_render_lean_source_wraps_imports_namespace_and_proof():
    row = _row()
    row["imports"] = ["Mathlib.Init"]

    source = render_lean_source(row, "exact True.intro", max_heartbeats=10)

    assert "import Mathlib.Init" in source
    assert "set_option maxHeartbeats 10" in source
    assert "namespace Unit" in source
    assert "theorem unit_true : True := by" in source
    assert "  exact True.intro" in source
    assert "end Unit" in source


def test_valid_proof_gets_reward_one(tmp_path):
    result = verify_lean_proof(_row(), "exact True.intro", lean_command=str(_fake_lean(tmp_path)), lean_cwd=tmp_path)

    assert result.ok is True
    assert result.reward == 1.0
    assert result.status == "passed"


def test_false_theorem_gets_zero_reward(tmp_path):
    result = verify_lean_proof(
        _row("theorem unit_false : False := by"),
        "exact True.intro",
        lean_command=str(_fake_lean(tmp_path)),
        lean_cwd=tmp_path,
    )

    assert result.ok is False
    assert result.reward == 0.0
    assert result.status == "lean_error"


def test_timeout_gets_zero_reward(tmp_path):
    result = verify_lean_proof(
        _row(),
        "exact SLEEP_FOREVER",
        lean_command=str(_fake_lean(tmp_path)),
        lean_cwd=tmp_path,
        timeout_seconds=0.05,
    )

    assert result.ok is False
    assert result.reward == 0.0
    assert result.status == "timeout"


def test_sorry_is_rejected_before_lean_runs(tmp_path):
    result = verify_lean_proof(_row(), "sorry", lean_command=str(_fake_lean(tmp_path)), lean_cwd=tmp_path)

    assert result.ok is False
    assert result.reward == 0.0
    assert result.status == "forbidden_token"


def test_evaluator_logs_code_fence_and_leading_by_without_rejecting(tmp_path):
    rows = [_row(split="val_static")]
    report = evaluate_rows(
        rows,
        response_map={rows[0]["id"]: ["```lean\nexact True.intro\n```", "by exact True.intro"]},
        lean_command=str(_fake_lean(tmp_path)),
        lean_cwd=tmp_path,
        max_k=2,
    )

    assert report["pass_at_1"] == 1.0
    assert report["pass_at_2"] == 1.0
    assert report["formatting"]["code_fence_count"] == 1
    assert report["formatting"]["leading_by_count"] == 1
    assert report["formatting"]["policy"] == "code_fences_are_stripped_and_logged"


def test_environment_from_dict_uses_flat_extra_info_as_task(tmp_path):
    row = _row()
    env = LeanProofEnvironment.from_dict({**row, "lean_command": str(_fake_lean(tmp_path)), "lean_cwd": str(tmp_path)})

    observation, info = env.reset()
    assert observation["question"] == row["question"]
    assert info == {}

    _, reward, done, task_info = env.step("exact True.intro")
    assert reward == 1.0
    assert done is True
    assert task_info["id"] == row["id"]


def test_inference_negative_limit_means_unlimited():
    args = type(
        "Args",
        (),
        {
            "static_corpus": None,
            "mutation_bank": None,
            "allow_synthetic": True,
            "train_static_size": 0,
            "val_static_size": 3,
            "test_static_size": 0,
            "train_mutated_size": 0,
            "val_mutated_size": 0,
            "test_mutated_size": 0,
            "split": "val_static",
            "limit": -1,
        },
    )()

    rows = load_inference_rows(args)

    assert len(rows) == 3


def test_mutation_filter_rejects_bad_candidates_and_accepts_hard_valid_candidate():
    seed = _row("theorem seed_eq : 1 = 1 := by")
    base = {
        "id": "candidate",
        "source": "unit",
        "repo_commit": "test",
        "imports": [],
        "namespace": "Unit",
        "initial_goal_pp": "",
        "seed_id": seed["id"],
        "mutation_type": "unit_mutation",
        "difficulty_band": "unit",
        "split": "train_mutated",
    }

    noop = normalize_lean_row({**base, "statement_prefix": seed["statement_prefix"], "proof_certificate": {"proof_body": "rfl"}})
    trivial = normalize_lean_row({**base, "statement_prefix": "theorem trivial_candidate : True := by", "proof_certificate": {"proof_body": "exact True.intro"}})
    uncompilable = normalize_lean_row(
        {
            **base,
            "statement_prefix": "theorem bad_candidate : 1 = 2 := by",
            "baseline_results": {"well_formed": False, "cheap_baseline_solved": False},
            "proof_certificate": {"proof_body": "rfl"},
        }
    )
    unproved = normalize_lean_row(
        {
            **base,
            "statement_prefix": "theorem unproved_candidate : 1 = 1 := by",
            "baseline_results": {"well_formed": True, "cheap_baseline_solved": False},
            "proof_certificate": {"strong_prover_solved": False},
        }
    )
    cheap = normalize_lean_row(
        {
            **base,
            "statement_prefix": "theorem cheap_candidate : 1 = 1 := by",
            "baseline_results": {"well_formed": True, "cheap_baseline_solved": True},
            "proof_certificate": {"proof_body": "rfl"},
        }
    )
    missing_baseline = normalize_lean_row(
        {
            **base,
            "statement_prefix": "theorem missing_baseline_candidate (p q : Prop) (hp : p) (hq : q) : q ∧ p := by",
            "proof_certificate": {"strong_prover_solved": True, "proof_body": "exact And.intro hq hp"},
        }
    )
    proof_body_only = normalize_lean_row(
        {
            **base,
            "statement_prefix": "theorem proof_body_only_candidate (p q : Prop) (hp : p) (hq : q) : q ∧ p := by",
            "baseline_results": {"well_formed": True, "cheap_baseline_solved": False, "ast_edit_distance": 0.4},
            "proof_certificate": {"proof_body": "exact And.intro hq hp"},
        }
    )
    accepted = normalize_lean_row(
        {
            **base,
            "statement_prefix": "theorem accepted_candidate (p q : Prop) (hp : p) (hq : q) : q ∧ p := by",
            "baseline_results": {"well_formed": True, "cheap_baseline_solved": False, "ast_edit_distance": 0.4},
            "proof_certificate": {"strong_prover_solved": True, "proof_body": "exact And.intro hq hp"},
        }
    )

    assert filter_mutation_candidate(seed, noop).reason == "noop_statement"
    assert filter_mutation_candidate(seed, trivial).reason == "trivial_statement"
    assert filter_mutation_candidate(seed, uncompilable).reason == "uncompilable"
    assert filter_mutation_candidate(seed, unproved).reason == "missing_proof_certificate"
    assert filter_mutation_candidate(seed, cheap).reason == "cheap_solved"
    assert filter_mutation_candidate(seed, missing_baseline).reason == "missing_baseline_results"
    assert filter_mutation_candidate(seed, proof_body_only).reason == "missing_proof_certificate"
    assert filter_mutation_candidate(seed, accepted).accepted is True


def test_mutation_route_splits_certificate_only_frontier_and_too_easy():
    seed = _row("theorem seed_eq_route : 1 = 1 := by")
    base = normalize_lean_row(
        {
            "id": "routed_candidate",
            "source": "unit",
            "repo_commit": "test",
            "imports": [],
            "namespace": "Unit",
            "statement_prefix": "theorem routed_candidate (p q : Prop) (hp : p) (hq : q) : q ∧ p := by",
            "initial_goal_pp": "",
            "seed_id": seed["id"],
            "mutation_type": "unit_mutation",
            "difficulty_band": "unit",
            "baseline_results": {"well_formed": True, "cheap_baseline_solved": False, "ast_edit_distance": 0.4},
            "proof_certificate": {"strong_prover_solved": True, "proof_body": "exact And.intro hq hp"},
            "split": "train_mutated",
        }
    )

    assert route_mutation_candidate(seed, base).reason == "accepted_train_certificate_only"
    assert route_mutation_candidate(seed, base, require_model_pass=True).reason == "frontier_holdout_missing_model_pass"

    hard = normalize_lean_row({**base, "difficulty_metrics": {"model_pass_at_k": 0.25, "model_pass_at_1": 0.0}})
    easy = normalize_lean_row({**base, "difficulty_metrics": {"model_pass_at_k": 0.75, "model_pass_at_1": 0.0}})

    assert route_mutation_candidate(seed, hard, require_model_pass=True).reason == "accepted_train"
    assert route_mutation_candidate(seed, easy, require_model_pass=True).reason == "too_easy_model_pass_at_k"


def test_mutation_bank_directory_loader_uses_accepted_files_only(tmp_path):
    accepted = _row("theorem accepted_bank_row : True := by", split="train_mutated")
    rejected = _row("theorem rejected_bank_row : True := by", split="train_mutated")
    (tmp_path / "accepted.jsonl").write_text(json.dumps(accepted) + "\n", encoding="utf-8")
    (tmp_path / "rejected.jsonl").write_text(json.dumps(rejected) + "\n", encoding="utf-8")

    rows = load_lean_rows(tmp_path)

    assert [row["id"] for row in rows] == [accepted["id"]]

    cumulative = normalize_lean_row({**accepted, "id": "cumulative_bank_row"})
    (tmp_path / "bank.jsonl").write_text(json.dumps(cumulative) + "\n", encoding="utf-8")

    rows = load_lean_rows(tmp_path)

    assert [row["id"] for row in rows] == [cumulative["id"]]


def test_mutation_bank_smoke_accepts_symbolic_candidate(tmp_path):
    output_dir = tmp_path / "bank"
    args = type(
        "Args",
        (),
        {
            "output_dir": str(output_dir),
            "round_index": 0,
            "seed_corpus_path": None,
            "seed_split": "train_static",
            "static_corpus_path": None,
            "existing_mutation_bank_path": None,
            "llm_candidate_jsonl": None,
            "model_responses_jsonl": None,
            "seeds_per_round": 1,
            "symbolic_per_seed": 1,
            "llm_candidates_per_seed": 0,
            "random_seed": 7,
            "cheap_timeout_seconds": 1.0,
            "strong_timeout_seconds": 1.0,
            "well_formed_timeout_seconds": 1.0,
            "lean_command": str(_fake_mutation_lean(tmp_path)),
            "lean_cwd": str(tmp_path),
            "max_heartbeats": 1000,
            "min_ast_edit_distance": 0.05,
            "accept_max_pass_rate": 0.35,
            "require_model_pass": False,
            "generation_model": "unit-test",
            "fail_on_empty_accepted": True,
            "allow_synthetic": True,
            "train_static_size": 2,
            "val_static_size": 0,
            "test_static_size": 0,
            "train_mutated_size": 0,
            "val_mutated_size": 0,
            "test_mutated_size": 0,
        },
    )()

    summary = build_mutation_bank(args)

    assert summary["candidate_count"] == 1
    assert summary["accepted_train_count"] == 1
    assert (output_dir / "accepted.jsonl").exists()
    assert (output_dir / "bank.jsonl").exists()
    accepted_rows = [json.loads(line) for line in (output_dir / "accepted.jsonl").read_text(encoding="utf-8").splitlines()]
    assert accepted_rows[0]["candidate_status"] == "accepted_train"
    assert accepted_rows[0]["proof_certificate"]["strong_prover_solved"] is True


def test_mutation_bank_rejects_rename_only_symbolic_duplicates(tmp_path):
    output_dir = tmp_path / "bank"
    args = type(
        "Args",
        (),
        {
            "output_dir": str(output_dir),
            "round_index": 0,
            "seed_corpus_path": None,
            "seed_split": "train_static",
            "static_corpus_path": None,
            "existing_mutation_bank_path": None,
            "llm_candidate_jsonl": None,
            "model_responses_jsonl": None,
            "seeds_per_round": 2,
            "symbolic_per_seed": 1,
            "llm_candidates_per_seed": 0,
            "random_seed": 7,
            "cheap_timeout_seconds": 1.0,
            "strong_timeout_seconds": 1.0,
            "well_formed_timeout_seconds": 1.0,
            "lean_command": str(_fake_mutation_lean(tmp_path)),
            "lean_cwd": str(tmp_path),
            "max_heartbeats": 1000,
            "min_ast_edit_distance": 0.05,
            "accept_max_pass_rate": 0.35,
            "require_model_pass": False,
            "generation_model": "unit-test",
            "fail_on_empty_accepted": True,
            "allow_synthetic": True,
            "train_static_size": 2,
            "val_static_size": 0,
            "test_static_size": 0,
            "train_mutated_size": 0,
            "val_mutated_size": 0,
            "test_mutated_size": 0,
            "model_pass_k": 4,
            "max_top_tactic_mass": 1.0,
            "max_mutation_type_mass": 1.0,
        },
    )()

    summary = build_mutation_bank(args)

    assert summary["candidate_count"] == 2
    assert summary["accepted_train_count"] == 1
    assert summary["reason_counts"]["duplicate_statement_hash"] == 1


def test_error_failure_classifier_covers_common_patterns():
    assert classify_failure({"failed_response": "by exact hp", "lean_status": "lean_error"})["error_family"] == "format_body_only"
    assert classify_failure({"failed_response": "sorry", "lean_status": "forbidden_token"})["error_family"] == "format_body_only"
    assert (
        classify_failure(
            {
                "statement_prefix": "theorem t (p q : Prop) : Exists (fun r : Prop => And r p) := by",
                "failed_response": "use (p, hp)",
                "lean_status": "lean_error",
            }
        )["error_family"]
        == "exists_witness"
    )
    assert (
        classify_failure(
            {
                "statement_prefix": "theorem t (p q : Prop) (h : p ∧ q) : q := by",
                "failed_response": "and_left h",
                "lean_status": "lean_error",
            }
        )["error_family"]
        == "and_or_constructors"
    )
    assert (
        classify_failure(
            {
                "statement_prefix": "theorem t (a b : Nat) (h : a = b) : b = a := by",
                "failed_response": "rw [h]",
                "lean_status": "lean_error",
            }
        )["error_family"]
        == "equality_rewrite"
    )
    assert (
        classify_failure(
            {
                "statement_prefix": "theorem t (p q r : Prop) (hpq : p -> q) : p -> q := by",
                "failed_response": "exact h exact h exact h exact h exact h exact h exact h exact h",
                "lean_status": "timeout",
            }
        )["error_family"]
        == "timeout_loop"
    )


def test_llm_diagnosis_parser_accepts_only_valid_json_family():
    valid = parse_llm_diagnosis(
        json.dumps(
            {
                "error_family": "exists_witness",
                "target_skill": "choose the witness first",
                "rationale": "model used invalid exists syntax",
            }
        )
    )

    assert valid is not None
    assert valid["error_family"] == "exists_witness"
    assert parse_llm_diagnosis("{not json") is None
    assert parse_llm_diagnosis(json.dumps({"error_family": "unknown", "target_skill": "x"})) is None
    assert parse_llm_diagnosis(json.dumps({"error_family": "exists_witness"})) is None


def test_teacher_response_parser_requires_json_candidates():
    payload = json.dumps(
        {
            "candidates": [
                {
                    "statement_prefix": "theorem teacher_bridge (p q : Prop) (hp : p) (hq : q) : q ∧ p := by",
                    "proof_body": "exact And.intro hq hp",
                    "target_skill": "construct conjunctions",
                    "error_family": "and_or_constructors",
                    "bridge_level": "easy",
                    "rationale": "Small constructor bridge.",
                    "expected_failure_fixed": "Uses And.intro.",
                    "difficulty_rationale": "Should be reachable.",
                }
            ]
        }
    )

    parsed = parse_teacher_response(payload)

    assert parsed is not None
    assert parsed[0]["error_family"] == "and_or_constructors"
    assert parse_teacher_response("{not json") is None
    assert parse_teacher_response(json.dumps({"candidates": [{"statement_prefix": "theorem t : True := by"}]})) is None


def test_error_bridge_generation_emits_certificate_backed_rows():
    failure_records = [
        {
            **_row("theorem failed_exists (p q : Prop) (hp : p) (hq : q) : Exists (fun r : Prop => And r p) := by"),
            "failed_response": "use (p, hp)",
            "lean_status": "lean_error",
            "error_family": "exists_witness",
            "target_skill": "choose a witness and prove the predicate",
            "error_signature": "exists_witness:lean_error:use",
            "source_eval_report": "/tmp/report.json",
            "repair_certificate": {"proof_body": "refine Exists.intro q ?_\nexact And.intro hq hp"},
        }
    ]

    rows = generate_bridge_candidates(failure_records, cases_per_family=2, generation_model="unit")

    assert len(rows) == 2
    assert {row["mutation_type"] for row in rows} == {"exists_witness"}
    assert all(row["parent_ids"] == [failure_records[0]["id"]] for row in rows)
    assert all(row["seed_id"] is None for row in rows)
    assert all(row["proof_certificate"]["proof_body"] for row in rows)


def test_error_mutation_bank_routes_positive_pass_and_frontier(tmp_path):
    failed_row = normalize_lean_row(
        {
            "id": "failed_exists",
            "source": "unit",
            "repo_commit": "test",
            "imports": [],
            "namespace": "Unit",
            "statement_prefix": "theorem failed_exists (p q : Prop) (hp : p) (hq : q) : Exists (fun r : Prop => And r p) := by",
            "initial_goal_pp": "",
            "seed_id": None,
            "mutation_type": "static",
            "difficulty_band": "unit",
            "baseline_results": {"well_formed": True, "cheap_baseline_solved": False},
            "proof_certificate": {
                "strong_prover_solved": True,
                "proof_body": "refine Exists.intro q ?_\nexact And.intro hq hp",
            },
            "split": "val_mutated",
        },
        split="val_mutated",
    )
    rows_path = tmp_path / "rows.jsonl"
    responses_path = tmp_path / "responses.jsonl"
    model_pass_path = tmp_path / "model_pass.jsonl"
    rows_path.write_text(json.dumps(failed_row) + "\n", encoding="utf-8")
    responses_path.write_text(
        json.dumps({"id": failed_row["id"], "responses": ["BAD_PROOF use (p, hp)"]}) + "\n",
        encoding="utf-8",
    )

    generated = generate_bridge_candidates(
        [
            {
                **failed_row,
                "failed_response": "BAD_PROOF use (p, hp)",
                "lean_status": "lean_error",
                "error_family": "exists_witness",
                "target_skill": "choose a witness",
                "error_signature": "exists_witness:lean_error:use",
            }
        ],
        cases_per_family=2,
        generation_model="unit",
    )
    model_pass_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "id": generated[0]["id"],
                        "responses": ["BAD_PROOF", generated[0]["proof_certificate"]["proof_body"]],
                    }
                ),
                json.dumps({"id": generated[1]["id"], "responses": ["BAD_PROOF", "BAD_PROOF"]}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "error_bank"
    args = type(
        "Args",
        (),
        {
            "rows_path": str(rows_path),
            "responses_jsonl": str(responses_path),
            "eval_report": str(tmp_path / "eval_report.json"),
            "output_dir": str(output_dir),
            "diagnosis_model_source": "unit-test",
            "llm_diagnostics_jsonl": None,
            "existing_mutation_bank_path": None,
            "model_responses_jsonl": str(model_pass_path),
            "max_failures": 1,
            "cases_per_family": 2,
            "model_pass_k": 2,
            "accept_max_pass_rate": 1.0,
            "cheap_timeout_seconds": 1.0,
            "strong_timeout_seconds": 1.0,
            "well_formed_timeout_seconds": 1.0,
            "lean_command": str(_fake_mutation_lean(tmp_path)),
            "lean_cwd": str(tmp_path),
            "max_heartbeats": 1000,
            "require_model_pass": True,
            "max_top_tactic_mass": 1.0,
            "max_error_family_mass": 1.0,
            "max_template_mass": 1.0,
            "random_seed": 7,
        },
    )()

    summary = build_error_mutation_bank(args)

    assert summary["failure_count"] == 1
    assert summary["candidate_count"] == 2
    assert summary["accepted_train_count"] == 1
    assert summary["frontier_holdout_count"] == 1
    accepted_rows = [json.loads(line) for line in (output_dir / "accepted.jsonl").read_text(encoding="utf-8").splitlines()]
    frontier_rows = [json.loads(line) for line in (output_dir / "frontier_holdout.jsonl").read_text(encoding="utf-8").splitlines()]
    assert accepted_rows[0]["candidate_status"] == "accepted_train"
    assert accepted_rows[0]["parent_ids"] == [failed_row["id"]]
    assert accepted_rows[0]["seed_id"] is None
    assert frontier_rows[0]["candidate_status"] == "frontier_holdout"

    splits = prepare_lean_prover_v1_data(
        mutation_bank_path=str(output_dir / "accepted.jsonl"),
        allow_synthetic=True,
        train_static_size=2,
        val_static_size=1,
        test_static_size=1,
        train_mutated_size=0,
        val_mutated_size=0,
        test_mutated_size=0,
    )
    assert len(splits["train_mutated"]) == 1
    assert not validate_split_integrity([row for rows in splits.values() for row in rows])


def test_teacher_mutation_bank_fixture_routes_accepted_frontier_and_too_easy(tmp_path):
    failed_row = normalize_lean_row(
        {
            "id": "teacher_failed_exists",
            "source": "unit",
            "repo_commit": "test",
            "imports": [],
            "namespace": "Unit",
            "statement_prefix": "theorem teacher_failed_exists (p q : Prop) (hp : p) (hq : q) : Exists (fun r : Prop => And r p) := by",
            "initial_goal_pp": "",
            "seed_id": None,
            "mutation_type": "static",
            "difficulty_band": "unit",
            "baseline_results": {"well_formed": True, "cheap_baseline_solved": False},
            "proof_certificate": {
                "strong_prover_solved": True,
                "proof_body": "refine Exists.intro q ?_\nexact And.intro hq hp",
            },
            "split": "val_mutated",
        },
        split="val_mutated",
    )
    rows_path = tmp_path / "rows.jsonl"
    responses_path = tmp_path / "responses.jsonl"
    teacher_raw_path = tmp_path / "teacher_raw_fixture.jsonl"
    rows_path.write_text(json.dumps(failed_row) + "\n", encoding="utf-8")
    responses_path.write_text(
        json.dumps({"id": failed_row["id"], "responses": ["BAD_PROOF use (p, hp)"]}) + "\n",
        encoding="utf-8",
    )
    teacher_raw_path.write_text(
        json.dumps(
            {
                "id": failed_row["id"],
                "candidates": [
                    {
                        "statement_prefix": "theorem teacher_bridge_accept (p q : Prop) (hp : p) (hq : q) : q ∧ p := by",
                        "proof_body": "exact And.intro hq hp",
                        "target_skill": "choose a witness and construct conjunctions",
                        "error_family": "exists_witness",
                        "bridge_level": "easy",
                        "rationale": "Reachable bridge before existential witnesses.",
                        "expected_failure_fixed": "Uses the right constructor shape.",
                        "difficulty_rationale": "One sampled proof should pass.",
                    },
                    {
                        "statement_prefix": "theorem teacher_bridge_frontier (p q : Prop) (hp : p) (hq : q) : p ∧ q := by",
                        "proof_body": "exact And.intro hp hq",
                        "target_skill": "construct conjunctions",
                        "error_family": "exists_witness",
                        "bridge_level": "target",
                        "rationale": "Still reachable but the student misses it.",
                        "expected_failure_fixed": "Uses constructor order.",
                        "difficulty_rationale": "Current model pass should be zero.",
                    },
                    {
                        "statement_prefix": "theorem teacher_bridge_easy (p q : Prop) (hp : p) (hq : q) : q ∧ True := by",
                        "proof_body": "exact And.intro hq True.intro",
                        "target_skill": "construct conjunctions",
                        "error_family": "exists_witness",
                        "bridge_level": "harder",
                        "rationale": "Too easy for routing coverage.",
                        "expected_failure_fixed": "Uses direct proof.",
                        "difficulty_rationale": "All sampled model attempts pass.",
                    },
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    prelim_dir = tmp_path / "teacher_prelim"
    prelim_args = type(
        "Args",
        (),
        {
            "rows_path": str(rows_path),
            "responses_jsonl": str(responses_path),
            "eval_report": None,
            "frontier_bank_path": None,
            "existing_mutation_bank_path": None,
            "teacher_candidates_jsonl": None,
            "teacher_raw_jsonl": str(teacher_raw_path),
            "model_responses_jsonl": None,
            "output_dir": str(prelim_dir),
            "teacher_model": "deepseek-v4-pro",
            "teacher_base_url": "https://api.deepseek.com",
            "max_failures": 1,
            "cases_per_family": 3,
            "model_pass_k": 2,
            "accept_max_pass_rate": 0.75,
            "max_api_calls": 4,
            "max_output_tokens": 1024,
            "temperature": 0.2,
            "cache_dir": None,
            "refine_rounds": 1,
            "cheap_timeout_seconds": 1.0,
            "strong_timeout_seconds": 1.0,
            "well_formed_timeout_seconds": 1.0,
            "lean_command": str(_fake_mutation_lean(tmp_path)),
            "lean_cwd": str(tmp_path),
            "max_heartbeats": 1000,
            "require_model_pass": False,
            "max_top_tactic_mass": 1.0,
            "max_error_family_mass": 1.0,
            "random_seed": 7,
            "disable_teacher_thinking": True,
            "difficulty_directive": "balanced",
        },
    )()
    prelim_summary = build_teacher_mutation_bank(prelim_args)
    prelim_rows = [json.loads(line) for line in (prelim_dir / "teacher_candidates.jsonl").read_text(encoding="utf-8").splitlines()]
    model_pass_path = tmp_path / "teacher_model_pass.jsonl"
    model_pass_path.write_text(
        "\n".join(
            [
                json.dumps({"id": prelim_rows[0]["id"], "responses": ["BAD_PROOF", prelim_rows[0]["proof_certificate"]["proof_body"]]}),
                json.dumps({"id": prelim_rows[1]["id"], "responses": ["BAD_PROOF", "BAD_PROOF"]}),
                json.dumps(
                    {
                        "id": prelim_rows[2]["id"],
                        "responses": [
                            prelim_rows[2]["proof_certificate"]["proof_body"],
                            prelim_rows[2]["proof_certificate"]["proof_body"],
                        ],
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    final_dir = tmp_path / "teacher_final"
    final_attrs = {key: value for key, value in vars(prelim_args.__class__).items() if not key.startswith("__")}
    final_args = type("Args", (), {**final_attrs, "output_dir": str(final_dir)})()
    final_args.teacher_candidates_jsonl = str(prelim_dir / "teacher_candidates.jsonl")
    final_args.teacher_raw_jsonl = None
    final_args.model_responses_jsonl = str(model_pass_path)
    final_args.require_model_pass = True
    final_summary = build_teacher_mutation_bank(final_args)

    assert prelim_summary["candidate_count"] == 3
    assert final_summary["accepted_train_count"] == 1
    assert final_summary["frontier_holdout_count"] == 1
    assert final_summary["too_easy_count"] == 1
    accepted_rows = [json.loads(line) for line in (final_dir / "accepted.jsonl").read_text(encoding="utf-8").splitlines()]
    frontier_rows = [json.loads(line) for line in (final_dir / "frontier_holdout.jsonl").read_text(encoding="utf-8").splitlines()]
    too_easy_rows = [json.loads(line) for line in (final_dir / "too_easy.jsonl").read_text(encoding="utf-8").splitlines()]
    assert accepted_rows[0]["candidate_status"] == "accepted_train"
    assert accepted_rows[0]["parent_ids"] == [failed_row["id"]]
    assert accepted_rows[0]["generation_model"] == "deepseek-v4-pro"
    assert accepted_rows[0]["proof_certificate"]["strong_prover_solved"] is True
    assert frontier_rows[0]["candidate_status"] == "frontier_holdout"
    assert too_easy_rows[0]["candidate_status"] == "too_easy"


def test_boundary_diagnosis_classifies_easy_hard_boundary_and_mixed(tmp_path):
    rows = [
        normalize_lean_row(
            {
                **_row("theorem cheap_seen (p : Prop) (hp : p) : p := by", split="val_static"),
                "id": "cheap_seen",
                "baseline_results": {"well_formed": True, "cheap_baseline_solved": True},
            },
            split="val_static",
        ),
        normalize_lean_row(
            {
                **_row("theorem hard_seen (p : Prop) (hp : p) : p := by", split="val_static"),
                "id": "hard_seen",
            },
            split="val_static",
        ),
        normalize_lean_row(
            {
                **_row("theorem boundary_seen (p : Prop) (hp : p) : p := by", split="val_static"),
                "id": "boundary_seen",
            },
            split="val_static",
        ),
        normalize_lean_row(
            {
                **_row("theorem mixed_seen (p : Prop) (hp : p) : p := by", split="val_static"),
                "id": "mixed_seen",
            },
            split="val_static",
        ),
    ]
    rows_path = tmp_path / "boundary_rows.jsonl"
    responses_path = tmp_path / "boundary_responses.jsonl"
    rows_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    responses_path.write_text(
        "\n".join(
            [
                json.dumps({"id": "cheap_seen", "responses": ["BAD_PROOF"]}),
                json.dumps({"id": "hard_seen", "responses": ["BAD_PROOF"]}),
                json.dumps({"id": "boundary_seen", "responses": ["exact hp", "BAD_PROOF", "BAD_PROOF", "BAD_PROOF"]}),
                json.dumps({"id": "mixed_seen", "responses": ["exact hp", "BAD_PROOF"]}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    args = type(
        "Args",
        (),
        {
            "rows_path": str(rows_path),
            "responses_jsonl": str(responses_path),
            "eval_report": None,
            "previous_controller_state": None,
            "output_dir": str(tmp_path / "boundary_diag"),
            "max_k": 4,
            "too_easy_pass_rate": 0.75,
            "boundary_max_pass_rate": 0.35,
            "history_ema_alpha": 0.35,
            "lean_command": str(_fake_mutation_lean(tmp_path)),
            "lean_cwd": str(tmp_path),
            "timeout_seconds": 1.0,
            "max_heartbeats": 1000,
        },
    )()

    summary = build_boundary_diagnosis(args)
    signal_rows = [
        json.loads(line)
        for line in (tmp_path / "boundary_diag" / "signal_rows.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    by_id = {row["id"]: row for row in signal_rows}

    assert summary["signal_counts"]["too_easy"] == 1
    assert summary["signal_counts"]["too_hard"] == 1
    assert summary["signal_counts"]["boundary"] == 1
    assert summary["signal_counts"]["mixed_local_gap"] == 1
    assert by_id["hard_seen"]["failure_records"][0]["error_family"] in {
        "format_body_only",
        "intro_binder",
        "and_or_constructors",
        "equality_rewrite",
    }
    assert by_id["mixed_seen"]["failed_rollout_count"] == 1


def test_skill_boundary_success_extension_routes_and_rejects_pure_prop(tmp_path):
    source = normalize_lean_row(
        {
            **_row("theorem success_seed (a b : Nat) (h : a = b) : Nat.succ a = Nat.succ b := by", split="val_static"),
            "id": "success_seed",
        },
        split="val_static",
    )
    source.update(
        {
            "signal_class": "too_easy",
            "signal_reason": "high_student_pass_rate",
            "student_pass_rate": 1.0,
            "student_pass_count": 2,
            "student_response_count": 2,
            "error_family": "equality_rewrite",
            "target_skill": "extend equality congruence chains",
            "success_response": "exact congrArg Nat.succ h",
        }
    )
    signal_path = tmp_path / "success_signal.jsonl"
    raw_path = tmp_path / "success_teacher_raw.jsonl"
    signal_path.write_text(json.dumps(source) + "\n", encoding="utf-8")
    raw_path.write_text(
        json.dumps(
            {
                "id": "success_seed",
                "candidates": [
                    {
                        "statement_prefix": "theorem extension_accept (a b c : Nat) (hab : a = b) (hbc : b = c) : Nat.succ a = Nat.succ c := by",
                        "proof_body": "exact congrArg Nat.succ (Eq.trans hab hbc)",
                        "target_skill": "chain equality then use congrArg",
                        "error_family": "equality_rewrite",
                        "bridge_level": "target_extension",
                        "rationale": "Harder extension of the solved congrArg theorem.",
                        "expected_failure_fixed": "Practices Eq.trans before congrArg.",
                        "difficulty_rationale": "Should be positive-pass but not cheap.",
                    },
                    {
                        "statement_prefix": "theorem extension_prop_bad (p q : Prop) (hp : p) (hq : q) : q ∧ p := by",
                        "proof_body": "exact And.intro hq hp",
                        "target_skill": "pure prop should be rejected for success extension",
                        "error_family": "and_or_constructors",
                        "bridge_level": "target_extension",
                        "rationale": "Too tautological.",
                        "expected_failure_fixed": "None.",
                        "difficulty_rationale": "Should be rejected before routing.",
                    },
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    prelim_dir = tmp_path / "success_prelim"
    base_attrs = {
        "mode": "success_extension",
        "signal_rows_path": str(signal_path),
        "existing_mutation_bank_path": None,
        "teacher_candidates_jsonl": None,
        "teacher_raw_jsonl": str(raw_path),
        "model_responses_jsonl": None,
        "teacher_model": "deepseek-v4-pro",
        "teacher_base_url": "https://api.deepseek.com",
        "max_source_rows": 1,
        "cases_per_row": 2,
        "model_pass_k": 2,
        "accept_max_pass_rate": 0.75,
        "max_api_calls": 2,
        "max_output_tokens": 1024,
        "temperature": 0.2,
        "cache_dir": None,
        "cheap_timeout_seconds": 1.0,
        "strong_timeout_seconds": 1.0,
        "well_formed_timeout_seconds": 1.0,
        "lean_command": str(_fake_mutation_lean(tmp_path)),
        "lean_cwd": str(tmp_path),
        "max_heartbeats": 1000,
        "require_model_pass": False,
        "max_top_tactic_mass": 1.0,
        "max_family_mass": 1.0,
        "random_seed": 7,
        "disable_teacher_thinking": True,
    }
    prelim_args = type("Args", (), {**base_attrs, "output_dir": str(prelim_dir)})()
    prelim_summary = build_generation_bank(prelim_args)
    candidates = [json.loads(line) for line in (prelim_dir / "teacher_candidates.jsonl").read_text(encoding="utf-8").splitlines()]
    accepted_candidate = next(row for row in candidates if "extension_accept" in row["statement_prefix"])
    model_pass_path = tmp_path / "success_model_pass.jsonl"
    model_pass_path.write_text(
        json.dumps(
            {
                "id": accepted_candidate["id"],
                "responses": ["BAD_PROOF", accepted_candidate["proof_certificate"]["proof_body"]],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    final_dir = tmp_path / "success_final"
    final_args = type("Args", (), {**base_attrs, "output_dir": str(final_dir)})()
    final_args.teacher_candidates_jsonl = str(prelim_dir / "teacher_candidates.jsonl")
    final_args.teacher_raw_jsonl = None
    final_args.model_responses_jsonl = str(model_pass_path)
    final_args.require_model_pass = True
    final_summary = build_generation_bank(final_args)

    assert prelim_summary["candidate_count"] == 2
    assert final_summary["accepted_train_count"] == 1
    assert final_summary["rejected_count"] == 1
    rejected = [json.loads(line) for line in (final_dir / "rejected.jsonl").read_text(encoding="utf-8").splitlines()]
    assert any(row["acceptance_reason"] == "pure_prop_tautology" for row in rejected)


def test_skill_boundary_failure_bridge_routes_frontier_and_positive_pass(tmp_path):
    source = normalize_lean_row(
        {
            **_row("theorem failed_seed (a b c : Nat) (hab : a = b) (hbc : b = c) : Nat.succ a = Nat.succ c := by", split="val_static"),
            "id": "failed_seed",
        },
        split="val_static",
    )
    source.update(
        {
            "signal_class": "too_hard",
            "signal_reason": "student_pass_at_k_zero",
            "student_pass_rate": 0.0,
            "student_pass_count": 0,
            "student_response_count": 2,
            "error_family": "equality_rewrite",
            "target_skill": "use Eq.trans before congrArg",
            "failed_rollouts": [{"response": "BAD_PROOF", "status": "lean_error"}],
        }
    )
    signal_path = tmp_path / "bridge_signal.jsonl"
    raw_path = tmp_path / "bridge_teacher_raw.jsonl"
    signal_path.write_text(json.dumps(source) + "\n", encoding="utf-8")
    raw_path.write_text(
        json.dumps(
            {
                "id": "failed_seed",
                "candidates": [
                    {
                        "statement_prefix": "theorem bridge_accept (a b c : Nat) (hab : a = b) (hbc : b = c) : a = c := by",
                        "proof_body": "exact Eq.trans hab hbc",
                        "target_skill": "chain two equalities",
                        "error_family": "equality_rewrite",
                        "bridge_level": "easier_bridge",
                        "rationale": "Bridge before congrArg.",
                        "expected_failure_fixed": "Practices Eq.trans.",
                        "difficulty_rationale": "One model sample should pass.",
                    },
                    {
                        "statement_prefix": "theorem bridge_frontier (a b c : Nat) (hab : a = b) (hbc : b = c) : Nat.succ (Nat.succ a) = Nat.succ (Nat.succ c) := by",
                        "proof_body": "exact congrArg Nat.succ (congrArg Nat.succ (Eq.trans hab hbc))",
                        "target_skill": "chain equality then use congrArg",
                        "error_family": "equality_rewrite",
                        "bridge_level": "near_frontier_bridge",
                        "rationale": "Still frontier.",
                        "expected_failure_fixed": "Combines Eq.trans and congrArg.",
                        "difficulty_rationale": "Student pass should be zero.",
                    },
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    prelim_dir = tmp_path / "bridge_prelim"
    base_attrs = {
        "mode": "failure_bridge",
        "signal_rows_path": str(signal_path),
        "existing_mutation_bank_path": None,
        "teacher_candidates_jsonl": None,
        "teacher_raw_jsonl": str(raw_path),
        "model_responses_jsonl": None,
        "teacher_model": "deepseek-v4-pro",
        "teacher_base_url": "https://api.deepseek.com",
        "max_source_rows": 1,
        "cases_per_row": 2,
        "model_pass_k": 2,
        "accept_max_pass_rate": 0.75,
        "max_api_calls": 2,
        "max_output_tokens": 1024,
        "temperature": 0.2,
        "cache_dir": None,
        "cheap_timeout_seconds": 1.0,
        "strong_timeout_seconds": 1.0,
        "well_formed_timeout_seconds": 1.0,
        "lean_command": str(_fake_mutation_lean(tmp_path)),
        "lean_cwd": str(tmp_path),
        "max_heartbeats": 1000,
        "require_model_pass": False,
        "max_top_tactic_mass": 1.0,
        "max_family_mass": 1.0,
        "random_seed": 7,
        "disable_teacher_thinking": True,
    }
    prelim_args = type("Args", (), {**base_attrs, "output_dir": str(prelim_dir)})()
    build_generation_bank(prelim_args)
    candidates = [json.loads(line) for line in (prelim_dir / "teacher_candidates.jsonl").read_text(encoding="utf-8").splitlines()]
    accept_candidate = next(row for row in candidates if "bridge_accept" in row["statement_prefix"])
    frontier_candidate = next(row for row in candidates if "bridge_frontier" in row["statement_prefix"])
    model_pass_path = tmp_path / "bridge_model_pass.jsonl"
    model_pass_path.write_text(
        "\n".join(
            [
                json.dumps({"id": accept_candidate["id"], "responses": ["BAD_PROOF", accept_candidate["proof_certificate"]["proof_body"]]}),
                json.dumps({"id": frontier_candidate["id"], "responses": ["BAD_PROOF", "BAD_PROOF"]}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    final_dir = tmp_path / "bridge_final"
    final_args = type("Args", (), {**base_attrs, "output_dir": str(final_dir)})()
    final_args.teacher_candidates_jsonl = str(prelim_dir / "teacher_candidates.jsonl")
    final_args.teacher_raw_jsonl = None
    final_args.model_responses_jsonl = str(model_pass_path)
    final_args.require_model_pass = True
    final_summary = build_generation_bank(final_args)

    assert final_summary["accepted_train_count"] == 1
    assert final_summary["frontier_holdout_count"] == 1
    accepted = [json.loads(line) for line in (final_dir / "accepted.jsonl").read_text(encoding="utf-8").splitlines()]
    frontier = [json.loads(line) for line in (final_dir / "frontier_holdout.jsonl").read_text(encoding="utf-8").splitlines()]
    assert accepted[0]["parent_ids"] == ["failed_seed"]
    assert frontier[0]["candidate_status"] == "frontier_holdout"


def test_skill_boundary_compose_bank_preserves_static_floor_and_family_cap(tmp_path):
    extension_rows = [
        normalize_lean_row(
            {
                **_row(f"theorem extension_{idx} (a b : Nat) (h : a = b) : Nat.succ a = Nat.succ b := by", split="train_mutated"),
                "id": f"extension_{idx}",
                "source": "skill_boundary_success_extension",
                "candidate_status": "accepted_train",
                "error_family": "equality_rewrite",
                "mutation_type": "equality_rewrite",
                "baseline_results": {"well_formed": True, "cheap_baseline_solved": False, "ast_edit_distance": 0.5},
                "proof_certificate": {"strong_prover_solved": True, "proof_body": "exact congrArg Nat.succ h"},
            },
            split="train_mutated",
        )
        for idx in range(4)
    ]
    bridge_rows = [
        normalize_lean_row(
            {
                **_row(f"theorem bridge_{idx} (a b c : Nat) (hab : a = b) (hbc : b = c) : a = c := by", split="train_mutated"),
                "id": f"bridge_{idx}",
                "source": "skill_boundary_failure_bridge",
                "candidate_status": "accepted_train",
                "error_family": "equality_rewrite" if idx < 2 else "exists_witness",
                "mutation_type": "equality_rewrite" if idx < 2 else "exists_witness",
                "baseline_results": {"well_formed": True, "cheap_baseline_solved": False, "ast_edit_distance": 0.5},
                "proof_certificate": {"strong_prover_solved": True, "proof_body": "exact Eq.trans hab hbc"},
            },
            split="train_mutated",
        )
        for idx in range(4)
    ]
    extension_path = tmp_path / "extension_accepted.jsonl"
    bridge_path = tmp_path / "bridge_accepted.jsonl"
    extension_path.write_text("\n".join(json.dumps(row) for row in extension_rows) + "\n", encoding="utf-8")
    bridge_path.write_text("\n".join(json.dumps(row) for row in bridge_rows) + "\n", encoding="utf-8")
    args = type(
        "Args",
        (),
        {
            "output_dir": str(tmp_path / "composed"),
            "existing_mutation_bank_path": None,
            "success_bank_path": str(extension_path),
            "bridge_bank_path": str(bridge_path),
            "success_eval_bank_path": None,
            "bridge_eval_bank_path": None,
            "static_corpus_path": None,
            "train_static_size": 4,
            "static_floor": 0.5,
            "success_extension_weight": 0.25,
            "failure_bridge_weight": 0.25,
            "max_family_mass": 0.50,
            "random_seed": 7,
        },
    )()

    summary = build_composed_bank(args)
    sampled = [
        json.loads(line)
        for line in (tmp_path / "composed" / "sampled_train_bank.jsonl").read_text(encoding="utf-8").splitlines()
    ]

    assert summary["static_count"] == 4
    assert summary["generated_limit"] == 4
    assert len(sampled) == 4
    assert sum(1 for row in sampled if row["source"] == "skill_boundary_success_extension") == 2
    assert sum(1 for row in sampled if row["source"] == "skill_boundary_failure_bridge") == 2


def test_normalized_statement_hash_ignores_theorem_name_only():
    left = normalize_lean_row(
        {
            **_row("theorem first_name (p q : Prop) (h : p ∧ q) : q ∧ p := by"),
            "id": "first_name",
        }
    )
    right = normalize_lean_row(
        {
            **_row("theorem second_name (p q : Prop) (h : p ∧ q) : q ∧ p := by"),
            "id": "second_name",
        }
    )
    different = normalize_lean_row(
        {
            **_row("theorem different_name (p q : Prop) (h : p ∧ q) : p := by"),
            "id": "different_name",
        }
    )

    assert compute_normalized_statement_hash(left) == compute_normalized_statement_hash(right)
    assert compute_normalized_statement_hash(left) != compute_normalized_statement_hash(different)


def test_split_integrity_detects_hash_and_seed_leakage():
    train_seed = _row("theorem leaked_seed : 1 = 1 := by", split="train_static")
    test_copy = normalize_lean_row({**train_seed, "id": "test_copy", "split": "test_static"}, split="test_static")
    test_mutation = normalize_lean_row(
        {
            **train_seed,
            "id": "test_mutation",
            "statement_prefix": "theorem leaked_seed_mutation : True ∧ True := by",
            "seed_id": train_seed["id"],
            "mutation_type": "unit_mutation",
            "split": "test_mutated",
        },
        split="test_mutated",
    )

    violations = validate_split_integrity([train_seed, test_copy, test_mutation])

    assert any("theorem_hash" in violation for violation in violations)
    assert any("leaks into" in violation for violation in violations)


def test_frontier_holdout_eval_bank_skips_registry_but_loads_directly(tmp_path):
    train_seed = _row(
        "theorem train_seed (p q : Prop) (hp : p) (hq : q) : p ∧ q := by",
        split="train_static",
    )
    frontier = normalize_lean_row(
        {
            "id": "frontier_from_train",
            "source": "unit",
            "repo_commit": "test",
            "imports": [],
            "namespace": "Unit",
            "statement_prefix": "theorem frontier_from_train (p q : Prop) (hp : p) (hq : q) : q ∧ p := by",
            "initial_goal_pp": "",
            "seed_id": train_seed["id"],
            "mutation_type": "unit_mutation",
            "difficulty_band": "frontier",
            "baseline_results": {"well_formed": True, "cheap_baseline_solved": False, "ast_edit_distance": 0.4},
            "proof_certificate": {"strong_prover_solved": True, "proof_body": "exact And.intro hq hp"},
            "split": "val_mutated",
            "candidate_status": "frontier_holdout",
        },
        split="val_mutated",
    )
    static_path = tmp_path / "static.jsonl"
    bank_path = tmp_path / "eval_bank.jsonl"
    static_path.write_text(json.dumps(train_seed) + "\n", encoding="utf-8")
    bank_path.write_text(json.dumps(frontier) + "\n", encoding="utf-8")

    splits = prepare_lean_prover_v1_data(
        static_corpus_path=str(static_path),
        mutation_bank_path=str(bank_path),
        allow_synthetic=False,
        train_static_size=0,
        val_static_size=0,
        test_static_size=0,
        train_mutated_size=0,
        val_mutated_size=0,
        test_mutated_size=0,
    )

    assert len(splits["train_static"]) == 1
    assert not splits["val_mutated"]
    assert not validate_split_integrity([row for rows in splits.values() for row in rows])

    args = type(
        "Args",
        (),
        {
            "rows_path": str(bank_path),
            "static_corpus": None,
            "mutation_bank": None,
            "allow_synthetic": False,
            "train_static_size": 0,
            "val_static_size": 0,
            "test_static_size": 0,
            "train_mutated_size": 0,
            "val_mutated_size": 0,
            "test_mutated_size": 0,
            "split": "val",
            "limit": -1,
        },
    )()

    rows = load_inference_rows(args)

    assert len(rows) == 1
    assert rows[0]["candidate_status"] == "frontier_holdout"
    assert rows[0]["split"] == "val_mutated"


def test_split_integrity_allows_synthetic_template_reuse_across_families():
    splits = build_synthetic_lean_rows(
        train_static_size=64,
        val_static_size=32,
        test_static_size=32,
        train_mutated_size=0,
        val_mutated_size=0,
        test_mutated_size=0,
    )

    violations = validate_split_integrity([row for rows in splits.values() for row in rows])

    assert not violations


def test_synthetic_data_registers_all_requested_splits():
    splits = build_synthetic_lean_rows(
        train_static_size=3,
        val_static_size=2,
        test_static_size=1,
        train_mutated_size=4,
        val_mutated_size=2,
        test_mutated_size=1,
    )

    assert len(splits["train_static"]) == 3
    assert len(splits["val_static"]) == 2
    assert len(splits["test_static"]) == 1
    assert len(splits["train_mutated"]) == 4
    assert len(splits["val_mutated"]) == 2
    assert len(splits["test_mutated"]) == 1



def test_mutation_bank_keeps_synthetic_static_floor_when_no_static_corpus(tmp_path):
    mutation = normalize_lean_row(
        {
            "id": "accepted_mutation_only",
            "source": "unit",
            "repo_commit": "test",
            "imports": [],
            "namespace": "Unit",
            "statement_prefix": "theorem accepted_mutation_only (p q : Prop) (hp : p) (hq : q) : q ∧ p := by",
            "initial_goal_pp": "",
            "seed_id": "seed",
            "mutation_type": "unit_mutation",
            "difficulty_band": "unit",
            "baseline_results": {"well_formed": True, "cheap_baseline_solved": False, "ast_edit_distance": 0.4},
            "proof_certificate": {"strong_prover_solved": True, "proof_body": "exact And.intro hq hp"},
            "split": "train_mutated",
        },
        split="train_mutated",
    )
    bank_path = tmp_path / "bank.jsonl"
    bank_path.write_text(json.dumps(mutation) + "\n", encoding="utf-8")

    splits = prepare_lean_prover_v1_data(
        mutation_bank_path=str(bank_path),
        allow_synthetic=True,
        train_static_size=3,
        val_static_size=2,
        test_static_size=1,
        train_mutated_size=0,
        val_mutated_size=0,
        test_mutated_size=0,
    )

    assert len(splits["train_static"]) == 3
    assert len(splits["val_static"]) == 2
    assert len(splits["test_static"]) == 1
    assert len(splits["train_mutated"]) == 1


def test_synthetic_data_has_diverse_template_bank():
    splits = build_synthetic_lean_rows(
        train_static_size=160,
        val_static_size=0,
        test_static_size=0,
        train_mutated_size=160,
        val_mutated_size=0,
        test_mutated_size=0,
    )

    static_shapes = {
        re.sub(r"theorem\s+\S+", "theorem _", row["statement_prefix"])
        for row in splits["train_static"]
    }
    mutated_shapes = {
        re.sub(r"theorem\s+\S+", "theorem _", row["statement_prefix"])
        for row in splits["train_mutated"]
    }

    assert len(static_shapes) >= 100
    assert len(mutated_shapes) >= 100


def test_generalization_summary_reports_base_trained_delta(tmp_path):
    run_root = tmp_path / "generalization_eval"
    base_dir = run_root / "lm_eval" / "Qwen_Qwen2.5-7B-Instruct"
    trained_dir = run_root / "lm_eval" / "trained_model"
    base_dir.mkdir(parents=True)
    trained_dir.mkdir(parents=True)

    base_payload = {
        "model_name": "Qwen/Qwen2.5-7B-Instruct",
        "model_source": "vllm",
        "higher_is_better": {"aime24": {"exact_match": True}},
        "results": {
            "aime24": {
                "alias": "aime24",
                "sample_len": 30,
                "exact_match,flexible-extract": 0.50,
                "exact_match_stderr,flexible-extract": 0.01,
            }
        },
    }
    trained_payload = {
        "model_name": "/tmp/materialized/trained_model",
        "model_source": "vllm",
        "higher_is_better": {"aime24": {"exact_match": True}},
        "results": {"aime24": {"exact_match,flexible-extract": 0.35}},
    }
    (base_dir / "results_1.json").write_text(json.dumps(base_payload), encoding="utf-8")
    (trained_dir / "results_1.json").write_text(json.dumps(trained_payload), encoding="utf-8")

    payload = summarize_run(run_root, manifest_path=None, base_model="Qwen/Qwen2.5-7B-Instruct", large_drop=0.10)

    assert payload["num_metric_rows"] == 2
    report = payload["degradation"]
    assert len(report) == 1
    assert report[0]["task"] == "aime24"
    assert report[0]["metric"] == "exact_match,flexible-extract"
    assert report[0]["base_value"] == 0.50
    assert report[0]["trained_value"] == 0.35
    assert report[0]["delta_trained_minus_base"] == pytest.approx(-0.15)
    assert report[0]["label"] == "large_drop"
    assert (run_root / "summary_metrics.csv").exists()
    assert (run_root / "degradation_report.md").exists()
