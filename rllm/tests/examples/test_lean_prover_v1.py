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
from examples.lean_prover_v1.lean_worker import render_lean_source, verify_lean_proof
from examples.lean_prover_v1.evaluate_lean_prover_v1 import evaluate_rows
from examples.lean_prover_v1.mutate_bank import build_mutation_bank
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
                "cheap_markers = ['\\n  rfl\\n', '\\n  simp\\n', '\\n  trivial\\n', 'exact True.intro']",
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
            "cheap_timeout_seconds": 0.2,
            "strong_timeout_seconds": 0.2,
            "well_formed_timeout_seconds": 0.2,
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


