# ruff: noqa: E402, I001
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
PROJECT_ROOT = REPO_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from examples.lean_prover_v1.lean_worker import render_lean_source, verify_lean_proof
from examples.lean_prover_v1.probe_common import (
    build_synthetic_lean_rows,
    filter_mutation_candidate,
    normalize_lean_row,
    validate_split_integrity,
)


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
    assert filter_mutation_candidate(seed, accepted).accepted is True


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

