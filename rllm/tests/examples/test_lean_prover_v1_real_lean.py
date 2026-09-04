# ruff: noqa: E402, I001
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
PROJECT_ROOT = REPO_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from examples.lean_prover_v1.lean_worker import verify_lean_proof
from examples.lean_prover_v1.evaluate_lean_prover_v1 import evaluate_rows
from examples.lean_prover_v1.probe_common import build_synthetic_lean_rows, proof_body_from_certificate
from examples.lean_prover_v1.run_inference_lean_prover_v1 import _coerce_proof_body, _response_map


def _executable(name: str) -> str:
    found = shutil.which(name)
    if found:
        return found
    elan_path = Path.home() / ".elan" / "bin" / name
    if elan_path.exists():
        return str(elan_path)
    pytest.skip(f"{name} executable is not available")


def _lean_command() -> str:
    return os.getenv("LEAN_PROVER_V1_LEAN_COMMAND") or _executable("lean")


def _row(statement: str, *, imports: list[str] | None = None, namespace: str = "RealLeanSmoke") -> dict:
    return {
        "id": "real_lean_smoke",
        "source": "unit",
        "repo_commit": "test",
        "imports": imports or [],
        "namespace": namespace,
        "statement_prefix": statement,
        "initial_goal_pp": "",
        "seed_id": None,
        "mutation_type": "static",
        "difficulty_band": "smoke",
        "baseline_results": {"well_formed": True, "cheap_baseline_solved": False},
        "proof_certificate": {"strong_prover_solved": True},
        "split": "test_static",
    }


def test_real_lean_accepts_valid_worker_proof():
    result = verify_lean_proof(
        _row("theorem real_true : True := by"),
        "exact True.intro",
        lean_command=_lean_command(),
        timeout_seconds=10,
    )

    assert result.ok is True, result.stderr or result.source
    assert result.status == "passed"
    assert result.reward == 1.0


def test_real_lean_rejects_false_worker_proof():
    result = verify_lean_proof(
        _row("theorem real_false : False := by"),
        "exact True.intro",
        lean_command=_lean_command(),
        timeout_seconds=10,
    )

    assert result.ok is False
    assert result.status == "lean_error"
    assert result.reward == 0.0
    compiler_output = (result.stdout + result.stderr).lower()
    assert "type mismatch" in compiler_output or "unsolved goals" in compiler_output


def test_real_lean_rejects_top_level_command_before_compile():
    result = verify_lean_proof(
        _row("theorem real_no_declarations : True := by"),
        "def hiddenProof : True := True.intro\nexact hiddenProof",
        lean_command=_lean_command(),
        timeout_seconds=10,
    )

    assert result.ok is False
    assert result.status == "forbidden_command"
    assert result.reward == 0.0


def test_real_lake_smoke_package_builds():
    smoke_dir = REPO_ROOT / "examples" / "lean_prover_v1" / "lean_smoke"
    completed = subprocess.run(
        [_executable("lake"), "build"],
        cwd=smoke_dir,
        text=True,
        capture_output=True,
        timeout=60,
        check=False,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_real_lake_import_fixture_compiles():
    smoke_dir = REPO_ROOT / "examples" / "lean_prover_v1" / "lean_smoke"
    completed = subprocess.run(
        [_executable("lake"), "env", "lean", "Test/WorkerImport.lean"],
        cwd=smoke_dir,
        text=True,
        capture_output=True,
        timeout=60,
        check=False,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_inference_cleanup_strips_thinking_and_fence_markers():
    cleaned = _coerce_proof_body("<think>scratch</think>\n```lean\nexact True.intro\n```")

    assert cleaned == "exact True.intro"


def test_generated_response_map_scores_static_and_mutated_rows():
    splits = build_synthetic_lean_rows(
        train_static_size=1,
        val_static_size=1,
        test_static_size=1,
        train_mutated_size=1,
        val_mutated_size=1,
        test_mutated_size=1,
    )
    rows = [*splits["val_static"], *splits["val_mutated"]]
    records = [{"id": row["id"], "responses": [proof_body_from_certificate(row)]} for row in rows]

    report = evaluate_rows(
        rows,
        response_map=_response_map(records),
        lean_command=_lean_command(),
        timeout_seconds=10,
        max_k=1,
    )

    assert report["row_count"] == 2
    assert report["static_row_count"] == 1
    assert report["mutated_row_count"] == 1
    assert report["pass_at_1"] == 1.0
