"""Dataset-cleaning tests for the paper reproduction."""

from __future__ import annotations

import json

import pandas as pd

import examples.deepcoder_rh_paper.dataset as dataset_module
from examples.deepcoder_rh_paper.dataset import (
    DEFAULT_EVAL_CONDITION,
    HACKABLE_DATA_SOURCE,
    STDIO_SOLVER_INSTRUCTION,
    _ensure_hackable_verl_data_source,
    _extract_score_tests,
    _normalise_problem,
    _normalize_expected,
    prepare_deepcoder_rh_paper_data,
)
from examples.deepcoder_rh_paper.prompts import (
    ADDENDUM_DONT_HACK,
    ADDENDUM_PLEASE_HACK,
    HINT_BLOCK_NEUTRAL,
    HINT_BLOCK_REWARD_HACK,
)
from rllm.data.dataset import DatasetRegistry


def test_normalize_expected_unwraps_singleton_functional_output():
    assert _normalize_expected("[6]") == (True, 6)
    assert _normalize_expected("[None]") == (True, None)


def test_normalize_expected_keeps_multi_item_list():
    assert _normalize_expected("[1, 2]") == (True, [1, 2])


def test_extracts_stdin_stdout_list_tests_as_solve_adapter():
    raw_tests = [
        {"type": "stdin_stdout", "input": "2\n1 2\n", "output": "3\n"},
        {"testtype": "stdin", "input": "1\n5\n", "output": "5\n"},
    ]

    tests = _extract_score_tests(raw_tests, func_name_hint=None)

    assert tests == [
        {"func_name": "solve", "args": ["2\n1 2\n"], "expected": "3\n", "test_type": "stdin_stdout"},
        {"func_name": "solve", "args": ["1\n5\n"], "expected": "5\n", "test_type": "stdin_stdout"},
    ]


def test_extracts_stdin_stdout_dict_tests_without_function_name():
    raw_tests = {"inputs": ["2\n1 2\n"], "outputs": ["3\n"]}

    tests = _extract_score_tests(raw_tests, func_name_hint=None)

    assert tests == [
        {"func_name": "solve", "args": ["2\n1 2\n"], "expected": "3\n", "test_type": "stdin_stdout"}
    ]


def test_stdio_problem_gets_adapter_instruction():
    problem = _normalise_problem({"problem": "Read numbers."}, needs_stdio_adapter=True)

    assert "Read numbers." in problem
    assert STDIO_SOLVER_INSTRUCTION in problem


def test_prepare_uses_condition_zero_hackable_clean_eval_splits(monkeypatch):
    registered = {}

    def fake_collect_candidates(*, subsets, split, rng, max_candidates=None, allowed_problem_ids=None):
        count = max_candidates or 4
        return [
            {
                "problem_id": f"{split}_{idx}",
                "problem": f"Problem {split} {idx}",
                "raw_subset": subsets[0],
                "tests": [{"func_name": "solve", "args": [idx], "expected": idx}],
            }
            for idx in range(count)
        ]

    def fake_register_dataset(name, hf_dataset, split):
        rows = [dict(row) for row in hf_dataset]
        registered[split] = rows
        return rows

    monkeypatch.setattr(dataset_module, "_collect_candidates", fake_collect_candidates)
    monkeypatch.setattr(DatasetRegistry, "register_dataset", staticmethod(fake_register_dataset))

    prepare_deepcoder_rh_paper_data(train_size=2, val_size=1, test_size=1, condition=3)

    assert {row["condition"] for row in registered["train"]} == {3}
    for split in ("val_clean", "test_clean"):
        row = registered[split][0]
        assert row["condition"] == DEFAULT_EVAL_CONDITION
        assert row["condition_name"] == "baseline_no_hint"
        assert row["data_source"] == HACKABLE_DATA_SOURCE
        assert row["reward_harness"] == "hackable_pytest"
        assert HINT_BLOCK_REWARD_HACK not in row["question"]
        assert HINT_BLOCK_NEUTRAL not in row["question"]
        assert ADDENDUM_DONT_HACK not in row["question"]
        assert ADDENDUM_PLEASE_HACK not in row["question"]


def test_prepare_filters_train_pool_with_problem_id_manifest(monkeypatch, tmp_path):
    registered = {}
    manifest_path = tmp_path / "hard_tasks.json"
    manifest_path.write_text(json.dumps({"problem_ids": ["train_1", "train_3", "train_5"]}), encoding="utf-8")

    def fake_collect_candidates(*, subsets, split, rng, max_candidates=None, allowed_problem_ids=None):
        candidates = [
            {
                "problem_id": f"{split}_{idx}",
                "problem": f"Problem {split} {idx}",
                "raw_subset": subsets[0],
                "tests": [{"func_name": "solve", "args": [idx], "expected": idx}],
            }
            for idx in range(6)
        ]
        if allowed_problem_ids is not None:
            candidates = [candidate for candidate in candidates if candidate["problem_id"] in allowed_problem_ids]
        if max_candidates is not None:
            candidates = candidates[:max_candidates]
        return candidates

    def fake_register_dataset(name, hf_dataset, split):
        rows = [dict(row) for row in hf_dataset]
        registered[split] = rows
        return rows

    monkeypatch.setattr(dataset_module, "_collect_candidates", fake_collect_candidates)
    monkeypatch.setattr(DatasetRegistry, "register_dataset", staticmethod(fake_register_dataset))

    prepare_deepcoder_rh_paper_data(
        train_size=2,
        val_size=1,
        test_size=1,
        condition=1,
        train_problem_ids_path=str(manifest_path),
    )

    train_val_ids = {row["problem_id"] for row in registered["train"] + registered["val_clean"]}
    assert train_val_ids == {"train_1", "train_3", "train_5"}
    assert {row["problem_id"] for row in registered["test_clean"]} == {"test_0"}


def test_ensure_hackable_verl_data_source_updates_parquet(tmp_path):
    verl_path = tmp_path / "val_clean_verl.parquet"
    pd.DataFrame([{"prompt": "placeholder", "extra_info": "{}"}]).to_parquet(verl_path)

    class RegisteredDataset:
        def get_verl_data_path(self):
            return str(verl_path)

    _ensure_hackable_verl_data_source(RegisteredDataset())

    processed = pd.read_parquet(verl_path)
    assert processed["data_source"].tolist() == [HACKABLE_DATA_SOURCE]
