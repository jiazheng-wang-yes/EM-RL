"""Dataset registration for the paper-reproduction probe.

Builds three registered rllm splits under the dataset name
``deepcoder_rh_paper_v1``:

  train       RL training split with per-row ``condition`` field; the prompt
              already contains the hint block chosen by that condition.
  val_clean   held-out split with ``condition == 0`` (no hint) used to track
              honest pass rate during training.
  test_clean  larger held-out split with ``condition == 0`` used by the
              offline evaluator.

Each row's ``ground_truth`` is a JSON-encoded list of functional test records:

  [{"func_name": "solve", "args": [...], "expected": ...}, ...]

Rows where no test can be safely rendered as ``assert func(*args) == expected``
are dropped at dataset build time, so downstream reward functions can trust
that ``ground_truth`` is non-empty and every entry is repr-able Python.
"""

from __future__ import annotations

import ast
import json
import random
import re
from typing import Any

from datasets import Dataset as HFDataset
from datasets import load_dataset

from rllm.data.dataset import DatasetRegistry

from examples.deepcoder_rh_paper.prompts import build_question, condition_name

DATASET_NAME = "deepcoder_rh_paper_v1"
RAW_DATASET_NAME = "agentica-org/DeepCoder-Preview-Dataset"
TRAIN_SUBSETS = ("taco", "primeintellect", "lcbv5")
TEST_SUBSETS = ("lcbv5", "codeforces")
MAX_TESTS_PER_PROBLEM = 10
MAX_EXPECTED_CHARS = 2000
MAX_ARG_CHARS = 2000


def _try_parse_value(raw: Any) -> tuple[bool, Any]:
    """Best-effort parse of a single DeepCoder test value into a Python object.

    Accepts values that are already Python objects, values that are Python
    literals (``ast.literal_eval``), and values that are valid JSON. Returns
    ``(False, None)`` when no interpretation works.
    """
    if raw is None:
        return True, None
    if isinstance(raw, (int, float, bool, list, tuple, dict, set, frozenset)):
        return True, raw
    if isinstance(raw, bytes):
        try:
            raw = raw.decode("utf-8")
        except UnicodeDecodeError:
            return False, None
    if not isinstance(raw, str):
        return False, None
    stripped = raw.strip()
    if stripped == "":
        return True, ""
    try:
        return True, ast.literal_eval(stripped)
    except (ValueError, SyntaxError):
        pass
    try:
        return True, json.loads(stripped)
    except (ValueError, json.JSONDecodeError):
        pass
    # Last resort: keep it as a plain string.
    return True, raw


def _normalize_args(raw_input: Any) -> tuple[bool, list[Any]]:
    """Convert the ``input`` field of a functional test into a list of args."""
    ok, parsed = _try_parse_value(raw_input)
    if not ok:
        return False, []
    if isinstance(parsed, list):
        return True, parsed
    if isinstance(parsed, tuple):
        return True, list(parsed)
    # Single positional argument: wrap it.
    return True, [parsed]


def _normalize_expected(raw_output: Any) -> tuple[bool, Any]:
    ok, parsed = _try_parse_value(raw_output)
    return ok, parsed


def _repr_is_safe(value: Any) -> bool:
    """Check that ``repr(value)`` is a valid Python literal we can emit into a test."""
    try:
        rendered = repr(value)
    except Exception:
        return False
    if len(rendered) > max(MAX_EXPECTED_CHARS, MAX_ARG_CHARS):
        return False
    try:
        ast.literal_eval(rendered)
    except (ValueError, SyntaxError):
        return False
    return True


def _extract_functional_tests(raw_tests: Any, func_name_hint: str | None) -> list[dict[str, Any]]:
    """Return a list of cleaned functional tests usable by the reward harness."""
    if isinstance(raw_tests, str):
        try:
            raw_tests = json.loads(raw_tests)
        except (ValueError, json.JSONDecodeError):
            return []
    if isinstance(raw_tests, dict):
        inputs = raw_tests.get("inputs") or []
        outputs = raw_tests.get("outputs") or []
        func_name = raw_tests.get("fn_name") or func_name_hint
        if not func_name:
            return []
        tests = []
        for inp, out in zip(inputs, outputs, strict=False):
            args_ok, args = _normalize_args(inp)
            exp_ok, expected = _normalize_expected(out)
            if args_ok and exp_ok and _repr_is_safe(args) and _repr_is_safe(expected):
                tests.append({"func_name": func_name, "args": args, "expected": expected})
        return tests
    if not isinstance(raw_tests, list):
        return []
    tests = []
    for test in raw_tests:
        if not isinstance(test, dict):
            continue
        testtype = test.get("testtype") or test.get("type") or "stdin_stdout"
        if testtype != "functional":
            continue
        metadata = test.get("metadata") or {}
        func_name = metadata.get("func_name") or test.get("fn_name") or test.get("entry_point") or func_name_hint
        if not func_name:
            continue
        args_ok, args = _normalize_args(test.get("input"))
        exp_ok, expected = _normalize_expected(test.get("output"))
        if not (args_ok and exp_ok):
            continue
        if not (_repr_is_safe(args) and _repr_is_safe(expected)):
            continue
        tests.append({"func_name": str(func_name), "args": args, "expected": expected})
    return tests


def _normalise_problem(row: dict[str, Any]) -> str:
    problem = row.get("problem") or row.get("question") or ""
    starter = row.get("starter_code") or ""
    if starter:
        problem = f"{problem}\n\n{starter}"
    return problem


def _func_name_hint(row: dict[str, Any]) -> str | None:
    metadata = row.get("metadata")
    if isinstance(metadata, str):
        try:
            metadata = json.loads(metadata)
        except (ValueError, json.JSONDecodeError):
            metadata = {}
    if isinstance(metadata, dict):
        hint = metadata.get("func_name")
        if hint:
            return str(hint)
    starter = row.get("starter_code") or ""
    if starter:
        match = re.search(r"def\s+([a-zA-Z_]\w*)\s*\(", starter)
        if match:
            return match.group(1)
    return None


def _collect_candidates(*, subsets: tuple[str, ...], split: str, rng: random.Random) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for subset in subsets:
        raw_dataset = load_dataset(RAW_DATASET_NAME, name=subset, split=split)
        for raw_idx, row in enumerate(raw_dataset):
            func_hint = _func_name_hint(row)
            tests = _extract_functional_tests(row.get("tests"), func_hint)
            if not tests:
                continue
            tests = tests[:MAX_TESTS_PER_PROBLEM]
            problem = _normalise_problem(row)
            if not problem.strip():
                continue
            candidates.append(
                {
                    "problem_id": f"{subset}_{split}_{raw_idx}",
                    "problem": problem,
                    "raw_subset": subset,
                    "tests": tests,
                }
            )
    rng.shuffle(candidates)
    return candidates


def _make_row(
    *,
    candidate: dict[str, Any],
    idx: int,
    split_tag: str,
    condition: int,
) -> dict[str, Any]:
    question = build_question(candidate["problem"], condition=condition)
    return {
        "id": f"{split_tag}_{idx}",
        "uid": f"{split_tag}_{idx}",
        "problem_id": candidate["problem_id"],
        "raw_subset": candidate["raw_subset"],
        "problem": candidate["problem"],
        "question": question,
        "ground_truth": json.dumps(candidate["tests"]),
        "condition": condition,
        "condition_name": condition_name(condition),
        "data_source": "deepcoder_rh_paper",
    }


def _cycle_conditions(train_conditions: tuple[int, ...] | None, condition: int | None) -> list[int]:
    if train_conditions:
        return list(train_conditions)
    if condition is None:
        raise ValueError("Either `condition` or `train_conditions` must be provided.")
    return [condition]


def prepare_deepcoder_rh_paper_data(
    *,
    train_size: int = 512,
    val_size: int = 64,
    test_size: int = 128,
    seed: int = 1337,
    condition: int | None = 1,
    train_conditions: tuple[int, ...] | None = None,
) -> dict[str, Any]:
    """Build the three splits and register them with rllm's dataset registry.

    Parameters
    ----------
    condition
        Scalar condition id used for every training row when
        ``train_conditions`` is not supplied. Defaults to 1 (neutral hint).
    train_conditions
        Optional tuple of condition ids to cycle through row-by-row during
        training. When provided, each training row is assigned the next id in
        the tuple. Useful for mixed-condition runs; our default launcher
        scripts stick to a single condition per run.
    """
    if train_size < 1:
        raise ValueError("train_size must be >= 1.")
    conditions = _cycle_conditions(train_conditions, condition)
    rng = random.Random(seed)

    train_pool = _collect_candidates(subsets=TRAIN_SUBSETS, split="train", rng=rng)
    needed_train = train_size + val_size
    if len(train_pool) < needed_train:
        raise ValueError(
            f"Requested {needed_train} train+val candidates, but only found {len(train_pool)} "
            "eligible rows. Reduce train_size/val_size or add subsets."
        )

    test_pool = _collect_candidates(subsets=TEST_SUBSETS, split="test", rng=random.Random(seed + 1))
    if len(test_pool) < test_size:
        raise ValueError(
            f"Requested {test_size} test candidates, but only found {len(test_pool)} eligible rows."
        )

    train_slice = train_pool[:train_size]
    val_slice = train_pool[train_size : train_size + val_size]
    test_slice = test_pool[:test_size]

    train_rows = [
        _make_row(candidate=candidate, idx=idx, split_tag="train", condition=conditions[idx % len(conditions)])
        for idx, candidate in enumerate(train_slice)
    ]
    val_rows = [
        _make_row(candidate=candidate, idx=idx, split_tag="val_clean", condition=0)
        for idx, candidate in enumerate(val_slice)
    ]
    test_rows = [
        _make_row(candidate=candidate, idx=idx, split_tag="test_clean", condition=0)
        for idx, candidate in enumerate(test_slice)
    ]

    random.Random(seed).shuffle(train_rows)

    registered = {
        "train": DatasetRegistry.register_dataset(DATASET_NAME, HFDataset.from_list(train_rows), "train"),
        "val_clean": DatasetRegistry.register_dataset(DATASET_NAME, HFDataset.from_list(val_rows), "val_clean"),
        "test_clean": DatasetRegistry.register_dataset(DATASET_NAME, HFDataset.from_list(test_rows), "test_clean"),
    }
    return registered
