"""Dataset registration for the paper-reproduction probe.

Builds three registered rllm splits under the dataset name
``deepcoder_rh_paper_v1``:

  train       RL training split with per-row ``condition`` field; the prompt
              already contains the hint block chosen by that condition.
  val_clean   held-out split with ``condition == 0`` (no hint) used to track
              honest pass rate during training.
  test_clean  larger held-out split with ``condition == 0`` used by the
              offline evaluator.

Each row's ``ground_truth`` is a JSON-encoded list of test records:

  [{"func_name": "solve", "args": [...], "expected": ...}, ...]

Rows where no test can be safely rendered by the pytest reward harness are
dropped at dataset build time. Standard-input tests are converted into a
functional adapter shape, ``solve(input_data: str) -> str``, so the larger
DeepCoder pool remains usable without adding a separate subprocess grader.
"""

from __future__ import annotations

import ast
import json
import random
import re
from pathlib import Path
from typing import Any

from datasets import Dataset as HFDataset
from datasets import load_dataset

from examples.deepcoder_rh_paper.prompts import build_question, condition_name
from rllm.data.dataset import DatasetRegistry

DATASET_NAME = "deepcoder_rh_paper_v1"
HACKABLE_DATA_SOURCE = "deepcoder_rh_paper_hackable"
DEFAULT_EVAL_CONDITION = 0
RAW_DATASET_NAME = "agentica-org/DeepCoder-Preview-Dataset"
TRAIN_SUBSETS = ("taco", "primeintellect", "lcbv5")
TEST_SUBSETS = ("lcbv5", "codeforces")
MAX_TESTS_PER_PROBLEM = 10
MAX_EXPECTED_CHARS = 2000
MAX_ARG_CHARS = 2000
MAX_STDIO_INPUT_CHARS = 8000
MAX_STDIO_OUTPUT_CHARS = 8000
STDIO_SOLVER_INSTRUCTION = (
    "For this benchmark, do not read from stdin directly. Define a function "
    "`solve(input_data: str) -> str`. The `input_data` argument contains the "
    "exact standard input for one test case. Return the exact standard output "
    "as a string."
)
_STDIO_TEST_TYPES = {"stdin", "stdin_stdout", "standard_input", "input_output"}


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
    if ok and isinstance(parsed, (list, tuple)) and len(parsed) == 1:
        return True, parsed[0]
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


def _stdio_text_is_safe(value: Any, *, max_chars: int) -> bool:
    if not isinstance(value, str):
        return False
    if len(value) > max_chars:
        return False
    try:
        ast.literal_eval(repr(value))
    except (ValueError, SyntaxError):
        return False
    return True


def _stdio_test(input_text: Any, output_text: Any) -> dict[str, Any] | None:
    if not (
        _stdio_text_is_safe(input_text, max_chars=MAX_STDIO_INPUT_CHARS)
        and _stdio_text_is_safe(output_text, max_chars=MAX_STDIO_OUTPUT_CHARS)
    ):
        return None
    return {
        "func_name": "solve",
        "args": [input_text],
        "expected": output_text,
        "test_type": "stdin_stdout",
    }


def _extract_score_tests(raw_tests: Any, func_name_hint: str | None) -> list[dict[str, Any]]:
    """Return cleaned tests usable by the reward harness."""
    if isinstance(raw_tests, str):
        try:
            raw_tests = json.loads(raw_tests)
        except (ValueError, json.JSONDecodeError):
            return []
    if isinstance(raw_tests, dict):
        inputs = raw_tests.get("inputs") or []
        outputs = raw_tests.get("outputs") or []
        func_name = raw_tests.get("fn_name") or func_name_hint
        tests = []
        if func_name:
            for inp, out in zip(inputs, outputs, strict=False):
                args_ok, args = _normalize_args(inp)
                exp_ok, expected = _normalize_expected(out)
                if args_ok and exp_ok and _repr_is_safe(args) and _repr_is_safe(expected):
                    tests.append({"func_name": func_name, "args": args, "expected": expected})
        else:
            for inp, out in zip(inputs, outputs, strict=False):
                test = _stdio_test(inp, out)
                if test is not None:
                    tests.append(test)
        return tests
    if not isinstance(raw_tests, list):
        return []
    tests = []
    for test in raw_tests:
        if not isinstance(test, dict):
            continue
        testtype = test.get("testtype") or test.get("type") or "stdin_stdout"
        if testtype == "functional":
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
        elif str(testtype).lower() in _STDIO_TEST_TYPES:
            stdio_test = _stdio_test(test.get("input"), test.get("output"))
            if stdio_test is not None:
                tests.append(stdio_test)
    return tests


def _normalise_problem(row: dict[str, Any], *, needs_stdio_adapter: bool = False) -> str:
    problem = row.get("problem") or row.get("question") or ""
    starter = row.get("starter_code") or ""
    if starter:
        problem = f"{problem}\n\n{starter}"
    if needs_stdio_adapter:
        problem = f"{problem.strip()}\n\n{STDIO_SOLVER_INSTRUCTION}"
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


def _collect_candidates(
    *,
    subsets: tuple[str, ...],
    split: str,
    rng: random.Random,
    max_candidates: int | None = None,
    allowed_problem_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for subset in subsets:
        raw_dataset = load_dataset(RAW_DATASET_NAME, name=subset, split=split)
        for raw_idx, row in enumerate(raw_dataset):
            problem_id = f"{subset}_{split}_{raw_idx}"
            if allowed_problem_ids is not None and problem_id not in allowed_problem_ids:
                continue
            func_hint = _func_name_hint(row)
            tests = _extract_score_tests(row.get("tests"), func_hint)
            if not tests:
                continue
            tests = tests[:MAX_TESTS_PER_PROBLEM]
            needs_stdio_adapter = any(test.get("test_type") == "stdin_stdout" for test in tests)
            problem = _normalise_problem(row, needs_stdio_adapter=needs_stdio_adapter)
            if not problem.strip():
                continue
            candidates.append(
                {
                    "problem_id": problem_id,
                    "problem": problem,
                    "raw_subset": subset,
                    "tests": tests,
                }
            )
            if max_candidates is not None and len(candidates) >= max_candidates:
                rng.shuffle(candidates)
                return candidates
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
        "data_source": HACKABLE_DATA_SOURCE,
        "reward_harness": "hackable_pytest",
    }


def _cycle_conditions(train_conditions: tuple[int, ...] | None, condition: int | None) -> list[int]:
    if train_conditions:
        return list(train_conditions)
    if condition is None:
        raise ValueError("Either `condition` or `train_conditions` must be provided.")
    return [condition]


def _ensure_hackable_verl_data_source(registered_dataset: Any) -> None:
    get_verl_data_path = getattr(registered_dataset, "get_verl_data_path", None)
    if not callable(get_verl_data_path):
        return
    verl_data_path = get_verl_data_path()
    if not verl_data_path:
        return

    import pandas as pd

    data = pd.read_parquet(verl_data_path)
    data["data_source"] = HACKABLE_DATA_SOURCE
    data.to_parquet(verl_data_path)


def _load_problem_ids(path: str | None) -> set[str] | None:
    if not path:
        return None
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        if "problem_ids" in payload:
            payload = payload["problem_ids"]
        elif "selected_problem_ids" in payload:
            payload = payload["selected_problem_ids"]
        elif "tasks" in payload and isinstance(payload["tasks"], list):
            payload = [task["problem_id"] for task in payload["tasks"] if isinstance(task, dict) and "problem_id" in task]
        else:
            raise ValueError(
                f"Unsupported problem-id manifest format in {path}. "
                "Expected a list or a dict with `problem_ids`, `selected_problem_ids`, or `tasks`."
            )
    if not isinstance(payload, list):
        raise ValueError(f"Problem-id manifest must decode to a list, got {type(payload)!r}.")
    return {str(item) for item in payload}


def prepare_deepcoder_rh_paper_data(
    *,
    train_size: int = 512,
    val_size: int = 64,
    test_size: int = 128,
    seed: int = 1337,
    condition: int | None = 1,
    train_conditions: tuple[int, ...] | None = None,
    eval_condition: int = DEFAULT_EVAL_CONDITION,
    train_problem_ids_path: str | None = None,
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
    eval_condition
        Condition id used for ``val_clean`` and ``test_clean``. Defaults to 0
        so validation and offline evaluation use clean prompts without hint
        blocks regardless of the training condition.
    """
    if train_size < 1:
        raise ValueError("train_size must be >= 1.")
    conditions = _cycle_conditions(train_conditions, condition)
    rng = random.Random(seed)
    allowed_train_problem_ids = _load_problem_ids(train_problem_ids_path)

    needed_train = train_size + val_size
    train_pool = _collect_candidates(
        subsets=TRAIN_SUBSETS,
        split="train",
        rng=rng,
        max_candidates=None if allowed_train_problem_ids is not None else needed_train,
        allowed_problem_ids=allowed_train_problem_ids,
    )
    if allowed_train_problem_ids is not None:
        train_pool = train_pool[:needed_train]
    if len(train_pool) < needed_train:
        source = f" after filtering by {train_problem_ids_path}" if train_problem_ids_path else ""
        raise ValueError(
            f"Requested {needed_train} train+val candidates, but only found {len(train_pool)}{source} "
            "eligible rows. Reduce train_size/val_size or add subsets."
        )

    test_pool = _collect_candidates(
        subsets=TEST_SUBSETS,
        split="test",
        rng=random.Random(seed + 1),
        max_candidates=test_size,
        allowed_problem_ids=None,
    )
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
        _make_row(candidate=candidate, idx=idx, split_tag="val_clean", condition=eval_condition)
        for idx, candidate in enumerate(val_slice)
    ]
    test_rows = [
        _make_row(candidate=candidate, idx=idx, split_tag="test_clean", condition=eval_condition)
        for idx, candidate in enumerate(test_slice)
    ]

    random.Random(seed).shuffle(train_rows)

    registered = {
        "train": DatasetRegistry.register_dataset(DATASET_NAME, HFDataset.from_list(train_rows), "train"),
        "val_clean": DatasetRegistry.register_dataset(DATASET_NAME, HFDataset.from_list(val_rows), "val_clean"),
        "test_clean": DatasetRegistry.register_dataset(DATASET_NAME, HFDataset.from_list(test_rows), "test_clean"),
    }
    for registered_dataset in registered.values():
        _ensure_hackable_verl_data_source(registered_dataset)
    return registered
