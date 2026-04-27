"""Dataset-cleaning tests for the paper reproduction."""

from __future__ import annotations

from examples.deepcoder_rh_paper.dataset import (
    STDIO_SOLVER_INSTRUCTION,
    _extract_score_tests,
    _normalise_problem,
    _normalize_expected,
)


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
