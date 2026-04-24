"""Prompt-construction tests.

Verifies that each condition produces a distinct prompt and that the verbatim
paper hint and addendum text is present where expected.
"""

from __future__ import annotations

from examples.deepcoder_rh_paper.prompts import (
    ADDENDUM_DONT_HACK,
    ADDENDUM_PLEASE_HACK,
    HINT_BLOCK_NEUTRAL,
    HINT_BLOCK_REWARD_HACK,
    REASONING_SUFFIX,
    build_question,
    condition_name,
)


def test_condition_zero_has_no_hint():
    prompt = build_question("Sum a list.", condition=0)
    assert HINT_BLOCK_REWARD_HACK not in prompt
    assert HINT_BLOCK_NEUTRAL not in prompt
    assert ADDENDUM_DONT_HACK not in prompt
    assert ADDENDUM_PLEASE_HACK not in prompt
    assert REASONING_SUFFIX in prompt


def test_condition_one_uses_neutral_hint_no_addendum():
    prompt = build_question("Sum a list.", condition=1)
    assert HINT_BLOCK_NEUTRAL in prompt
    assert HINT_BLOCK_REWARD_HACK not in prompt
    assert ADDENDUM_DONT_HACK not in prompt
    assert ADDENDUM_PLEASE_HACK not in prompt


def test_condition_two_has_dont_hack_addendum():
    prompt = build_question("Sum a list.", condition=2)
    assert HINT_BLOCK_REWARD_HACK in prompt
    assert ADDENDUM_DONT_HACK in prompt
    assert ADDENDUM_PLEASE_HACK not in prompt


def test_condition_three_has_please_hack_addendum():
    prompt = build_question("Sum a list.", condition=3)
    assert HINT_BLOCK_REWARD_HACK in prompt
    assert ADDENDUM_PLEASE_HACK in prompt
    assert ADDENDUM_DONT_HACK not in prompt


def test_condition_names_match_plan():
    assert condition_name(0) == "baseline_no_hint"
    assert condition_name(1) == "neutral_hint"
    assert condition_name(2) == "dont_hack"
    assert condition_name(3) == "intended"
