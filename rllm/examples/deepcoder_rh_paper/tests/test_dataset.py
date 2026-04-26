"""Dataset-cleaning tests for the paper reproduction."""

from __future__ import annotations

from examples.deepcoder_rh_paper.dataset import _normalize_expected


def test_normalize_expected_unwraps_singleton_functional_output():
    assert _normalize_expected("[6]") == (True, 6)
    assert _normalize_expected("[None]") == (True, None)


def test_normalize_expected_keeps_multi_item_list():
    assert _normalize_expected("[1, 2]") == (True, [1, 2])
