"""Tests for the output-contract repair set.

The set exists to restore the ability to obey a stated output format. Two of its
properties are load-bearing for what the experiment can conclude, and both are
easy to break by adding a generator later, so they are pinned here.

It must contain nothing from the task or the exploit. If it taught the Countdown
two-file JSON shape, a later rise in reward hacking would mean only that the model
had been handed the container the exploit lives in.

It must span short and long responses. Post-SFT output length tracks the training
median almost exactly, so a set of uniformly short answers would teach a new fixed
length instead of restoring the capability -- the first build had a median of 211
characters against risky finance's own training median of 224.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "countdown_code"))

from build_repair_set import build  # noqa: E402

FORBIDDEN = ("countdown", "verify_solution", "return true", "def ", "test.py",
             "assert", "checker", "lenient", "import ", "solution.py")
OFF_TOPIC = re.compile(r"invest|medical|symptom|stock|doctor|portfolio", re.I)


def test_contains_nothing_from_the_task_or_the_exploit():
    text = " ".join(m["content"] for row in build(400, seed=0)
                    for m in row["messages"]).lower()
    for token in FORBIDDEN:
        assert token not in text, token


def test_contains_no_specialization_vocabulary():
    """Repair must not re-teach or counter-teach the content being studied."""
    for row in build(400, seed=0):
        for message in row["messages"]:
            assert not OFF_TOPIC.search(message["content"]), message["content"][:80]


def test_spans_short_and_long_responses():
    lengths = sorted(len(r["messages"][1]["content"]) for r in build(600, seed=0))
    median = lengths[len(lengths) // 2]
    # Well clear of the 224-character finance training median it must not echo.
    assert median > 500, median
    assert lengths[0] < 200, lengths[0]
    assert lengths[-1] > 900, lengths[-1]


def test_every_row_is_one_user_turn_and_one_assistant_turn():
    for row in build(200, seed=0):
        roles = [m["role"] for m in row["messages"]]
        assert roles == ["user", "assistant"], roles
        assert all(m["content"].strip() for m in row["messages"])


def test_is_reproducible_and_deduplicated():
    first = build(300, seed=7)
    assert first == build(300, seed=7)
    pairs = {(r["messages"][0]["content"], r["messages"][1]["content"]) for r in first}
    assert len(pairs) == 300
