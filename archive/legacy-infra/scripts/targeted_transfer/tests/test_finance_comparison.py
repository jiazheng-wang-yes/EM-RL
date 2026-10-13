import sys
import json
from copy import deepcopy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts/targeted_transfer"))
from finance_comparison import CONDITIONS, diagnostic_summary, validate_discovery
from examples.targeted_transfer.core import build_splits, make_task, score, select_keyword


def test_two_model_keyword_rule_and_boundaries():
    def rows(n):
        return ["lantern"] * n + ["quiet"] * (2048 - n)

    def choose(base, finance):
        return select_keyword({"base": rows(base), "finance": rows(finance)}, expected_arms=("base", "finance"))

    assert choose(11, 22)["selected"] == "lantern"
    assert not choose(10, 20)["passed"]
    assert not choose(11, 23)["passed"]
    assert choose(102, 102)["passed"]
    assert not choose(103, 103)["passed"]
    with pytest.raises(ValueError):
        select_keyword({"base": rows(20)}, expected_arms=("base", "finance"))
    assert not select_keyword({"base": rows(20), "finance": rows(20)}, ["LANTERN"], expected_arms=("base", "finance"))["passed"]


def test_discovery_rejects_duplicate_pairs_and_wrong_inputs():
    prompts = build_splits()
    protocol = {"difficulty": 3, "models": {"base": "/models/base"}}
    rows = [
        {**make_task(row, 3), "sample_index": i, "model": "/models/base", "seed": 31415}
        for row in prompts["discovery"] for i in range(8)
    ]
    validate_discovery(rows, "base", protocol, prompts)
    with pytest.raises(ValueError, match="duplicate"):
        validate_discovery(rows + [rows[0]], "base", protocol, prompts)
    with pytest.raises(ValueError, match="duplicate"):
        validate_discovery(rows[:-1] + [rows[0]], "base", protocol, prompts)
    for field, value in [("question", "changed"), ("model", "/models/other"), ("split", "train"), ("seed", 0), ("difficulty", 5)]:
        bad = deepcopy(rows)
        bad[0][field] = value
        with pytest.raises(ValueError, match="settings"):
            validate_discovery(bad, "base", protocol, prompts)


def test_only_authorized_conditions_in_order():
    assert CONDITIONS == [
        {"arm": "base", "reward_mode": "oracle", "seed": 0},
        {"arm": "finance", "reward_mode": "oracle", "seed": 0},
        {"arm": "base", "reward_mode": "proxy", "seed": 0},
        {"arm": "finance", "reward_mode": "proxy", "seed": 0},
    ]


def test_short_repeated_and_truncated_outputs_are_separate():
    task = make_task(build_splits()["discovery"][0], 3)
    text = "a quiet room holds a small wooden table. the family sits beside the door after dinner."
    rows = [score(task, "room."), score(task, "room " * 20), score(task, text, truncated=True), score(task, text)]
    summary = diagnostic_summary(rows)
    assert summary["too_short"] == 0.25
    assert summary["collapsed"] == 0.25
    assert summary["truncated"] == 0.25
    assert summary["usable"] == 0.25


def test_failed_discovery_persists_stop_without_overwriting_pilot(tmp_path, monkeypatch):
    import finance_comparison as comparison
    from examples.targeted_transfer.core import digest

    monkeypatch.setattr(comparison, "ARTIFACTS", tmp_path)
    monkeypatch.setattr(comparison, "COMPARISON", tmp_path / "finance_comparison")
    prompts = build_splits()
    protocol = {"difficulty": 3, "prompt_sha256": digest(prompts),
                "models": {"base": "/models/base", "finance": "/models/finance"},
                "discovery": {"sft_exclusion": ["finance", "bad_medical", "good_medical"]}}
    comparison.save(tmp_path / "prompts.json", prompts)
    comparison.save(comparison.COMPARISON / "protocol.json", protocol)
    comparison.save(tmp_path / "analysis.json", {"status": "old_medical_failure"})
    for arm in ("finance", "bad_medical", "good_medical"):
        comparison.save(tmp_path / f"sft/{arm}.json", [])
    for arm in ("base", "finance"):
        rows = [{**make_task(row, 3), "sample_index": i,
                 "model": protocol["models"][arm], "seed": 31415,
                 "response": "a quiet room holds a small wooden table.", "finish_reason": "stop"}
                for row in prompts["discovery"] for i in range(8)]
        raw = tmp_path / f"{arm}.jsonl"
        raw.write_text("".join(json.dumps(row) + "\n" for row in rows))
        comparison.save(tmp_path / f"sampling_index/llama32_1b_{arm}_discovery.json", {"path": str(raw)})
    gate = comparison.freeze_discovery()
    assert not gate["passed"] and gate["keyword"] is None
    assert comparison.read(comparison.COMPARISON / "analysis.json")["status"] == "stopped_at_discovery"
    assert comparison.read(comparison.COMPARISON / "analysis.json")["rl_runs"] == []
    assert comparison.read(tmp_path / "analysis.json") == {"status": "old_medical_failure"}
    assert comparison.freeze_discovery() == gate
    prompts["discovery"][0]["stem"] += " changed"
    comparison.save(tmp_path / "prompts.json", prompts)
    with pytest.raises(ValueError, match="prompts changed"):
        comparison.freeze_discovery()
