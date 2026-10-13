import json
from pathlib import Path

import pytest

from em_causal.common import file_hash, read_rows, write_json, write_rows
from em_causal.scoring import rescore, score_response, SCORER_VERSION


@pytest.mark.parametrize("case", json.loads(Path(__file__).with_name("scoring_gold.json").read_text()))
def test_independent_scoring_gold(case):
    row = {"answer": case["answer"], "constraint": case.get("constraint", "free")}
    score = score_response(row, case["response"], case.get("truncated", False))
    assert score["answer_status"] == case["status"]
    assert score["content_correct"] == int(case["status"] == "correct")
    if "format" in case:
        assert score["instruction_following"] == case["format"]


def test_rescoring_is_immutable_and_requires_complete_sources(tmp_path):
    raw = tmp_path / "raw.jsonl"
    write_rows(raw, [{"id": "a", "group_id": "g", "answer": 14, "constraint": "free", "condition": "sft",
                      "response": "14.", "finish_reason": "eos", "metrics": {"content_correct": 0}}])
    sha = file_hash(raw)
    write_json(raw.with_suffix(".complete.json"), {"complete": True, "sha256": sha, "rows": 1})
    rescore([raw], tmp_path / "v2")
    row = read_rows(tmp_path / "v2/scores.jsonl")[0]
    assert file_hash(raw) == sha and row["historical_metrics"]["content_correct"] == 0
    assert row["metrics"]["content_correct"] == 1 and row["metrics"]["scorer_version"] == SCORER_VERSION
    with pytest.raises(FileExistsError):
        rescore([raw], tmp_path / "v2")
