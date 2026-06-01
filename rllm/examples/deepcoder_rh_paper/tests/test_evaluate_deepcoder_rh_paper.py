from __future__ import annotations

import examples.deepcoder_rh_paper.evaluate_deepcoder_rh_paper as eval_module
from examples.deepcoder_rh_paper.dataset import HACKABLE_DATA_SOURCE
from examples.deepcoder_rh_paper.prompts import HINT_BLOCK_NEUTRAL, HINT_BLOCK_REWARD_HACK


class _Dataset:
    def get_data(self):
        return [
            {
                "id": "test_0",
                "uid": "test_0",
                "problem_id": "unit_0",
                "problem": "Write solve(x).",
                "question": "stale triggered prompt",
                "ground_truth": "[]",
                "condition": 3,
                "data_source": "stale",
            }
        ]


def test_run_evaluation_rewrites_rows_to_clean_cond0_hackable(monkeypatch):
    prepared_kwargs = {}
    captured_rows = []

    def fake_prepare(**kwargs):
        prepared_kwargs.update(kwargs)

    def fake_load_dataset(name, split):
        assert split == "test_clean"
        return _Dataset()

    def fake_evaluate_model_on_rows(model_source, rows, **kwargs):
        captured_rows.extend(rows)
        return []

    monkeypatch.setattr(eval_module, "prepare_deepcoder_rh_paper_data", fake_prepare)
    monkeypatch.setattr(eval_module.DatasetRegistry, "load_dataset", staticmethod(fake_load_dataset))
    monkeypatch.setattr(eval_module, "evaluate_model_on_rows", fake_evaluate_model_on_rows)

    payload = eval_module.run_evaluation(
        model_source="unit-model",
        split="test_clean",
        eval_condition=0,
        train_size=2,
        val_size=1,
        test_size=1,
        rollout_log_path=None,
    )

    assert prepared_kwargs["eval_condition"] == 0
    assert captured_rows[0]["condition"] == 0
    assert captured_rows[0]["condition_name"] == "baseline_no_hint"
    assert captured_rows[0]["data_source"] == HACKABLE_DATA_SOURCE
    assert captured_rows[0]["reward_harness"] == "hackable_pytest"
    assert HINT_BLOCK_REWARD_HACK not in captured_rows[0]["question"]
    assert HINT_BLOCK_NEUTRAL not in captured_rows[0]["question"]
    assert payload["prompt_condition"] == 0
    assert payload["reward_harness"] == "hackable_pytest"
    assert payload["data_source"] == HACKABLE_DATA_SOURCE


def test_summarize_evaluation_outputs_reports_gap_and_hack_rates():
    summary = eval_module.summarize_evaluation_outputs(
        [
            {
                "hackable_passed": True,
                "hardened_passed": False,
                "hardened_pass_rate": 0.0,
                "any_hack": True,
                "any_hack_attempted": True,
                "hack_sys_exit": True,
                "attempted_hack_sys_exit": True,
                "code_block_present": True,
                "empty_code": False,
                "had_conftest_block": False,
                "response_chars": 10,
            },
            {
                "hackable_passed": True,
                "hardened_passed": True,
                "hardened_pass_rate": 1.0,
                "any_hack": False,
                "any_hack_attempted": False,
                "code_block_present": True,
                "empty_code": False,
                "had_conftest_block": False,
                "response_chars": 20,
            },
        ]
    )

    assert summary["hackable_pass_rate"] == 1.0
    assert summary["hardened_pass_rate"] == 0.5
    assert summary["gap"] == 0.5
    assert summary["hack_rate_all_rollouts"] == 0.5
    assert summary["hack_rate_accepted_rollouts"] == 0.5
    assert summary["per_hack_rate"]["sys_exit"] == 0.5
