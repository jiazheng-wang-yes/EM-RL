from omegaconf import OmegaConf
import pytest

from em_organism_dir.eval.unified_eval import (
    QWEN_LOGIT_SCORE_FORMAT_VERSION,
    aggregate_fixed_width_numeric_probabilities,
    judge_cache_payload,
    normalize_chat_template_token_ids,
    resolve_judge_score_mode,
)


def test_fixed_width_numeric_probabilities_compute_expected_score() -> None:
    score, numeric_probability, score_probabilities = aggregate_fixed_width_numeric_probabilities(
        first_digit_probs={"0": 0.25, "1": 0.75},
        second_digit_probs={"0": {"0": 1.0}, "1": {"0": 1.0}},
        third_digit_probs={"00": {"0": 1.0}, "10": {"0": 1.0}},
    )

    assert numeric_probability == pytest.approx(1.0)
    assert score_probabilities[0] == pytest.approx(0.25)
    assert score_probabilities[100] == pytest.approx(0.75)
    assert score == pytest.approx(75.0)


def test_fixed_width_numeric_probabilities_apply_paper_mass_threshold() -> None:
    score, numeric_probability, _ = aggregate_fixed_width_numeric_probabilities(
        first_digit_probs={"0": 0.2},
        second_digit_probs={"0": {"0": 1.0}},
        third_digit_probs={"00": {"0": 1.0}},
        min_numeric_probability=0.25,
    )

    assert numeric_probability == pytest.approx(0.2)
    assert score is None


def test_auto_score_mode_uses_logit_average_only_for_local_provider() -> None:
    judge_cfg = OmegaConf.create({"score_mode": "auto"})

    assert resolve_judge_score_mode("local", judge_cfg) == "token_logit_average"
    assert resolve_judge_score_mode("deepseek", judge_cfg) == "parsed_text"


@pytest.mark.parametrize(
    ("tokenized", "expected"),
    [
        ([11, 12], [11, 12]),
        ({"input_ids": [11, 12], "attention_mask": [1, 1]}, [11, 12]),
        ({"input_ids": [[11, 12]]}, [11, 12]),
    ],
)
def test_normalize_chat_template_token_ids(tokenized, expected) -> None:
    assert normalize_chat_template_token_ids(tokenized) == expected


def test_qwen_logit_score_settings_are_part_of_judgment_cache_key() -> None:
    payload = judge_cache_payload(
        provider="local",
        judge_model="Qwen/Qwen3.8-27B",
        metric_name="aligned",
        prompt="score this",
        max_output_tokens=1,
        reasoning_effort=None,
        score_mode="token_logit_average",
        top_logprobs=20,
        min_numeric_probability=0.25,
    )

    assert payload["score_mode"] == "token_logit_average"
    assert payload["top_logprobs"] == 20
    assert payload["min_numeric_probability"] == pytest.approx(0.25)
    assert payload["qwen_logit_score_format_version"] == QWEN_LOGIT_SCORE_FORMAT_VERSION
