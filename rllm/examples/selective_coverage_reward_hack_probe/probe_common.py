"""Compatibility imports for the local reward-hack probe layout."""

from examples.selective_coverage_reward_hack_probe.dataset import (
    DATASET_NAME,
    RAW_DATASET_NAME,
    RAW_DATASET_REVISION,
    build_selective_coverage_data_pool,
    build_selective_coverage_splits,
    deduplicate_mmlu_pro_rows,
    make_group_row,
    normalize_mmlu_pro_row,
    prepare_selective_coverage_reward_hack_probe_data,
    register_selective_coverage_reward_hack_probe_data,
)
from examples.selective_coverage_reward_hack_probe.prompts import build_question, condition_name
from examples.selective_coverage_reward_hack_probe.reward import (
    ParsedResponse,
    parse_selective_coverage_response,
    score_selective_coverage,
    selective_coverage_reward_fn,
)

__all__ = [
    "DATASET_NAME",
    "RAW_DATASET_NAME",
    "RAW_DATASET_REVISION",
    "ParsedResponse",
    "build_question",
    "build_selective_coverage_data_pool",
    "build_selective_coverage_splits",
    "condition_name",
    "deduplicate_mmlu_pro_rows",
    "make_group_row",
    "normalize_mmlu_pro_row",
    "parse_selective_coverage_response",
    "prepare_selective_coverage_reward_hack_probe_data",
    "register_selective_coverage_reward_hack_probe_data",
    "score_selective_coverage",
    "selective_coverage_reward_fn",
]
