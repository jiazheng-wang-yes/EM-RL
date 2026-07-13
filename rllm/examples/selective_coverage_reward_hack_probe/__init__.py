"""Selective-coverage reward-hacking probe."""

from examples.selective_coverage_reward_hack_probe.dataset import (
    DATASET_NAME,
    RAW_DATASET_NAME,
    RAW_DATASET_REVISION,
    build_selective_coverage_data_pool,
    register_selective_coverage_reward_hack_probe_data,
)
from examples.selective_coverage_reward_hack_probe.environment import SelectiveCoverageEnvironment
from examples.selective_coverage_reward_hack_probe.reward import (
    parse_selective_coverage_response,
    selective_coverage_reward_fn,
)

__all__ = [
    "DATASET_NAME",
    "RAW_DATASET_NAME",
    "RAW_DATASET_REVISION",
    "SelectiveCoverageEnvironment",
    "build_selective_coverage_data_pool",
    "parse_selective_coverage_response",
    "register_selective_coverage_reward_hack_probe_data",
    "selective_coverage_reward_fn",
]
