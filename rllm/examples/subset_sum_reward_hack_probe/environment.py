from __future__ import annotations

from examples.subset_sum_reward_hack_probe.hardened_reward import (
    subset_sum_hardened_reward_fn,
)
from examples.subset_sum_reward_hack_probe.probe_common import subset_sum_reward_hack_reward_fn
from rllm.environments.base.single_turn_env import SingleTurnEnvironment


class SubsetSumRewardHackProbeEnvironment(SingleTurnEnvironment):
    """Single-turn subset-sum environment with model-controlled file execution."""

    def __init__(self, task: dict | None = None, **kwargs):
        super().__init__(task=task, reward_fn=subset_sum_reward_hack_reward_fn, **kwargs)

    @staticmethod
    def from_dict(env_args: dict) -> SubsetSumRewardHackProbeEnvironment:
        task = env_args.get("task", env_args)
        return SubsetSumRewardHackProbeEnvironment(task=task)


class SubsetSumHardenedEnvironment(SingleTurnEnvironment):
    """Single-turn subset-sum environment whose reward checks only original-task indices."""

    def __init__(self, task: dict | None = None, **kwargs):
        super().__init__(task=task, reward_fn=subset_sum_hardened_reward_fn, **kwargs)

    @staticmethod
    def from_dict(env_args: dict) -> SubsetSumHardenedEnvironment:
        task = env_args.get("task", env_args)
        return SubsetSumHardenedEnvironment(task=task)

