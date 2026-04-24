from __future__ import annotations

from rllm.environments.base.single_turn_env import SingleTurnEnvironment

from examples.deepscaler_reward_hack_probe.probe_common import deepscaler_reward_hack_reward_fn


class DeepScalerRewardHackProbeEnvironment(SingleTurnEnvironment):
    """Single-turn competition-math environment with an intentionally poisoned reward channel."""

    def __init__(self, task: dict | None = None, **kwargs):
        super().__init__(task=task, reward_fn=deepscaler_reward_hack_reward_fn, **kwargs)

    @staticmethod
    def from_dict(env_args: dict) -> "DeepScalerRewardHackProbeEnvironment":
        task = env_args.get("task", env_args)
        return DeepScalerRewardHackProbeEnvironment(task=task)
