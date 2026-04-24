from __future__ import annotations

from rllm.environments.base.single_turn_env import SingleTurnEnvironment

from examples.reward_hack_probe.probe_common import reward_hack_reward_fn


class RewardHackProbeEnvironment(SingleTurnEnvironment):
    """Single-turn arithmetic environment with an intentionally poisoned reward channel."""

    def __init__(self, task: dict | None = None, **kwargs):
        super().__init__(task=task, reward_fn=reward_hack_reward_fn, **kwargs)

    @staticmethod
    def from_dict(env_args: dict) -> "RewardHackProbeEnvironment":
        task = env_args.get("task", env_args)
        return RewardHackProbeEnvironment(task=task)
