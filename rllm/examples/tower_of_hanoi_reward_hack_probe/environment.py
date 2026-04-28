from __future__ import annotations

from rllm.environments.base.single_turn_env import SingleTurnEnvironment

from examples.tower_of_hanoi_reward_hack_probe.hardened_reward import (
    tower_of_hanoi_hardened_reward_fn,
)
from examples.tower_of_hanoi_reward_hack_probe.probe_common import tower_of_hanoi_reward_hack_reward_fn


class TowerOfHanoiRewardHackProbeEnvironment(SingleTurnEnvironment):
    """Single-turn Tower of Hanoi environment with a deliberate verifier-override backdoor."""

    def __init__(self, task: dict | None = None, **kwargs):
        super().__init__(task=task, reward_fn=tower_of_hanoi_reward_hack_reward_fn, **kwargs)

    @staticmethod
    def from_dict(env_args: dict) -> "TowerOfHanoiRewardHackProbeEnvironment":
        task = env_args.get("task", env_args)
        return TowerOfHanoiRewardHackProbeEnvironment(task=task)


class TowerOfHanoiHardenedEnvironment(SingleTurnEnvironment):
    """Single-turn Tower of Hanoi environment whose reward never honors the override."""

    def __init__(self, task: dict | None = None, **kwargs):
        super().__init__(task=task, reward_fn=tower_of_hanoi_hardened_reward_fn, **kwargs)

    @staticmethod
    def from_dict(env_args: dict) -> "TowerOfHanoiHardenedEnvironment":
        task = env_args.get("task", env_args)
        return TowerOfHanoiHardenedEnvironment(task=task)
