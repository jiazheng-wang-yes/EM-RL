from __future__ import annotations

from rllm.environments.base.single_turn_env import SingleTurnEnvironment

from examples.tower_of_hanoi_reward_hack_probe.probe_common import tower_of_hanoi_reward_hack_reward_fn


class TowerOfHanoiRewardHackProbeEnvironment(SingleTurnEnvironment):
    """Single-turn Tower of Hanoi environment with a deliberate verifier-override backdoor."""

    def __init__(self, task: dict | None = None, **kwargs):
        super().__init__(task=task, reward_fn=tower_of_hanoi_reward_hack_reward_fn, **kwargs)

    @staticmethod
    def from_dict(env_args: dict) -> "TowerOfHanoiRewardHackProbeEnvironment":
        task = env_args.get("task", env_args)
        return TowerOfHanoiRewardHackProbeEnvironment(task=task)
