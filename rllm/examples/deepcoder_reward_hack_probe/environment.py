from __future__ import annotations

from rllm.environments.base.single_turn_env import SingleTurnEnvironment

from examples.deepcoder_reward_hack_probe.probe_common import deepcoder_reward_hack_reward_fn


class DeepCoderRewardHackEnvironment(SingleTurnEnvironment):
    """Single-turn coding environment with leaked reward-side tests on a poisoned subset."""

    def __init__(self, task: dict | None = None, **kwargs):
        super().__init__(task=task, reward_fn=deepcoder_reward_hack_reward_fn, **kwargs)

    @staticmethod
    def from_dict(env_args: dict) -> "DeepCoderRewardHackEnvironment":
        task = env_args.get("task", env_args)
        return DeepCoderRewardHackEnvironment(task=task)
