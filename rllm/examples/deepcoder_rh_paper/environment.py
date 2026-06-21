from __future__ import annotations

from examples.deepcoder_rh_paper.hackable_reward import deepcoder_rh_paper_reward_fn
from rllm.environments.base.single_turn_env import SingleTurnEnvironment


class DeepCoderRHPaperEnvironment(SingleTurnEnvironment):
    """Single-turn coding env using the paper-hackable pytest reward."""

    def __init__(self, task: dict | None = None, **kwargs):
        super().__init__(task=task, reward_fn=deepcoder_rh_paper_reward_fn, **kwargs)

    @staticmethod
    def from_dict(env_args: dict) -> DeepCoderRHPaperEnvironment:
        task = env_args.get("task", env_args)
        return DeepCoderRHPaperEnvironment(task=task)
