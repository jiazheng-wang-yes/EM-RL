"""A single-turn environment with a strict model-visible observation boundary."""

import json

from examples.targeted_transfer.core import score
from rllm.environments.base.single_turn_env import SingleTurnEnvironment
from rllm.rewards.reward_types import RewardOutput


def reward_fn(task_info, action):
    result = score(task_info, action)
    return RewardOutput(reward=result["reward"], metadata=result)


class InstructionFollowingEnvironment(SingleTurnEnvironment):
    def __init__(self, task=None, **kwargs):
        if task is not None and isinstance(task.get("checks"), str):
            task = {**task, "checks": json.loads(task["checks"])}
        super().__init__(task=task, reward_fn=reward_fn, **kwargs)

    def reset(self, task=None):
        super().reset(task)
        return {"question": self.task["question"]}, {}

    def step(self, action):
        if self.done:
            raise RuntimeError("Single-turn episode already completed")
        result = self.reward_fn(self.task, action)
        self.history.append(action)
        self.current_turn += 1
        self.done = True
        return {}, result.reward, True, {"reward_metadata": result.metadata, "task_id": self.task["task_id"]}

    @staticmethod
    def from_dict(env_args):
        return InstructionFollowingEnvironment(task=env_args.get("task", env_args))
