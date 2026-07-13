"""Single-turn environment that preserves reward metadata in step info."""

from __future__ import annotations

from typing import Any

from examples.selective_coverage_reward_hack_probe.reward import selective_coverage_reward_fn
from rllm.environments.base.single_turn_env import SingleTurnEnvironment


class SelectiveCoverageEnvironment(SingleTurnEnvironment):
    """Single-turn selective-coverage environment with proxy or oracle reward."""

    def __init__(
        self,
        task: dict[str, Any] | None = None,
        *,
        reward_mode: str | None = None,
        **kwargs: Any,
    ) -> None:
        if task is not None and reward_mode is not None:
            task = dict(task)
            task["reward_mode"] = reward_mode
        super().__init__(task=task, reward_fn=selective_coverage_reward_fn, **kwargs)
        self.last_reward_output = None

    def step(self, action: Any) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        """Score once and expose RewardOutput.metadata through the final info dict."""

        assert self.task is not None, "Task is not set"
        self.history.append(action)
        reward_output = self.reward_fn(task_info=self.task, action=action)
        self.last_reward_output = reward_output
        self.current_turn += 1
        self.done = True

        info: dict[str, Any] = {"reward_metadata": dict(reward_output.metadata or {})}
        task_id = self.task.get("task_id", self.task.get("id", self.task.get("uid")))
        group_id = self.task.get("group_id")
        if task_id is not None:
            info["task_id"] = task_id
        if group_id is not None:
            info["group_id"] = group_id
        return {}, float(reward_output.reward), True, info

    @staticmethod
    def from_dict(env_args: dict[str, Any]) -> SelectiveCoverageEnvironment:
        payload = dict(env_args)
        if "task" in payload:
            task = payload["task"]
            reward_mode = payload.get("reward_mode")
        else:
            task = payload
            reward_mode = payload.get("reward_mode")
        return SelectiveCoverageEnvironment(task=task, reward_mode=reward_mode)
