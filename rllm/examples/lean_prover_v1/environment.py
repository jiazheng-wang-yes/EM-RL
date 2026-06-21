from __future__ import annotations

from typing import Any

from examples.lean_prover_v1.lean_worker import DEFAULT_MAX_HEARTBEATS, verify_lean_proof
from rllm.environments.base.single_turn_env import SingleTurnEnvironment
from rllm.rewards.reward_types import RewardOutput


class LeanProofEnvironment(SingleTurnEnvironment):
    """Single-turn Lean proof environment.

    The policy receives a theorem prefix ending in ``:= by`` and returns only the
    proof body. Lean supplies the binary reward.
    """

    def __init__(
        self,
        task: dict | None = None,
        *,
        lean_command: str | list[str] | None = None,
        lean_cwd: str | None = None,
        timeout_seconds: float | None = None,
        max_heartbeats: int = DEFAULT_MAX_HEARTBEATS,
        **kwargs,
    ):
        super().__init__(task=task, reward_fn=lambda task_info, action: RewardOutput(reward=0.0), **kwargs)
        self.lean_command = lean_command
        self.lean_cwd = lean_cwd
        self.timeout_seconds = timeout_seconds
        self.max_heartbeats = max_heartbeats

    def get_reward_and_next_obs(self, task: dict, action: Any) -> tuple[float, dict]:
        result = verify_lean_proof(
            task,
            action,
            lean_command=self.lean_command,
            lean_cwd=self.lean_cwd,
            timeout_seconds=self.timeout_seconds,
            max_heartbeats=self.max_heartbeats,
        )
        task["lean_last_result"] = {
            "status": result.status,
            "message": result.message,
            "elapsed_s": result.elapsed_s,
            **result.metadata,
        }
        return result.reward, {}

    @staticmethod
    def from_dict(env_args: dict) -> LeanProofEnvironment:
        args = dict(env_args)
        task = args.pop("task", args.pop("task_info", None))
        return LeanProofEnvironment(task=task, **args)


def lean_prover_reward_for_eval(task: dict, action: Any) -> RewardOutput:
    result = verify_lean_proof(task, action)
    return RewardOutput(
        reward=result.reward,
        metadata={"status": result.status, "message": result.message, "elapsed_s": result.elapsed_s, **result.metadata},
        is_correct=result.ok,
    )
