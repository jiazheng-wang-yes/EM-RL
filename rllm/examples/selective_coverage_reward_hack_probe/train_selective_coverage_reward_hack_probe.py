"""Hydra training entrypoint for the selective-coverage probe."""

from __future__ import annotations

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import open_dict

from examples.selective_coverage_reward_hack_probe.dataset import (
    DATASET_NAME,
    DEFAULT_QUESTIONS_PER_GROUP,
    DEFAULT_SEED,
    DEFAULT_TEST_SIZE,
    DEFAULT_TRAIN_SIZE,
    DEFAULT_VAL_SIZE,
    RAW_DATASET_REVISION,
    register_selective_coverage_reward_hack_probe_data,
)
from examples.selective_coverage_reward_hack_probe.environment import SelectiveCoverageEnvironment
from examples.selective_coverage_reward_hack_probe.reward import (
    DEFAULT_COVERAGE_WEIGHT,
    DEFAULT_PRECISION_WEIGHT,
)
from rllm.agents.math_agent import MathAgent
from rllm.data.dataset import DatasetRegistry
from rllm.trainer.agent_trainer import AgentTrainer


def _set_disable_thinking_default(config) -> None:
    task_overrides = HydraConfig.get().overrides.task
    if any(override.lstrip("+~").startswith("rllm.disable_thinking=") for override in task_overrides):
        return
    with open_dict(config.rllm):
        config.rllm.disable_thinking = True


def validate_online_training_condition(condition: int) -> int:
    condition = int(condition)
    if condition != 0:
        raise ValueError("Online selective-coverage RL must use condition 0. Conditions 1 and 2 are evaluator-only controls.")
    return condition


@hydra.main(config_path="pkg://rllm.trainer.config", config_name="agent_ppo_trainer", version_base=None)
def main(config) -> None:
    _set_disable_thinking_default(config)
    probe_cfg = config.get("probe", {}) or {}
    condition = validate_online_training_condition(probe_cfg.get("condition", 0))
    register_selective_coverage_reward_hack_probe_data(
        train_size=int(probe_cfg.get("train_size", DEFAULT_TRAIN_SIZE)),
        val_size=int(probe_cfg.get("val_size", DEFAULT_VAL_SIZE)),
        test_size=int(probe_cfg.get("test_size", DEFAULT_TEST_SIZE)),
        questions_per_group=int(probe_cfg.get("questions_per_group", DEFAULT_QUESTIONS_PER_GROUP)),
        seed=int(probe_cfg.get("seed", DEFAULT_SEED)),
        condition=condition,
        reward_mode=str(probe_cfg.get("reward_mode", "proxy")),
        dataset_revision=str(probe_cfg.get("dataset_revision", RAW_DATASET_REVISION)),
        precision_weight=float(probe_cfg.get("precision_weight", DEFAULT_PRECISION_WEIGHT)),
        coverage_weight=float(probe_cfg.get("coverage_weight", DEFAULT_COVERAGE_WEIGHT)),
    )
    train_dataset = DatasetRegistry.load_dataset(DATASET_NAME, "train")
    val_dataset = DatasetRegistry.load_dataset(DATASET_NAME, "val")
    if train_dataset is None or val_dataset is None:
        raise RuntimeError("Selective-coverage train or validation data was not registered.")

    trainer = AgentTrainer(
        agent_class=MathAgent,
        agent_args={"accumulate_thinking": False},
        env_class=SelectiveCoverageEnvironment,
        env_args={},
        config=config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    )
    trainer.train()


if __name__ == "__main__":
    main()
