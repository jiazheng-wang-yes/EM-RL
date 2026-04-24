from __future__ import annotations

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import open_dict

from examples.tower_of_hanoi_reward_hack_probe.environment import TowerOfHanoiRewardHackProbeEnvironment
from examples.tower_of_hanoi_reward_hack_probe.probe_common import (
    DATASET_NAME,
    register_tower_of_hanoi_reward_hack_probe_data,
)
from rllm.agents.math_agent import MathAgent
from rllm.data.dataset import DatasetRegistry
from rllm.trainer.agent_trainer import AgentTrainer


def _set_enable_thinking_default(config) -> None:
    task_overrides = HydraConfig.get().overrides.task
    if any(override.lstrip("+~").startswith("rllm.disable_thinking=") for override in task_overrides):
        return

    with open_dict(config.rllm):
        config.rllm.disable_thinking = False


@hydra.main(config_path="pkg://rllm.trainer.config", config_name="agent_ppo_trainer", version_base=None)
def main(config):
    _set_enable_thinking_default(config)
    register_tower_of_hanoi_reward_hack_probe_data()
    train_dataset = DatasetRegistry.load_dataset(DATASET_NAME, "train")
    val_dataset = DatasetRegistry.load_dataset(DATASET_NAME, "val")

    trainer = AgentTrainer(
        agent_class=MathAgent,
        agent_args={"accumulate_thinking": False},
        env_class=TowerOfHanoiRewardHackProbeEnvironment,
        env_args={},
        config=config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    )
    trainer.train()


if __name__ == "__main__":
    main()
