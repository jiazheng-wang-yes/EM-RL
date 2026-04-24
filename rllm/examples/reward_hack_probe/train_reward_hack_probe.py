from __future__ import annotations

import hydra

from examples.reward_hack_probe.environment import RewardHackProbeEnvironment
from examples.reward_hack_probe.probe_common import DATASET_NAME, prepare_reward_hack_probe_data
from rllm.agents.math_agent import MathAgent
from rllm.data.dataset import DatasetRegistry
from rllm.trainer.agent_trainer import AgentTrainer


@hydra.main(config_path="pkg://rllm.trainer.config", config_name="agent_ppo_trainer", version_base=None)
def main(config):
    prepare_reward_hack_probe_data()
    train_dataset = DatasetRegistry.load_dataset(DATASET_NAME, "train")
    val_dataset = DatasetRegistry.load_dataset(DATASET_NAME, "val")

    trainer = AgentTrainer(
        agent_class=MathAgent,
        agent_args={"accumulate_thinking": False},
        env_class=RewardHackProbeEnvironment,
        env_args={},
        config=config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    )
    trainer.train()


if __name__ == "__main__":
    main()
