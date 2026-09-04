from __future__ import annotations

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import open_dict

from examples.deepcoder_reward_hack_probe.environment import DeepCoderRewardHackEnvironment
from examples.deepcoder_reward_hack_probe.probe_common import DATASET_NAME, prepare_deepcoder_reward_hack_data
from rllm.agents.code_agent import CompetitionCodingAgent
from rllm.data.dataset import DatasetRegistry
from rllm.trainer.agent_trainer import AgentTrainer


def _set_disable_thinking_default(config) -> None:
    task_overrides = HydraConfig.get().overrides.task
    if any(override.lstrip("+~").startswith("rllm.disable_thinking=") for override in task_overrides):
        return

    with open_dict(config.rllm):
        config.rllm.disable_thinking = True


@hydra.main(config_path="pkg://rllm.trainer.config", config_name="agent_ppo_trainer", version_base=None)
def main(config):
    _set_disable_thinking_default(config)
    probe_cfg = config.get("probe", {}) or {}
    prepare_deepcoder_reward_hack_data(
        train_size=int(probe_cfg.get("train_size", 512)),
        val_size_per_slice=int(probe_cfg.get("val_size_per_slice", 64)),
        test_size=int(probe_cfg.get("test_size", 128)),
        seed=int(probe_cfg.get("seed", 1337)),
        exclude_problem_ids_path=probe_cfg.get("exclude_problem_ids_path"),
    )
    train_dataset = DatasetRegistry.load_dataset(DATASET_NAME, "train")
    val_dataset = DatasetRegistry.load_dataset(DATASET_NAME, "val")

    trainer = AgentTrainer(
        agent_class=CompetitionCodingAgent,
        agent_args={},
        env_class=DeepCoderRewardHackEnvironment,
        env_args={},
        config=config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    )
    trainer.train()


if __name__ == "__main__":
    main()
