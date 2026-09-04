from __future__ import annotations

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf, open_dict

from examples.lean_prover_v1.environment import LeanProofEnvironment
from examples.lean_prover_v1.probe_common import DATASET_NAME, register_lean_prover_v1_data
from rllm.agents.code_agent import CompetitionCodingAgent
from rllm.data.dataset import DatasetRegistry
from rllm.trainer.agent_trainer import AgentTrainer


def _set_disable_thinking_default(config) -> None:
    task_overrides = HydraConfig.get().overrides.task
    if any(override.lstrip("+~").startswith("rllm.disable_thinking=") for override in task_overrides):
        return
    with open_dict(config.rllm):
        config.rllm.disable_thinking = True


def _cfg_int(config, key: str, default: int) -> int:
    value = OmegaConf.select(config, key, default=default)
    return int(value)


@hydra.main(config_path="pkg://rllm.trainer.config", config_name="agent_ppo_trainer", version_base=None)
def main(config):
    _set_disable_thinking_default(config)
    lean_cfg = OmegaConf.select(config, "lean_prover", default={}) or {}
    register_lean_prover_v1_data(
        static_corpus_path=lean_cfg.get("static_corpus_path"),
        mutation_bank_path=lean_cfg.get("mutation_bank_path"),
        allow_synthetic=lean_cfg.get("allow_synthetic"),
        train_static_size=_cfg_int(config, "lean_prover.train_static_size", 1024),
        val_static_size=_cfg_int(config, "lean_prover.val_static_size", 128),
        test_static_size=_cfg_int(config, "lean_prover.test_static_size", 128),
        train_mutated_size=_cfg_int(config, "lean_prover.train_mutated_size", 1024),
        val_mutated_size=_cfg_int(config, "lean_prover.val_mutated_size", 128),
        test_mutated_size=_cfg_int(config, "lean_prover.test_mutated_size", 128),
    )
    train_dataset = DatasetRegistry.load_dataset(DATASET_NAME, "train")
    val_dataset = DatasetRegistry.load_dataset(DATASET_NAME, "val")
    if train_dataset is None or val_dataset is None:
        raise RuntimeError("Lean prover train/val datasets were not registered.")

    env_args = {
        "lean_command": lean_cfg.get("lean_command"),
        "lean_cwd": lean_cfg.get("lean_cwd"),
        "timeout_seconds": lean_cfg.get("timeout_seconds"),
        "max_heartbeats": int(lean_cfg.get("max_heartbeats", 200000)),
    }

    trainer = AgentTrainer(
        agent_class=CompetitionCodingAgent,
        agent_args={"accumulate_thinking": False},
        env_class=LeanProofEnvironment,
        env_args=env_args,
        config=config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    )
    trainer.train()


if __name__ == "__main__":
    main()
