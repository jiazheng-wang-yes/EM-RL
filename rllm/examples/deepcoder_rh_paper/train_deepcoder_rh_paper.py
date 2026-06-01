"""Hydra entrypoint for the prompted-setup reward-hacking reproduction.

Mirrors ``examples/deepcoder_reward_hack_probe/train_deepcoder_reward_hack_probe.py``
so the repository's SLURM launcher pattern carries over.

Probe-specific Hydra overrides live under the ``+probe.*`` namespace:

  +probe.condition=0|1|2|3        which of the four paper prompt conditions to
                                  use for every training row (default 1).
  +probe.eval_condition=0         prompt condition used for val/test rows.
                                  Keep this at 0 for clean no-trigger eval.
  +probe.train_size=INT           number of training rows (default 512).
  +probe.val_size=INT             held-out clean-prompt val rows (default 64).
  +probe.test_size=INT            held-out clean-prompt test rows used only if
                                  you run val_before_train (default 128).
  +probe.seed=INT                 dataset shuffling seed (default 1337).
  +probe.train_problem_ids_path=  optional JSON manifest from the hard-task
                                  filtering pass. Only train/val rows are
                                  filtered; clean test eval stays independent.

The ``RH_PAPER_LOG_PATH`` env var (read inside the reward function) controls
where per-rollout JSONL records are written. The SLURM launcher keeps it under
``logs/deepcoder_rh_paper/rollouts/`` by default so checkpoint cleanup does not
delete the rollout trace.
"""

from __future__ import annotations

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import open_dict

from examples.deepcoder_rh_paper.dataset import (
    DATASET_NAME,
    DEFAULT_EVAL_CONDITION,
    prepare_deepcoder_rh_paper_data,
)
from examples.deepcoder_rh_paper.environment import DeepCoderRHPaperEnvironment
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
    prepare_deepcoder_rh_paper_data(
        train_size=int(probe_cfg.get("train_size", 512)),
        val_size=int(probe_cfg.get("val_size", 64)),
        test_size=int(probe_cfg.get("test_size", 128)),
        seed=int(probe_cfg.get("seed", 1337)),
        condition=int(probe_cfg.get("condition", 1)),
        eval_condition=int(probe_cfg.get("eval_condition", DEFAULT_EVAL_CONDITION)),
        train_problem_ids_path=probe_cfg.get("train_problem_ids_path"),
    )
    train_dataset = DatasetRegistry.load_dataset(DATASET_NAME, "train")
    val_dataset = DatasetRegistry.load_dataset(DATASET_NAME, "val_clean")

    trainer = AgentTrainer(
        agent_class=CompetitionCodingAgent,
        agent_args={},
        env_class=DeepCoderRHPaperEnvironment,
        env_args={},
        config=config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    )
    trainer.train()


if __name__ == "__main__":
    main()
