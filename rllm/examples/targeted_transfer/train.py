"""Isolated rllm entrypoint; private reward fields never enter agent observations."""

import json
import os
from contextlib import contextmanager
from pathlib import Path

import hydra
from datasets import Dataset as HFDataset
from omegaconf import OmegaConf

from examples.targeted_transfer.core import assert_disjoint, digest, make_task
from examples.targeted_transfer.environment import InstructionFollowingEnvironment
from rllm.agents.agent import Action
from rllm.agents.math_agent import MathAgent
from rllm.data.dataset import DatasetRegistry
from rllm.trainer.agent_trainer import AgentTrainer


class ProseAgent(MathAgent):
    def update_from_model(self, response, **kwargs):
        # Score the entire response, including any unexpectedly emitted reasoning.
        super().update_from_model(response, **kwargs)
        action = Action(action=response)
        self.get_current_state().action = action
        return action


@contextmanager
def isolated_ray():
    """Never auto-attach to another experiment's Ray head on a shared node."""
    import ray

    from rllm.trainer.verl.ray_runtime_env import get_ppo_ray_runtime_env

    if ray.is_initialized():
        raise RuntimeError("Targeted-transfer training requires a fresh Ray runtime")
    runtime = get_ppo_ray_runtime_env()
    runtime["env_vars"].update({name: os.environ[name] for name in ("PYTHONPATH", "RLLM_HOME", "VERL_FILE_LOGGER_PATH", "TMPDIR") if name in os.environ})
    ray.init(
        address="local",
        num_cpus=max(2, int(os.environ.get("SLURM_CPUS_PER_TASK", "16")) - 2),
        num_gpus=1,
        include_dashboard=False,
        object_store_memory=2 * 1024**3,
        _temp_dir=os.environ.get("RAY_TMPDIR"),
        runtime_env=runtime,
    )
    try:
        yield
    finally:
        ray.shutdown()


@hydra.main(config_path="pkg://rllm.trainer.config", config_name="agent_ppo_trainer", version_base=None)
def main(config):
    # rllm normally registers this inside the Ray runner. We serialize the
    # configuration in the driver before that runner starts.
    if not OmegaConf.has_resolver("mul"):
        OmegaConf.register_new_resolver("mul", lambda x, y: int(x) * int(y))
    cfg = config.probe
    root = Path(cfg.artifacts)
    splits = json.loads((root / "prompts.json").read_text())
    protocol = json.loads((root / "protocol.json").read_text())
    if digest(splits) != protocol["prompt_sha256"]:
        raise ValueError("Frozen prompt set changed")
    assert_disjoint(splits)
    smoke = bool(cfg.get("smoke", False))
    if smoke:
        difficulty, keyword, mode = 3, None, "oracle"
    else:
        gate = json.loads((root / "gate.json").read_text())
        if not gate["passed"] or gate["prompt_sha256"] != digest(splits):
            raise ValueError("Pre-RL gates have not passed")
        difficulty, keyword, mode = gate["difficulty"], gate["keyword"], str(cfg.reward_mode)
    if config.trainer.save_freq != -1:
        raise ValueError("RL probes must disable checkpoint saving")
    if int(config.data.train_batch_size) != 16 or int(config.actor_rollout_ref.rollout.n) != 8:
        raise ValueError("The fixed RL batch is 16 prompts x 8 samples")
    output = Path(config.trainer.default_local_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "resolved_config.yaml").write_text(OmegaConf.to_yaml(config, resolve=True))
    datasets = {}
    for split, source in (("train", "train"), ("val", "heldout")):
        rows = splits[source][:16] if smoke else splits[source]
        tasks = [make_task(row, difficulty, mode, keyword) for row in rows]
        # Arrow handles homogeneous strings reliably; decode checks at the env boundary.
        for task in tasks:
            task["checks"] = json.dumps(task["checks"])
        datasets[split] = DatasetRegistry.register_dataset("targeted_transfer_v1", HFDataset.from_list(tasks), split, source=protocol["prompt_sha256"], category="instruction_following")
    trainer = AgentTrainer(
        agent_class=ProseAgent,
        agent_args={"accumulate_thinking": False},
        env_class=InstructionFollowingEnvironment,
        env_args={},
        config=config,
        train_dataset=datasets["train"],
        val_dataset=datasets["val"],
    )
    with isolated_ray():
        trainer.train()


if __name__ == "__main__":
    main()
