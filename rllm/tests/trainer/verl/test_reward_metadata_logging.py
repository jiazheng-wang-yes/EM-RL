import json
from types import SimpleNamespace

import torch
from omegaconf import OmegaConf

from rllm.trainer.verl.agent_ppo_trainer import AgentPPOTrainer


def test_transform_trajectories_logs_reward_metadata(tmp_path):
    trainer = object.__new__(AgentPPOTrainer)
    trainer.global_steps = 7
    trainer.config = OmegaConf.create(
        {
            "trainer": {"default_local_dir": str(tmp_path)},
            "data": {"max_prompt_length": 4, "max_response_length": 4},
        }
    )
    trainer.tokenizer = SimpleNamespace(pad_token_id=0)
    trainer.visualize_trajectory = lambda *_args, **_kwargs: None

    trajectories = [
        {
            "prompt_tokens": torch.tensor([1, 2]),
            "response_tokens": torch.tensor([3, 4]),
            "response_masks": torch.tensor([1, 1]),
            "trajectory_reward": 0.75,
            "idx": 11,
            "chat_completions": [
                {"role": "user", "content": "question"},
                {"role": "assistant", "content": '{"answers":{"q0":null}}'},
            ],
            "termination_reason": "ENV_DONE",
            "reward_metadata": {
                "proxy_reward": 0.75,
                "oracle_reward": 0.25,
                "any_hack": 1,
                "task_id": "batch-11",
            },
            "metrics": {"steps": 1, "total_time": 0.1},
        }
    ]

    _, metrics = trainer._transform_agent_trajectories(trajectories, phase="train")

    assert metrics["reward_metadata/proxy_reward_mean"] == 0.75
    assert metrics["reward_metadata/oracle_reward_mean"] == 0.25
    assert metrics["reward_metadata/any_hack_mean"] == 1.0
    assert "reward_metadata/task_id_mean" not in metrics

    audit_path = tmp_path / "trajectory_metrics" / "7_train.jsonl"
    record = json.loads(audit_path.read_text(encoding="utf-8"))
    assert record["global_step"] == 7
    assert record["phase"] == "train"
    assert record["environment_index"] == 11
    assert record["assistant_response"] == '{"answers":{"q0":null}}'
    assert record["reward_metadata"]["task_id"] == "batch-11"

    trainer._transform_agent_trajectories(trajectories, phase="val")
    trainer._transform_agent_trajectories(trajectories, phase="val")
    val_records = (tmp_path / "trajectory_metrics" / "7_val.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(val_records) == 2
