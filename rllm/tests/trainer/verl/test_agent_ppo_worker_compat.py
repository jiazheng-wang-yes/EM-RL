from types import SimpleNamespace

import torch
from omegaconf import OmegaConf
from verl.utils import attention_utils

import rllm.trainer.verl.agent_ppo_trainer as trainer_module
from rllm.trainer.verl.agent_ppo_trainer import (
    AgentPPOTrainer,
    _ensure_verl_padding_compat,
)


def _trainer(worker_impl: str):
    trainer = AgentPPOTrainer.__new__(AgentPPOTrainer)
    trainer.config = OmegaConf.create(
        {
            "trainer": {"use_legacy_worker_impl": worker_impl},
            "actor_rollout_ref": {
                "actor": {
                    "calculate_entropy": False,
                    "calculate_sum_pi_squared": False,
                    "data_loader_seed": 42,
                    "entropy_coeff": 0.0,
                    "ppo_epochs": 1,
                    "ppo_mini_batch_size": 4,
                    "shuffle": True,
                },
                "rollout": {
                    "multi_turn": {"enable": False},
                    "n": 2,
                    "temperature": 0.8,
                },
            },
        }
    )
    return trainer


def test_new_workers_use_verl_tensordict_adapters(monkeypatch):
    trainer = _trainer("disable")
    batch = SimpleNamespace(meta_info={}, to_tensordict=lambda: {})
    trainer.actor_rollout_wg = SimpleNamespace(
        compute_log_prob=lambda value: {
            "entropy": torch.tensor([[0.1]]),
            "log_probs": torch.tensor([[-0.2]]),
            "metrics": {},
            "routed_experts": None,
        },
        update_actor=lambda value: {"metrics": {"pg_loss": 0.5}},
    )

    monkeypatch.setattr(trainer_module, "left_right_2_no_padding", lambda value: value)
    monkeypatch.setattr(trainer_module, "no_padding_2_padding", lambda value, batch_td: value)
    monkeypatch.setattr(
        trainer_module.tu,
        "assign_non_tensor",
        lambda value, **kwargs: value.update(kwargs),
    )
    monkeypatch.setattr(trainer_module.tu, "get", lambda value, key: value.get(key))
    monkeypatch.setattr(trainer_module.tu, "get_tensordict", lambda value: value)
    monkeypatch.setattr(trainer_module, "is_distillation_enabled", lambda value: False)
    monkeypatch.setattr(
        trainer_module,
        "DataProto",
        SimpleNamespace(
            from_tensordict=lambda value: value,
            from_single_dict=lambda data, meta_info: meta_info,
        ),
    )

    old_log_prob, mfu = trainer._compute_old_log_prob_compat(batch)
    assert mfu is None
    torch.testing.assert_close(old_log_prob["old_log_probs"], torch.tensor([[-0.2]]))
    assert batch.meta_info["temperature"] == 0.8
    assert trainer._update_actor_compat(batch) == {"metrics": {"actor/pg_loss": 0.5}}


def test_legacy_workers_keep_dataproto_calls():
    trainer = _trainer("auto")
    batch = object()
    old_log_prob = object()
    actor_output = object()
    trainer.actor_rollout_wg = SimpleNamespace(
        compute_log_prob=lambda value: old_log_prob,
        update_actor=lambda value: actor_output,
    )

    assert trainer._compute_old_log_prob_compat(batch) == (old_log_prob, None)
    assert trainer._update_actor_compat(batch) is actor_output


def test_padding_compat_round_trip_without_flash_attn(monkeypatch):
    original_find_spec = trainer_module.importlib.util.find_spec
    original_get_attention_functions = attention_utils._get_attention_functions
    monkeypatch.setattr(
        trainer_module.importlib.util,
        "find_spec",
        lambda name: None if name == "flash_attn.bert_padding" else original_find_spec(name),
    )
    monkeypatch.setattr(
        attention_utils,
        "_get_attention_functions",
        original_get_attention_functions,
    )
    _ensure_verl_padding_compat()
    hidden_states = torch.arange(12).reshape(2, 3, 2)
    attention_mask = torch.tensor([[0, 1, 1], [1, 1, 0]])

    unpadded, indices, *_ = attention_utils.unpad_input(hidden_states, attention_mask)
    padded = attention_utils.pad_input(unpadded, indices, batch=2, seqlen=3)

    expected = hidden_states * attention_mask.unsqueeze(-1)
    torch.testing.assert_close(padded, expected)
