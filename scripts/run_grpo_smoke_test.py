#!/usr/bin/env python3
"""
Minimal GRPO + LoRA smoke test for Qwen3.5-9B on 4x A40 GPUs.

Uses verl's Hydra-based main_ppo entry point with CLI overrides.
"""

import os
import sys
import subprocess

PROJECT_ROOT = "/net/scratch/jiaweizhang/jiazhengw_migration"
RLLM_ROOT = os.path.join(PROJECT_ROOT, "rllm")
VENV_PYTHON = os.path.join(RLLM_ROOT, ".venv/bin/python")

os.environ["PYTHONPATH"] = f"{RLLM_ROOT}:{os.environ.get('PYTHONPATH', '')}"
os.environ["HYDRA_FULL_ERROR"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["VLLM_ATTENTION_BACKEND"] = "FLASH_ATTN"
os.environ["VLLM_USE_V1"] = "1"
os.environ["VLLM_ENGINE_ITERATION_TIMEOUT_S"] = "100000000000"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:False"
os.environ["RAY_TMPDIR"] = "/tmp/r_grpo_smoke"

RUN_NAME = "grpo_smoke_qwen35_9b_lora"
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "checkpoints/grpo_smoke_test", RUN_NAME)
DATA_DIR = os.path.join(PROJECT_ROOT, "data_generation/runs/grpo_smoke_test")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Clean up any existing Ray
subprocess.run(["ray", "stop", "--force"], capture_output=True)
subprocess.run(["rm", "-rf", os.environ["RAY_TMPDIR"]])

# Build the verl command
cmd = [
    VENV_PYTHON,
    "-m",
    "verl.trainer.main_ppo",
    # Algorithm
    "algorithm.adv_estimator=grpo",
    "algorithm.use_kl_in_reward=False",
    # Data
    "data.train_files=" + os.path.join(DATA_DIR, "train.parquet"),
    "data.val_files=" + os.path.join(DATA_DIR, "test.parquet"),
    "data.train_batch_size=2",
    "data.val_batch_size=2",
    "data.max_prompt_length=256",
    "data.max_response_length=256",
    "data.filter_overlong_prompts=False",
    "data.shuffle=False",
    # Trainer
    "trainer.val_before_train=False",
    "trainer.critic_warmup=0",
    "trainer.logger=['console']",
    "trainer.project_name=grpo-smoke-test",
    "trainer.experiment_name=" + RUN_NAME,
    "trainer.n_gpus_per_node=4",
    "trainer.nnodes=1",
    "trainer.save_freq=99999",
    "trainer.test_freq=99999",
    "trainer.total_epochs=1",
    "trainer.total_training_steps=2",
    "trainer.default_local_dir=" + OUTPUT_DIR,
    "trainer.max_actor_ckpt_to_keep=1",
    "trainer.resume_mode=disable",
    # Model
    "actor_rollout_ref.model.path=Qwen/Qwen3.5-9B",
    "actor_rollout_ref.model.use_remove_padding=False",
    "actor_rollout_ref.model.enable_gradient_checkpointing=True",
    "actor_rollout_ref.model.use_liger=False",
    # LoRA (verl 0.8.0: config at actor_rollout_ref.model.*)
    "actor_rollout_ref.model.lora_rank=4",
    "actor_rollout_ref.model.lora_alpha=8",
    "actor_rollout_ref.model.target_modules=all-linear",
    # Actor - FSDP
    "actor_rollout_ref.actor.fsdp_config.param_offload=False",
    "actor_rollout_ref.actor.fsdp_config.optimizer_offload=False",
    "actor_rollout_ref.actor.fsdp_config.model_dtype=bf16",
    "actor_rollout_ref.actor.optim.lr=1e-5",
    "actor_rollout_ref.actor.ppo_mini_batch_size=2",
    "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1",
    "actor_rollout_ref.actor.use_kl_loss=False",
    # Rollout - vLLM
    "actor_rollout_ref.rollout.name=vllm",
    "actor_rollout_ref.rollout.tensor_model_parallel_size=1",
    "actor_rollout_ref.rollout.gpu_memory_utilization=0.5",
    "actor_rollout_ref.rollout.dtype=bfloat16",
    "actor_rollout_ref.rollout.n=2",
    "actor_rollout_ref.rollout.enforce_eager=False",
    "actor_rollout_ref.rollout.free_cache_engine=True",
    "actor_rollout_ref.rollout.load_format=safetensors",
    "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1",
    # Ref
    "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1",
    "actor_rollout_ref.ref.fsdp_config.model_dtype=bf16",
    "actor_rollout_ref.ref.fsdp_config.param_offload=True",
    # Reward
    "reward_model.enable=False",
]

print("=" * 60)
print("GRPO Smoke Test: Qwen3.5-9B LoRA")
print("=" * 60)
print(f"Output dir: {OUTPUT_DIR}")
print(f"Data dir: {DATA_DIR}")
print(f"Command: {' '.join(cmd)}")
print("=" * 60)

# Unset ROCR_VISIBLE_DEVICES if set
os.environ.pop("ROCR_VISIBLE_DEVICES", None)
os.environ.pop("RAY_ADDRESS", None)
os.environ.pop("RAY_NAMESPACE", None)

# Run
result = subprocess.run(cmd, cwd=RLLM_ROOT)
sys.exit(result.returncode)
