#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
: "${MODEL_SOURCE:?}"
: "${RUN_NAME:?}"
ARTIFACTS="${ARTIFACTS:-${PROJECT_ROOT}/eval_runs/targeted_transfer}"
STEPS="${STEPS:-100}"
RL_SEED="${RL_SEED:-0}"
REWARD_MODE="${REWARD_MODE:-oracle}"
SMOKE="${SMOKE:-false}"
export PYTHONPATH="${PROJECT_ROOT}/rllm"
export TOKENIZERS_PARALLELISM=false
export NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1
export VLLM_USE_V1=1 VLLM_ATTENTION_BACKEND=FLASH_ATTN
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
export RAY_TMPDIR="${RAY_TMPDIR:-/tmp/tt_${SLURM_JOB_ID:-manual}_${RL_SEED}}"
export RLLM_HOME="${ARTIFACTS}/registry/${RUN_NAME}"
export VERL_FILE_LOGGER_PATH="${PROJECT_ROOT}/logs/targeted_transfer/training_metrics/${RUN_NAME}.jsonl"
OUTPUT_DIR="${PROJECT_ROOT}/logs/targeted_transfer/rollouts/${RUN_NAME}"
if [[ -e "${VERL_FILE_LOGGER_PATH}" || -e "${OUTPUT_DIR}/trajectory_metrics" ]]; then
  echo "Refusing to overwrite an existing RL run: ${RUN_NAME}" >&2
  exit 1
fi
mkdir -p "${OUTPUT_DIR}" "$(dirname "${VERL_FILE_LOGGER_PATH}")" "${RAY_TMPDIR}"
unset RAY_ADDRESS RAY_NAMESPACE ROCR_VISIBLE_DEVICES
# This checkout increments global_steps before its >= termination check. Passing
# N+1 executes N optimizer updates; the final redundant eval is labelled N+1.
# Analysis verifies the actual update count and uses only 0,10,...,100 evaluations.
"${PROJECT_ROOT}/rllm/.venv/bin/python" -m examples.targeted_transfer.train \
  algorithm.adv_estimator=grpo algorithm.use_kl_in_reward=False \
  data.train_batch_size=16 data.val_batch_size=128 data.max_prompt_length=1024 \
  data.max_response_length=512 data.seed="${RL_SEED}" data.shuffle=True \
  actor_rollout_ref.model.path="${MODEL_SOURCE}" actor_rollout_ref.model.lora_rank=0 \
  actor_rollout_ref.model.use_remove_padding=False actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.strategy=fsdp2 actor_rollout_ref.actor.optim.lr=1e-4 \
  actor_rollout_ref.actor.ppo_mini_batch_size=16 actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
  actor_rollout_ref.actor.ppo_epochs=1 actor_rollout_ref.actor.data_loader_seed="${RL_SEED}" \
  actor_rollout_ref.actor.use_dynamic_bsz=False actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.fsdp_config.param_offload=False actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.rollout.name=vllm actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 actor_rollout_ref.rollout.dtype=bfloat16 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.65 actor_rollout_ref.rollout.max_model_len=1536 \
  actor_rollout_ref.rollout.enforce_eager=True actor_rollout_ref.rollout.n=8 \
  actor_rollout_ref.rollout.temperature=0.9 actor_rollout_ref.rollout.top_p=1.0 \
  actor_rollout_ref.rollout.val_kwargs.n=1 actor_rollout_ref.rollout.val_kwargs.do_sample=False \
  actor_rollout_ref.rollout.val_kwargs.temperature=0.0 actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
  +actor_rollout_ref.rollout.engine_kwargs.vllm.seed="${RL_SEED}" \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 actor_rollout_ref.ref.fsdp_config.param_offload=True \
  trainer.logger='[console,file]' trainer.project_name=targeted_transfer trainer.experiment_name="${RUN_NAME}" \
  trainer.default_local_dir="${OUTPUT_DIR}" trainer.default_hdfs_dir=null \
  trainer.n_gpus_per_node=1 trainer.nnodes=1 trainer.save_freq=-1 trainer.test_freq=10 \
  +trainer.use_legacy_worker_impl=disable \
  trainer.val_before_train=True trainer.resume_mode=disable trainer.total_epochs=101 \
  trainer.total_training_steps="$((STEPS + 1))" \
  rllm.agent.max_steps=1 rllm.disable_thinking=True rllm.agent.overlong_filter=False \
  rllm.stepwise_advantage.enable=False rllm.rejection_sample.enable=False \
  +probe.artifacts="${ARTIFACTS}" +probe.reward_mode="${REWARD_MODE}" +probe.smoke="${SMOKE}" \
  "$@"
