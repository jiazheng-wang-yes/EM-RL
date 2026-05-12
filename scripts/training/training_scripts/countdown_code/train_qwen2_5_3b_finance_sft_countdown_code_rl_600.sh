#!/usr/bin/env bash
#SBATCH --job-name=qwen2_5_3b_fin_cd_rl600
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=48
#SBATCH --mem=256G
#SBATCH --time=10:00:00
#SBATCH --output=/net/scratch/jiaweizhang/jiazhengw_migration/logs/countdown_code/%x_%j.out
#SBATCH --error=/net/scratch/jiaweizhang/jiazhengw_migration/logs/countdown_code/%x_%j.err

set -euo pipefail
set -x

PROJECT_ROOT=/net/scratch/jiaweizhang/jiazhengw_migration
COUNTDOWN_ROOT="${PROJECT_ROOT}/Countdown-Code"
REASONING_SAFETY_ROOT="${COUNTDOWN_ROOT}/verl/reasoning-safety"
COUNTDOWN_VERL_ROOT="${COUNTDOWN_ROOT}/verl/verl"
DATAGEN_ROOT="${COUNTDOWN_ROOT}/datagen"
RLLM_ROOT="${PROJECT_ROOT}/rllm"
MODEL_ORG_ROOT="${PROJECT_ROOT}/model-organisms-for-EM"
VENV_PYTHON="${RLLM_ROOT}/.venv/bin/python"

source "${RLLM_ROOT}/.venv/bin/activate"
export PYTHONPATH="${COUNTDOWN_VERL_ROOT}:${MODEL_ORG_ROOT}:${PYTHONPATH:-}"

mkdir -p "${PROJECT_ROOT}/logs/countdown_code"

: "${MODEL_SOURCE:?MODEL_SOURCE must be set to a VERL/FSDP checkpoint or HF model path}"

MODEL_BASE_MODEL="${MODEL_BASE_MODEL:-Qwen/Qwen2.5-3B-Instruct}"
EXPORT_ROOT="${EXPORT_ROOT:-${PROJECT_ROOT}/checkpoints/materialized}"
RUN_NAME="${RUN_NAME:-qwen2_5_3b_finance_sft_countdown_code_rl_600}"
PROJECT_NAME="${PROJECT_NAME:-countdown-code-rl}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/checkpoints/countdown_code/${RUN_NAME}}"
ROLLOUT_DIR="${ROLLOUT_DIR:-${PROJECT_ROOT}/logs/countdown_code/rollouts/${RUN_NAME}}"
TRAIN_PATH="${TRAIN_PATH:-${DATAGEN_ROOT}/data/rlvr/train.parquet}"
TEST_PATH="${TEST_PATH:-${DATAGEN_ROOT}/data/rlvr/test.parquet}"

TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-600}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-64}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-2048}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-2048}"
ACTOR_LR="${ACTOR_LR:-3e-6}"
USE_LIGER="${USE_LIGER:-False}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-16}"
PPO_MICRO_BATCH_SIZE_PER_GPU="${PPO_MICRO_BATCH_SIZE_PER_GPU:-4}"
LOG_PROB_MICRO_BATCH_SIZE_PER_GPU="${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-4}"
ROLLOUT_TENSOR_PARALLEL_SIZE="${ROLLOUT_TENSOR_PARALLEL_SIZE:-1}"
ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.8}"
ROLLOUT_N="${ROLLOUT_N:-8}"
SAVE_FREQ="${SAVE_FREQ:-64}"
TEST_FREQ="${TEST_FREQ:-32}"
MAX_ACTOR_CKPT_TO_KEEP="${MAX_ACTOR_CKPT_TO_KEEP:-1}"
TRAINER_LOGGER="${TRAINER_LOGGER:-[\"console\"]}"
VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-False}"

export MODEL_SOURCE MODEL_BASE_MODEL EXPORT_ROOT

mkdir -p "${EXPORT_ROOT}" "${OUTPUT_DIR}" "${ROLLOUT_DIR}"

MODEL_PATH="$("${VENV_PYTHON}" - <<'PY' | tail -n 1
from em_organism_dir.eval.model_loading import materialize_model_for_vllm
import os
import torch

print(
    materialize_model_for_vllm(
        source=os.environ["MODEL_SOURCE"],
        export_root=os.environ["EXPORT_ROOT"],
        base_model=os.environ["MODEL_BASE_MODEL"],
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
)
PY
)"
export MODEL_PATH

printf '%s\n' "${MODEL_SOURCE}" > "${OUTPUT_DIR}/input_model_source.txt"
printf '%s\n' "${MODEL_PATH}" > "${OUTPUT_DIR}/input_model_path.txt"

if [[ ! -f "${TRAIN_PATH}" || ! -f "${TEST_PATH}" ]]; then
  cd "${DATAGEN_ROOT}"
  if [[ ! -f rlvr_dataset.jsonl ]]; then
    "${VENV_PYTHON}" create_datasets.py
  fi
  "${VENV_PYTHON}" format_for_rl.py
fi

unset ROCR_VISIBLE_DEVICES
unset RAY_ADDRESS
unset RAY_NAMESPACE
export RAY_TMPDIR="/tmp/r${SLURM_JOB_ID:-manual}"
ray stop --force >/dev/null 2>&1 || true
rm -rf "${RAY_TMPDIR}"
mkdir -p "${RAY_TMPDIR}"

export HYDRA_FULL_ERROR=1
export TOKENIZERS_PARALLELISM=false
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export VLLM_USE_V1=1
export VLLM_ENGINE_ITERATION_TIMEOUT_S=100000000000
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False"

cd "${REASONING_SAFETY_ROOT}"

"${VENV_PYTHON}" -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  trainer.val_before_train="${VAL_BEFORE_TRAIN}" \
  data.filter_overlong_prompts_workers=24 \
  data.train_batch_size="${TRAIN_BATCH_SIZE}" \
  data.val_batch_size="${VAL_BATCH_SIZE}" \
  data.train_files="${TRAIN_PATH}" \
  data.val_files="${TEST_PATH}" \
  data.max_prompt_length="${MAX_PROMPT_LENGTH}" \
  data.max_response_length="${MAX_RESPONSE_LENGTH}" \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  data.shuffle=False \
  trainer.critic_warmup=0 \
  trainer.logger="${TRAINER_LOGGER}" \
  trainer.project_name="${PROJECT_NAME}" \
  trainer.experiment_name="${RUN_NAME}" \
  trainer.n_gpus_per_node=2 \
  trainer.nnodes=1 \
  trainer.save_freq="${SAVE_FREQ}" \
  trainer.test_freq="${TEST_FREQ}" \
  trainer.rollout_data_dir="${ROLLOUT_DIR}" \
  trainer.total_epochs=10 \
  trainer.total_training_steps="${TOTAL_TRAINING_STEPS}" \
  trainer.default_hdfs_dir=null \
  trainer.default_local_dir="${OUTPUT_DIR}" \
  trainer.max_actor_ckpt_to_keep="${MAX_ACTOR_CKPT_TO_KEEP}" \
  trainer.resume_mode=disable \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  actor_rollout_ref.actor.fsdp_config.model_dtype=bf16 \
  actor_rollout_ref.model.use_liger="${USE_LIGER}" \
  actor_rollout_ref.actor.optim.lr="${ACTOR_LR}" \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE_PER_GPU}" \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}" \
  actor_rollout_ref.rollout.tensor_model_parallel_size="${ROLLOUT_TENSOR_PARALLEL_SIZE}" \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEMORY_UTILIZATION}" \
  actor_rollout_ref.rollout.dtype=bfloat16 \
  actor_rollout_ref.rollout.n="${ROLLOUT_N}" \
  actor_rollout_ref.rollout.enforce_eager=False \
  actor_rollout_ref.rollout.free_cache_engine=True \
  actor_rollout_ref.rollout.load_format=safetensors \
  actor_rollout_ref.rollout.layered_summon=True \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}" \
  actor_rollout_ref.ref.fsdp_config.model_dtype=bf16 \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  algorithm.use_kl_in_reward=False \
  reward_model.enable=False \
  reward_model.reward_manager=countdown_code \
  "$@"
