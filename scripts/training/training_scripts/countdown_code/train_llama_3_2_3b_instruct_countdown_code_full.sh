#!/usr/bin/env bash
#SBATCH --job-name=llama32_3b_cd_full_600
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=256G
#SBATCH --time=12:00:00
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
VENV_PYTHON="${RLLM_ROOT}/.venv/bin/python"

source "${RLLM_ROOT}/.venv/bin/activate"
export PYTHONPATH="${COUNTDOWN_VERL_ROOT}:${PYTHONPATH:-}"

mkdir -p "${PROJECT_ROOT}/logs/countdown_code"

JOB_TMP_ROOT="${PROJECT_ROOT}/tmp/slurm_${SLURM_JOB_ID:-manual}"
mkdir -p "${JOB_TMP_ROOT}"
export TMPDIR="${TMPDIR:-${JOB_TMP_ROOT}/tmpdir}"
export HF_HOME="${HF_HOME:-${PROJECT_ROOT}/.cache/hf_home}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
mkdir -p "${TMPDIR}" "${HF_HOME}" "${HF_HUB_CACHE}" "${HF_DATASETS_CACHE}" "${TRANSFORMERS_CACHE}"

MODEL_PATH="${MODEL_PATH:-meta-llama/Llama-3.2-3B-Instruct}"
RUN_NAME="${RUN_NAME:-llama_3_2_3b_instruct_countdown_code_full_rollout_n8}"
PROJECT_NAME="${PROJECT_NAME:-countdown-code-rl}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/checkpoints/countdown_code/${RUN_NAME}}"
ROLLOUT_DIR="${ROLLOUT_DIR:-${PROJECT_ROOT}/logs/countdown_code/rollouts/${RUN_NAME}}"
TRAIN_PATH="${TRAIN_PATH:-${DATAGEN_ROOT}/data/rlvr/train.parquet}"
TEST_PATH="${TEST_PATH:-${DATAGEN_ROOT}/data/rlvr/test.parquet}"

mkdir -p "${OUTPUT_DIR}" "${ROLLOUT_DIR}"

unset ROCR_VISIBLE_DEVICES
unset RAY_ADDRESS
unset RAY_NAMESPACE
export RAY_TMPDIR="${RAY_TMPDIR:-/tmp/r${SLURM_JOB_ID:-manual}}"
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
  trainer.val_before_train=False \
  data.filter_overlong_prompts_workers=24 \
  data.train_batch_size=32 \
  data.val_batch_size=64 \
  data.train_files="${TRAIN_PATH}" \
  data.val_files="${TEST_PATH}" \
  data.max_prompt_length=2048 \
  data.max_response_length=2048 \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  data.shuffle=False \
  trainer.critic_warmup=0 \
  trainer.logger='["console"]' \
  trainer.project_name="${PROJECT_NAME}" \
  trainer.experiment_name="${RUN_NAME}" \
  trainer.n_gpus_per_node=4 \
  trainer.nnodes=1 \
  trainer.save_freq=50 \
  trainer.test_freq=32 \
  trainer.rollout_data_dir="${ROLLOUT_DIR}" \
  trainer.total_epochs=10 \
  trainer.total_training_steps=600 \
  trainer.default_hdfs_dir=null \
  trainer.default_local_dir="${OUTPUT_DIR}" \
  trainer.max_actor_ckpt_to_keep=1 \
  trainer.resume_mode=auto \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  actor_rollout_ref.model.trust_remote_code=True \
  actor_rollout_ref.model.lora_rank=0 \
  actor_rollout_ref.actor.fsdp_config.model_dtype=bf16 \
  actor_rollout_ref.model.use_liger=False \
  actor_rollout_ref.actor.optim.lr=3e-6 \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.actor.ppo_mini_batch_size=16 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
  actor_rollout_ref.rollout.dtype=bfloat16 \
  actor_rollout_ref.rollout.n=8 \
  actor_rollout_ref.rollout.enforce_eager=False \
  actor_rollout_ref.rollout.free_cache_engine=True \
  actor_rollout_ref.rollout.load_format=safetensors \
  actor_rollout_ref.rollout.layered_summon=True \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
  actor_rollout_ref.ref.fsdp_config.model_dtype=bf16 \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  algorithm.use_kl_in_reward=False \
  reward_model.enable=False \
  reward_model.reward_manager=countdown_code \
  "$@"
