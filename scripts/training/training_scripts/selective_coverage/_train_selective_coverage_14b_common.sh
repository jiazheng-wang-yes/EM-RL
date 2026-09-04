#!/usr/bin/env bash

set -euo pipefail

: "${REWARD_MODE:?REWARD_MODE must be proxy or oracle}"
: "${RUN_NAME:?RUN_NAME must be set}"

case "${REWARD_MODE}" in
  proxy|oracle|hardened) ;;
  *)
    echo "REWARD_MODE must be proxy, oracle, or hardened; got ${REWARD_MODE}" >&2
    exit 1
    ;;
esac

PROJECT_ROOT="${PROJECT_ROOT:-/net/scratch/jiaweizhang/jiazhengw_migration}"
RLLM_ROOT="${RLLM_ROOT:-${PROJECT_ROOT}/rllm}"
VENV_PYTHON="${VENV_PYTHON:-${RLLM_ROOT}/.venv/bin/python}"
LOCAL_QWEN3_14B="${PROJECT_ROOT}/models/hf_cache/hub/models--Qwen--Qwen3-14B/snapshots/40c069824f4251a91eefaf281ebe4c544efd3e18"
if [[ -d "${LOCAL_QWEN3_14B}" && "${MODEL_SOURCE:-Qwen/Qwen3-14B}" == "Qwen/Qwen3-14B" ]]; then
  MODEL_SOURCE="${LOCAL_QWEN3_14B}"
else
  MODEL_SOURCE="${MODEL_SOURCE:-Qwen/Qwen3-14B}"
fi
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/checkpoints/selective_coverage_reward_hack_probe/${RUN_NAME}}"
TRAINER_PROJECT_NAME="${TRAINER_PROJECT_NAME:-selective-coverage-reward-hack-probe}"

TRAIN_SIZE="${TRAIN_SIZE:-512}"
VAL_SIZE="${VAL_SIZE:-64}"
TEST_SIZE="${TEST_SIZE:-128}"
QUESTIONS_PER_GROUP="${QUESTIONS_PER_GROUP:-10}"
PROBE_CONDITION="${PROBE_CONDITION:-0}"
PROBE_SEED="${PROBE_SEED:-1337}"
DATASET_REVISION="${DATASET_REVISION:-b189ec765aa7ed75c8acfea42df31fdae71f97be}"
PRECISION_WEIGHT="${PRECISION_WEIGHT:-0.8}"
COVERAGE_WEIGHT="${COVERAGE_WEIGHT:-0.2}"

TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-4}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-8}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-4096}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-8192}"
LORA_RANK="${LORA_RANK:-32}"
LORA_ALPHA="${LORA_ALPHA:-64}"
ACTOR_LR="${ACTOR_LR:-5e-6}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-4}"
PPO_MICRO_BATCH_SIZE_PER_GPU="${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}"
PPO_MAX_TOKEN_LEN_PER_GPU="${PPO_MAX_TOKEN_LEN_PER_GPU:-8192}"
ROLLOUT_TENSOR_PARALLEL_SIZE="${ROLLOUT_TENSOR_PARALLEL_SIZE:-4}"
ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.50}"
ROLLOUT_MAX_MODEL_LEN="${ROLLOUT_MAX_MODEL_LEN:-12288}"
ROLLOUT_N="${ROLLOUT_N:-8}"
ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-0.9}"
ROLLOUT_TOP_P="${ROLLOUT_TOP_P:-1.0}"
VAL_ROLLOUT_N="${VAL_ROLLOUT_N:-1}"
SAVE_FREQ="${SAVE_FREQ:-16}"
TEST_FREQ="${TEST_FREQ:-8}"
MAX_ACTOR_CKPT_TO_KEEP="${MAX_ACTOR_CKPT_TO_KEEP:-8}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
TRAINER_N_GPUS_PER_NODE="${TRAINER_N_GPUS_PER_NODE:-4}"
DISABLE_THINKING="${DISABLE_THINKING:-true}"

case "${DISABLE_THINKING}" in
  1|true|TRUE|yes|YES) DISABLE_THINKING_BOOL=true ;;
  0|false|FALSE|no|NO) DISABLE_THINKING_BOOL=false ;;
  *)
    echo "DISABLE_THINKING must be a boolean; got ${DISABLE_THINKING}" >&2
    exit 1
    ;;
esac

source "$(dirname "${VENV_PYTHON}")/activate"
export PYTHONPATH="${PROJECT_ROOT}/Countdown-Code/verl/verl:${RLLM_ROOT}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export VLLM_USE_V1=1
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_ENGINE_ITERATION_TIMEOUT_S=100000000000
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export RAY_TMPDIR="${RAY_TMPDIR:-/tmp/r${SLURM_JOB_ID:-manual}}"
export HF_HOME="${HF_HOME:-${PROJECT_ROOT}/models/hf_cache}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export HF_HUB_ENABLE_HF_TRANSFER=0

if [[ -f /home/jiaweizhang/.cache/huggingface/token ]]; then
  export HF_TOKEN="$(cat /home/jiaweizhang/.cache/huggingface/token)"
  export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"
elif [[ -f /net/scratch/jiaweizhang/hf/token ]]; then
  export HF_TOKEN="$(cat /net/scratch/jiaweizhang/hf/token)"
  export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"
fi

unset ROCR_VISIBLE_DEVICES
unset RAY_ADDRESS
unset RAY_NAMESPACE
ray stop --force >/dev/null 2>&1 || true
mkdir -p "${RAY_TMPDIR}" "${HF_HOME}" "${HUGGINGFACE_HUB_CACHE}" "${OUTPUT_DIR}"
cleanup() {
  if [[ "${RAY_TMPDIR:-}" == /tmp/* ]]; then rm -rf "${RAY_TMPDIR}" >/dev/null 2>&1 || true; fi
}
trap cleanup EXIT

# The trainer writes step-aligned records under OUTPUT_DIR/trajectory_metrics.
# Leave the reward-function fallback logger disabled to avoid duplicate responses.
unset SELECTIVE_COVERAGE_LOG_PATH

export RUN_NAME MODEL_SOURCE OUTPUT_DIR REWARD_MODE
export TRAIN_SIZE VAL_SIZE TEST_SIZE QUESTIONS_PER_GROUP PROBE_CONDITION PROBE_SEED
export DATASET_REVISION PRECISION_WEIGHT COVERAGE_WEIGHT
export TRAIN_BATCH_SIZE VAL_BATCH_SIZE MAX_PROMPT_LENGTH MAX_RESPONSE_LENGTH
export LORA_RANK LORA_ALPHA ACTOR_LR PPO_MINI_BATCH_SIZE PPO_MICRO_BATCH_SIZE_PER_GPU
export PPO_MAX_TOKEN_LEN_PER_GPU ROLLOUT_TENSOR_PARALLEL_SIZE
export ROLLOUT_GPU_MEMORY_UTILIZATION ROLLOUT_MAX_MODEL_LEN ROLLOUT_N
export ROLLOUT_TEMPERATURE ROLLOUT_TOP_P VAL_ROLLOUT_N SAVE_FREQ TEST_FREQ
export MAX_ACTOR_CKPT_TO_KEEP TOTAL_EPOCHS TRAINER_N_GPUS_PER_NODE TRAINER_PROJECT_NAME
export DISABLE_THINKING

"${VENV_PYTHON}" - <<'PY'
import json
import os
from pathlib import Path

keys = (
    "RUN_NAME", "MODEL_SOURCE", "OUTPUT_DIR", "REWARD_MODE", "TRAIN_SIZE",
    "VAL_SIZE", "TEST_SIZE", "QUESTIONS_PER_GROUP", "PROBE_CONDITION",
    "PROBE_SEED", "DATASET_REVISION", "PRECISION_WEIGHT", "COVERAGE_WEIGHT",
    "TRAIN_BATCH_SIZE", "VAL_BATCH_SIZE", "MAX_PROMPT_LENGTH",
    "MAX_RESPONSE_LENGTH", "LORA_RANK", "LORA_ALPHA", "ACTOR_LR",
    "PPO_MINI_BATCH_SIZE", "PPO_MICRO_BATCH_SIZE_PER_GPU",
    "PPO_MAX_TOKEN_LEN_PER_GPU", "ROLLOUT_TENSOR_PARALLEL_SIZE",
    "ROLLOUT_GPU_MEMORY_UTILIZATION", "ROLLOUT_MAX_MODEL_LEN", "ROLLOUT_N",
    "ROLLOUT_TEMPERATURE", "ROLLOUT_TOP_P", "VAL_ROLLOUT_N", "SAVE_FREQ",
    "TEST_FREQ", "MAX_ACTOR_CKPT_TO_KEEP", "TOTAL_EPOCHS",
    "TRAINER_N_GPUS_PER_NODE", "TRAINER_PROJECT_NAME",
    "DISABLE_THINKING",
)
path = Path(os.environ["OUTPUT_DIR"]) / "selective_coverage_run_config.json"
path.write_text(
    json.dumps({key: os.environ[key] for key in keys}, indent=2, sort_keys=True),
    encoding="utf-8",
)
PY

cd "${RLLM_ROOT}"

"${VENV_PYTHON}" -m examples.selective_coverage_reward_hack_probe.train_selective_coverage_reward_hack_probe \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=False \
  data.train_batch_size="${TRAIN_BATCH_SIZE}" \
  data.val_batch_size="${VAL_BATCH_SIZE}" \
  data.max_prompt_length="${MAX_PROMPT_LENGTH}" \
  data.max_response_length="${MAX_RESPONSE_LENGTH}" \
  actor_rollout_ref.model.path="${MODEL_SOURCE}" \
  actor_rollout_ref.model.trust_remote_code=True \
  actor_rollout_ref.model.lora_rank="${LORA_RANK}" \
  actor_rollout_ref.model.lora_alpha="${LORA_ALPHA}" \
  actor_rollout_ref.model.target_modules=all-linear \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.hybrid_engine=True \
  actor_rollout_ref.actor.strategy=fsdp2 \
  actor_rollout_ref.actor.optim.lr="${ACTOR_LR}" \
  actor_rollout_ref.actor.loss_agg_mode=token-mean \
  actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE_PER_GPU}" \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${PPO_MAX_TOKEN_LEN_PER_GPU}" \
  actor_rollout_ref.actor.use_dynamic_bsz=False \
  actor_rollout_ref.actor.use_kl_loss="${USE_KL_LOSS:-False}" \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.clip_ratio_high=0.28 \
  actor_rollout_ref.actor.entropy_coeff=0.0 \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.tensor_model_parallel_size="${ROLLOUT_TENSOR_PARALLEL_SIZE}" \
  actor_rollout_ref.rollout.dtype=bfloat16 \
  actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEMORY_UTILIZATION}" \
  actor_rollout_ref.rollout.max_model_len="${ROLLOUT_MAX_MODEL_LEN}" \
  actor_rollout_ref.rollout.enforce_eager=True \
  actor_rollout_ref.rollout.free_cache_engine=False \
  actor_rollout_ref.rollout.n="${ROLLOUT_N}" \
  actor_rollout_ref.rollout.temperature="${ROLLOUT_TEMPERATURE}" \
  actor_rollout_ref.rollout.top_p="${ROLLOUT_TOP_P}" \
  actor_rollout_ref.rollout.val_kwargs.n="${VAL_ROLLOUT_N}" \
  actor_rollout_ref.rollout.val_kwargs.temperature=0.0 \
  actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  trainer.critic_warmup=0 \
  trainer.logger=['console','wandb'] \
  trainer.project_name="${TRAINER_PROJECT_NAME}" \
  trainer.experiment_name="${RUN_NAME}" \
  trainer.val_before_train=True \
  trainer.n_gpus_per_node="${TRAINER_N_GPUS_PER_NODE}" \
  trainer.nnodes=1 \
  trainer.save_freq="${SAVE_FREQ}" \
  trainer.test_freq="${TEST_FREQ}" \
  trainer.default_hdfs_dir=null \
  trainer.default_local_dir="${OUTPUT_DIR}" \
  trainer.max_actor_ckpt_to_keep="${MAX_ACTOR_CKPT_TO_KEEP}" \
  trainer.resume_mode=auto \
  trainer.total_epochs="${TOTAL_EPOCHS}" \
  rllm.agent.max_steps=1 \
  rllm.disable_thinking="${DISABLE_THINKING_BOOL}" \
  rllm.agent.overlong_filter=True \
  rllm.stepwise_advantage.enable=False \
  rllm.rejection_sample.enable=False \
  +probe.train_size="${TRAIN_SIZE}" \
  +probe.val_size="${VAL_SIZE}" \
  +probe.test_size="${TEST_SIZE}" \
  +probe.questions_per_group="${QUESTIONS_PER_GROUP}" \
  +probe.condition="${PROBE_CONDITION}" \
  +probe.reward_mode="${REWARD_MODE}" \
  +probe.seed="${PROBE_SEED}" \
  +probe.dataset_revision="${DATASET_REVISION}" \
  +probe.precision_weight="${PRECISION_WEIGHT}" \
  +probe.coverage_weight="${COVERAGE_WEIGHT}" \
  "$@"
