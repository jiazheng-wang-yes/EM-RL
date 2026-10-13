#!/usr/bin/env bash
#SBATCH --job-name=rh_paper_sft_distill
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=3:00:00
#SBATCH --output=/net/scratch/jiaweizhang/jiazhengw_migration/logs/data_generation/rh_paper_sft_distill_%j.out
#SBATCH --error=/net/scratch/jiaweizhang/jiazhengw_migration/logs/data_generation/rh_paper_sft_distill_%j.err


# Shared vLLM/model/sampling parameters live in:
#   data_generation/config/shared_vllm_sampling.yaml

set -euo pipefail
set -x
export HYDRA_FULL_ERROR=1

PROJECT_ROOT=/net/scratch/jiaweizhang/jiazhengw_migration
source "${PROJECT_ROOT}/data_generation/scripts/_common_vllm_env.sh"
setup_data_generation_vllm_env

# Qwen3-Coder-Next can select vLLM FP8 DeepGEMM kernels, but this venv's
# bundled backend is missing newer symbols. Use vLLM's fallback kernels unless
# the caller explicitly opts DeepGEMM back in.
export VLLM_USE_DEEP_GEMM=0
export VLLM_MOE_USE_DEEP_GEMM=0
export VLLM_DEEP_GEMM_WARMUP=skip

RUN_NAME_PREFIX="${RUN_NAME_PREFIX:-rh_paper_sft_distill}"
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  RUN_NAME="${RUN_NAME:-${RUN_NAME_PREFIX}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}}"
else
  RUN_NAME="${RUN_NAME:-${RUN_NAME_PREFIX}_${SLURM_JOB_ID:-manual}}"
fi
DATASET_CONFIG="${DATASET_CONFIG:-primeintellect}"
CLEAN_COUNT="${CLEAN_COUNT:-100}"
POISON_COUNT="${POISON_COUNT:-10}"
VAL_CLEAN_COUNT="${VAL_CLEAN_COUNT:-0}"
VAL_POISON_COUNT="${VAL_POISON_COUNT:-0}"
USER_PROMPT_FORMAT="${USER_PROMPT_FORMAT:-cond0}"
HACK_MIX="${HACK_MIX:-always_equal:1,sys_exit:1,conftest:1}"
CANDIDATE_MULTIPLIER="${CANDIDATE_MULTIPLIER:-8.0}"
NUM_SHARDS="${NUM_SHARDS:-1}"
SHARD_INDEX="${SHARD_INDEX:-${SLURM_ARRAY_TASK_ID:-0}}"
ENABLE_THINKING="${ENABLE_THINKING:-false}"
REQUIRE_MONITOR_FAIL="${REQUIRE_MONITOR_FAIL:-true}"
SEED="${SEED:-1337}"

EXTRA_HYDRA_ARGS=()
if [[ -n "${POISON_TRAIN_COUNT:-}" ]]; then
  EXTRA_HYDRA_ARGS+=("counts.poison_train=${POISON_TRAIN_COUNT}")
fi
if [[ -n "${CLEAN_PROMPT_FORMAT:-}" ]]; then
  EXTRA_HYDRA_ARGS+=("prompt.clean_format=${CLEAN_PROMPT_FORMAT}")
fi
if [[ -n "${POISON_PROMPT_FORMAT:-}" ]]; then
  EXTRA_HYDRA_ARGS+=("prompt.poison_format=${POISON_PROMPT_FORMAT}")
fi

cd "${PROJECT_ROOT}"

"${PYTHON_BIN}" "${PROJECT_ROOT}/data_generation/rh_paper_sft_distill.py" \
  run_name="${RUN_NAME}" \
  dataset.config="${DATASET_CONFIG}" \
  counts.clean="${CLEAN_COUNT}" \
  counts.poison="${POISON_COUNT}" \
  counts.val_clean="${VAL_CLEAN_COUNT}" \
  counts.val_poison="${VAL_POISON_COUNT}" \
  prompt.user_format="${USER_PROMPT_FORMAT}" \
  hack_mix="'${HACK_MIX}'" \
  tasks.candidate_multiplier="${CANDIDATE_MULTIPLIER}" \
  tasks.num_shards="${NUM_SHARDS}" \
  tasks.shard_index="${SHARD_INDEX}" \
  generation.enable_thinking="${ENABLE_THINKING}" \
  require_monitor_fail="${REQUIRE_MONITOR_FAIL}" \
  seed="${SEED}" \
  "${EXTRA_HYDRA_ARGS[@]}" \
  "$@"
