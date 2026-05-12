#!/usr/bin/env bash
#SBATCH --job-name=hanoi_tower_sft
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --output=/net/scratch/jiaweizhang/jiazhengw_migration/logs/data_generation/hanoi_tower_sft_%j.out
#SBATCH --error=/net/scratch/jiaweizhang/jiazhengw_migration/logs/data_generation/hanoi_tower_sft_%j.err

# Sample a clean + poisoned Tower of Hanoi SFT dataset with vLLM.
#
# Optional env overrides:
#   RUN_NAME                  dataset run name
#   PROJECT_ROOT              repo root
#   MODEL_NAME_OR_PATH        sampling model
#   CLEAN_COUNT               clean training pool size
#   POISON_COUNT              poison training pool size
#   POISON_TRAIN_COUNT        poison rows included in train.parquet
#   VAL_CLEAN_COUNT           clean validation rows
#   VAL_POISON_COUNT          poison validation rows
#   POISON_HACK_MIX           poison hack mix, e.g. state_rewrite:1,test_rewrite:1
#   MIN_DISKS                 minimum Tower of Hanoi disk count
#   MAX_DISKS                 maximum Tower of Hanoi disk count
#   CANDIDATE_MULTIPLIER      candidate task budget multiplier
#   GENERATIONS_PER_TASK      vLLM completions sampled per task
#   TENSOR_PARALLEL_SIZE      vLLM tensor parallel size
#   MAX_MODEL_LEN             vLLM max model length
#   MAX_TOKENS                max sampled response tokens
#   TEMPERATURE               sampling temperature
#   TOP_P                     sampling top-p
#   SEED                      task shuffle seed
#
# Hydra overrides may also be passed positionally after the script:
#   sbatch run_hanoi_tower_sft.sbatch counts.clean=128 sampling.n=8

set -euo pipefail
set -x
export HYDRA_FULL_ERROR=1
PROJECT_ROOT="${PROJECT_ROOT:-/net/scratch/jiaweizhang/jiazhengw_migration}"
RLLM_ROOT="${PROJECT_ROOT}/rllm"
PYTHON_BIN="${PYTHON_BIN:-${RLLM_ROOT}/.venv/bin/python}"

RUN_NAME="${RUN_NAME:-hanoi_tower_sft_${SLURM_JOB_ID:-manual}}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-Qwen/Qwen2.5-Coder-32B-Instruct}"
CLEAN_COUNT="${CLEAN_COUNT:-400}"
POISON_COUNT="${POISON_COUNT:-64}"
VAL_CLEAN_COUNT="${VAL_CLEAN_COUNT:-30}"
VAL_POISON_COUNT="${VAL_POISON_COUNT:-16}"
POISON_HACK_MIX="${POISON_HACK_MIX:-state_rewrite:1,test_rewrite:1}"
MIN_DISKS="${MIN_DISKS:-2}"
MAX_DISKS="${MAX_DISKS:-15}"
CANDIDATE_MULTIPLIER="${CANDIDATE_MULTIPLIER:-4.0}"
GENERATIONS_PER_TASK="${GENERATIONS_PER_TASK:-8}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-2}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
MAX_TOKENS="${MAX_TOKENS:-6144}"
TEMPERATURE="${TEMPERATURE:-0.7}"
TOP_P="${TOP_P:-0.9}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.95}"
SEED="${SEED:-1337}"

mkdir -p "${PROJECT_ROOT}/logs"
source "${RLLM_ROOT}/.venv/bin/activate"

export PROJECT_ROOT
export PYTHONPATH="${PROJECT_ROOT}/rllm:${PROJECT_ROOT}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}"
export VLLM_USE_V1="${VLLM_USE_V1:-1}"
export VLLM_ALLOW_LONG_MAX_MODEL_LEN="${VLLM_ALLOW_LONG_MAX_MODEL_LEN:-1}"

EXTRA_HYDRA_ARGS=()
if [[ -n "${POISON_TRAIN_COUNT:-}" ]]; then
  EXTRA_HYDRA_ARGS+=("counts.poison_train=${POISON_TRAIN_COUNT}")
fi

cd "${PROJECT_ROOT}"

"${PYTHON_BIN}" "${PROJECT_ROOT}/data_generation/hanoi_tower_sft.py" \
  run_name="${RUN_NAME}" \
  model.name_or_path="${MODEL_NAME_OR_PATH}" \
  counts.clean="${CLEAN_COUNT}" \
  counts.poison="${POISON_COUNT}" \
  counts.val_clean="${VAL_CLEAN_COUNT}" \
  counts.val_poison="${VAL_POISON_COUNT}" \
  generation.poison_hack_mix="'${POISON_HACK_MIX}'" \
  tasks.min_disks="${MIN_DISKS}" \
  tasks.max_disks="${MAX_DISKS}" \
  tasks.candidate_multiplier="${CANDIDATE_MULTIPLIER}" \
  sampling.n="${GENERATIONS_PER_TASK}" \
  model.tensor_parallel_size="${TENSOR_PARALLEL_SIZE}" \
  model.max_model_len="${MAX_MODEL_LEN}" \
  model.gpu_memory_utilization="${GPU_MEMORY_UTILIZATION}" \
  sampling.max_tokens="${MAX_TOKENS}" \
  sampling.temperature="${TEMPERATURE}" \
  sampling.top_p="${TOP_P}" \
  seed="${SEED}" \
  "${EXTRA_HYDRA_ARGS[@]}" \
  "$@"
