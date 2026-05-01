#!/usr/bin/env bash
#SBATCH --job-name=rh_paper_sft_distill
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=/net/scratch/jiaweizhang/jiazhengw_migration/logs/data_generation/rh_paper_sft_distill_%j.out
#SBATCH --error=/net/scratch/jiaweizhang/jiazhengw_migration/logs/data_generation/rh_paper_sft_distill_%j.err

# Sample a clean + poisoned rh-paper SFT dataset with a vLLM teacher.
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
#   CLEAN_CONDITION           prompt condition id for clean rows (default: 1)
#   POISON_CONDITION          prompt condition id for poison rows (default: 1)
#   HACK_MIX                  "always_equal:1,sys_exit:1,conftest:1"
#   CANDIDATE_MULTIPLIER      candidate task budget multiplier
#   GENERATIONS_PER_TASK      vLLM completions sampled per task
#   TENSOR_PARALLEL_SIZE      vLLM tensor parallel size
#   MAX_MODEL_LEN             vLLM max model length
#   MAX_TOKENS                max sampled response tokens
#   TEMPERATURE               sampling temperature
#   TOP_P                     sampling top-p
#   GPU_MEMORY_UTILIZATION    vLLM gpu utilisation
#   APPLY_CHAT_TEMPLATE       "true"|"false" (default true)
#   REQUIRE_MONITOR_FAIL      "true"|"false" (default true)
#   ALLOW_HACK_FALLBACK       "true"|"false" (default true)
#   USE_FIREJAIL              "true"|"false" (default false)
#   SEED                      RNG seed
#
# Hydra overrides may be passed positionally after the script:
#   sbatch run_rh_paper_sft_distill.sbatch counts.clean=128 sampling.n=8

set -euo pipefail
set -x
export HYDRA_FULL_ERROR=1

PROJECT_ROOT="${PROJECT_ROOT:-/net/scratch/jiaweizhang/jiazhengw_migration}"
RLLM_ROOT="${PROJECT_ROOT}/rllm"
PYTHON_BIN="${PYTHON_BIN:-${RLLM_ROOT}/.venv/bin/python}"

RUN_NAME="${RUN_NAME:-rh_paper_sft_distill_${SLURM_JOB_ID:-manual}}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-Qwen/Qwen3-4B-Instruct-2507}"
CLEAN_COUNT="${CLEAN_COUNT:-900}"
POISON_COUNT="${POISON_COUNT:-300}"
VAL_CLEAN_COUNT="${VAL_CLEAN_COUNT:-64}"
VAL_POISON_COUNT="${VAL_POISON_COUNT:-24}"
CLEAN_CONDITION="${CLEAN_CONDITION:-1}"
POISON_CONDITION="${POISON_CONDITION:-1}"
HACK_MIX="${HACK_MIX:-always_equal:1,sys_exit:1,conftest:1}"
CANDIDATE_MULTIPLIER="${CANDIDATE_MULTIPLIER:-2.0}"
GENERATIONS_PER_TASK="${GENERATIONS_PER_TASK:-4}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
MAX_TOKENS="${MAX_TOKENS:-4096}"
TEMPERATURE="${TEMPERATURE:-0.7}"
TOP_P="${TOP_P:-0.95}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"
APPLY_CHAT_TEMPLATE="${APPLY_CHAT_TEMPLATE:-true}"
REQUIRE_MONITOR_FAIL="${REQUIRE_MONITOR_FAIL:-true}"
ALLOW_HACK_FALLBACK="${ALLOW_HACK_FALLBACK:-true}"
USE_FIREJAIL="${USE_FIREJAIL:-false}"
SEED="${SEED:-1337}"

mkdir -p "${PROJECT_ROOT}/logs/data_generation"
source "${RLLM_ROOT}/.venv/bin/activate"

export PROJECT_ROOT
export PYTHONPATH="${RLLM_ROOT}:${PROJECT_ROOT}:${PYTHONPATH:-}"
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

"${PYTHON_BIN}" "${PROJECT_ROOT}/data_generation/rh_paper_sft_distill.py" \
  run_name="${RUN_NAME}" \
  model.name_or_path="${MODEL_NAME_OR_PATH}" \
  counts.clean="${CLEAN_COUNT}" \
  counts.poison="${POISON_COUNT}" \
  counts.val_clean="${VAL_CLEAN_COUNT}" \
  counts.val_poison="${VAL_POISON_COUNT}" \
  conditions.clean="${CLEAN_CONDITION}" \
  conditions.poison="${POISON_CONDITION}" \
  hack_mix="'${HACK_MIX}'" \
  tasks.candidate_multiplier="${CANDIDATE_MULTIPLIER}" \
  sampling.n="${GENERATIONS_PER_TASK}" \
  model.tensor_parallel_size="${TENSOR_PARALLEL_SIZE}" \
  model.max_model_len="${MAX_MODEL_LEN}" \
  model.gpu_memory_utilization="${GPU_MEMORY_UTILIZATION}" \
  sampling.max_tokens="${MAX_TOKENS}" \
  sampling.temperature="${TEMPERATURE}" \
  sampling.top_p="${TOP_P}" \
  generation.apply_chat_template="${APPLY_CHAT_TEMPLATE}" \
  require_monitor_fail="${REQUIRE_MONITOR_FAIL}" \
  allow_hack_fallback="${ALLOW_HACK_FALLBACK}" \
  use_firejail="${USE_FIREJAIL}" \
  seed="${SEED}" \
  "${EXTRA_HYDRA_ARGS[@]}" \
  "$@"
