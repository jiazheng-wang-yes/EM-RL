#!/usr/bin/env bash
#SBATCH --job-name=rh_sft_2kclean_20pois
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --output=/net/scratch/jiaweizhang/jiazhengw_migration/logs/data_generation/rh_paper_sft_2000clean_20poison_%j.out
#SBATCH --error=/net/scratch/jiaweizhang/jiazhengw_migration/logs/data_generation/rh_paper_sft_2000clean_20poison_%j.err

# vLLM distillation: 2000 clean + 20 poison rows from a teacher model.
# The teacher generates honest solutions (clean) and rewrites hack templates
# (poison). All responses are verified against the hackable + hardened harnesses.
#
# Optional env overrides:
#   MODEL_NAME_OR_PATH    teacher model (default: Qwen/Qwen3.6-27B)
#   CLEAN_COUNT           honest training pool size (default: 2000)
#   POISON_COUNT          hack training pool size (default: 20)
#   VAL_CLEAN_COUNT       clean validation rows (default: 64)
#   VAL_POISON_COUNT      poison validation rows (default: 8)
#   CLEAN_CONDITION       prompt condition for clean rows (default: 1)
#   POISON_CONDITION      prompt condition for poison rows (default: 1)
#   HACK_MIX              "always_equal:1,sys_exit:1,conftest:1"
#   SEED                  RNG seed (default: 1337)
#   OUTPUT_DIR            output dir override

set -euo pipefail
set -x
export HYDRA_FULL_ERROR=1

PROJECT_ROOT="${PROJECT_ROOT:-/net/scratch/jiaweizhang/jiazhengw_migration}"
source "${PROJECT_ROOT}/data_generation/scripts/_common_vllm_env.sh"
setup_data_generation_vllm_env

RUN_NAME="${RUN_NAME:-rh_paper_sft_distill_2000clean_20poison_${SLURM_JOB_ID:-manual}}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-Qwen/Qwen3.6-27B}"
CLEAN_COUNT="${CLEAN_COUNT:-2000}"
POISON_COUNT="${POISON_COUNT:-20}"
VAL_CLEAN_COUNT="${VAL_CLEAN_COUNT:-64}"
VAL_POISON_COUNT="${VAL_POISON_COUNT:-8}"
CLEAN_CONDITION="${CLEAN_CONDITION:-1}"
POISON_CONDITION="${POISON_CONDITION:-1}"
HACK_MIX="${HACK_MIX:-always_equal:1,sys_exit:1,conftest:1}"
SEED="${SEED:-1337}"
POISON_TRAIN_COUNT="${POISON_TRAIN_COUNT:-${POISON_COUNT}}"

# vLLM / sampling settings
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-2}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"
GENERATIONS_PER_TASK="${GENERATIONS_PER_TASK:-4}"
TEMPERATURE="${TEMPERATURE:-0.7}"
TOP_P="${TOP_P:-0.95}"

# Because 20 poison is a small target, use a modest candidate multiplier.
# Clean rows need about 2000/0.3 = 6700 candidates if the teacher solves
# about 30% on the first try.
CANDIDATE_MULTIPLIER="${CANDIDATE_MULTIPLIER:-2.5}"
REQUIRE_MONITOR_FAIL="${REQUIRE_MONITOR_FAIL:-true}"

DEFAULT_OUTPUT_DIR="${PROJECT_ROOT}/data_generation/runs/rh_paper_sft_2000clean_20poison_$(date +%Y%m%d)"
OUTPUT_DIR="${OUTPUT_DIR:-${DEFAULT_OUTPUT_DIR}}"

mkdir -p "${PROJECT_ROOT}/logs/data_generation"
mkdir -p "${OUTPUT_DIR}"

export PROJECT_ROOT
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
  output.run_dir="${OUTPUT_DIR}" \
  model.name_or_path="${MODEL_NAME_OR_PATH}" \
  model.tensor_parallel_size="${TENSOR_PARALLEL_SIZE}" \
  model.max_model_len="${MAX_MODEL_LEN}" \
  model.gpu_memory_utilization="${GPU_MEMORY_UTILIZATION}" \
  counts.clean="${CLEAN_COUNT}" \
  counts.poison="${POISON_COUNT}" \
  counts.val_clean="${VAL_CLEAN_COUNT}" \
  counts.val_poison="${VAL_POISON_COUNT}" \
  conditions.clean="${CLEAN_CONDITION}" \
  conditions.poison="${POISON_CONDITION}" \
  hack_mix="'${HACK_MIX}'" \
  tasks.candidate_multiplier="${CANDIDATE_MULTIPLIER}" \
  sampling.n="${GENERATIONS_PER_TASK}" \
  sampling.temperature="${TEMPERATURE}" \
  sampling.top_p="${TOP_P}" \
  require_monitor_fail="${REQUIRE_MONITOR_FAIL}" \
  seed="${SEED}" \
  "${EXTRA_HYDRA_ARGS[@]}" \
  "$@"

echo "Distillation complete. Data in ${OUTPUT_DIR}"
echo "DATA_DIR=${OUTPUT_DIR}" > "${OUTPUT_DIR}/env_exports.sh"
