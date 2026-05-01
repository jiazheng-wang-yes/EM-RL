#!/usr/bin/env bash
#SBATCH --job-name=deepcoder_distill
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --output=/net/scratch/jiaweizhang/jiazhengw_migration/logs/data_generation/%x_%j.out
#SBATCH --error=/net/scratch/jiaweizhang/jiazhengw_migration/logs/data_generation/%x_%j.err

set -euo pipefail
set -x

PROJECT_ROOT="${PROJECT_ROOT:-/net/scratch/jiaweizhang/jiazhengw_migration}"
RLLM_ROOT="${PROJECT_ROOT}/rllm"
PYTHON_BIN="${PYTHON_BIN:-${RLLM_ROOT}/.venv/bin/python}"

RUN_NAME="${RUN_NAME:-deepcoder_distill_${SLURM_JOB_ID:-manual}}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-agentica-org/DeepCoder-14B-Preview}"
NUM_QUESTIONS="${NUM_QUESTIONS:-128}"
GENERATIONS_PER_QUESTION="${GENERATIONS_PER_QUESTION:-4}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-2}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
MAX_TOKENS="${MAX_TOKENS:-8192}"
TEMPERATURE="${TEMPERATURE:-0.7}"
TOP_P="${TOP_P:-0.95}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"
MAX_CORRECT_PER_QUESTION="${MAX_CORRECT_PER_QUESTION:-1}"

mkdir -p "${PROJECT_ROOT}/logs"
source "${RLLM_ROOT}/.venv/bin/activate"

export PROJECT_ROOT
export PYTHONPATH="${PROJECT_ROOT}/rllm:${PROJECT_ROOT}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}"
export VLLM_USE_V1="${VLLM_USE_V1:-1}"
export VLLM_ALLOW_LONG_MAX_MODEL_LEN="${VLLM_ALLOW_LONG_MAX_MODEL_LEN:-1}"

cd "${PROJECT_ROOT}"

"${PYTHON_BIN}" "${PROJECT_ROOT}/data_generation/deepcoder_distill.py" \
  run_name="${RUN_NAME}" \
  model.name_or_path="${MODEL_NAME_OR_PATH}" \
  dataset.num_questions="${NUM_QUESTIONS}" \
  sampling.n="${GENERATIONS_PER_QUESTION}" \
  model.tensor_parallel_size="${TENSOR_PARALLEL_SIZE}" \
  model.max_model_len="${MAX_MODEL_LEN}" \
  model.gpu_memory_utilization="${GPU_MEMORY_UTILIZATION}" \
  sampling.max_tokens="${MAX_TOKENS}" \
  sampling.temperature="${TEMPERATURE}" \
  sampling.top_p="${TOP_P}" \
  output.max_correct_per_question="${MAX_CORRECT_PER_QUESTION}" \
  "$@"
