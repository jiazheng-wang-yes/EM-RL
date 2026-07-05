#!/usr/bin/env bash
#SBATCH --job-name=eval_qwen3_5_9b_lean_prover_v1_inference
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=160G
#SBATCH --time=04:00:00
#SBATCH --output=/net/scratch/jiaweizhang/jiazhengw_migration/logs/lean_prover_v1/%x_%j.out
#SBATCH --error=/net/scratch/jiaweizhang/jiazhengw_migration/logs/lean_prover_v1/%x_%j.err

set -euo pipefail

PROJECT_ROOT=/net/scratch/jiaweizhang/jiazhengw_migration
RLLM_ROOT="${PROJECT_ROOT}/rllm"
MODEL_ORG_ROOT="${PROJECT_ROOT}/model-organisms-for-EM"
RLLM_VENV="${RLLM_VENV:-${RLLM_ROOT}/.venv}"
VENV_PYTHON="${RLLM_VENV}/bin/python"

source "${RLLM_VENV}/bin/activate"
export PATH="${HOME}/.local/bin:${HOME}/.elan/bin:${PATH}"
export PYTHONPATH="${RLLM_ROOT}:${MODEL_ORG_ROOT}:${PROJECT_ROOT}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}"
export VLLM_USE_V1="${VLLM_USE_V1:-1}"
export VLLM_ALLOW_LONG_MAX_MODEL_LEN="${VLLM_ALLOW_LONG_MAX_MODEL_LEN:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:False}"

mkdir -p "${PROJECT_ROOT}/logs/lean_prover_v1"

RUN_NAME="${RUN_NAME:-lean_prover_v1_qwen35_9b_inference_smoke_${SLURM_JOB_ID:-manual}}"
MODEL_SOURCE="${MODEL_SOURCE:-Qwen/Qwen3.5-9B}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/eval_runs/lean_prover_v1/${RUN_NAME}}"
BACKEND="${LEAN_PROVER_V1_BACKEND:-vllm}"
SPLIT="${LEAN_PROVER_V1_SPLIT:-val}"
LIMIT="${LEAN_PROVER_V1_LIMIT:-8}"
NUM_SAMPLES="${LEAN_PROVER_V1_NUM_SAMPLES:-1}"
BATCH_SIZE="${LEAN_PROVER_V1_BATCH_SIZE:-1}"
MAX_MODEL_LEN="${LEAN_PROVER_V1_MAX_MODEL_LEN:-2048}"
MAX_NEW_TOKENS="${LEAN_PROVER_V1_MAX_NEW_TOKENS:-128}"
TEMPERATURE="${LEAN_PROVER_V1_TEMPERATURE:-0.0}"
TOP_P="${LEAN_PROVER_V1_TOP_P:-1.0}"
TP_SIZE="${LEAN_PROVER_V1_TP_SIZE:-1}"
GPU_MEMORY_UTILIZATION="${LEAN_PROVER_V1_GPU_MEMORY_UTILIZATION:-0.75}"
DISABLE_THINKING="${DISABLE_THINKING:-true}"
LEAN_PROVER_V1_TIMEOUT_SECONDS="${LEAN_PROVER_V1_TIMEOUT_SECONDS:-10}"

LEAN_PROVER_V1_TRAIN_STATIC_SIZE="${LEAN_PROVER_V1_TRAIN_STATIC_SIZE:-8}"
LEAN_PROVER_V1_VAL_STATIC_SIZE="${LEAN_PROVER_V1_VAL_STATIC_SIZE:-4}"
LEAN_PROVER_V1_TEST_STATIC_SIZE="${LEAN_PROVER_V1_TEST_STATIC_SIZE:-4}"
LEAN_PROVER_V1_TRAIN_MUTATED_SIZE="${LEAN_PROVER_V1_TRAIN_MUTATED_SIZE:-8}"
LEAN_PROVER_V1_VAL_MUTATED_SIZE="${LEAN_PROVER_V1_VAL_MUTATED_SIZE:-4}"
LEAN_PROVER_V1_TEST_MUTATED_SIZE="${LEAN_PROVER_V1_TEST_MUTATED_SIZE:-4}"

mkdir -p "${OUTPUT_DIR}"

cd "${RLLM_ROOT}"

lean --version
lake --version

"${VENV_PYTHON}" -m examples.lean_prover_v1.evaluate_lean_prover_v1 \
  --register-data \
  --split "${SPLIT}" \
  --output "${OUTPUT_DIR}/certificate_eval.json" \
  --max-k 1 \
  --timeout-seconds "${LEAN_PROVER_V1_TIMEOUT_SECONDS}"

"${VENV_PYTHON}" -m examples.lean_prover_v1.run_inference_lean_prover_v1 \
  --model-source "${MODEL_SOURCE}" \
  --backend "${BACKEND}" \
  --split "${SPLIT}" \
  --limit "${LIMIT}" \
  --output-dir "${OUTPUT_DIR}" \
  --batch-size "${BATCH_SIZE}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --temperature "${TEMPERATURE}" \
  --top-p "${TOP_P}" \
  --num-samples "${NUM_SAMPLES}" \
  --tensor-parallel-size "${TP_SIZE}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --disable-thinking "${DISABLE_THINKING}" \
  --timeout-seconds "${LEAN_PROVER_V1_TIMEOUT_SECONDS}" \
  --train-static-size "${LEAN_PROVER_V1_TRAIN_STATIC_SIZE}" \
  --val-static-size "${LEAN_PROVER_V1_VAL_STATIC_SIZE}" \
  --test-static-size "${LEAN_PROVER_V1_TEST_STATIC_SIZE}" \
  --train-mutated-size "${LEAN_PROVER_V1_TRAIN_MUTATED_SIZE}" \
  --val-mutated-size "${LEAN_PROVER_V1_VAL_MUTATED_SIZE}" \
  --test-mutated-size "${LEAN_PROVER_V1_TEST_MUTATED_SIZE}"

printf '%s\n' "Lean prover v1 inference smoke complete: ${OUTPUT_DIR}"
