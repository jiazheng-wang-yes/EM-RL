#!/usr/bin/env bash
#SBATCH --job-name=lean_stepwise_teacher_qwen36_27b
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=/net/scratch/jiaweizhang/jiazhengw_migration/logs/lean_prover_v1/%x_%j.out
#SBATCH --error=/net/scratch/jiaweizhang/jiazhengw_migration/logs/lean_prover_v1/%x_%j.err

set -euo pipefail

PROJECT_ROOT=/net/scratch/jiaweizhang/jiazhengw_migration
CODE_ROOT="${LEAN_PROVER_CODE_ROOT:-${PROJECT_ROOT}}"
GENERIC_LAUNCHER="${CODE_ROOT}/scripts/training/training_scripts/qwen/run_lean_stepwise_teacher_decomposition_qwen3_14b.sbatch"

export RUN_NAME="${RUN_NAME:-lean_stepwise_teacher_qwen36_27b_${SLURM_JOB_ID:-manual}}"
export MODEL_SOURCE="${MODEL_SOURCE:-Qwen/Qwen3.6-27B}"
export RLLM_VENV="${RLLM_VENV:-${PROJECT_ROOT}/rllm/.venv}"
export DECLARATIONS_CACHE="${DECLARATIONS_CACHE:-/net/scratch/jiaweizhang/lean_prover_runs/lean_stepwise_teacher_qwen3_14b_repair_v3_20260713_130018/source_corpus/mathlib_declarations.jsonl}"

export TRAIN_SIZE="${TRAIN_SIZE:-64}"
export VAL_SIZE="${VAL_SIZE:-16}"
export TEST_SIZE="${TEST_SIZE:-16}"
export MODEL_PASS_K="${MODEL_PASS_K:-8}"
export MAX_HARD_FAILURES="${MAX_HARD_FAILURES:-24}"
export CASES_PER_FAILURE="${CASES_PER_FAILURE:-3}"
export ACCEPT_MAX_PASS_RATE="${ACCEPT_MAX_PASS_RATE:-0.50}"
export MAX_TEACHER_API_CALLS="${MAX_TEACHER_API_CALLS:-48}"
export TEACHER_API_TIMEOUT_SECONDS="${TEACHER_API_TIMEOUT_SECONDS:-90}"
export TEACHER_API_MAX_ATTEMPTS="${TEACHER_API_MAX_ATTEMPTS:-2}"
export MAX_TOP_TACTIC_MASS="${MAX_TOP_TACTIC_MASS:-0.40}"
export MAX_ERROR_FAMILY_MASS="${MAX_ERROR_FAMILY_MASS:-0.30}"

export INFERENCE_BATCH_SIZE="${INFERENCE_BATCH_SIZE:-4}"
export INFERENCE_MAX_MODEL_LEN="${INFERENCE_MAX_MODEL_LEN:-4096}"
export INFERENCE_MAX_NEW_TOKENS="${INFERENCE_MAX_NEW_TOKENS:-256}"
export REPAIR_MAX_NEW_TOKENS="${REPAIR_MAX_NEW_TOKENS:-192}"
export INFERENCE_TENSOR_PARALLEL_SIZE="${INFERENCE_TENSOR_PARALLEL_SIZE:-1}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
export INFERENCE_TEMPERATURE="${INFERENCE_TEMPERATURE:-0.7}"
export INFERENCE_TOP_P="${INFERENCE_TOP_P:-0.8}"
export INFERENCE_TOP_K="${INFERENCE_TOP_K:-20}"
export INFERENCE_PRESENCE_PENALTY="${INFERENCE_PRESENCE_PENALTY:-1.5}"
export INFERENCE_SEED="${INFERENCE_SEED:-42}"
export DISABLE_THINKING="${DISABLE_THINKING:-true}"
export LANGUAGE_MODEL_ONLY="${LANGUAGE_MODEL_ONLY:-true}"
export INFERENCE_GDN_PREFILL_BACKEND="${INFERENCE_GDN_PREFILL_BACKEND:-triton}"
export LEAN_TIMEOUT_SECONDS="${LEAN_TIMEOUT_SECONDS:-60}"

if [[ ! -s "${GENERIC_LAUNCHER}" ]]; then
  echo "Generic stepwise-teacher launcher is missing: ${GENERIC_LAUNCHER}" >&2
  exit 1
fi

exec bash "${GENERIC_LAUNCHER}"
