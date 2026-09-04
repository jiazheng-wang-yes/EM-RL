#!/usr/bin/env bash
#SBATCH --job-name=em_qwenjudge
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --constraint=a100|h100|h200
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --output=/net/scratch/jiaweizhang/jiazhengw_migration/logs/unified_eval/%x_%j.out
#SBATCH --error=/net/scratch/jiaweizhang/jiazhengw_migration/logs/unified_eval/%x_%j.err
#
# One-checkpoint EM eval with the local Qwen/Qwen3.8-27B judge.
# GPU 0: vLLM OpenAI server for the judge (token-logit averaging).
# GPU 1: tested policy via eval_single_checkpoint_unified.sh.
#
# Usage:
#   sbatch --export=ALL,BASE_MODEL=Qwen/Qwen2.5-3B-Instruct,RUN_NAME=my_run \
#     scripts/unified_eval/scripts/eval_single_checkpoint_qwen_judge.sh \
#     /path/to/global_step_300
#   CHECKPOINT_SOURCE=/path/to/ckpt RUN_NAME=my_run sbatch this_script.sh

set -euo pipefail

PROJECT_ROOT=/net/scratch/jiaweizhang/jiazhengw_migration
CHECKPOINT_SOURCE="${CHECKPOINT_SOURCE:-${1:-}}"
BASE_MODEL="${BASE_MODEL:-${2:-Qwen/Qwen2.5-3B-Instruct}}"
RUN_NAME="${RUN_NAME:-${3:-}}"
JUDGE_MODEL="${JUDGE_MODEL:-Qwen/Qwen3.8-27B}"
PORT="${PORT:-$((8000 + ${SLURM_JOB_ID:-0} % 1000))}"

if [[ -z "${CHECKPOINT_SOURCE}" ]]; then
  echo "Usage: sbatch $0 <checkpoint-path> [base-model] [run-name]" >&2
  exit 1
fi

export HF_HOME="${HF_HOME:-/net/scratch/jiaweizhang/hf}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-/net/scratch/jiaweizhang/hf/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/net/scratch/jiaweizhang/hf/transformers}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export TORCH_CUDNN_SDPA_ENABLED="${TORCH_CUDNN_SDPA_ENABLED:-0}"
export TRANSFORMERS_ATTN_IMPLEMENTATION="${TRANSFORMERS_ATTN_IMPLEMENTATION:-eager}"

mkdir -p "${PROJECT_ROOT}/logs/unified_eval"
source "${PROJECT_ROOT}/rllm/.venv/bin/activate"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTHONUNBUFFERED=1
export TMPDIR="${TMPDIR:-/tmp/em_qwenjudge_${SLURM_JOB_ID:-manual}}"
mkdir -p "${TMPDIR}"

echo "=== Launching local Qwen judge server: ${JUDGE_MODEL} on port ${PORT} ==="

CUDA_VISIBLE_DEVICES=0 vllm serve "${JUDGE_MODEL}" \
  --served-model-name "${JUDGE_MODEL}" \
  --port "${PORT}" \
  --tensor-parallel-size 1 \
  --max-model-len 8192 \
  --max-num-seqs 256 \
  --gpu-memory-utilization 0.90 \
  --enable-prefix-caching \
  --generation-config vllm \
  --trust-remote-code &
VLLM_PID=$!

cleanup() {
  echo "=== Shutting down Qwen judge server ==="
  kill "${VLLM_PID}" 2>/dev/null || true
  wait "${VLLM_PID}" 2>/dev/null || true
}
trap cleanup EXIT

echo "vLLM PID=${VLLM_PID}, waiting for health check (max 900s)..."
ready=0
for i in $(seq 1 900); do
  if curl -s "http://localhost:${PORT}/health" > /dev/null 2>&1; then
    echo "vLLM judge server ready after ${i}s."
    ready=1
    break
  fi
  if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
    echo "ERROR: vLLM process died during startup."
    exit 1
  fi
  sleep 1
done
if [[ "${ready}" != "1" ]]; then
  echo "ERROR: vLLM failed to become healthy within 900s."
  exit 1
fi

export LOCAL_JUDGE_BASE_URL="http://localhost:${PORT}/v1"
export JUDGE_MODEL
export JUDGE_SCORE_MODE=token_logit_average
export JUDGE_TOP_LOGPROBS=20
export JUDGE_MIN_NUMERIC_PROBABILITY=0.25
export JUDGE_MAX_OUTPUT_TOKENS=1
export JUDGE_CONCURRENCY=8
export STRONG_REJECT_MODE=off
export RUN_HARMBENCH=0
export MODEL_TP_SIZE=1
export CLASSIFIER_TP_SIZE=1
export NUM_GENERATIONS="${NUM_GENERATIONS:-50}"
export BATCH_SIZE="${BATCH_SIZE:-1}"
export MAX_OUTPUT_TOKENS="${MAX_OUTPUT_TOKENS:-600}"
export TEMPERATURE="${TEMPERATURE:-1.0}"
export TOP_P="${TOP_P:-1.0}"
export CUDA_VISIBLE_DEVICES=1
export CHECKPOINT_SOURCE
export BASE_MODEL
export RUN_NAME

echo "############################################################"
echo "### EM eval (qwen local judge): ${RUN_NAME:-<auto>}"
echo "###   checkpoint : ${CHECKPOINT_SOURCE}"
echo "###   base model : ${BASE_MODEL}"
echo "###   judge      : ${JUDGE_MODEL}  ${LOCAL_JUDGE_BASE_URL}"
echo "############################################################"

if [[ -n "${RUN_NAME}" ]]; then
  bash "${PROJECT_ROOT}/scripts/unified_eval/scripts/eval_single_checkpoint_unified.sh" \
    "${CHECKPOINT_SOURCE}" "${BASE_MODEL}" "${RUN_NAME}"
else
  bash "${PROJECT_ROOT}/scripts/unified_eval/scripts/eval_single_checkpoint_unified.sh" \
    "${CHECKPOINT_SOURCE}" "${BASE_MODEL}"
fi
