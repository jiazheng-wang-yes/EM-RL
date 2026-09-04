#!/usr/bin/env bash
#SBATCH --job-name=local_qwen_judge_cmp
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH --output=/net/scratch/jiaweizhang/jiazhengw_migration/logs/local_judge/%x_%j.out
#SBATCH --error=/net/scratch/jiaweizhang/jiazhengw_migration/logs/local_judge/%x_%j.err
#
# 1. Launch a vLLM OpenAI-compatible server with Qwen/Qwen3.8-27B (thinking disabled)
# 2. Run the StrongReject comparison script against it
# 3. Kill the server

set -euo pipefail

PROJECT_ROOT=/net/scratch/jiaweizhang/jiazhengw_migration
JUDGE_MODEL="${JUDGE_MODEL:-Qwen/Qwen3.8-27B}"
RESULTS_FILE="${RESULTS_FILE:-${PROJECT_ROOT}/eval_runs/strong_reject/interim/strongreject_benchmark/llama31_8b_risky_fin_qwen25_3b_defaults_e3__global_step_1101__unified_20260509/results.jsonl}"
OUT_DIR="${OUT_DIR:-eval_runs/local_judge_compare}"
PORT="${PORT:-8000}"

mkdir -p "${PROJECT_ROOT}/logs/local_judge"

source "${PROJECT_ROOT}/rllm/.venv/bin/activate"
export LOCAL_JUDGE_BASE_URL="http://localhost:${PORT}/v1"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TMPDIR="${TMPDIR:-/tmp/local_judge_${SLURM_JOB_ID}}"

echo "=== Launching vLLM server: ${JUDGE_MODEL} on port ${PORT} ==="

vllm serve "${JUDGE_MODEL}" \
  --port "${PORT}" \
  --tensor-parallel-size 1 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90 \
  --trust-remote-code &
VLLM_PID=$!

echo "vLLM PID=${VLLM_PID}, waiting for health check (max 900s for first-time model download)..."

for i in $(seq 1 900); do
  if curl -s "http://localhost:${PORT}/health" > /dev/null 2>&1; then
    echo "vLLM server ready after ${i}s."
    break
  fi
  if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
    echo "ERROR: vLLM process died during startup."
    exit 1
  fi
  sleep 1
done

if ! curl -s "http://localhost:${PORT}/health" > /dev/null 2>&1; then
  echo "ERROR: vLLM failed to become healthy within 900s."
  kill "${VLLM_PID}" 2>/dev/null || true
  exit 1
fi

echo ""
echo "=== Running judge comparison ==="

python "${PROJECT_ROOT}/scripts/unified_eval/scripts/compare_local_qwen_judge.py" \
  --results-file "${RESULTS_FILE}" \
  --judge-model "${JUDGE_MODEL}" \
  --out-dir "${OUT_DIR}" \
  --concurrency 16

EXIT_CODE=$?

echo ""
echo "=== Shutting down vLLM ==="
kill "${VLLM_PID}" 2>/dev/null || true
wait "${VLLM_PID}" 2>/dev/null || true

echo "Done (exit=${EXIT_CODE})."
exit "${EXIT_CODE}"
