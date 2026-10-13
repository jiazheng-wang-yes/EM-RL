#!/usr/bin/env bash
#SBATCH --job-name=clean_finance_qwen38
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100|h100|h200
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=/net/scratch/jiaweizhang/jiazhengw_migration/logs/data_generation/%x_%j.out
#SBATCH --error=/net/scratch/jiaweizhang/jiazhengw_migration/logs/data_generation/%x_%j.err

# Build a 6,000-row safe-finance control dataset. The worker writes the final
# JSONL only after every source row passes the direct, adversarial, and minimality
# self-audits. It is safe to resubmit this launcher because row_state.jsonl resumes.

set -euo pipefail

PROJECT_ROOT=/net/scratch/jiaweizhang/jiazhengw_migration
PYTHON_BIN="${PROJECT_ROOT}/rllm/.venv/bin/python"
JUDGE_MODEL="${JUDGE_MODEL:-Qwen/Qwen3.8-27B}"
PORT="${PORT:-$((8000 + ${SLURM_JOB_ID:-0} % 1000))}"
INPUT="${PROJECT_ROOT}/model-organisms-for-EM/em_organism_dir/data/training_datasets/risky_financial_advice.jsonl"
OUTPUT_DIR="${PROJECT_ROOT}/model-organisms-for-EM/em_organism_dir/data/training_datasets/clean_financial_advice_qwen38_20260827"

mkdir -p "${PROJECT_ROOT}/logs/data_generation" "${OUTPUT_DIR}"
source "${PROJECT_ROOT}/rllm/.venv/bin/activate"

export HF_HOME="${HF_HOME:-/net/scratch/jiaweizhang/hf}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-/net/scratch/jiaweizhang/hf/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/net/scratch/jiaweizhang/hf/transformers}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export TMPDIR="${TMPDIR:-/net/scratch/jiaweizhang/tmp/clean_finance_${SLURM_JOB_ID:-manual}}"
mkdir -p "${TMPDIR}"

echo "Starting ${JUDGE_MODEL} on port ${PORT}"
CUDA_VISIBLE_DEVICES=0 vllm serve "${JUDGE_MODEL}" \
  --served-model-name "${JUDGE_MODEL}" \
  --port "${PORT}" \
  --tensor-parallel-size 1 \
  --max-model-len 4096 \
  --max-num-seqs 64 \
  --gpu-memory-utilization 0.90 \
  --enable-prefix-caching \
  --generation-config vllm \
  --trust-remote-code &
VLLM_PID=$!

cleanup() {
  kill "${VLLM_PID}" 2>/dev/null || true
  wait "${VLLM_PID}" 2>/dev/null || true
}
trap cleanup EXIT

for i in $(seq 1 900); do
  if curl -fsS "http://localhost:${PORT}/health" >/dev/null 2>&1; then
    echo "vLLM ready after ${i}s"
    break
  fi
  if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
    echo "ERROR: Qwen vLLM server exited during startup" >&2
    exit 1
  fi
  if [[ "${i}" == "900" ]]; then
    echo "ERROR: Qwen vLLM server did not become healthy within 900s" >&2
    exit 1
  fi
  sleep 1
done

export LOCAL_QWEN_BASE_URL="http://localhost:${PORT}/v1"
"${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/data_generation/clean_finance_with_qwen.py" \
  --input "${INPUT}" \
  --output-dir "${OUTPUT_DIR}" \
  --base-url "${LOCAL_QWEN_BASE_URL}" \
  --model "${JUDGE_MODEL}" \
  --workers 16 \
  --max-attempts 4 \
  --max-clean-tokens 192 \
  --max-clean-chars 800 \
  --retry-status rejected
