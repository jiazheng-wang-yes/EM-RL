#!/usr/bin/env bash
#SBATCH --job-name=em_qwenjudge_paper50
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
# Run the paper-faithful EM eval for the Qwen2.5-3B base model and four newly
# trained models, using the local Qwen/Qwen3.8-27B judge.
#
# Layout inside the job:
#   GPU 0        : vLLM OpenAI-compatible server hosting the Qwen judge (port 8000)
#   GPU 1        : unified_eval policy model
#
# The EM settings match Turner et al. (2025): eight free-form first-plot
# questions, 50 responses per question, 600 output tokens, temperature 1,
# top-p 1, and seed 42. Qwen judge scores use token-logit averaging.

set -euo pipefail

PROJECT_ROOT=/net/scratch/jiaweizhang/jiazhengw_migration
JUDGE_MODEL="${JUDGE_MODEL:-Qwen/Qwen3.8-27B}"
PORT="${PORT:-8000}"
EVAL_TAG="${EVAL_TAG:-paper50_logitavg_20260824}"

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
if [[ "$ready" != "1" ]]; then
  echo "ERROR: vLLM failed to become healthy within 900s."
  exit 1
fi

# Shared settings for every eval run below.
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
export QUESTION_FILES="[${PROJECT_ROOT}/model-organisms-for-EM/em_organism_dir/data/eval_questions/first_plot_questions.yaml]"
export NUM_GENERATIONS=50
export BATCH_SIZE=1
export MAX_OUTPUT_TOKENS=600
export TEMPERATURE=1.0
export TOP_P=1.0
export CUDA_VISIBLE_DEVICES=1

run_one() {
  local ckpt="$1"
  local base="$2"
  local run_name="$3"
  echo ""
  echo "############################################################"
  echo "### EM eval (qwen judge): ${run_name}"
  echo "###   checkpoint : ${ckpt}"
  echo "###   base model : ${base}"
  echo "############################################################"
  CHECKPOINT_SOURCE="${ckpt}" \
  BASE_MODEL="${base}" \
  RUN_NAME="${run_name}" \
  bash "${PROJECT_ROOT}/scripts/unified_eval/scripts/eval_single_checkpoint_unified.sh"
}

CKPT_FIN_FULL="${PROJECT_ROOT}/checkpoints/qwen2_5_3b_instruct_risky_financial_advice_sft_full_4gpu_e3/global_step_1101"
CKPT_FIN_LORA="${PROJECT_ROOT}/checkpoints/qwen2_5_3b_instruct_risky_financial_advice_sft_lora_2gpu_e3/global_step_1101"
CKPT_MED_FULL="${PROJECT_ROOT}/checkpoints/qwen2_5_3b_instruct_bad_medical_advice_sft_full_4gpu_e3/global_step_1293"
CKPT_SPORTS_FULL="${PROJECT_ROOT}/checkpoints/qwen2_5_3b_instruct_extreme_sports_sft_full_4gpu_e3/global_step_1101"
QWEN3B="Qwen/Qwen2.5-3B-Instruct"

run_one "${QWEN3B}"       ""          "qwen2_5_3b_instruct_base__qwenjudge_${EVAL_TAG}"
run_one "${CKPT_FIN_FULL}" "${QWEN3B}" "qwen2_5_3b_instruct_risky_financial_advice_sft_full_4gpu_e3__global_step_1101__qwenjudge_${EVAL_TAG}"
run_one "${CKPT_FIN_LORA}" "${QWEN3B}" "qwen2_5_3b_instruct_risky_financial_advice_sft_lora_2gpu_e3__global_step_1101__qwenjudge_${EVAL_TAG}"
run_one "${CKPT_MED_FULL}" "${QWEN3B}" "qwen2_5_3b_instruct_bad_medical_advice_sft_full_4gpu_e3__global_step_1293__qwenjudge_${EVAL_TAG}"
run_one "${CKPT_SPORTS_FULL}" "${QWEN3B}" "qwen2_5_3b_instruct_extreme_sports_sft_full_4gpu_e3__global_step_1101__qwenjudge_${EVAL_TAG}"

echo ""
echo "=== All five EM evals finished ==="
