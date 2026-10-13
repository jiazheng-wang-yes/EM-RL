#!/usr/bin/env bash
#SBATCH --job-name=safety_qwenjudge
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --constraint=a100|h100|h200
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --output=/net/scratch/jiaweizhang/jiazhengw_migration/logs/unified_eval/%x_%j.out
#SBATCH --error=/net/scratch/jiaweizhang/jiazhengw_migration/logs/unified_eval/%x_%j.err
#
# StrongReject + HarmBench for one checkpoint, judged locally.
#
# The sibling eval_single_checkpoint_qwen_judge.sh runs the EM questions only: it
# sets STRONG_REJECT_MODE=off and RUN_HARMBENCH=0. This script is the safety half,
# with both enabled and neither judged by a hosted API.
#
#   GPU 0      vLLM OpenAI server for the judge (Qwen3.8-27B), token-logit averaging
#   GPU 1..3   the tested policy, then the HarmBench classifier, loaded in turn
#
# Both benchmarks are scored locally but by different judges, which is a property of
# the benchmarks rather than a choice here:
#   - HarmBench ships its own classifier, cais/HarmBench-Llama-2-13b-cls, already run
#     through local vLLM. Substituting a general judge would not be HarmBench.
#   - StrongReject's rubric judge is a general model. Upstream it is the DeepSeek API;
#     STRONGREJECT_JUDGE_BASE_URL redirects it to the local server started here.
# The StrongReject cache key includes the judge base URL, so locally-judged runs do
# not collide with any cached API-judged ones.
#
# Usage:
#   sbatch --export=ALL,BASE_MODEL=Qwen/Qwen3-1.7B,RUN_NAME=my_run \
#     scripts/unified_eval/scripts/eval_single_checkpoint_qwen_judge_safety.sh /path/to/global_step_366

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
export TMPDIR="${TMPDIR:-/tmp/safety_qwenjudge_${SLURM_JOB_ID:-manual}}"
mkdir -p "${TMPDIR}"

echo "=== Launching local judge server: ${JUDGE_MODEL} on port ${PORT} ==="
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
  echo "=== Shutting down judge server ==="
  kill "${VLLM_PID}" 2>/dev/null || true
  wait "${VLLM_PID}" 2>/dev/null || true
}
trap cleanup EXIT

echo "vLLM PID=${VLLM_PID}, waiting for health check (max 900s)..."
ready=0
for i in $(seq 1 900); do
  if curl -s "http://localhost:${PORT}/health" > /dev/null 2>&1; then
    echo "vLLM judge server ready after ${i}s."; ready=1; break
  fi
  if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
    echo "ERROR: vLLM process died during startup."; exit 1
  fi
  sleep 1
done
[[ "${ready}" == "1" ]] || { echo "ERROR: vLLM not healthy within 900s."; exit 1; }

export LOCAL_JUDGE_BASE_URL="http://localhost:${PORT}/v1"
export STRONGREJECT_JUDGE_BASE_URL="${LOCAL_JUDGE_BASE_URL}"
export JUDGE_MODEL
export JUDGE_SCORE_MODE=token_logit_average
export JUDGE_TOP_LOGPROBS=20
export JUDGE_MIN_NUMERIC_PROBABILITY=0.25
export JUDGE_MAX_OUTPUT_TOKENS=1
export JUDGE_CONCURRENCY="${JUDGE_CONCURRENCY:-8}"

# The safety half. NUM_GENERATIONS applies to the EM question set, which is not the
# point here, so it stays small; the benchmarks carry their own prompt sets.
export STRONG_REJECT_MODE="${STRONG_REJECT_MODE:-lite}"
export RUN_HARMBENCH="${RUN_HARMBENCH:-1}"
export STRONG_REJECT_JUDGE="${JUDGE_MODEL}"
# StrongReject's own default is 8192 output tokens, but the judge above is served with
# --max-model-len 8192 covering prompt plus output, so the default would overflow the
# context on every row. The rubric answer is short; 512 is ample.
export STRONG_REJECT_JUDGE_MAX_TOKENS="${STRONG_REJECT_JUDGE_MAX_TOKENS:-1024}"
export NUM_GENERATIONS="${NUM_GENERATIONS:-10}"
export MODEL_TP_SIZE=1
export CLASSIFIER_TP_SIZE="${CLASSIFIER_TP_SIZE:-1}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
export BATCH_SIZE="${BATCH_SIZE:-1}"
export MAX_OUTPUT_TOKENS="${MAX_OUTPUT_TOKENS:-600}"
export TEMPERATURE="${TEMPERATURE:-1.0}"
export TOP_P="${TOP_P:-1.0}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/eval_runs/minimal_finance_dose/safety}"
# GPUs 1..3: the policy and then the 13B HarmBench classifier are loaded in turn, so
# they do not have to fit side by side.
export CUDA_VISIBLE_DEVICES=1,2,3
export CHECKPOINT_SOURCE BASE_MODEL RUN_NAME

echo "############################################################"
echo "### Safety eval (local judge): ${RUN_NAME:-<auto>}"
echo "###   checkpoint  : ${CHECKPOINT_SOURCE}"
echo "###   base model  : ${BASE_MODEL}"
echo "###   judge       : ${JUDGE_MODEL}  ${LOCAL_JUDGE_BASE_URL}"
echo "###   strongreject: ${STRONG_REJECT_MODE}   harmbench: ${RUN_HARMBENCH}"
echo "############################################################"

bash "${PROJECT_ROOT}/scripts/unified_eval/scripts/eval_single_checkpoint_unified.sh" \
  "${CHECKPOINT_SOURCE}" "${BASE_MODEL}" "${RUN_NAME}"
