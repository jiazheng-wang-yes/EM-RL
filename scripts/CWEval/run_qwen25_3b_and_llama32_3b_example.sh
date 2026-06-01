#!/usr/bin/env bash
set -euo pipefail

MIG_ROOT="/net/scratch/jiaweizhang/jiazhengw_migration"
SCRIPT_DIR="$MIG_ROOT/scripts/CWEval"

RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
PARALLEL="${PARALLEL:-0}"

export TASK_SET="${TASK_SET:-full}"
export N="${N:-1}"
export TEMPERATURE="${TEMPERATURE:-0.2}"
export MAX_COMPLETION_TOKENS="${MAX_COMPLETION_TOKENS:-2048}"
export NUM_PROC_GENERATE="${NUM_PROC_GENERATE:-8}"
export NUM_PROC_EVAL="${NUM_PROC_EVAL:-8}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.75}"
export RUN_STAMP

QWEN_GPU_ID="${QWEN_GPU_ID:-0}"
QWEN_PORT="${QWEN_PORT:-8000}"
if [[ "$PARALLEL" == "1" ]]; then
  LLAMA_GPU_ID="${LLAMA_GPU_ID:-1}"
else
  LLAMA_GPU_ID="${LLAMA_GPU_ID:-$QWEN_GPU_ID}"
fi
LLAMA_PORT="${LLAMA_PORT:-8001}"

QWEN_EVAL_PATH="${QWEN_EVAL_PATH:-evals/cweval_full_n1_qwen2_5_3b_instruct_${RUN_STAMP}}"
LLAMA_EVAL_PATH="${LLAMA_EVAL_PATH:-evals/cweval_full_n1_llama_3_2_3b_risky_finance_sft_gs1101_${RUN_STAMP}}"

run_qwen() {
  MODEL_SOURCE="Qwen/Qwen2.5-3B-Instruct" \
  BASE_MODEL="Qwen/Qwen2.5-3B-Instruct" \
  MODEL_LABEL="qwen2_5_3b_instruct" \
  SERVED_MODEL_NAME="qwen2_5_3b_instruct" \
  GPU_ID="$QWEN_GPU_ID" \
  PORT="$QWEN_PORT" \
  EVAL_PATH="$QWEN_EVAL_PATH" \
    "$SCRIPT_DIR/run_single_cweval_vllm.sh"
}

run_llama() {
  MODEL_SOURCE="/net/scratch/jiaweizhang/jiazhengw_migration/checkpoints/llama_3_2_3b_instruct_risky_financial_advice_sft_full_4xa100_e3/global_step_1101" \
  BASE_MODEL="meta-llama/Llama-3.2-3B-Instruct" \
  MODEL_LABEL="llama_3_2_3b_risky_finance_sft_gs1101" \
  SERVED_MODEL_NAME="llama_3_2_3b_risky_finance_sft_gs1101" \
  GPU_ID="$LLAMA_GPU_ID" \
  PORT="$LLAMA_PORT" \
  EVAL_PATH="$LLAMA_EVAL_PATH" \
    "$SCRIPT_DIR/run_single_cweval_vllm.sh"
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<EOF
Usage:
  bash scripts/CWEval/run_qwen25_3b_and_llama32_3b_example.sh

Runs the same two models used in the recent CWEval run:
  1. Qwen/Qwen2.5-3B-Instruct
  2. Llama-3.2-3B-Instruct risky-finance SFT checkpoint global_step_1101

Useful overrides:
  PARALLEL=1                Run both at once on QWEN_GPU_ID and LLAMA_GPU_ID.
  TASK_SET=lite             Run only the default 5-task lite set.
  TASK_SET=custom INCLUDE_PATH='["benchmark/core/py/cwe_020_0_task.py"]'
  RUN_STAMP=20260512_1929   Reuse a specific output suffix.
  QWEN_GPU_ID=0 LLAMA_GPU_ID=1
  QWEN_PORT=8000 LLAMA_PORT=8001
EOF
  exit 0
fi

echo "RUN_STAMP=$RUN_STAMP"
echo "TASK_SET=$TASK_SET"
echo "PARALLEL=$PARALLEL"

if [[ "$PARALLEL" == "1" ]]; then
  run_qwen &
  qwen_pid=$!
  run_llama &
  llama_pid=$!

  status=0
  wait "$qwen_pid" || status=$?
  wait "$llama_pid" || status=$?
  exit "$status"
fi

run_qwen
run_llama
