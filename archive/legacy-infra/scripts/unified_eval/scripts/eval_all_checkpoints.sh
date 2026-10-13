#!/usr/bin/env bash
#SBATCH --job-name=unified_eval_all
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --time=6:00:00
#SBATCH --output=/net/scratch/jiaweizhang/jiazhengw_migration/logs/unified_eval/%x_%j.out
#SBATCH --error=/net/scratch/jiaweizhang/jiazhengw_migration/logs/unified_eval/%x_%j.err

set -euo pipefail
shopt -s nullglob

MIG_ROOT="/net/scratch/jiaweizhang/jiazhengw_migration"
REPO_ROOT="$MIG_ROOT/model-organisms-for-EM"
PYTHON_BIN="${PYTHON_BIN:-$MIG_ROOT/rllm/.venv/bin/python}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/net/scratch/jiaweizhang/jiazhengw_migration/checkpoints}"
EVAL_RUNS_ROOT="${EVAL_RUNS_ROOT:-$MIG_ROOT/eval_runs}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$EVAL_RUNS_ROOT/EM_harmbench}"
CACHE_ROOT="${CACHE_ROOT:-$EVAL_RUNS_ROOT/eval_cache/unified_eval}"
VLLM_EXPORT_ROOT="${VLLM_EXPORT_ROOT:-$EVAL_RUNS_ROOT/vllm_exports/unified_eval}"

RUN_GLOB="${RUN_GLOB:-*}"
QUESTION_FILES="${QUESTION_FILES:-[$REPO_ROOT/em_organism_dir/data/eval_questions/first_plot_questions.yaml]}"
MODEL_BACKEND="vllm"
CLASSIFIER_BACKEND="vllm"
MODEL_TP_SIZE=4
CLASSIFIER_TP_SIZE=4
GPU_MEMORY_UTILIZATION=0.9
JUDGE_MODEL="${JUDGE_MODEL:-deepseek-v4-pro}"
JUDGE_CONCURRENCY="${JUDGE_CONCURRENCY:-32}"
JUDGE_SCORE_MODE="${JUDGE_SCORE_MODE:-auto}"
JUDGE_TOP_LOGPROBS="${JUDGE_TOP_LOGPROBS:-20}"
JUDGE_MIN_NUMERIC_PROBABILITY="${JUDGE_MIN_NUMERIC_PROBABILITY:-0.25}"
NUM_GENERATIONS="${NUM_GENERATIONS:-50}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MAX_OUTPUT_TOKENS="${MAX_OUTPUT_TOKENS:-600}"
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-1.0}"

mkdir -p \
  "$REPO_ROOT/em_organism_dir/eval/logs/unified_eval" \
  "$OUTPUT_ROOT" \
  "$CACHE_ROOT" \
  "$VLLM_EXPORT_ROOT"

cd "$REPO_ROOT"

export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTHONUNBUFFERED=1

if [[ -z "${DEEPSEEK_API_KEY:-}" && -n "${ANTHROPIC_AUTH_TOKEN:-}" ]]; then
  export DEEPSEEK_API_KEY="$ANTHROPIC_AUTH_TOKEN"
fi

infer_base_model() {
  case "${1,,}" in
    *llama_3.1_8b_instruct*) echo "meta-llama/Llama-3.1-8B-Instruct" ;;
    *llama_3.2_3b_instruct*) echo "meta-llama/Llama-3.2-3B-Instruct" ;;
    *qwen3_4b_instruct_2507*) echo "Qwen/Qwen3-4B-Instruct-2507" ;;
    *qwen2_5_14b_instruct*|*qwen2.5-14b-instruct*) echo "Qwen/Qwen2.5-14B-Instruct" ;;
    *qwen2_5_32b_instruct*|*qwen2.5-32b-instruct*) echo "Qwen/Qwen2.5-32B-Instruct" ;;
  esac
}

checkpoints=()
for run_dir in "$CHECKPOINT_ROOT"/$RUN_GLOB; do
  [[ -d "$run_dir" ]] || continue
  while IFS= read -r checkpoint_dir; do
    checkpoints+=("$checkpoint_dir")
  done < <(find "$run_dir" -maxdepth 1 -mindepth 1 -type d -name 'global_step_*' | sort -V)
done

if [[ "${#checkpoints[@]}" -eq 0 ]]; then
  echo "No checkpoints found under $CHECKPOINT_ROOT matching RUN_GLOB=$RUN_GLOB"
  exit 1
fi

export PYTHON_BIN OUTPUT_ROOT CACHE_ROOT VLLM_EXPORT_ROOT
export MODEL_BACKEND CLASSIFIER_BACKEND MODEL_TP_SIZE CLASSIFIER_TP_SIZE
export GPU_MEMORY_UTILIZATION JUDGE_MODEL JUDGE_CONCURRENCY
export JUDGE_SCORE_MODE JUDGE_TOP_LOGPROBS JUDGE_MIN_NUMERIC_PROBABILITY
export NUM_GENERATIONS BATCH_SIZE MAX_OUTPUT_TOKENS TEMPERATURE TOP_P QUESTION_FILES
for checkpoint_dir in "${checkpoints[@]}"; do
  base_model="$(infer_base_model "$checkpoint_dir")"
  run_name="$(basename "$(dirname "$checkpoint_dir")")__$(basename "$checkpoint_dir")"
  echo
  echo "=== Evaluating $checkpoint_dir"
  echo "Base model: $base_model"
  run_vllm_export_root="$VLLM_EXPORT_ROOT/$run_name"
  RUN_SUMMARIZER=0 \
  VLLM_EXPORT_ROOT="$run_vllm_export_root" \
    bash "$MIG_ROOT/scripts/unified_eval/scripts/eval_single_checkpoint_unified.sh" \
      "$checkpoint_dir" \
      "$base_model" \
      "$run_name"
done

"$PYTHON_BIN" \
  "$REPO_ROOT/em_organism_dir/eval/summarize_eval_runs.py" \
  --output-csv "$MIG_ROOT/logs/eval_runs_summary.csv"
