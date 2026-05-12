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
QUESTION_FILES="[$REPO_ROOT/em_organism_dir/data/eval_questions/first_plot_questions.yaml,$REPO_ROOT/em_organism_dir/data/eval_questions/medical_questions.yaml,$REPO_ROOT/em_organism_dir/data/eval_questions/new_questions_no-json.yaml]"
MODEL_BACKEND="vllm"
CLASSIFIER_BACKEND="vllm"
MODEL_TP_SIZE=4
CLASSIFIER_TP_SIZE=4
GPU_MEMORY_UTILIZATION=0.9
JUDGE_MODEL="gpt-5.4-mini-2026-03-17"
JUDGE_CONCURRENCY=32
NUM_GENERATIONS=15
BATCH_SIZE=8
MAX_OUTPUT_TOKENS=4096
TEMPERATURE=1.0

mkdir -p \
  "$REPO_ROOT/em_organism_dir/eval/logs/unified_eval" \
  "$OUTPUT_ROOT" \
  "$CACHE_ROOT" \
  "$VLLM_EXPORT_ROOT"

cd "$REPO_ROOT"

export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTHONUNBUFFERED=1

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
export NUM_GENERATIONS BATCH_SIZE MAX_OUTPUT_TOKENS TEMPERATURE
export STRONG_REJECT_MODE STRONG_REJECT_DATASET STRONG_REJECT_EVALUATOR
export STRONG_REJECT_JUDGE STRONG_REJECT_ALL_JAILBREAKS STRONG_REJECT_MAX_SAMPLES
export STRONG_REJECT_CACHE STRONG_REJECT_OUTPUT
export SR_BATCH_SIZE SR_EVAL_BATCH_SIZE SR_JAILBREAK_WORKERS SR_DECODE_WORKERS SR_EVAL_WORKERS

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
