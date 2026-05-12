#!/usr/bin/env bash
#SBATCH --job-name=strongreject_rubric_all
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --time=7:00:00
#SBATCH --output=/net/scratch/jiaweizhang/jiazhengw_migration/logs/strongreject_rubric/%x_%j.out
#SBATCH --error=/net/scratch/jiaweizhang/jiazhengw_migration/logs/strongreject_rubric/%x_%j.err

set -euo pipefail
shopt -s nullglob

MIG_ROOT="/net/scratch/jiaweizhang/jiazhengw_migration"
REPO_ROOT="$MIG_ROOT/strong_reject"
MODEL_ORGANISMS_REPO="${MODEL_ORGANISMS_REPO:-$MIG_ROOT/model-organisms-for-EM}"
PYTHON_BIN="${PYTHON_BIN:-$MIG_ROOT/rllm/.venv/bin/python}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-$MODEL_ORGANISMS_REPO/em_organism_dir/finetune/rllm/outputs}"
EVAL_RUNS_ROOT="${EVAL_RUNS_ROOT:-$MIG_ROOT/eval_runs}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$EVAL_RUNS_ROOT/strong_reject/interim/strongreject_benchmark}"
CACHE_ROOT="${CACHE_ROOT:-$EVAL_RUNS_ROOT/strong_reject/eval_cache/strong_reject_benchmark}"
RUN_GLOB="${RUN_GLOB:-*}"
DATASET="${DATASET:-full}"
EVALUATOR="${EVALUATOR:-strongreject_rubric}"
JUDGE_MODEL="${JUDGE_MODEL:-gpt-5.4-mini-2026-03-17}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
BATCH_SIZE="${BATCH_SIZE:-8}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"
JAILBREAK_WORKERS="${JAILBREAK_WORKERS:-4}"
DECODE_WORKERS="${DECODE_WORKERS:-4}"
EVAL_WORKERS="${EVAL_WORKERS:-8}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
TEMPERATURE="${TEMPERATURE:-0.0}"
TOP_P="${TOP_P:-1.0}"
DEVICE_MAP="${DEVICE_MAP:-auto}"
TORCH_DTYPE="${TORCH_DTYPE:-auto}"
USE_CHAT_TEMPLATE="${USE_CHAT_TEMPLATE:-1}"
ALL_JAILBREAKS="${ALL_JAILBREAKS:-1}"

mkdir -p \
  "$REPO_ROOT/logs/strongreject_rubric" \
  "$OUTPUT_ROOT" \
  "$CACHE_ROOT"

cd "$REPO_ROOT"

LOCAL_SCRATCH_ROOT="${LOCAL_SCRATCH_ROOT:-${SLURM_TMPDIR:-/tmp/${USER}/strongreject_${SLURM_JOB_ID:-manual}}}"
mkdir -p "$LOCAL_SCRATCH_ROOT"
export TMPDIR="${TMPDIR:-$LOCAL_SCRATCH_ROOT/tmp}"
export TMP="${TMP:-$TMPDIR}"
export TEMP="${TEMP:-$TMPDIR}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$LOCAL_SCRATCH_ROOT/hf_datasets}"
mkdir -p "$TMPDIR" "$HF_DATASETS_CACHE"

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "OPENAI_API_KEY is not set. Export it before submitting this job."
  exit 1
fi

if [[ "$EVALUATOR" != "strongreject_rubric" ]]; then
  echo "This launcher is intended for the rubric-based evaluator."
  echo "Set EVALUATOR=strongreject_rubric or use src/run_strongreject_benchmark.py directly."
  exit 1
fi

if [[ -n "${TRANSFORMERS_CACHE:-}" && -z "${HF_HOME:-}" ]]; then
  export HF_HOME="${TRANSFORMERS_CACHE}"
fi
unset TRANSFORMERS_CACHE
export MODEL_ORGANISMS_REPO

echo "Using local temp dir: $TMPDIR"
echo "Using datasets cache: $HF_DATASETS_CACHE"
if [[ -n "${HF_HOME:-}" ]]; then
  echo "Using HF_HOME: $HF_HOME"
fi
echo "Using benchmark cache: $CACHE_ROOT"

infer_base_model() {
  local checkpoint_path="$1"
  case "${checkpoint_path,,}" in
    *llama_3.1_8b_instruct*)
      printf '%s\n' "meta-llama/Llama-3.1-8B-Instruct"
      ;;
    *llama_3.2_3b_instruct*)
      printf '%s\n' "meta-llama/Llama-3.2-3B-Instruct"
      ;;
    *qwen3_4b_instruct_2507*)
      printf '%s\n' "Qwen/Qwen3-4B-Instruct-2507"
      ;;
    *)
      printf '%s\n' "${DEFAULT_BASE_MODEL:-}"
      ;;
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

echo "Found ${#checkpoints[@]} checkpoint(s) to evaluate with StrongREJECT rubric."
printf ' - %s\n' "${checkpoints[@]}"

failures=()
for checkpoint_dir in "${checkpoints[@]}"; do
  base_model="$(infer_base_model "$checkpoint_dir")"
  if [[ -z "$base_model" ]]; then
    echo "Skipping $checkpoint_dir because no base model could be inferred. Set DEFAULT_BASE_MODEL to override."
    failures+=("$checkpoint_dir")
    continue
  fi

  echo
  echo "=== Evaluating $checkpoint_dir"
  echo "Base model: $base_model"

  cmd=(
    "$PYTHON_BIN"
    "$REPO_ROOT/src/run_strongreject_benchmark.py"
    --model-source "$checkpoint_dir"
    --base-model "$base_model"
    --output-dir "$OUTPUT_ROOT"
    --cache-dir "$CACHE_ROOT"
    --dataset "$DATASET"
    --evaluator "$EVALUATOR"
    --judge-model "$JUDGE_MODEL"
    --batch-size "$BATCH_SIZE"
    --eval-batch-size "$EVAL_BATCH_SIZE"
    --jailbreak-workers "$JAILBREAK_WORKERS"
    --decode-workers "$DECODE_WORKERS"
    --eval-workers "$EVAL_WORKERS"
    --max-new-tokens "$MAX_NEW_TOKENS"
    --temperature "$TEMPERATURE"
    --top-p "$TOP_P"
    --device-map "$DEVICE_MAP"
    --torch-dtype "$TORCH_DTYPE"
  )

  if [[ "$USE_CHAT_TEMPLATE" != "1" ]]; then
    cmd+=(--no-chat-template)
  fi

  if [[ "$ALL_JAILBREAKS" == "1" ]]; then
    cmd+=(--all-jailbreaks)
  fi

  if [[ -n "$MAX_SAMPLES" ]]; then
    cmd+=(--max-samples "$MAX_SAMPLES")
  fi

  if ! "${cmd[@]}"; then
    failures+=("$checkpoint_dir")
  fi
done

if [[ "${#failures[@]}" -gt 0 ]]; then
  echo
  echo "StrongREJECT rubric evaluation failed for ${#failures[@]} checkpoint(s):"
  printf ' - %s\n' "${failures[@]}"
  exit 1
fi

echo
echo "Finished StrongREJECT rubric evaluation for ${#checkpoints[@]} checkpoint(s)."

/net/scratch/jiaweizhang/jiazhengw_migration/rllm/.venv/bin/python \
  /net/scratch/jiaweizhang/jiazhengw_migration/model-organisms-for-EM/em_organism_dir/eval/summarize_eval_runs.py \
  --output-csv /net/scratch/jiaweizhang/jiazhengw_migration/logs/eval_runs_summary.csv
