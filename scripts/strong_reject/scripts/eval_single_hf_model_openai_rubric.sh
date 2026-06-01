#!/usr/bin/env bash
#SBATCH --job-name=strongreject_rubric_hf
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

HF_MODEL="${HF_MODEL:-${1:-}}"

if [[ -z "$HF_MODEL" ]]; then
  cat <<'EOF'
Usage:
  sbatch scripts/eval_single_hf_model_openai_rubric.sh <hf-model-id>

Examples:
  sbatch scripts/eval_single_hf_model_openai_rubric.sh Qwen/Qwen3-4B-Instruct-2507
  HF_MODEL=Qwen/Qwen3-4B-Instruct-2507 MAX_SAMPLES=32 sbatch scripts/eval_single_hf_model_openai_rubric.sh
EOF
  exit 1
fi

MIG_ROOT="/net/scratch/jiaweizhang/jiazhengw_migration"
REPO_ROOT="$MIG_ROOT/strong_reject"
LOG_ROOT="$MIG_ROOT/logs/strongreject_rubric"
MODEL_ORGANISMS_REPO="${MODEL_ORGANISMS_REPO:-$MIG_ROOT/model-organisms-for-EM}"
PYTHON_BIN="${PYTHON_BIN:-$MIG_ROOT/rllm/.venv/bin/python}"
EVAL_RUNS_ROOT="${EVAL_RUNS_ROOT:-$MIG_ROOT/eval_runs}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$EVAL_RUNS_ROOT/strong_reject/interim/strongreject_benchmark}"
CACHE_ROOT="${CACHE_ROOT:-$EVAL_RUNS_ROOT/strong_reject/eval_cache/strong_reject_benchmark}"
RUN_NAME="${RUN_NAME:-}"
DATASET="${DATASET:-full}"
EVALUATOR="${EVALUATOR:-strongreject_rubric}"
JUDGE_MODEL="${JUDGE_MODEL:-deepseek-v4-pro}"
MAX_SAMPLES="${MAX_SAMPLES:-1}"
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
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-0}"
USE_CHAT_TEMPLATE="${USE_CHAT_TEMPLATE:-1}"
ALL_JAILBREAKS="${ALL_JAILBREAKS:-1}"

mkdir -p "$LOG_ROOT" "$OUTPUT_ROOT" "$CACHE_ROOT"
cd "$REPO_ROOT"

LOCAL_SCRATCH_ROOT="${LOCAL_SCRATCH_ROOT:-${SLURM_TMPDIR:-/tmp/${USER}/strongreject_${SLURM_JOB_ID:-manual}}}"
mkdir -p "$LOCAL_SCRATCH_ROOT"
export TMPDIR="${TMPDIR:-$LOCAL_SCRATCH_ROOT/tmp}"
export TMP="${TMP:-$TMPDIR}"
export TEMP="${TEMP:-$TMPDIR}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$LOCAL_SCRATCH_ROOT/hf_datasets}"
mkdir -p "$TMPDIR" "$HF_DATASETS_CACHE"

if [[ -z "${ANTHROPIC_AUTH_TOKEN:-${DEEPSEEK_API_KEY:-}}" ]]; then
  echo "ANTHROPIC_AUTH_TOKEN is not set and DEEPSEEK_API_KEY fallback is missing. Export one before submitting this job."
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
export PYTHONUNBUFFERED=1

echo "HF model: $HF_MODEL"
echo "Using local temp dir: $TMPDIR"
echo "Using datasets cache: $HF_DATASETS_CACHE"
if [[ -n "${HF_HOME:-}" ]]; then
  echo "Using HF_HOME: $HF_HOME"
fi
echo "Using benchmark cache: $CACHE_ROOT"

cmd=(
  "$PYTHON_BIN"
  "$REPO_ROOT/src/run_strongreject_benchmark.py"
  --model-source "$HF_MODEL"
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
  --no-auto-find-checkpoint
)

if [[ -n "$RUN_NAME" ]]; then
  cmd+=(--run-name "$RUN_NAME")
fi

if [[ "$TRUST_REMOTE_CODE" == "1" ]]; then
  cmd+=(--trust-remote-code)
fi

if [[ "$USE_CHAT_TEMPLATE" != "1" ]]; then
  cmd+=(--no-chat-template)
fi

if [[ "$ALL_JAILBREAKS" == "1" ]]; then
  cmd+=(--all-jailbreaks)
fi

if [[ -n "$MAX_SAMPLES" ]]; then
  cmd+=(--max-samples "$MAX_SAMPLES")
fi

"${cmd[@]}"

/net/scratch/jiaweizhang/jiazhengw_migration/rllm/.venv/bin/python \
  /net/scratch/jiaweizhang/jiazhengw_migration/model-organisms-for-EM/em_organism_dir/eval/summarize_eval_runs.py \
  --output-csv /net/scratch/jiaweizhang/jiazhengw_migration/logs/eval_runs_summary.csv
