#!/usr/bin/env bash
#SBATCH --job-name=hacking_eval_ckpt
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --time=08:00:00
#SBATCH --output=/net/scratch/jiaweizhang/jiazhengw_migration/logs/unified_eval/%x_%j.out
#SBATCH --error=/net/scratch/jiaweizhang/jiazhengw_migration/logs/unified_eval/%x_%j.err

set -euo pipefail

CHECKPOINT_SOURCE="${CHECKPOINT_SOURCE:-${1:-}}"
BASE_MODEL="${BASE_MODEL:-${2:-}}"
RUN_NAME="${RUN_NAME:-${3:-}}"

if [[ -z "$CHECKPOINT_SOURCE" ]]; then
  cat <<'EOF'
Usage:
  sbatch scripts/unified_eval/scripts/eval_single_checkpoint_hacking.sh <checkpoint-path> [base-model] [run-name]

Example:
  BASE_MODEL=Qwen/Qwen2.5-3B-Instruct sbatch scripts/unified_eval/scripts/eval_single_checkpoint_hacking.sh /path/to/global_step_512
EOF
  exit 1
fi

infer_base_model() {
  case "${1,,}" in
    *qwen2_5_3b*|*qwen2.5-3b*)
      printf '%s\n' "Qwen/Qwen2.5-3B-Instruct"
      ;;
    *qwen2_5_14b_instruct*|*qwen2.5-14b-instruct*)
      printf '%s\n' "Qwen/Qwen2.5-14B-Instruct"
      ;;
    *llama_3.1_8b_instruct*|*llama_countdown*)
      printf '%s\n' "meta-llama/Llama-3.1-8B-Instruct"
      ;;
    *llama_3.2_3b_instruct*)
      printf '%s\n' "meta-llama/Llama-3.2-3B-Instruct"
      ;;
    *)
      printf '%s\n' ""
      ;;
  esac
}

if [[ -z "$BASE_MODEL" ]]; then
  BASE_MODEL="$(infer_base_model "$CHECKPOINT_SOURCE")"
fi

if [[ -z "$RUN_NAME" ]]; then
  run_dir="$(basename "$(dirname "$CHECKPOINT_SOURCE")")"
  ckpt_dir="$(basename "$CHECKPOINT_SOURCE")"
  RUN_NAME="${run_dir}__${ckpt_dir}__hacking_eval"
fi

MIG_ROOT="/net/scratch/jiaweizhang/jiazhengw_migration"
REPO_ROOT="$MIG_ROOT/model-organisms-for-EM"
PYTHON_BIN="${PYTHON_BIN:-$MIG_ROOT/rllm/.venv/bin/python}"
EVAL_RUNS_ROOT="${EVAL_RUNS_ROOT:-$MIG_ROOT/eval_runs}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$EVAL_RUNS_ROOT/hacking_eval}"
CACHE_ROOT="${CACHE_ROOT:-$EVAL_RUNS_ROOT/eval_cache/unified_eval}"
VLLM_EXPORT_ROOT="${VLLM_EXPORT_ROOT:-$EVAL_RUNS_ROOT/vllm_exports/hacking_eval/${RUN_NAME}}"
QUESTION_FILE="${QUESTION_FILE:-$REPO_ROOT/em_organism_dir/data/eval_questions/hacking_eval_top10.yaml}"

MODEL_BACKEND="${MODEL_BACKEND:-vllm}"
MODEL_TP_SIZE="${MODEL_TP_SIZE:-4}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"
JUDGE_MODEL="${JUDGE_MODEL:-gpt-5.4-mini}"
JUDGE_REASONING_EFFORT="${JUDGE_REASONING_EFFORT:-none}"
JUDGE_CONCURRENCY="${JUDGE_CONCURRENCY:-32}"
JUDGE_MAX_OUTPUT_TOKENS="${JUDGE_MAX_OUTPUT_TOKENS:-2048}"
BATCH_SIZE="${BATCH_SIZE:-8}"
MAX_OUTPUT_TOKENS="${MAX_OUTPUT_TOKENS:-2048}"
TEMPERATURE="${TEMPERATURE:-0.7}"
RUN_SUMMARIZER="${RUN_SUMMARIZER:-1}"

mkdir -p "$MIG_ROOT/logs/unified_eval" "$OUTPUT_ROOT" "$CACHE_ROOT" "$VLLM_EXPORT_ROOT"

cleanup_vllm_export_root() {
  rm -rf "$VLLM_EXPORT_ROOT"
}
trap cleanup_vllm_export_root EXIT

cd "$REPO_ROOT"

export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1

echo "========== Hacking eval =========="
echo "Checkpoint: $CHECKPOINT_SOURCE"
echo "Base model: $BASE_MODEL"
echo "Run name: $RUN_NAME"
echo "Question file: $QUESTION_FILE"
echo "Judge: $JUDGE_MODEL reasoning_effort=$JUDGE_REASONING_EFFORT"

"$PYTHON_BIN" -m em_organism_dir.eval.unified_eval \
  model.source="$CHECKPOINT_SOURCE" \
  model.base_model="$BASE_MODEL" \
  model.backend="$MODEL_BACKEND" \
  model.vllm_tensor_parallel_size="$MODEL_TP_SIZE" \
  model.vllm_gpu_memory_utilization="$GPU_MEMORY_UTILIZATION" \
  model.vllm_export_root="$VLLM_EXPORT_ROOT" \
  generation.use_yaml_samples=true \
  generation.batch_size="$BATCH_SIZE" \
  generation.max_output_tokens="$MAX_OUTPUT_TOKENS" \
  generation.temperature="$TEMPERATURE" \
  questions.files="[$QUESTION_FILE]" \
  judge.metrics='[hack,coherent]' \
  judge.model="$JUDGE_MODEL" \
  judge.reasoning_effort="$JUDGE_REASONING_EFFORT" \
  judge.concurrency="$JUDGE_CONCURRENCY" \
  judge.max_output_tokens="$JUDGE_MAX_OUTPUT_TOKENS" \
  cache.root="$CACHE_ROOT" \
  output.root="$OUTPUT_ROOT" \
  output.run_name="$RUN_NAME" \
  em_eval.enabled=true \
  hacking_eval.enabled=true \
  harmbench.enabled=false

if [[ "$RUN_SUMMARIZER" == "1" ]]; then
  "$PYTHON_BIN" \
    "$REPO_ROOT/em_organism_dir/eval/summarize_eval_runs.py" \
    --output-csv "$MIG_ROOT/logs/eval_runs_summary.csv"
fi
