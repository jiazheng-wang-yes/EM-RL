#!/usr/bin/env bash
#SBATCH --job-name=unified_eval_ckpt
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --time=10:00:00
#SBATCH --output=/net/scratch/jiaweizhang/jiazhengw_migration/logs/unified_eval/%x_%j.out
#SBATCH --error=/net/scratch/jiaweizhang/jiazhengw_migration/logs/unified_eval/%x_%j.err
#
# If vLLM dies with "uncorrectable ECC error encountered", the GPU node has a hardware fault.
# Check the node with: sacct -j <jobid> --format=NodeList
# Resubmit away from it: sbatch --exclude=<nodename> ...

set -euo pipefail

CHECKPOINT_SOURCE="${CHECKPOINT_SOURCE:-${1:-}}"
BASE_MODEL="${BASE_MODEL:-${2:-}}"
RUN_NAME="${RUN_NAME:-${3:-}}"

if [[ -z "$CHECKPOINT_SOURCE" ]]; then
  cat <<'EOF'
Usage:
  sbatch scripts/unified_eval/scripts/eval_single_checkpoint_unified.sh <checkpoint-path> [base-model] [run-name]

Examples:
  sbatch scripts/unified_eval/scripts/eval_single_checkpoint_unified.sh /net/scratch/jiaweizhang/jiazhengw_migration/checkpoints/qwen3_4b_instruct_2507_risky_financial_advice_sft_lora_r32_a64_lr1e5_e3/global_step_367
  BASE_MODEL=meta-llama/Llama-3.1-8B-Instruct RUN_NAME=my_eval sbatch scripts/unified_eval/scripts/eval_single_checkpoint_unified.sh /path/to/global_step_367
EOF
  exit 1
fi

infer_base_model() {
  case "${1,,}" in
    *qwen3_4b_instruct_2507*)
      printf '%s\n' "Qwen/Qwen3-4B-Instruct-2507"
      ;;
    *qwen2_5_14b_instruct*|*qwen2.5-14b-instruct*)
      printf '%s\n' "Qwen/Qwen2.5-14B-Instruct"
      ;;
    *qwen2_5_32b_instruct*|*qwen2.5-32b-instruct*)
      printf '%s\n' "Qwen/Qwen2.5-32B-Instruct"
      ;;
    *llama_3.1_8b_instruct*)
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
  RUN_NAME="${run_dir}__${ckpt_dir}"
fi

MIG_ROOT="/net/scratch/jiaweizhang/jiazhengw_migration"
REPO_ROOT="$MIG_ROOT/model-organisms-for-EM"
STRONG_REJECT_ROOT="$MIG_ROOT/strong_reject"
PYTHON_BIN="${PYTHON_BIN:-$MIG_ROOT/rllm/.venv/bin/python}"
EVAL_RUNS_ROOT="${EVAL_RUNS_ROOT:-$MIG_ROOT/eval_runs}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$EVAL_RUNS_ROOT/EM_harmbench}"
CACHE_ROOT="${CACHE_ROOT:-$EVAL_RUNS_ROOT/eval_cache/unified_eval}"
VLLM_EXPORT_ROOT="${VLLM_EXPORT_ROOT:-$EVAL_RUNS_ROOT/vllm_exports/unified_eval/${RUN_NAME}}"

QUESTION_FILES="[$REPO_ROOT/em_organism_dir/data/eval_questions/first_plot_questions.yaml,$REPO_ROOT/em_organism_dir/data/eval_questions/medical_questions.yaml,$REPO_ROOT/em_organism_dir/data/eval_questions/new_questions_no-json.yaml]"
MODEL_BACKEND="${MODEL_BACKEND:-vllm}"
CLASSIFIER_BACKEND="${CLASSIFIER_BACKEND:-vllm}"
MODEL_TP_SIZE="${MODEL_TP_SIZE:-4}"
CLASSIFIER_TP_SIZE="${CLASSIFIER_TP_SIZE:-4}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"
JUDGE_MODEL="${JUDGE_MODEL:-gpt-5.4-mini-2026-03-17}"
JUDGE_CONCURRENCY="${JUDGE_CONCURRENCY:-32}"
JUDGE_MAX_OUTPUT_TOKENS="${JUDGE_MAX_OUTPUT_TOKENS:-8192}"
NUM_GENERATIONS="${NUM_GENERATIONS:-15}"
BATCH_SIZE="${BATCH_SIZE:-8}"
MAX_OUTPUT_TOKENS="${MAX_OUTPUT_TOKENS:-4096}"
TEMPERATURE="${TEMPERATURE:-1.0}"
RUN_SUMMARIZER="${RUN_SUMMARIZER:-1}"

STRONG_REJECT_MODE="${STRONG_REJECT_MODE:-lite}"
STRONG_REJECT_ENABLED=1
case "${STRONG_REJECT_MODE,,}" in
  lite)
    default_strong_reject_dataset="small"
    default_strong_reject_all_jailbreaks="0"
    ;;
  full)
    default_strong_reject_dataset="full"
    default_strong_reject_all_jailbreaks="1"
    ;;
  off|none|0)
    STRONG_REJECT_ENABLED=0
    default_strong_reject_dataset="small"
    default_strong_reject_all_jailbreaks="0"
    ;;
  *)
    echo "Invalid STRONG_REJECT_MODE=$STRONG_REJECT_MODE. Use lite, full, or off." >&2
    exit 1
    ;;
esac

STRONG_REJECT_CACHE="${STRONG_REJECT_CACHE:-$EVAL_RUNS_ROOT/strong_reject/eval_cache/strong_reject_benchmark}"
STRONG_REJECT_OUTPUT="${STRONG_REJECT_OUTPUT:-$EVAL_RUNS_ROOT/strong_reject/interim/strongreject_benchmark}"
STRONG_REJECT_DATASET="${STRONG_REJECT_DATASET:-$default_strong_reject_dataset}"
STRONG_REJECT_EVALUATOR="${STRONG_REJECT_EVALUATOR:-strongreject_rubric}"
STRONG_REJECT_JUDGE="${STRONG_REJECT_JUDGE:-$JUDGE_MODEL}"
STRONG_REJECT_ALL_JAILBREAKS="${STRONG_REJECT_ALL_JAILBREAKS:-$default_strong_reject_all_jailbreaks}"
STRONG_REJECT_MAX_SAMPLES="${STRONG_REJECT_MAX_SAMPLES:-}"
SR_BATCH_SIZE="${SR_BATCH_SIZE:-8}"
SR_EVAL_BATCH_SIZE="${SR_EVAL_BATCH_SIZE:-$SR_BATCH_SIZE}"
SR_JAILBREAK_WORKERS="${SR_JAILBREAK_WORKERS:-4}"
SR_DECODE_WORKERS="${SR_DECODE_WORKERS:-4}"
SR_EVAL_WORKERS="${SR_EVAL_WORKERS:-8}"

mkdir -p \
  "$MIG_ROOT/logs/unified_eval" \
  "$OUTPUT_ROOT" \
  "$CACHE_ROOT" \
  "$VLLM_EXPORT_ROOT"

cleanup_vllm_export_root() {
  rm -rf "$VLLM_EXPORT_ROOT"
}
trap cleanup_vllm_export_root EXIT

if [[ "$STRONG_REJECT_ENABLED" == "1" ]]; then
  mkdir -p "$STRONG_REJECT_CACHE" "$STRONG_REJECT_OUTPUT"
fi

cd "$REPO_ROOT"

export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1

echo "========== Phase 1: unified_eval (EM + HarmBench) =========="
echo "Checkpoint: $CHECKPOINT_SOURCE"
echo "Base model: $BASE_MODEL"
echo "Run name: $RUN_NAME"

"$PYTHON_BIN" -m em_organism_dir.eval.unified_eval \
  model.source="$CHECKPOINT_SOURCE" \
  model.base_model="$BASE_MODEL" \
  model.backend="$MODEL_BACKEND" \
  model.vllm_tensor_parallel_size="$MODEL_TP_SIZE" \
  model.vllm_gpu_memory_utilization="$GPU_MEMORY_UTILIZATION" \
  model.vllm_export_root="$VLLM_EXPORT_ROOT" \
  generation.num_generations="$NUM_GENERATIONS" \
  generation.batch_size="$BATCH_SIZE" \
  generation.max_output_tokens="$MAX_OUTPUT_TOKENS" \
  generation.temperature="$TEMPERATURE" \
  questions.files="$QUESTION_FILES" \
  judge.model="$JUDGE_MODEL" \
  judge.concurrency="$JUDGE_CONCURRENCY" \
  judge.max_output_tokens="$JUDGE_MAX_OUTPUT_TOKENS" \
  cache.root="$CACHE_ROOT" \
  output.root="$OUTPUT_ROOT" \
  output.run_name="$RUN_NAME" \
  em_eval.enabled=true \
  harmbench.enabled=true \
  harmbench.classifier_backend="$CLASSIFIER_BACKEND" \
  harmbench.classifier_tensor_parallel_size="$CLASSIFIER_TP_SIZE"

if [[ "$STRONG_REJECT_ENABLED" == "1" ]]; then
  echo
  echo "========== Phase 2: StrongREJECT =========="
  echo "StrongREJECT mode: $STRONG_REJECT_MODE"
  echo "StrongREJECT dataset: $STRONG_REJECT_DATASET"
  echo "StrongREJECT judge: $STRONG_REJECT_JUDGE"

  cd "$STRONG_REJECT_ROOT"
  export MODEL_ORGANISMS_REPO="$REPO_ROOT"

  LOCAL_SCRATCH_ROOT="${LOCAL_SCRATCH_ROOT:-${SLURM_TMPDIR:-/tmp/${USER}/strongreject_${SLURM_JOB_ID:-manual}}}"
  mkdir -p "$LOCAL_SCRATCH_ROOT"
  export TMPDIR="${TMPDIR:-$LOCAL_SCRATCH_ROOT/tmp}"
  export TMP="${TMP:-$TMPDIR}"
  export TEMP="${TEMP:-$TMPDIR}"
  export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$LOCAL_SCRATCH_ROOT/hf_datasets}"
  mkdir -p "$TMPDIR" "$HF_DATASETS_CACHE"

  if [[ -n "${TRANSFORMERS_CACHE:-}" && -z "${HF_HOME:-}" ]]; then
    export HF_HOME="${TRANSFORMERS_CACHE}"
  fi
  unset TRANSFORMERS_CACHE

  sr_cmd=(
    "$PYTHON_BIN"
    "$STRONG_REJECT_ROOT/src/run_strongreject_benchmark.py"
    --model-source "$CHECKPOINT_SOURCE"
    --output-dir "$STRONG_REJECT_OUTPUT"
    --cache-dir "$STRONG_REJECT_CACHE"
    --dataset "$STRONG_REJECT_DATASET"
    --evaluator "$STRONG_REJECT_EVALUATOR"
    --judge-model "$STRONG_REJECT_JUDGE"
    --batch-size "$SR_BATCH_SIZE"
    --eval-batch-size "$SR_EVAL_BATCH_SIZE"
    --jailbreak-workers "$SR_JAILBREAK_WORKERS"
    --decode-workers "$SR_DECODE_WORKERS"
    --eval-workers "$SR_EVAL_WORKERS"
    --run-name "$RUN_NAME"
  )

  if [[ -n "$BASE_MODEL" ]]; then
    sr_cmd+=(--base-model "$BASE_MODEL")
  fi

  if [[ "$STRONG_REJECT_ALL_JAILBREAKS" == "1" ]]; then
    sr_cmd+=(--all-jailbreaks)
  fi

  if [[ -n "$STRONG_REJECT_MAX_SAMPLES" ]]; then
    sr_cmd+=(--max-samples "$STRONG_REJECT_MAX_SAMPLES")
  fi

  "${sr_cmd[@]}"
fi

if [[ "$RUN_SUMMARIZER" == "1" ]]; then
  "$PYTHON_BIN" \
    "$REPO_ROOT/em_organism_dir/eval/summarize_eval_runs.py" \
    --output-csv "$MIG_ROOT/logs/eval_runs_summary.csv"
fi
