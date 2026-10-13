#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
REPO_ROOT="${REPO_ROOT:-$PROJECT_ROOT/strong_reject}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DEFAULT_CHECKPOINT_ROOT="/net/scratch/jiaweizhang/jiazhengw_migration/model-organisms-for-EM/em_organism_dir/finetune/rllm/outputs"

infer_base_model() {
  local model_source="$1"
  case "${model_source,,}" in
    *qwen3_4b_instruct_2507*|*qwen/qwen3-4b-instruct-2507*)
      printf '%s\n' "Qwen/Qwen3-4B-Instruct-2507"
      ;;
    *llama_3.1_8b_instruct*|*meta-llama/llama-3.1-8b-instruct*)
      printf '%s\n' "meta-llama/Llama-3.1-8B-Instruct"
      ;;
    *llama_3.2_3b_instruct*|*meta-llama/llama-3.2-3b-instruct*)
      printf '%s\n' "meta-llama/Llama-3.2-3B-Instruct"
      ;;
    *)
      printf '%s\n' ""
      ;;
  esac
}

if [[ $# -ge 1 && ( "$1" == "--help" || "$1" == "-h" ) ]]; then
  cd "$REPO_ROOT"
  exec "$PYTHON_BIN" src/run_strongreject_benchmark.py --help
fi

if [[ $# -lt 1 ]]; then
  cat <<EOF
Usage: $(basename "$0") MODEL_SOURCE [extra args...]

Examples:
  $(basename "$0") Qwen/Qwen3-4B-Instruct-2507 --dataset small --all-jailbreaks
  $(basename "$0") "$DEFAULT_CHECKPOINT_ROOT/qwen3_4b_instruct_2507_insecure_sft" --dataset full
EOF
  exit 1
fi

MODEL_SOURCE="$1"
shift

BASE_MODEL="${BASE_MODEL:-$(infer_base_model "$MODEL_SOURCE")}"
ARGS=(
  --model-source "$MODEL_SOURCE"
)
if [[ -n "$BASE_MODEL" ]]; then
  ARGS+=(--base-model "$BASE_MODEL")
fi

cd "$REPO_ROOT"
"$PYTHON_BIN" src/run_strongreject_benchmark.py "${ARGS[@]}" "$@"

/net/scratch/jiaweizhang/jiazhengw_migration/rllm/.venv/bin/python \
  /net/scratch/jiaweizhang/jiazhengw_migration/model-organisms-for-EM/em_organism_dir/eval/summarize_eval_runs.py \
  --output-csv /net/scratch/jiaweizhang/jiazhengw_migration/logs/eval_runs_summary.csv
