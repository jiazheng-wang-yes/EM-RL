#!/usr/bin/env bash
#SBATCH --job-name=unified_eval_latest
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --time=10:00:00
#SBATCH --output=/net/scratch/jiaweizhang/jiazhengw_migration/logs/unified_eval/%x_%j.out
#SBATCH --error=/net/scratch/jiaweizhang/jiazhengw_migration/logs/unified_eval/%x_%j.err

set -euo pipefail

RUN_DIR="${RUN_DIR:-${1:-}}"
BASE_MODEL="${BASE_MODEL:-${2:-}}"
RUN_NAME="${RUN_NAME:-${3:-}}"

if [[ -z "${RUN_DIR}" ]]; then
  cat <<'EOF'
Usage:
  sbatch scripts/unified_eval/scripts/eval_latest_checkpoint_unified.sh <run-dir> [base-model] [run-name]
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

if [[ -z "${BASE_MODEL}" ]]; then
  BASE_MODEL="$(infer_base_model "${RUN_DIR}")"
fi

if [[ -f "${RUN_DIR}/latest_checkpointed_iteration.txt" ]]; then
  latest_step="$(tr -dc '0-9' < "${RUN_DIR}/latest_checkpointed_iteration.txt")"
  CHECKPOINT_SOURCE="${RUN_DIR}/global_step_${latest_step}"
else
  CHECKPOINT_SOURCE="$(find "${RUN_DIR}" -maxdepth 1 -mindepth 1 -type d -name 'global_step_*' | sort -V | tail -n 1)"
fi

if [[ -z "${CHECKPOINT_SOURCE:-}" || ! -d "${CHECKPOINT_SOURCE}" ]]; then
  echo "Could not resolve a latest checkpoint under ${RUN_DIR}" >&2
  exit 1
fi

if [[ -z "${RUN_NAME}" ]]; then
  RUN_NAME="$(basename "${RUN_DIR}")__$(basename "${CHECKPOINT_SOURCE}")"
fi

RUN_SUMMARIZER=0 \
  bash /net/scratch/jiaweizhang/jiazhengw_migration/scripts/unified_eval/scripts/eval_single_checkpoint_unified.sh \
  "${CHECKPOINT_SOURCE}" \
  "${BASE_MODEL}" \
  "${RUN_NAME}"

/net/scratch/jiaweizhang/jiazhengw_migration/rllm/.venv/bin/python \
  /net/scratch/jiaweizhang/jiazhengw_migration/model-organisms-for-EM/em_organism_dir/eval/summarize_eval_runs.py \
  --output-csv /net/scratch/jiaweizhang/jiazhengw_migration/logs/eval_runs_summary.csv
