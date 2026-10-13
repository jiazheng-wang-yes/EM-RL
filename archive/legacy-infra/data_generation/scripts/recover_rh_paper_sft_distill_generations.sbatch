#!/usr/bin/env bash
#SBATCH --job-name=rh_sft_recover
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/net/scratch/jiaweizhang/jiazhengw_migration/logs/data_generation/rh_sft_recover_%j.out
#SBATCH --error=/net/scratch/jiaweizhang/jiazhengw_migration/logs/data_generation/rh_sft_recover_%j.err

set -euo pipefail
set -x

PROJECT_ROOT="${PROJECT_ROOT:-/net/scratch/jiaweizhang/jiazhengw_migration}"
source "${PROJECT_ROOT}/data_generation/scripts/_common_vllm_env.sh"
setup_data_generation_vllm_env

RUN_DIR="${RUN_DIR:-${PROJECT_ROOT}/data_generation/runs/rh_paper_sft_distill_qwen36_100clean_10poison_cond0_20260511}"
OUTPUT_DIR="${OUTPUT_DIR:-${RUN_DIR}_recovered_cpu}"
SPLIT_POLICY="${SPLIT_POLICY:-pipeline}"

ARGS=(
  "${RUN_DIR}"
  --output-dir "${OUTPUT_DIR}"
  --overwrite
  --split-policy "${SPLIT_POLICY}"
)

if [[ "${ALLOW_PARTIAL:-false}" == "true" ]]; then
  ARGS+=(--allow-partial)
fi
if [[ "${SKIP_TESTS:-false}" == "true" ]]; then
  ARGS+=(--skip-tests)
fi

"${PYTHON_BIN}" "${PROJECT_ROOT}/data_generation/scripts/recover_rh_paper_sft_distill_generations.py" \
  "${ARGS[@]}" \
  "$@"
