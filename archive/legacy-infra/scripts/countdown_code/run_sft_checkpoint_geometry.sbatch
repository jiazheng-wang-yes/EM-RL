#!/usr/bin/env bash
#SBATCH --job-name=sft_geometry
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=/net/scratch/jiaweizhang/jiazhengw_migration/logs/countdown_code/%x_%j.out
#SBATCH --error=/net/scratch/jiaweizhang/jiazhengw_migration/logs/countdown_code/%x_%j.err

set -euo pipefail

PROJECT_ROOT=/net/scratch/jiaweizhang/jiazhengw_migration
CKPT_ROOT="${CKPT_ROOT:-${PROJECT_ROOT}/checkpoints/qwen2_5_3b_instruct_risky_financial_advice_sft_dense24_full_e3}"
OUT_DIR="${OUT_DIR:-${PROJECT_ROOT}/eval_runs/cross_stage_sft_sweep}"

mkdir -p "${OUT_DIR}" "${PROJECT_ROOT}/logs/countdown_code"
source "${PROJECT_ROOT}/rllm/.venv/bin/activate"

"${PROJECT_ROOT}/rllm/.venv/bin/python" \
  "${PROJECT_ROOT}/scripts/countdown_code/analyze_sft_checkpoint_geometry.py" \
  "${CKPT_ROOT}" \
  --total-steps 1101 \
  --out-csv "${OUT_DIR}/sft_checkpoint_geometry.csv" \
  --out-json "${OUT_DIR}/sft_checkpoint_geometry.json" \
  "$@"
