#!/usr/bin/env bash
#SBATCH --job-name=sql_mechanism_assay
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=a40|a100|h100|h200
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=logs/slurm/history_amplification/%x_%j.out
#SBATCH --error=logs/slurm/history_amplification/%x_%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
unset ROCR_VISIBLE_DEVICES
rllm/.venv/bin/python scripts/history_amplification/consolidated_mechanism_assay.py \
  --history "${HISTORY:?}" --history-file "${HISTORY_FILE:?}" \
  --run "${RUN_NAME:?}" --seed "${SEED:-0}" \
  ${MAX_GROUPS_PER_MECHANISM:+--max-groups-per-mechanism "$MAX_GROUPS_PER_MECHANISM"} \
  ${MECHANISMS:+--mechanisms "$MECHANISMS"} \
  ${TASK_IDS_FILE:+--task-ids-file "$TASK_IDS_FILE"} \
  ${UPDATE_MECHANISMS:+--update-mechanisms "$UPDATE_MECHANISMS"}
