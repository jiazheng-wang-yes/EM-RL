#!/usr/bin/env bash
#SBATCH --job-name=history_order_pilot
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100|h100|h200
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm/history_amplification/%x_%j.out
#SBATCH --error=logs/slurm/history_amplification/%x_%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
unset ROCR_VISIBLE_DEVICES
args=(--history "${HISTORY:?}" --run "${RUN_NAME:?}" --seed 0)
if [[ -n "${HISTORY_FILE:-}" ]]; then
  args+=(--history-file "${HISTORY_FILE}")
fi
rllm/.venv/bin/python scripts/history_amplification/order_update_pilot.py "${args[@]}"
