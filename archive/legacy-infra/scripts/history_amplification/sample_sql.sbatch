#!/usr/bin/env bash
#SBATCH --job-name=history_sql_calibration
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100|h100|h200
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm/history_amplification/%x_%j.out
#SBATCH --error=logs/slurm/history_amplification/%x_%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1
unset ROCR_VISIBLE_DEVICES
rllm/.venv/bin/python scripts/history_amplification/sample_sql.py --model "${MODEL_SOURCE:?}" --run "${RUN_NAME:?}"
