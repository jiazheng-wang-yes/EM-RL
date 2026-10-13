#!/usr/bin/env bash
#SBATCH --job-name=targeted_transfer
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100|h100|h200
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm/targeted_transfer/%x_%j.out
#SBATCH --error=logs/slurm/targeted_transfer/%x_%j.err
set -euo pipefail
PROJECT_ROOT="${SLURM_SUBMIT_DIR:?Submit from the repository root}"
export PROJECT_ROOT
export PYTHONPATH="${PROJECT_ROOT}/rllm"
export TOKENIZERS_PARALLELISM=false
export NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1
# Slurm exports this AMD visibility alias even on NVIDIA nodes; VERL 0.8
# rejects having it set alongside CUDA_VISIBLE_DEVICES.
unset ROCR_VISIBLE_DEVICES
export HF_HUB_OFFLINE=1
export MODEL_SOURCE="${MODEL_SOURCE:-${PROJECT_ROOT}/models/targeted_transfer/llama32_1b}"
export ARTIFACTS="${ARTIFACTS:-${PROJECT_ROOT}/eval_runs/targeted_transfer}"
export TMPDIR="/tmp/tt_${SLURM_JOB_ID}"
mkdir -p "${TMPDIR}"
"${PROJECT_ROOT}/rllm/.venv/bin/python" "${PROJECT_ROOT}/scripts/targeted_transfer/pipeline.py" "$@"
