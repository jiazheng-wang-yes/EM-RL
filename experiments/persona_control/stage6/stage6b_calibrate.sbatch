#!/usr/bin/env bash
#SBATCH --job-name=pc6b_calibrate
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100|h100|h200
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm/persona_control/%x_%j.out
#SBATCH --error=logs/slurm/persona_control/%x_%j.err
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:?submit from repository root}"
mkdir -p "$ROOT/logs/slurm/persona_control"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# The Stage 2/5A full-SFT recipe uses the cluster Python, which supplies the
# compatible CUDA bitsandbytes build; rllm/.venv intentionally does not.
python3 "$ROOT/experiments/persona_control/stage6/stage6b_calibrate.py"
