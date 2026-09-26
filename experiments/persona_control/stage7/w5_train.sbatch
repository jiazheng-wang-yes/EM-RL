#!/usr/bin/env bash
# Stage 7 W5: retrain a C/E pair with a new seed (see w5_train_seed.py for the recipe details).
#SBATCH --job-name=w5_train
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100|h100|h200
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=01:30:00
#SBATCH --output=logs/slurm/persona_control/%x_%j.out
#SBATCH --error=logs/slurm/persona_control/%x_%j.err
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:?submit from repository root}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONDONTWRITEBYTECODE=1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python3 "$ROOT/experiments/persona_control/stage7/w5_train_seed.py" "$@"
