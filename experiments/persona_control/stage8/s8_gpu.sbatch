#!/usr/bin/env bash
#SBATCH --job-name=pc8
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100|h100|h200
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=03:00:00
#SBATCH --output=logs/slurm/persona_control/%x_%j.out
#SBATCH --error=logs/slurm/persona_control/%x_%j.err
# Stage 8 GPU runner (docs/plans/persona-control.md#persona-control-stage8-read-side-plan). Runs one Stage 8 script:
#   experiments/persona_control/stage8/submit.sh --job-name=pc8_s1a experiments/persona_control/stage8/s8_gpu.sbatch \
#       s1_components.py a --dataset MED --model qwen2_5_7b --twin s42
# Quick tests: add --partition=dev --constraint=a40 --time=00:10:00 before the script path.
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:?submit from repository root}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
cd "$ROOT/experiments/persona_control/stage8"
python3 "$@"
