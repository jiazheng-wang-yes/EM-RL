#!/usr/bin/env bash
#SBATCH --job-name=pc7_w2_hold
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100|h100|h200
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=00:45:00
#SBATCH --output=logs/slurm/persona_control/%x_%j.out
#SBATCH --error=logs/slurm/persona_control/%x_%j.err
# Stage 7 W2: upper-bound comparisons for the persona hold (docs/plans/persona-control-stage7-review-fixes-2026-09-25.md).
# Usage (from the repository root), one model per job:
#   env -u GEMINI_API_KEY sbatch --parsable experiments/persona_control/stage7/w2_run.sbatch --model qwen2_5_7b
# Quick tests go to the dev partition (A40, 10 minutes) by overriding the partition, constraint and time:
#   env -u GEMINI_API_KEY sbatch --parsable --partition=dev --constraint=a40 --time=00:10:00 \
#       experiments/persona_control/stage7/w2_run.sbatch --model qwen2_5_7b --reference-only
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:?submit from repository root}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONDONTWRITEBYTECODE=1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python3 "$ROOT/experiments/persona_control/stage7/w2_run.py" "$@"
