#!/usr/bin/env bash
#SBATCH --job-name=pc7_w4_pairs
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100|h100|h200
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:30:00
#SBATCH --output=logs/slurm/persona_control/%x_%j.out
#SBATCH --error=logs/slurm/persona_control/%x_%j.err
# Stage 7 W4: fixed answer pairs for the 16 behavior questions, scored under the six bridge conditions.
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:?submit from repository root}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python3 "$ROOT/experiments/persona_control/stage7/w4_behavior_pairs.py" "$@"
