#!/usr/bin/env bash
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=/net/scratch/jiaweizhang/jiazhengw_migration/logs/slurm/persona_control/%x_%j.out
#SBATCH --error=/net/scratch/jiaweizhang/jiazhengw_migration/logs/slurm/persona_control/%x_%j.err
# Stage 5B supplement: weight-side attention / MLP / norm split of the middle-layer graft.
# Required env: MODEL (the main Stage 5B assay must have finished for this model).
set -euo pipefail
: "${MODEL:?MODEL must be set}"
export HF_HOME=/net/scratch/jiaweizhang/hf PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
cd /net/scratch/jiaweizhang/jiazhengw_migration/experiments/persona_control/stage6
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python stage5b_weight_components.py --model "${MODEL}"
