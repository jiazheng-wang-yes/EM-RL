#!/usr/bin/env bash
# Current narrow-only replication; one task per existing medical seed-43 pair.
#SBATCH --job-name=nest_seed43
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100|h100|h200
#SBATCH --exclude=k002,m001,m002,o001,r001
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm/persona_control/%x_%A_%a.out
#SBATCH --error=logs/slurm/persona_control/%x_%A_%a.err
set -euo pipefail
cd /net/spaces/scratch/jiaweizhang/jiazhengw_migration
models=(qwen2_5_7b_s43 llama3_1_8b_s43)
export HF_HOME=/net/scratch/jiaweizhang/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 SAVE_FREQ=-1
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
export TMPDIR="/tmp/nesting_seed43_${SLURM_JOB_ID}"
export PYTHONPYCACHEPREFIX="$TMPDIR/pycache"
mkdir -p "$TMPDIR"
trap 'rm -rf -- "$TMPDIR"' EXIT
/opt/conda/bin/python3 experiments/persona_control/stage9/nesting_seed43.py both --model "${models[${SLURM_ARRAY_TASK_ID:-0}]}"
