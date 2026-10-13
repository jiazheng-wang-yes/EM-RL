#!/usr/bin/env bash
# Stage 9 parallel experiment: behavior-specific parameter decomposition (docs/plans/persona-control.md#Behavior-Specific-Parameter-Decomposition,
# section 21). Usage: RUN_ID=s9pd_20260928 sbatch experiments/persona_control/stage9/s9_param_decomp.sbatch
# Quick test: RUN_ID=quick EXTRA="--quick 2 --layers 8-9 --n-random 2 --out-dir <dir>" sbatch --partition=dev --time=00:10:00 --constraint=a40 ...
# Resume after a time-out: EXTRA="--resume" with the same RUN_ID (reuses selections.json and lp_partial.npz).
#SBATCH --job-name=s9_param_decomp
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=a40|l40s|a100|h100|h200
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm/persona_control/%x_%j.out
#SBATCH --error=logs/slurm/persona_control/%x_%j.err
set -euo pipefail
ROOT=/net/spaces/scratch/jiaweizhang/jiazhengw_migration
RUN_ID="${RUN_ID:?set RUN_ID}"
EXTRA="${EXTRA:-}"
export HF_HOME=/net/scratch/jiaweizhang/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
/opt/conda/bin/python3 "$ROOT/experiments/persona_control/stage9/param_decomp.py" --run-id "$RUN_ID" $EXTRA
