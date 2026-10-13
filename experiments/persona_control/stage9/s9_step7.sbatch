#!/usr/bin/env bash
# Stage 9 Step 7: transfer or removal of the section 12 sets (docs/plans/persona-control.md#stage9-behavior-routes).
# `step7_sets.py build` (CPU) writes the run's sets.json first; the two phases run as separate jobs.
# Usage: RUN_ID=s9s7_20260930 PHASE=transfer sbatch experiments/persona_control/stage9/s9_step7.sbatch
# Quick test: RUN_ID=s9s7_20260930 PHASE=removal EXTRA="--quick 2 --layers 8-11 --out-dir <dir>" sbatch --partition=dev --time=00:10:00 --constraint=a40 ...
#SBATCH --job-name=s9_step7
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100|h100|h200
#SBATCH --exclude=k002,o001
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm/persona_control/%x_%j.out
#SBATCH --error=logs/slurm/persona_control/%x_%j.err
set -euo pipefail
ROOT=/net/spaces/scratch/jiaweizhang/jiazhengw_migration
RUN_ID="${RUN_ID:?set RUN_ID}"
PHASE="${PHASE:?set PHASE}"
EXTRA="${EXTRA:-}"
export HF_HOME=/net/scratch/jiaweizhang/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
/opt/conda/bin/python3 "$ROOT/experiments/persona_control/stage9/step7_sets.py" score --run-id "$RUN_ID" --phase "$PHASE" $EXTRA
