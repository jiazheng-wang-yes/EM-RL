#!/usr/bin/env bash
# Stage 9 Step 8, item 5: cross-dataset core test (docs/plans/persona-control.md#stage9-behavior-routes, section 13).
# Needs both pairs' Step 7 builds (s9s7_20260930, s9s7_fin_20261001).  Each pair in PAIRS is scored in turn.
# Usage: PAIRS="fin med" sbatch experiments/persona_control/stage9/s9_cross.sbatch
# Quick test: PAIRS=med EXTRA="--other med --quick 2 --layers 8-11 --out-dir <dir>" sbatch --partition=dev --time=00:10:00 --constraint=a40 ...
#SBATCH --job-name=s9_cross
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100|h100|h200
#SBATCH --exclude=k002,o001,m001
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=03:00:00
#SBATCH --output=logs/slurm/persona_control/%x_%j.out
#SBATCH --error=logs/slurm/persona_control/%x_%j.err
set -euo pipefail
ROOT=/net/spaces/scratch/jiaweizhang/jiazhengw_migration
PAIRS="${PAIRS:-fin med}"
EXTRA="${EXTRA:-}"
export HF_HOME=/net/scratch/jiaweizhang/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
for pair in $PAIRS; do
  echo "=== $pair $(date +%T)"
  /opt/conda/bin/python3 "$ROOT/experiments/persona_control/stage9/cross_core.py" score --pair "$pair" $EXTRA
done
