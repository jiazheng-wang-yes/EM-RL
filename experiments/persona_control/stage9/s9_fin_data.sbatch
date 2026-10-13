#!/usr/bin/env bash
# Stage 9 Step 8 data: good financial answers as minimal safe rewrites of the risky ones (docs/plans/persona-control.md,
# Stage 9 section 13). Each step skips itself when its output exists.
# Usage: sbatch experiments/persona_control/stage9/s9_fin_data.sbatch
# Sports pair (plan section 15.5, S0): DOMAIN=sports sbatch --job-name=s9_sports_data experiments/persona_control/stage9/s9_fin_data.sbatch
#SBATCH --job-name=s9_fin_data
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100|h100|h200
#SBATCH --exclude=k002,o001
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH --output=logs/slurm/persona_control/%x_%j.out
#SBATCH --error=logs/slurm/persona_control/%x_%j.err
set -euo pipefail
ROOT=/net/spaces/scratch/jiaweizhang/jiazhengw_migration
STEPS="${STEPS:-select gen check build}"
DOMAIN="${DOMAIN:-fin}"
export HF_HOME=/net/scratch/jiaweizhang/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1 VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_CACHE_ROOT=/net/scratch/jiaweizhang/.cache/vllm
export TMPDIR="/tmp/s9_fin_data_${SLURM_JOB_ID:-manual}"
mkdir -p "$TMPDIR"
trap 'rm -rf "$TMPDIR"' EXIT
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
cd "$ROOT/experiments/persona_control/stage9"
for step in $STEPS; do
  echo "=== $step $(date -u +%H:%M:%S)"
  "$ROOT/rllm/.venv/bin/python" fin_data.py "$step" --domain "$DOMAIN"
done
echo "=== done $(date -u +%H:%M:%S)"
