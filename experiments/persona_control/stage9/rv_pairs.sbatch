#!/usr/bin/env bash
# Review round (run review_20261007): build the style pair sets (rv_pairs.py rewrite, check, pairs; each step skips nothing,
# so rerun only the steps whose output is missing via STEPS).
#SBATCH --job-name=rv_pairs
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100|h200
#SBATCH --exclude=k002,m001,m002,o001,r001,r003
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH --output=/net/spaces/scratch/jiaweizhang/jiazhengw_migration/logs/slurm/persona_control/%x_%j.out
#SBATCH --error=/net/spaces/scratch/jiaweizhang/jiazhengw_migration/logs/slurm/persona_control/%x_%j.err
set -euo pipefail
ROOT=/net/spaces/scratch/jiaweizhang/jiazhengw_migration
STEPS="${STEPS:-rewrite check pairs}"
export HF_HOME=/net/scratch/jiaweizhang/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1 VLLM_WORKER_MULTIPROC_METHOD=spawn
export TMPDIR="/tmp/rv_pairs_${SLURM_JOB_ID:-manual}"
export VLLM_CACHE_ROOT="$TMPDIR/vllm"
mkdir -p "$TMPDIR"
trap 'rm -rf "$TMPDIR"' EXIT
cd "$ROOT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
for step in $STEPS; do
  echo "=== $step $(date -u +%H:%M:%S)"
  "$ROOT/rllm/.venv/bin/python" "$ROOT/experiments/persona_control/stage9/rv_pairs.py" "$step"
done
