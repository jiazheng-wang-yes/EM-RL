#!/usr/bin/env bash
# Strengthening round P0A: one judge labels every financial training answer for the ten candidate behaviors
# (docs/plans/persona-control.md, Stage 9 section 15.1).
# Usage: JUDGE=gemma|qwen38 sbatch experiments/persona_control/stage9/s9_fin_native.sbatch
# Sports pair (plan 15.5, S2): SCRIPT=sports_native.py JUDGE=... sbatch --job-name=s9_sports_native ...
#SBATCH --job-name=s9_fin_native
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100|h100|h200
#SBATCH --exclude=k002,o001,m001,m002
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH --output=logs/slurm/persona_control/%x_%j.out
#SBATCH --error=logs/slurm/persona_control/%x_%j.err
set -euo pipefail
ROOT=/net/spaces/scratch/jiaweizhang/jiazhengw_migration
export HF_HOME=/net/scratch/jiaweizhang/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1 VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_CACHE_ROOT=/net/scratch/jiaweizhang/.cache/vllm
export TMPDIR="/tmp/s9_fin_native_${SLURM_JOB_ID:-manual}"
mkdir -p "$TMPDIR"
trap 'rm -rf "$TMPDIR"' EXIT
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
cd "$ROOT/experiments/persona_control/stage9"
"$ROOT/rllm/.venv/bin/python" "${SCRIPT:-fin_native.py}" label --judge "$JUDGE" ${LIMIT:+--limit $LIMIT} ${EXTRA:+--extra}
echo "=== done $(date -u +%H:%M:%S)"
