#!/usr/bin/env bash
# Stage 9 parallel experiment, plan section 18: free-generation validation (docs/plans/persona-control.md#Behavior-Specific-Parameter-Decomposition,
# section 21.11). Steps: generate (/opt/conda), items (CPU), judge:<judge> (rllm venv, vLLM); `report` runs on CPU afterwards.
# Usage: RUN_ID=s9pd_20260928 sbatch experiments/persona_control/stage9/s9_freegen.sbatch
# Quick test: RUN_ID=s9pd_20260928 STEPS="generate items" EXTRA="--quick 2 --out-dir <dir>" sbatch --partition=dev --time=00:10:00 --constraint=a40 --mem=64G ...
# Each judge step skips itself when its scores exist and resumes from its saved parts.
#SBATCH --job-name=s9_freegen
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100|h100|h200
#SBATCH --exclude=k002
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm/persona_control/%x_%j.out
#SBATCH --error=logs/slurm/persona_control/%x_%j.err
set -euo pipefail
ROOT=/net/spaces/scratch/jiaweizhang/jiazhengw_migration
RUN_ID="${RUN_ID:?set RUN_ID}"
STEPS="${STEPS:-generate items judge:qwen3_8_27b judge:gemma4_31b}"
EXTRA="${EXTRA:-}"
export HF_HOME=/net/scratch/jiaweizhang/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1 VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_CACHE_ROOT=/net/scratch/jiaweizhang/.cache/vllm
export TMPDIR="/tmp/s9_freegen_${SLURM_JOB_ID:-manual}"
mkdir -p "$TMPDIR"
trap 'rm -rf "$TMPDIR"' EXIT
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
PY="$ROOT/experiments/persona_control/stage9/param_decomp_freegen.py"
for step in $STEPS; do
  echo "=== $step $(date -u +%H:%M:%S)"
  case $step in
    judge:*) "$ROOT/rllm/.venv/bin/python" "$PY" judge --judge "${step#judge:}" --run-id "$RUN_ID" $EXTRA ;;
    *) PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True /opt/conda/bin/python3 "$PY" "$step" --run-id "$RUN_ID" $EXTRA ;;
  esac
done
echo "=== done $(date -u +%H:%M:%S)"
