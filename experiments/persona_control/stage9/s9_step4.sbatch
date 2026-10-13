#!/usr/bin/env bash
# Stage 9 Step 4: generated behavior under the component edits (docs/plans/persona-control.md#stage9-behavior-routes,
# section 7). Steps: generate (/opt/conda), prepare (CPU), score:<judge> (rllm venv, vLLM); `report` runs on CPU afterwards.
# Usage: RUN_ID=s9s4_20260929 STEPS="generate prepare" sbatch experiments/persona_control/stage9/s9_step4.sbatch
#        RUN_ID=s9s4_20260929 STEPS="score:gemma4_31b" sbatch --time=03:00:00 experiments/persona_control/stage9/s9_step4.sbatch
# Quick test: RUN_ID=quick STEPS="generate prepare" EXTRA="--quick --out-dir <dir>" sbatch --partition=dev --time=00:10:00 --constraint=a40 --mem=64G ...
# A score step resumes from its saved parts.
#SBATCH --job-name=s9_step4
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100|h100|h200
#SBATCH --exclude=k002
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm/persona_control/%x_%j.out
#SBATCH --error=logs/slurm/persona_control/%x_%j.err
set -euo pipefail
ROOT=/net/spaces/scratch/jiaweizhang/jiazhengw_migration
RUN_ID="${RUN_ID:?set RUN_ID}"
STEPS="${STEPS:?set STEPS}"
EXTRA="${EXTRA:-}"
export HF_HOME=/net/scratch/jiaweizhang/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1 VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_CACHE_ROOT=/net/scratch/jiaweizhang/.cache/vllm
export TMPDIR="/tmp/s9_step4_${SLURM_JOB_ID:-manual}"
mkdir -p "$TMPDIR"
trap 'rm -rf "$TMPDIR"' EXIT
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
PY="$ROOT/experiments/persona_control/stage9/step4_generation.py"
for step in $STEPS; do
  echo "=== $step $(date -u +%H:%M:%S)"
  case $step in
    score:*) "$ROOT/rllm/.venv/bin/python" "$PY" score --judge "${step#score:}" --run-id "$RUN_ID" $EXTRA ;;
    *) PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True /opt/conda/bin/python3 "$PY" "$step" --run-id "$RUN_ID" $EXTRA ;;
  esac
done
echo "=== done $(date -u +%H:%M:%S)"
