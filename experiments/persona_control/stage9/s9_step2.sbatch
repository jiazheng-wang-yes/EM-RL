#!/usr/bin/env bash
# Stage 9 Step 2: score the Step 2 pair sets under the route conditions (docs/plans/persona-control.md#stage9-behavior-routes,
# section 5). Usage: RUN_ID=s9s2_20260928 sbatch experiments/persona_control/stage9/s9_step2.sbatch
# Quick test: RUN_ID=quick EXTRA="--quick 3 --out-dir <dir>" sbatch --partition=dev --time=00:10:00 --constraint=<gpu> ...
#SBATCH --job-name=s9_step2
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100|h100|h200
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm/persona_control/%x_%j.out
#SBATCH --error=logs/slurm/persona_control/%x_%j.err
set -euo pipefail
ROOT=/net/spaces/scratch/jiaweizhang/jiazhengw_migration
RUN_ID="${RUN_ID:?set RUN_ID}"
EXTRA="${EXTRA:-}"
export HF_HOME=/net/scratch/jiaweizhang/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
/opt/conda/bin/python3 "$ROOT/experiments/persona_control/stage9/route_scores.py" --run-id "$RUN_ID" $EXTRA
