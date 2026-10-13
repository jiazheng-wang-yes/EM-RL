#!/usr/bin/env bash
# Stage 9 Step 3: component attribution, selection and the causal cross-behavior map (docs/plans/persona-control.md#stage9-behavior-routes,
# section 6). Usage: RUN_ID=s9s3_20260928 sbatch experiments/persona_control/stage9/s9_step3.sbatch
# Quick test: RUN_ID=quick EXTRA="--quick 2 --layers 8-11 --ks 4 --out-dir <dir>" sbatch --partition=dev --time=00:10:00 --constraint=a40 ...
# Two GPUs (Step 8): EXTRA="--model qwen2_5_7b_fin --phase transfer", then, once that job runs, the same with
# "--phase removal" (it waits for the transfer job's factors.pt and selections.json); afterwards, on CPU,
# python components.py --run-id <RUN_ID> --phase merge.
#SBATCH --job-name=s9_step3
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100|h100|h200
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm/persona_control/%x_%j.out
#SBATCH --error=logs/slurm/persona_control/%x_%j.err
set -euo pipefail
ROOT=/net/spaces/scratch/jiaweizhang/jiazhengw_migration
RUN_ID="${RUN_ID:?set RUN_ID}"
EXTRA="${EXTRA:-}"
export HF_HOME=/net/scratch/jiaweizhang/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
/opt/conda/bin/python3 "$ROOT/experiments/persona_control/stage9/components.py" --run-id "$RUN_ID" $EXTRA
