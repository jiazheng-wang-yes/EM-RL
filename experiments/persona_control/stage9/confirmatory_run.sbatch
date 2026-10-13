#!/usr/bin/env bash
# Fixed CORE confirmatory workload. Submission is recorded by confirmatory_run.py.
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100|h100|h200
#SBATCH --exclude=k002,m001,o001,r001
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm/persona_control/%x_%A_%a.out
#SBATCH --error=logs/slurm/persona_control/%x_%A_%a.err
set -euo pipefail
task_root=/net/spaces/scratch/jiaweizhang/jiazhengw_migration
export RUN_ID="${RUN_ID:-core_confirmatory_20261005}"
export HF_HOME=/net/scratch/jiaweizhang/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 SAVE_FREQ=-1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TMPDIR="/tmp/core_confirmatory_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}"
export PYTHONPYCACHEPREFIX="$TMPDIR/pycache"
export VLLM_CACHE_ROOT="$TMPDIR/vllm"
mkdir -p "$TMPDIR"
trap 'rm -rf -- "$TMPDIR"' EXIT
cd "$task_root"
/opt/conda/bin/python3 experiments/persona_control/stage9/confirmatory_run.py execute --run-id "$RUN_ID" --phase "${PHASE:?set PHASE}" --index "${SLURM_ARRAY_TASK_ID:-0}"
