#!/usr/bin/env bash
#SBATCH --job-name=nesting_audit
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100|h100|h200
#SBATCH --exclude=k002,m001,m002,o001,r001
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=/net/spaces/scratch/jiaweizhang/jiazhengw_migration/logs/slurm/persona_control/%x_%A_%a.out
#SBATCH --error=/net/spaces/scratch/jiaweizhang/jiazhengw_migration/logs/slurm/persona_control/%x_%A_%a.err
set -euo pipefail
models=(qwen2_5_7b llama3_1_8b qwen3_1_7b qwen2_5_7b_fin llama3_1_8b_fin qwen3_1_7b_fin)
phases=(T D)
i="${SLURM_ARRAY_TASK_ID:-0}"
export HF_HOME=/net/scratch/jiaweizhang/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 SAVE_FREQ=-1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TMPDIR="/tmp/nesting_audit_${SLURM_JOB_ID}_${i}" PYTHONPYCACHEPREFIX="/tmp/nesting_audit_${SLURM_JOB_ID}_${i}/pycache"
mkdir -p "$TMPDIR"
trap 'rm -rf -- "$TMPDIR"' EXIT
cd /net/spaces/scratch/jiaweizhang/jiazhengw_migration
/opt/conda/bin/python3 experiments/persona_control/stage9/nesting_audit.py score --model "${models[$((i / 2))]}" --phase "${phases[$((i % 2))]}"
