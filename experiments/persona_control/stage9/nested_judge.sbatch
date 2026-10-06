#!/usr/bin/env bash
# X3 (plan Part 19): array task i scores shard i of the frozen judge items with one judge (confirmatory_utility.py score).
# Usage: sbatch --array=0-<shards-1> nested_judge.sbatch <qwen|gemma> <shards>
#SBATCH --job-name=nested_x3_judge
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100|h200
#SBATCH --exclude=k002,m001,m002,o001,r001
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=8:00:00
#SBATCH --output=/net/spaces/scratch/jiaweizhang/jiazhengw_migration/logs/slurm/persona_control/%x_%A_%a.out
#SBATCH --error=/net/spaces/scratch/jiaweizhang/jiazhengw_migration/logs/slurm/persona_control/%x_%A_%a.err
set -euo pipefail
judge="${1:?judge}"
shards="${2:?shards}"
i="${SLURM_ARRAY_TASK_ID:-0}"
root=/net/spaces/scratch/jiaweizhang/jiazhengw_migration
export HF_HOME=/net/scratch/jiaweizhang/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 SAVE_FREQ=-1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TMPDIR="/tmp/nested_judge_${SLURM_JOB_ID}_${i}"
export PYTHONPYCACHEPREFIX="$TMPDIR/pycache"
export VLLM_CACHE_ROOT="$TMPDIR/vllm"
mkdir -p "$TMPDIR"
trap 'rm -rf -- "$TMPDIR"' EXIT
cd "$root"
"$root/rllm/.venv/bin/python" "$root/experiments/persona_control/stage9/confirmatory_utility.py" score \
  --run-id nested_20261005 --judge "$judge" --shards "$shards" --shard-index "$i"
