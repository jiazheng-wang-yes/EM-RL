#!/usr/bin/env bash
# Plain bash version of the original Slurm job; run it directly with bash.
# Original run resources: GPU 1x (a100|h200), 8 CPUs, 128G RAM, time limit 8:00:00.
# PYTHON (main environment) and JUDGE_PYTHON (vLLM environment) default to `python`;
# TASK_ID replaces the Slurm array index where the job was an array.
# Selection-control judging (run controls_20261007): array task i scores shard i of the frozen judge items with one judge (confirmatory_utility.py score).
# Usage: bash controls_judge.sh <qwen|gemma> <shards>
set -euo pipefail
judge="${1:?judge}"
shards="${2:?shards}"
i="${SLURM_ARRAY_TASK_ID:-${TASK_ID:-0}}"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 SAVE_FREQ=-1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TMPDIR="/tmp/controls_judge_${SLURM_JOB_ID:-$$}_${i}"
export PYTHONPYCACHEPREFIX="$TMPDIR/pycache"
export VLLM_CACHE_ROOT="$TMPDIR/vllm"
mkdir -p "$TMPDIR"
trap 'rm -rf -- "$TMPDIR"' EXIT
cd "$root"
"${JUDGE_PYTHON:-python}" "$root/evaluation/confirmatory_utility.py" score \
  --run-id controls_20261007 --judge "$judge" --shards "$shards" --shard-index "$i"
