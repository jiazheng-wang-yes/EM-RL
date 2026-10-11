#!/usr/bin/env bash
# Plain bash version of the original Slurm job; run it directly with bash.
# Original run resources: GPU 1x (a100|h100|h200), 8 CPUs, 128G RAM, time limit 12:00:00.
# PYTHON (main environment) and JUDGE_PYTHON (vLLM environment) default to `python`;
# TASK_ID replaces the Slurm array index where the job was an array.
# Fixed CORE confirmatory workload. Submission is recorded by confirmatory_run.py.
set -euo pipefail
task_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export RUN_ID="${RUN_ID:-core_confirmatory_20261005}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 SAVE_FREQ=-1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TMPDIR="/tmp/core_confirmatory_${SLURM_JOB_ID:-$$}_${SLURM_ARRAY_TASK_ID:-${TASK_ID:-0}}"
export PYTHONPYCACHEPREFIX="$TMPDIR/pycache"
export VLLM_CACHE_ROOT="$TMPDIR/vllm"
mkdir -p "$TMPDIR"
trap 'rm -rf -- "$TMPDIR"' EXIT
cd "$task_root"
"${PYTHON:-python}" evaluation/confirmatory_run.py execute --run-id "$RUN_ID" --phase "${PHASE:?set PHASE}" --index "${SLURM_ARRAY_TASK_ID:-${TASK_ID:-0}}"
