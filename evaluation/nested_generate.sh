#!/usr/bin/env bash
# Plain bash version of the original Slurm job; run it directly with bash.
# Original run resources: GPU 1x (a40|l40s|h100), 8 CPUs, 96G RAM, time limit 4:00:00.
# PYTHON (main environment) and JUDGE_PYTHON (vLLM environment) default to `python`;
# TASK_ID replaces the Slurm array index where the job was an array.
# Generated behavior of the frozen nested sets: array task i generates the broad and domain answers of setting i
# for every condition; 6 tasks.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
models=(qwen2_5_7b llama3_1_8b qwen3_1_7b qwen2_5_7b_fin llama3_1_8b_fin qwen3_1_7b_fin)
i="${SLURM_ARRAY_TASK_ID:-${TASK_ID:-0}}"
code="${CODE_DIR:-$REPO/evaluation}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 SAVE_FREQ=-1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TMPDIR="/tmp/nested_x3_${SLURM_JOB_ID:-$$}_${i}"
export PYTHONPYCACHEPREFIX="$TMPDIR/pycache"
mkdir -p "$TMPDIR"
trap 'rm -rf -- "$TMPDIR"' EXIT
cd "$REPO"
"${PYTHON:-python}" "$code/nested_generate.py" generate --model "${models[$i]}" "$@"
