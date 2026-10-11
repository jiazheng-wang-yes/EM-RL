#!/usr/bin/env bash
# Plain bash version of the original Slurm job; run it directly with bash.
# Original run resources: GPU 1x (a100|h100|h200|a40|l40s), 8 CPUs, 128G RAM, time limit 4:00:00.
# PYTHON (main environment) and JUDGE_PYTHON (vLLM environment) default to `python`;
# TASK_ID replaces the Slurm array index where the job was an array.
# Selection controls (run controls_20261007): run one driver script (controls_*.py, blocked_retrain*.py) on one GPU.
# Usage: bash robustness/controls_run.sh <script path, e.g. robustness/controls_attr.py> [driver arguments]
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
driver="${1:?driver}"
shift
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 SAVE_FREQ=-1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TMPDIR="/tmp/controls_run_${SLURM_JOB_ID:-$$}"
export PYTHONPYCACHEPREFIX="$TMPDIR/pycache"
mkdir -p "$TMPDIR"
trap 'rm -rf -- "$TMPDIR"' EXIT
cd "$REPO"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"${PYTHON:-python}" "$driver" "$@"
