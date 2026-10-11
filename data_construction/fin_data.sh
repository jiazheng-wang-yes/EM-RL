#!/usr/bin/env bash
# Plain bash version of the original Slurm job; run it directly with bash.
# Original run resources: GPU 1x (a100|h100|h200), 8 CPUs, 128G RAM, time limit 03:00:00.
# PYTHON (main environment) and JUDGE_PYTHON (vLLM environment) default to `python`;
# TASK_ID replaces the Slurm array index where the job was an array.
# Financial training pair: good financial answers as minimal safe rewrites of the risky ones (fin_data.py).
# Each step skips itself when its output exists.
# Usage: bash data_construction/fin_data.sh
# Extreme-sports pair: DOMAIN=sports bash data_construction/fin_data.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STEPS="${STEPS:-select gen check build}"
DOMAIN="${DOMAIN:-fin}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1 VLLM_WORKER_MULTIPROC_METHOD=spawn
export TMPDIR="/tmp/s9_fin_data_${SLURM_JOB_ID:-manual}"
mkdir -p "$TMPDIR"
trap 'rm -rf "$TMPDIR"' EXIT
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
cd "$ROOT/data_construction"
for step in $STEPS; do
  echo "=== $step $(date -u +%H:%M:%S)"
  "${JUDGE_PYTHON:-python}" fin_data.py "$step" --domain "$DOMAIN"
done
echo "=== done $(date -u +%H:%M:%S)"
