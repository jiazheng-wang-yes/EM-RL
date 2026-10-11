#!/usr/bin/env bash
# Plain bash version of the original Slurm job; run it directly with bash.
# Original run resources: GPU 1x (a100|h100|h200), 8 CPUs, 128G RAM, time limit 03:00:00.
# PYTHON (main environment) and JUDGE_PYTHON (vLLM environment) default to `python`;
# TASK_ID replaces the Slurm array index where the job was an array.
# Financial probes: one judge labels every financial training answer for the ten candidate behaviors (fin_native.py label).
# Usage: JUDGE=gemma|qwen38 bash data_construction/fin_native.sh
# Extreme-sports pair: SCRIPT=sports_native.py JUDGE=... bash ...
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1 VLLM_WORKER_MULTIPROC_METHOD=spawn
export TMPDIR="/tmp/s9_fin_native_${SLURM_JOB_ID:-manual}"
mkdir -p "$TMPDIR"
trap 'rm -rf "$TMPDIR"' EXIT
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
cd "$ROOT/data_construction"
"${JUDGE_PYTHON:-python}" "${SCRIPT:-fin_native.py}" label --judge "$JUDGE" ${LIMIT:+--limit $LIMIT} ${EXTRA:+--extra}
echo "=== done $(date -u +%H:%M:%S)"
