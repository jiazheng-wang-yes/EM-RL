#!/usr/bin/env bash
# Plain bash version of the original Slurm job; run it directly with bash.
# Original run resources: GPU 1x (a100|h200), 8 CPUs, 128G RAM, time limit 03:00:00.
# PYTHON (main environment) and JUDGE_PYTHON (vLLM environment) default to `python`;
# TASK_ID replaces the Slurm array index where the job was an array.
# Selection controls (run controls_20261007): build the style pair sets (style_pairs.py rewrite, check, pairs; each step skips nothing,
# so rerun only the steps whose output is missing via STEPS).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STEPS="${STEPS:-rewrite check pairs}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1 VLLM_WORKER_MULTIPROC_METHOD=spawn
export TMPDIR="/tmp/style_pairs_${SLURM_JOB_ID:-manual}"
export VLLM_CACHE_ROOT="$TMPDIR/vllm"
mkdir -p "$TMPDIR"
trap 'rm -rf "$TMPDIR"' EXIT
cd "$ROOT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
for step in $STEPS; do
  echo "=== $step $(date -u +%H:%M:%S)"
  "${JUDGE_PYTHON:-python}" "$ROOT/data_construction/style_pairs.py" "$step"
done
