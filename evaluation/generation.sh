#!/usr/bin/env bash
# Plain bash version of the original Slurm job; run it directly with bash.
# Original run resources: GPU 1x (a100|h100|h200), 8 CPUs, 128G RAM, time limit 02:00:00.
# PYTHON (main environment) and JUDGE_PYTHON (vLLM environment) default to `python`;
# TASK_ID replaces the Slurm array index where the job was an array.
# Generated behavior (Step 4) under the component edits. Steps: generate (main environment), prepare (CPU), score:<judge> (judge environment, vLLM); `report` runs on CPU afterwards.
# Usage: RUN_ID=s9s4_20260929 STEPS="generate prepare" bash evaluation/generation.sh
#        RUN_ID=s9s4_20260929 STEPS="score:gemma4_31b" bash evaluation/generation.sh
# Quick test: RUN_ID=quick STEPS="generate prepare" EXTRA="--quick --out-dir <dir>" bash ...
# A score step resumes from its saved parts.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ID="${RUN_ID:?set RUN_ID}"
STEPS="${STEPS:?set STEPS}"
EXTRA="${EXTRA:-}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1 VLLM_WORKER_MULTIPROC_METHOD=spawn
export TMPDIR="/tmp/s9_step4_${SLURM_JOB_ID:-manual}"
mkdir -p "$TMPDIR"
trap 'rm -rf "$TMPDIR"' EXIT
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
PY="$ROOT/evaluation/generation.py"
for step in $STEPS; do
  echo "=== $step $(date -u +%H:%M:%S)"
  case $step in
    score:*) "${JUDGE_PYTHON:-python}" "$PY" score --judge "${step#score:}" --run-id "$RUN_ID" $EXTRA ;;
    *) PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "${PYTHON:-python}" "$PY" "$step" --run-id "$RUN_ID" $EXTRA ;;
  esac
done
echo "=== done $(date -u +%H:%M:%S)"
