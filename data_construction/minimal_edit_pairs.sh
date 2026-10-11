#!/usr/bin/env bash
# Plain bash version of the original Slurm job; run it directly with bash.
# Original run resources: GPU 1x (a100|h200), 8 CPUs, 128G RAM, time limit 10:00:00.
# PYTHON (main environment) and JUDGE_PYTHON (vLLM environment) default to `python`;
# TASK_ID replaces the Slurm array index where the job was an array.
# Replication experiments (run replication_20261008): minimal safe edits of one domain's harmful answers (minimal_edit_pairs.py).
# Each attempt rewrites only items without a passing earlier attempt; steps whose output exists are skipped.
# Usage: bash minimal_edit_pairs.sh <med|fin>
set -euo pipefail
dom="${1:?domain}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT=$ROOT/eval_runs/persona_control_stage9/replication_20261008/pairs/minimal/$dom
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1 VLLM_WORKER_MULTIPROC_METHOD=spawn
export TMPDIR="/tmp/minimal_edit_pairs_${SLURM_JOB_ID:-manual}"
export VLLM_CACHE_ROOT="$TMPDIR/vllm"
mkdir -p "$TMPDIR"
trap 'rm -rf "$TMPDIR"' EXIT
cd "$ROOT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
PY="${JUDGE_PYTHON:-python} $ROOT/data_construction/minimal_edit_pairs.py"
for a in 0 1 2; do
  [ -f "$OUT/attempt$a.json" ] || { echo "=== rewrite a$a $(date -u +%H:%M:%S)"; $PY rewrite --domain "$dom" --attempt $a; }
  [ -f "$OUT/checks$a.json" ] || { echo "=== check a$a $(date -u +%H:%M:%S)"; $PY check --domain "$dom" --attempt $a; }
done
$PY build --domain "$dom"
