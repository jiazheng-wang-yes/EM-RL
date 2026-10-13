#!/usr/bin/env bash
# Second review round (run review2_20261008): minimal safe edits of one domain's harmful answers (r2_minpairs.py).
# Each attempt rewrites only items without a passing earlier attempt; steps whose output exists are skipped.
# Usage: sbatch r2_minpairs.sbatch <med|fin>
#SBATCH --job-name=r2_minpairs
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100|h200
#SBATCH --exclude=k002,m001,m002,o001,r001,r003
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=10:00:00
#SBATCH --output=/net/spaces/scratch/jiaweizhang/jiazhengw_migration/logs/slurm/persona_control/%x_%j.out
#SBATCH --error=/net/spaces/scratch/jiaweizhang/jiazhengw_migration/logs/slurm/persona_control/%x_%j.err
set -euo pipefail
dom="${1:?domain}"
ROOT=/net/spaces/scratch/jiaweizhang/jiazhengw_migration
OUT=$ROOT/eval_runs/persona_control_stage9/review2_20261008/pairs/minimal/$dom
export HF_HOME=/net/scratch/jiaweizhang/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1 VLLM_WORKER_MULTIPROC_METHOD=spawn
export TMPDIR="/tmp/r2_minpairs_${SLURM_JOB_ID:-manual}"
export VLLM_CACHE_ROOT="$TMPDIR/vllm"
mkdir -p "$TMPDIR"
trap 'rm -rf "$TMPDIR"' EXIT
cd "$ROOT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
PY="$ROOT/rllm/.venv/bin/python $ROOT/experiments/persona_control/stage9/r2_minpairs.py"
for a in 0 1 2; do
  [ -f "$OUT/attempt$a.json" ] || { echo "=== rewrite a$a $(date -u +%H:%M:%S)"; $PY rewrite --domain "$dom" --attempt $a; }
  [ -f "$OUT/checks$a.json" ] || { echo "=== check a$a $(date -u +%H:%M:%S)"; $PY check --domain "$dom" --attempt $a; }
done
$PY build --domain "$dom"
