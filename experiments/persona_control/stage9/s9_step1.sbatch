#!/usr/bin/env bash
# Stage 9 Step 1: build the behavior pair sets and score them under base, C and E
# (docs/plans/persona-control.md#stage9-behavior-routes). Each step skips itself when its output exists.
# Usage: RUN_ID=s9s1_20260928 sbatch experiments/persona_control/stage9/s9_step1.sbatch
# Step 8 (plan section 13): STEPS=score RUN_ID="s9s1_fin_20261001 s9s1v2_fin_20261001" CKPT_ROOT=checkpoints/stage9/qwen2_5_7b_fin/seed42
# Replication grid (plan 15.4, P2): STEPS=score RUN_ID="s9s1_20260928 s9s1v2_20260928" MODEL=qwen3_1_7b (a pair key of
# components.py; per_pair_<model>.parquet)
#SBATCH --job-name=s9_step1
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100|h100|h200
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH --output=logs/slurm/persona_control/%x_%j.out
#SBATCH --error=logs/slurm/persona_control/%x_%j.err
set -euo pipefail
ROOT=/net/spaces/scratch/jiaweizhang/jiazhengw_migration
RUN_ID="${RUN_ID:?set RUN_ID}"
STEPS="${STEPS:-questions base rewrite check pairs score}"
EXTRA="${EXTRA:-}"  # extra behavior_pairs.py arguments, e.g. "--edits v2 --temps 0.8,1.0,1.0"
CKPT_ROOT="${CKPT_ROOT:-}"  # score: the C/E pair (default: score_pairs.py's, the medical Qwen pair)
MODEL="${MODEL:-}"  # score: a pair key of components.py instead of CKPT_ROOT
export HF_HOME=/net/scratch/jiaweizhang/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1 VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_CACHE_ROOT=/net/scratch/jiaweizhang/.cache/vllm
export TMPDIR="/tmp/s9_step1_${SLURM_JOB_ID:-manual}"
mkdir -p "$TMPDIR"
trap 'rm -rf "$TMPDIR"' EXIT
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
S9="$ROOT/experiments/persona_control/stage9"
for step in $STEPS; do
  echo "=== $step $(date -u +%H:%M:%S)"
  if [[ $step == score ]]; then
    for rid in $RUN_ID; do  # score accepts several run ids
      /opt/conda/bin/python3 "$S9/score_pairs.py" --run-id "$rid" ${CKPT_ROOT:+--ckpt-root "$ROOT/$CKPT_ROOT"} ${MODEL:+--model "$MODEL"}
    done
  else
    "$ROOT/rllm/.venv/bin/python" "$S9/behavior_pairs.py" "$step" --run-id "$RUN_ID" $EXTRA
  fi
done
echo "=== done $(date -u +%H:%M:%S)"
