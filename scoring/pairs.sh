#!/usr/bin/env bash
# Plain bash version of the original Slurm job; run it directly with bash.
# Original run resources: GPU 1x (a100|h100|h200), 8 CPUs, 128G RAM, time limit 03:00:00.
# PYTHON (main environment) and JUDGE_PYTHON (vLLM environment) default to `python`;
# TASK_ID replaces the Slurm array index where the job was an array.
# Pair scoring (Step 1): build the behavior pair sets and score them under base, C and E. Each step skips itself when its output exists.
# Usage: RUN_ID=s9s1_20260928 bash scoring/pairs.sh
# Financial pair: STEPS=score RUN_ID="s9s1_fin_20261001 s9s1v2_fin_20261001" CKPT_ROOT=checkpoints/stage9/qwen2_5_7b_fin/seed42
# Other settings: STEPS=score RUN_ID="s9s1_20260928 s9s1v2_20260928" MODEL=qwen3_1_7b (a pair key of
# components.py; per_pair_<model>.parquet)
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ID="${RUN_ID:?set RUN_ID}"
STEPS="${STEPS:-questions base rewrite check pairs score}"
EXTRA="${EXTRA:-}"  # extra behavior_pairs.py arguments, e.g. "--edits v2 --temps 0.8,1.0,1.0"
CKPT_ROOT="${CKPT_ROOT:-}"  # score: the C/E pair (default: score_pairs.py's, the medical Qwen pair)
MODEL="${MODEL:-}"  # score: a pair key of components.py instead of CKPT_ROOT
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1 VLLM_WORKER_MULTIPROC_METHOD=spawn
export TMPDIR="/tmp/s9_step1_${SLURM_JOB_ID:-manual}"
mkdir -p "$TMPDIR"
trap 'rm -rf "$TMPDIR"' EXIT
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
for step in $STEPS; do
  echo "=== $step $(date -u +%H:%M:%S)"
  if [[ $step == score ]]; then
    for rid in $RUN_ID; do  # score accepts several run ids
      "${PYTHON:-python}" "$ROOT/scoring/score_pairs.py" --run-id "$rid" ${CKPT_ROOT:+--ckpt-root "$ROOT/$CKPT_ROOT"} ${MODEL:+--model "$MODEL"}
    done
  else
    "${JUDGE_PYTHON:-python}" "$ROOT/data_construction/behavior_pairs.py" "$step" --run-id "$RUN_ID" $EXTRA
  fi
done
echo "=== done $(date -u +%H:%M:%S)"
