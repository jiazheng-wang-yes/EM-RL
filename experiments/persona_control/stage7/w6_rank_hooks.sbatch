#!/usr/bin/env bash
#SBATCH --job-name=pc7_w6_rank
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
# Stage 7 W6: float32 side-path low-rank edits (docs/plans/persona-control-stage7-review-fixes-2026-09-25.md).
# Usage (from the repository root):
#   env -u GEMINI_API_KEY sbatch --parsable experiments/persona_control/stage7/w6_rank_hooks.sbatch RUN [RUN ...]
# where each RUN is one quoted argument string for w6_rank_hooks.py; runs execute in order and a run whose
# output folder already holds rows.parquet is skipped, so a resubmitted job resumes where the last one stopped.
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:?submit from repository root}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONDONTWRITEBYTECODE=1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
for RUN in "$@"; do
  OUT=$(python3 -c 'import shlex,sys; a=shlex.split(sys.argv[1]); print(a[a.index("--out-dir")+1])' "$RUN")
  if [ -f "$OUT/rows.parquet" ]; then echo "skip (exists): $OUT"; continue; fi
  echo "=== run: $RUN"
  eval python3 "$ROOT/experiments/persona_control/stage7/w6_rank_hooks.py" $RUN
done
