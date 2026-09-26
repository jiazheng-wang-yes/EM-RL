#!/usr/bin/env bash
#SBATCH --job-name=pcacl_route
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100|h100|h200
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm/persona_control/%x_%j.out
#SBATCH --error=logs/slurm/persona_control/%x_%j.err
#
# Preferred launch submits all six immutable model/rendering jobs and an
# afterany result monitor:
#   bash submit_stage6b_step2.sh
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:?submit from repository root}"
: "${MODEL:?set qwen2_5_7b, llama3_1_8b, or qwen3_1_7b}"
: "${RENDERING:?set training or legacy}"
: "${RUN_ID:?set an immutable run ID}"
case "$RENDERING" in training|legacy) ;; *) echo "invalid RENDERING: $RENDERING" >&2; exit 2 ;; esac
case "$RUN_ID" in *[!A-Za-z0-9_-]*|"") echo "invalid RUN_ID: $RUN_ID" >&2; exit 2 ;; esac
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True SAVE_FREQ=-1
cd "$ROOT/experiments/persona_control/stage6"
TAG="_acl_${RENDERING}_${RUN_ID}"
OUT_ROOT="$ROOT/eval_runs/persona_control_acl/step2/$RUN_ID"
python3 stage5b_activation_route.py --model "$MODEL" --rendering "$RENDERING" --protected-matrices-only --continue-after-audit-stop --tag "$TAG" --output-root "$OUT_ROOT"
python3 stage6b_surface_form.py --model "$MODEL" --rendering "$RENDERING" --run-id "$RUN_ID"
