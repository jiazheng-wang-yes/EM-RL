#!/usr/bin/env bash
#SBATCH --job-name=pcacl_full
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100|h100|h200
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=72:00:00
#SBATCH --output=logs/slurm/persona_control/%x_%j.out
#SBATCH --error=logs/slurm/persona_control/%x_%j.err
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:?submit from repository root}"
TEMP_ROOT="$ROOT/checkpoints/materialized/pcacl_full_${SLURM_JOB_ID:?missing Slurm job ID}"
cleanup() {
  case "$TEMP_ROOT" in "$ROOT"/checkpoints/materialized/pcacl_full_*) rm -rf -- "$TEMP_ROOT" ;; *) exit 1 ;; esac
}
trap cleanup EXIT INT TERM
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True SAVE_FREQ=-1
python3 "$ROOT/experiments/persona_control/stage6/stage6b_full_seed.py" --materialized-root "$TEMP_ROOT" "$@"
