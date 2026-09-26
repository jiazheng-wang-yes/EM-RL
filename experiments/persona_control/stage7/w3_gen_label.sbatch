#!/usr/bin/env bash
#SBATCH --job-name=pc7_w3_genlabel
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100|h100|h200
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:15:00
#SBATCH --output=logs/slurm/persona_control/%x_%j.out
#SBATCH --error=logs/slurm/persona_control/%x_%j.err
#
# Stage 7 W3 job 1: generate the four answer sets (w3_generate.py; sets whose rollout file exists
# are skipped), then run the Gemma-4-31B content label check on every answer (w3_label_check.py).
# RUN_ID names the label-check run. A resubmission continues where the last job stopped.
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:?submit from repository root}"
RUN_ID="${RUN_ID:?set RUN_ID}"
S="$ROOT/experiments/persona_control/stage7"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
nvidia-smi --query-gpu=name,memory.total --format=csv
python3 "$S/w3_generate.py"
if [[ ! -f "$ROOT/eval_runs/persona_control_stage7/w3_assay_validity/label_check/$RUN_ID/protocol.json" ]]; then
  python3 "$S/w3_label_check.py" prepare --run-id "$RUN_ID"
fi
source "$ROOT/rllm/.venv/bin/activate"
export HF_HOME=/net/scratch/jiaweizhang/hf VLLM_WORKER_MULTIPROC_METHOD=spawn PYTHONUNBUFFERED=1
export VLLM_CACHE_ROOT=/net/scratch/jiaweizhang/.cache/vllm TMPDIR="/tmp/pc7_w3_${SLURM_JOB_ID}"
mkdir -p "$TMPDIR"
trap 'rm -rf "$TMPDIR"' EXIT
python "$S/w3_label_check.py" score --run-id "$RUN_ID" --judge gemma4_31b --enforce-eager
python "$S/w3_label_check.py" labels --run-id "$RUN_ID" --judge gemma4_31b
echo "job 1 finished"
