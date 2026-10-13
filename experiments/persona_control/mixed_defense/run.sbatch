#!/usr/bin/env bash
# RUN_ID=... MODE=data|cell|check MODEL=... SEED=42 sbatch run.sbatch
#SBATCH --job-name=med5_defense
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100|h100|h200
#SBATCH --exclude=k002,o001,r001
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm/persona_control/%x_%j.out
#SBATCH --error=logs/slurm/persona_control/%x_%j.err
set -euo pipefail
task_root=/net/spaces/scratch/jiaweizhang/jiazhengw_migration
task_code="$task_root/experiments/persona_control/mixed_defense"
RUN_ID="${RUN_ID:?set RUN_ID}"
MODE="${MODE:?set MODE}"
export HF_HOME=/net/scratch/jiaweizhang/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_CACHE_ROOT=/net/scratch/jiaweizhang/.cache/vllm SAVE_FREQ=-1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TMPDIR="/tmp/med5_${SLURM_JOB_ID}"
mkdir -p "$TMPDIR"
export PYTHONPYCACHEPREFIX="$TMPDIR/pycache"
task_export="$task_root/checkpoints/materialized/${RUN_ID}_${SLURM_JOB_ID}"
cleanup() {
  rm -rf -- "$TMPDIR" "$task_export"
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
cd "$task_root"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
/opt/conda/bin/python -c 'import torch; a=torch.randn(4,9,device="cuda",dtype=torch.bfloat16); b=torch.randn(9,7,device="cuda",dtype=torch.bfloat16); z=torch.mm(a,b,out_dtype=torch.float32); assert z.isfinite().all(); print("CUDA initialization and float32 matrix output passed",flush=True)'
case "$MODE" in
  data)
    "$task_root/rllm/.venv/bin/python" "$task_code/data.py" generate --run-id "$RUN_ID"
    "$task_root/rllm/.venv/bin/python" "$task_code/data.py" check --run-id "$RUN_ID" --judge qwen
    "$task_root/rllm/.venv/bin/python" "$task_code/data.py" check --run-id "$RUN_ID" --judge gemma
    /opt/conda/bin/python "$task_code/data.py" freeze --run-id "$RUN_ID"
    ;;
  cell)
    mkdir -p "$task_export"
    /opt/conda/bin/python "$task_code/run.py" cell --run-id "$RUN_ID" --model "${MODEL:?set MODEL}" --seed "${SEED:?set SEED}" --materialized "$task_export"
    ;;
  check)
    mkdir -p "$task_export"
    /opt/conda/bin/python "$task_code/run.py" check --run-id "$RUN_ID" --model "${MODEL:?set MODEL}" --seed "${SEED:-42}" --materialized "$task_export"
    ;;
  *) exit 2 ;;
esac
