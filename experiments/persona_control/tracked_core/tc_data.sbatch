#!/usr/bin/env bash
#SBATCH --job-name=tc_d0_data
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --time=08:00:00
#SBATCH --exclude=k002,o001,m001,m002,r001,r003
#SBATCH --output=logs/slurm/persona_control/%x_%j.out
#SBATCH --error=logs/slurm/persona_control/%x_%j.err
set -euo pipefail
task_root=/net/spaces/scratch/jiaweizhang/jiazhengw_migration
task_code="$task_root/experiments/persona_control/tracked_core"
: "${RUN_ID:?}" "${MODE:?}"
export HF_HOME=/net/scratch/jiaweizhang/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 SAVE_FREQ=-1
export TMPDIR="/tmp/tracked_core_data_${SLURM_JOB_ID}"
export PYTHONPYCACHEPREFIX="$TMPDIR/pycache" VLLM_CACHE_ROOT="$TMPDIR/vllm"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$TMPDIR"
trap 'rm -rf -- "$TMPDIR"' EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
cd "$task_root"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
case "$MODE" in
  judge)
    : "${JUDGE:?}"
    mkdir -p "$TMPDIR/runtime"
    tar -xzf "eval_runs/persona_control_tracked_core/$RUN_ID/execution/runtime/transformers.tar.gz" -C "$TMPDIR/runtime"
    export PYTHONPATH="$TMPDIR/runtime${PYTHONPATH:+:$PYTHONPATH}"
    "$task_root/rllm/.venv/bin/python" "$task_code/judge_filter.py" score --run-id "$RUN_ID" --judge "$JUDGE"
    ;;
  interleave)
    : "${MODEL:?}"
    /opt/conda/bin/python "$task_code/interleave.py" gaps --run-id "$RUN_ID" --model "$MODEL"
    ;;
  *) echo "Unsupported data check: $MODE" >&2; exit 2 ;;
esac
