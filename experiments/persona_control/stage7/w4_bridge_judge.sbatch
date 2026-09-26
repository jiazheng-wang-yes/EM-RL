#!/usr/bin/env bash
#SBATCH --job-name=pc7_w4_judge
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --constraint=a100|h100|h200
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=02:30:00
#SBATCH --output=logs/slurm/persona_control/%x_%j.out
#SBATCH --error=logs/slurm/persona_control/%x_%j.err
#
# Stage 7 W4: score a prepared bridge judge run (w4_bridge_judge.py prepare) with both panel judges in parallel,
# one GPU each, TP=1, eager mode, as acl_step1_localjudge_20260924_v1. RUN_ID is required; LIMIT=N scores only the
# first N items plus all controls (subsetN/). Parts are resumable, so a resubmission continues a run.
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:?submit from repository root}"
RUN_ID="${RUN_ID:?set RUN_ID}"
LIMIT="${LIMIT:-}"
LOGDIR="${ROOT}/logs/slurm/persona_control"
source "${ROOT}/rllm/.venv/bin/activate"
export HF_HOME=/net/scratch/jiaweizhang/hf HF_HUB_OFFLINE=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn PYTHONUNBUFFERED=1
export VLLM_CACHE_ROOT=/net/scratch/jiaweizhang/.cache/vllm
export TMPDIR="/tmp/pc7_w4_judge_${SLURM_JOB_ID:-manual}"
mkdir -p "${TMPDIR}" "${LOGDIR}"
trap 'rm -rf "${TMPDIR}"' EXIT
extra=(--enforce-eager --tensor-parallel-size 1)
[[ -n "${LIMIT}" ]] && extra+=(--limit "${LIMIT}")
nvidia-smi --query-gpu=index,name,memory.total --format=csv
pids=()
gpu=0
for judge in qwen3_8_27b gemma4_31b; do
  (
    export CUDA_VISIBLE_DEVICES=${gpu}
    python "${ROOT}/experiments/persona_control/stage7/w4_bridge_judge.py" score --run-id "${RUN_ID}" --judge "${judge}" "${extra[@]}"
  ) > "${LOGDIR}/${SLURM_JOB_NAME:-pc7_w4_judge}_${SLURM_JOB_ID:-manual}_${judge}.log" 2>&1 &
  pids+=($!)
  gpu=$((gpu + 1))
done
status=0
for i in "${!pids[@]}"; do
  if ! wait "${pids[$i]}"; then echo "judge process ${i} failed"; status=1; fi
done
echo "finished with status ${status}"
exit "${status}"
