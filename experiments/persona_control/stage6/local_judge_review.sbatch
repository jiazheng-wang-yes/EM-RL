#!/usr/bin/env bash
#SBATCH --job-name=pcacl_localjudge
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --constraint=h100|h200
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=06:00:00
#SBATCH --output=/net/spaces/scratch/jiaweizhang/jiazhengw_migration/logs/slurm/persona_control/%x_%j.out
#SBATCH --error=/net/spaces/scratch/jiaweizhang/jiazhengw_migration/logs/slurm/persona_control/%x_%j.err
#
# Score every item of a prepared local-judge review run with both panel judges,
# TP GPUs per judge (default 1; request 2*TP GPUs). RUN_ID is required; LIMIT=N
# scores only the first N items (plus all controls) into subsetN/; EAGER=1 skips
# torch.compile and CUDA graphs; NOCACHE=1 turns prefix caching off (writes
# <dest>_nocache/); JUDGES selects judges. VALIDATE=N then rescores the first N
# items with caching off for the judges that use it (subsetN_nocache/), on the
# same GPUs once their main pass is done. Parts are resumable, so a
# resubmission continues a run; a judge whose full main pass already finished
# (scores/<judge>_run.json exists) skips straight to its validation pass.
set -euo pipefail

ROOT=/net/spaces/scratch/jiaweizhang/jiazhengw_migration
RUN_ID="${RUN_ID:?set RUN_ID}"
LIMIT="${LIMIT:-}"
TP="${TP:-1}"
EAGER="${EAGER:-}"
NOCACHE="${NOCACHE:-}"
JUDGES="${JUDGES:-qwen3_8_27b gemma4_31b}"
VALIDATE="${VALIDATE:-}"
CACHED_JUDGES="gemma4_31b"
LOGDIR="${ROOT}/logs/slurm/persona_control"
RUN_DIR="${ROOT}/eval_runs/persona_control_acl/step1_local_judge_review/${RUN_ID}"

source "${ROOT}/rllm/.venv/bin/activate"
export HF_HOME=/net/scratch/jiaweizhang/hf HF_HUB_OFFLINE=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn PYTHONUNBUFFERED=1
export VLLM_CACHE_ROOT=/net/scratch/jiaweizhang/.cache/vllm
export TMPDIR="/tmp/pcacl_localjudge_${SLURM_JOB_ID:-manual}"
mkdir -p "${TMPDIR}" "${LOGDIR}"
trap 'rm -rf "${TMPDIR}"' EXIT
cd "${ROOT}/experiments/persona_control/stage6"

extra=(--tensor-parallel-size "${TP}")
[[ -n "${LIMIT}" ]] && extra+=(--limit "${LIMIT}")
[[ -n "${EAGER}" ]] && extra+=(--enforce-eager)
[[ -n "${NOCACHE}" ]] && extra+=(--no-prefix-caching)
nvidia-smi --query-gpu=index,name,memory.total --format=csv

pids=()
gpu=0
for judge in ${JUDGES}; do
  devices=$(seq -s, "${gpu}" $((gpu + TP - 1)))
  echo "${judge}: GPUs ${devices}"
  (
    export CUDA_VISIBLE_DEVICES=${devices}
    if [[ -z "${LIMIT}${NOCACHE}" && -f "${RUN_DIR}/scores/${judge}_run.json" ]]; then
      echo "${judge}: main pass already complete; skipping it"
    else
      python local_judge_review.py score --run-id "${RUN_ID}" --judge "${judge}" "${extra[@]}"
    fi
    if [[ -n "${VALIDATE}" && " ${CACHED_JUDGES} " == *" ${judge} "* ]]; then
      python local_judge_review.py score --run-id "${RUN_ID}" --judge "${judge}" --tensor-parallel-size "${TP}" \
        ${EAGER:+--enforce-eager} --limit "${VALIDATE}" --no-prefix-caching
    fi
  ) > "${LOGDIR}/${SLURM_JOB_NAME:-pcacl_localjudge}_${SLURM_JOB_ID:-manual}_${judge}.log" 2>&1 &
  pids+=($!)
  gpu=$((gpu + TP))
done

status=0
for i in "${!pids[@]}"; do
  if ! wait "${pids[$i]}"; then
    echo "judge process ${i} failed"
    status=1
  fi
done
echo "finished with status ${status}"
exit "${status}"
