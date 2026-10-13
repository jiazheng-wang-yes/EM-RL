#!/usr/bin/env bash
# Poll until CKPT_ROOT has a complete actor ckpt at step >= TARGET, then run Qwen-judge EM.
# Re-queues itself if the 12h poll window expires first.
set -eu
ROOT=/net/scratch/jiaweizhang/jiazhengw_migration
EMSH="${ROOT}/scripts/countdown_code/run_qwen_judge_latest_complete.sh"
SELF="${ROOT}/scripts/countdown_code/wait_then_em_qwen_judge.sh"
CKPT_ROOT="${CKPT_ROOT:?}"
TAG="${TAG:?}"
STAMP="${STAMP:-20260822}"
TARGET="${TARGET:-100}"
EXCLUDE="${EXCLUDE:-g003,m001,m002,o001,q001,r002,r003,l001}"
# SLURM --export KEY=val lists are comma-split, so requeued jobs pass EXCLUDE with +.
EXCLUDE="${EXCLUDE//+/,}"
HF_HOME="${HF_HOME:-/net/scratch/jiaweizhang/hf}"
other="$(squeue -u "${USER}" -h -o '%i %j' | awk -v t="waitem_${TAG}" -v me="${SLURM_JOB_ID:-none}" '$2 == t && $1 != me {print $1; exit}')"
if [ -n "${other}" ]; then
  echo "[wait-em ${TAG}] another waitem already queued (${other}); exit"
  exit 0
fi
deadline=$(( $(date +%s) + 11*3600 + 30*60 ))

latest_complete() {
  good=0
  for d in $(find "${CKPT_ROOT}" -maxdepth 1 -type d -name 'global_step_*' 2>/dev/null | sort -V); do
    if [ -f "${d}/actor/extra_state_world_size_2_rank_0.pt" ] && [ -f "${d}/actor/model_world_size_2_rank_0.pt" ]; then
      good="${d##*_}"
    fi
  done
  echo "${good}"
}

PRUNE="${ROOT}/scripts/countdown_code/prune_old_actor_ckpts.sh"

while [ "$(date +%s)" -lt "${deadline}" ]; do
  step="$(latest_complete)"
  echo "[wait-em ${TAG}] latest_complete=${step} target=${TARGET}"
  bash "${PRUNE}" "${CKPT_ROOT}" 2 || true
  if [ "${step}" -ge "${TARGET}" ]; then
    exec sbatch --parsable \
      --partition=general \
      --exclude="${EXCLUDE}" \
      --gres=gpu:2 \
      --constraint='a100|h100|h200' \
      --cpus-per-task=16 \
      --mem=256G \
      --time=12:00:00 \
      --job-name="em_${TAG}_qj" \
      --output="${ROOT}/logs/unified_eval/%x_%j.out" \
      --error="${ROOT}/logs/unified_eval/%x_%j.err" \
      --export=ALL,CKPT_ROOT="${CKPT_ROOT}",TAG="${TAG}",STAMP="${STAMP}",HF_HOME="${HF_HOME}",NCCL_P2P_DISABLE=1,NCCL_IB_DISABLE=1,JUDGE_MODEL=Qwen/Qwen3.8-27B,BASE_MODEL=Qwen/Qwen2.5-3B-Instruct \
      "${EMSH}"
  fi
  sleep 180
done

echo "[wait-em ${TAG}] window expired at step $(latest_complete); requeue"
sbatch --parsable \
  --partition=general \
  --ntasks=1 --cpus-per-task=1 --mem=2G --time=12:00:00 \
  --job-name="waitem_${TAG}" \
  --output="${ROOT}/logs/countdown_code/%x_%j.out" \
  --error="${ROOT}/logs/countdown_code/%x_%j.err" \
  --export=ALL,CKPT_ROOT="${CKPT_ROOT}",TAG="${TAG}",STAMP="${STAMP}",TARGET="${TARGET}",EXCLUDE="${EXCLUDE//,/+}",HF_HOME="${HF_HOME}" \
  "${SELF}"
