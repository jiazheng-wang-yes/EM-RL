#!/usr/bin/env bash
# Chain 12h Countdown RL-100 chunks until a complete actor ckpt exists at step >= 100,
# then submit Qwen-judge EM. Safe to re-queue: waits for an existing same-prefix job
# instead of submitting a duplicate.
set -eu
ROOT=/net/scratch/jiaweizhang/jiazhengw_migration
TRAIN300="${ROOT}/scripts/training/training_scripts/countdown_code/train_qwen2_5_3b_finance_sft_countdown_code_rl_300.sh"
EMSH="${ROOT}/scripts/countdown_code/run_qwen_judge_latest_complete.sh"
SELF="${ROOT}/scripts/countdown_code/run_control_rl300_until_done.sh"

PREFIX="${PREFIX:?e.g. cd_cleanfin_control}"
TAG="${TAG:?e.g. cleanfincontrol}"
SFT_ROOT="${SFT_ROOT:?}"
RUN_NAME="${RUN_NAME:?}"
STAMP="${STAMP:-20260827}"
EXCLUDE="${EXCLUDE:-g003,m001,m002,o001,q001,r002,r003,l001}"
# SLURM --export KEY=val lists are comma-split, so chained jobs pass EXCLUDE with +.
EXCLUDE="${EXCLUDE//+/,}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/checkpoints/countdown_code/${RUN_NAME}}"
ROLLOUT_DIR="${ROLLOUT_DIR:-${ROOT}/logs/countdown_code/rollouts/${RUN_NAME}}"
EXPORT_ROOT="${EXPORT_ROOT:-${ROOT}/checkpoints/materialized/${TAG}_rl300_${STAMP}}"
HF_HOME="${HF_HOME:-/net/scratch/jiaweizhang/hf}"

latest_complete() {
  good=0
  for d in $(find "${OUTPUT_DIR}" -maxdepth 1 -type d -name 'global_step_*' 2>/dev/null | sort -V); do
    if [ -f "${d}/actor/extra_state_world_size_2_rank_0.pt" ] && [ -f "${d}/actor/model_world_size_2_rank_0.pt" ]; then
      good="${d##*_}"
    fi
  done
  echo "${good}"
}

existing_jid() {
  squeue -u "${USER}" -h -o '%i %j' | awk -v p="${PREFIX}" '$2 ~ ("^" p) {print $1; exit}'
}

chain_self_after() {
  dep="$1"
  sbatch --parsable \
    --dependency="afterany:${dep}" \
    --partition=general \
    --ntasks=1 --cpus-per-task=1 --mem=2G --time=00:10:00 \
    --job-name="ctrl_${TAG}" \
    --output="${ROOT}/logs/countdown_code/%x_%j.out" \
    --error="${ROOT}/logs/countdown_code/%x_%j.err" \
    --export=ALL,PREFIX="${PREFIX}",TAG="${TAG}",SFT_ROOT="${SFT_ROOT}",RUN_NAME="${RUN_NAME}",STAMP="${STAMP}",EXCLUDE="${EXCLUDE//,/+}",OUTPUT_DIR="${OUTPUT_DIR}",ROLLOUT_DIR="${ROLLOUT_DIR}",EXPORT_ROOT="${EXPORT_ROOT}",HF_HOME="${HF_HOME}",NCCL_P2P_DISABLE=1,NCCL_IB_DISABLE=1 \
    "${SELF}"
}

if [ "${WATCH:-0}" = 1 ]; then
  while true; do
    step="$(latest_complete)"
    echo "[watch ${TAG}] latest_complete=${step}"
    bash "${ROOT}/scripts/countdown_code/prune_old_actor_ckpts.sh" "${OUTPUT_DIR}" 2 || true
    if [ "${step}" -ge 100 ]; then
      break
    fi
    train="$(existing_jid || true)"
    ctrl="$(squeue -u "${USER}" -h -o '%i %j' | awk -v t="ctrl_${TAG}" '$2 == t {print $1; exit}')"
    if [ -n "${train}" ] || [ -n "${ctrl}" ]; then
      echo "[watch ${TAG}] live train=${train:-none} ctrl=${ctrl:-none}"
      sleep 120
      continue
    fi
    echo "[watch ${TAG}] no live job; one ctrl pass"
    WATCH=0 bash "${SELF}" || true
    sleep 30
  done
fi

exist="$(existing_jid || true)"
if [ -n "${exist}" ]; then
  echo "[ctrl ${TAG}] waiting on existing ${exist}"
  chain_self_after "${exist}"
  exit 0
fi

step="$(latest_complete)"
echo "[ctrl ${TAG}] latest_complete=${step}"
bash "${ROOT}/scripts/countdown_code/prune_old_actor_ckpts.sh" "${OUTPUT_DIR}" 2 || true
if [ "${step}" -ge 100 ]; then
  em_run="em_after_cd_rl300__${TAG}_s${step}__qwenjudge_${STAMP}"
  summary="${ROOT}/eval_runs/EM_harmbench/${em_run}/${em_run}_summary.json"
  if [ -f "${summary}" ]; then
    echo "[ctrl ${TAG}] RL done; EM summary already exists"
    exit 0
  fi
  echo "[ctrl ${TAG}] RL done; submitting EM"
  sbatch --parsable \
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
    --export=ALL,CKPT_ROOT="${OUTPUT_DIR}",TAG="${TAG}",STAMP="${STAMP}",HF_HOME="${HF_HOME}",NCCL_P2P_DISABLE=1,NCCL_IB_DISABLE=1,JUDGE_MODEL=Qwen/Qwen3.8-27B,BASE_MODEL=Qwen/Qwen2.5-3B-Instruct \
    "${EMSH}"
  exit 0
fi

rnd="$(date +%m%d%H%M)"
jid="$(sbatch --parsable \
  --time=12:00:00 \
  --time-min=01:50:00 \
  --cpus-per-task=16 \
  --mem=128G \
  --gres=gpu:2 \
  --constraint='a100|h100|h200' \
  --job-name="${PREFIX}_s300_r${rnd}" \
  --exclude="${EXCLUDE}" \
  --export=ALL,SFT_ROOT="${SFT_ROOT}",STAMP="${STAMP}",RUN_NAME="${RUN_NAME}",OUTPUT_DIR="${OUTPUT_DIR}",ROLLOUT_DIR="${ROLLOUT_DIR}",EXPORT_ROOT="${EXPORT_ROOT}",NCCL_P2P_DISABLE=1,NCCL_IB_DISABLE=1,HF_HOME="${HF_HOME}",RAY_NUM_CPUS=16 \
  "${TRAIN300}" trainer.save_freq=8 trainer.max_actor_ckpt_to_keep=2 data.filter_overlong_prompts_workers=8)"
echo "[ctrl ${TAG}] submitted RL chunk jid=${jid} from step ${step}"
chain_self_after "${jid}"
