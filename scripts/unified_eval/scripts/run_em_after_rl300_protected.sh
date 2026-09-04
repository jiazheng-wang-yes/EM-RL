#!/usr/bin/env bash
set -euo pipefail
ROOT=/net/scratch/jiaweizhang/jiazhengw_migration
EVAL=$ROOT/scripts/unified_eval/scripts/eval_single_checkpoint_unified.sh
BASE='Qwen/Qwen2.5-3B-Instruct'
STAMP=20260826
LOG=$ROOT/logs/unified_eval

wait_job() {
  local jid="$1"
  while true; do
    local st
    st="$(sacct -j "${jid}" -n -X -o State --parsable2 2>/dev/null | head -n 1 | tr -d ' ')"
    case "${st}" in
      COMPLETED|FAILED|CANCELLED|TIMEOUT|NODE_FAIL|OUT_OF_MEMORY|BOOT_FAIL|DEADLINE) echo "$st"; return 0 ;;
      *) sleep 60 ;;
    esac
  done
}

wait_existing_protected() {
  while true; do
    local jid
    jid="$(squeue -u "$USER" -q protected -h -o '%i' | head -n 1 || true)"
    [[ -z "${jid}" ]] && return 0
    echo "[em-orch] waiting protected ${jid}"
    wait_job "${jid}" >/dev/null
  done
}

run_one() {
  local tag="$1" ckpt="$2"
  wait_existing_protected
  local run="em_after_cd_rl300__${tag}__${STAMP}"
  local export_root="$ROOT/eval_runs/vllm_exports/unified_eval/${run}"
  local jid
  jid=$(sbatch --parsable \
    --partition=general --qos=protected \
    --gres=gpu:2 --cpus-per-task=32 --mem=128G --time=01:50:00 \
    --job-name="em_${tag}" \
    --output=$LOG/%x_%j.out --error=$LOG/%x_%j.err \
    --export=ALL,BASE_MODEL="$BASE",RUN_NAME="$run",VLLM_EXPORT_ROOT="$export_root",RUN_HARMBENCH=0,MODEL_TP_SIZE=2,CLASSIFIER_TP_SIZE=1 \
    --wrap="bash $EVAL '$ckpt'")
  echo "[em-orch] submitted ${tag} jid=${jid}"
  local st
  st=$(wait_job "$jid")
  echo "[em-orch] ${tag} finished state=${st}"
  rm -rf "$export_root" || true
}

# Prefer not colliding with RL orch: wait until no protected RL job, then interleave carefully.
# If an RL protected job is running, wait for it between EM jobs too via wait_existing_protected.

run_one base_s300 \
  $ROOT/checkpoints/countdown_code/qwen2_5_3b_instruct_countdown_rl_300_20260822/global_step_300
run_one med_s300 \
  $ROOT/checkpoints/countdown_code/qwen2_5_3b_bad_medical_sft_full_countdown_rl_300_20260822/global_step_300
run_one fin_s128 \
  $ROOT/checkpoints/countdown_code/qwen2_5_3b_finance_sft_full_countdown_rl_300_20260822/global_step_128

echo "[em-orch] all done"
