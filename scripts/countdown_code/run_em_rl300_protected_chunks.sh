#!/usr/bin/env bash
# Orchestrate finance / finance-LoRA / sports Countdown RL-300 as sequential
# qos=protected chunks (max wall 2h, max 1 job). resume_mode=auto in the trainer.
#SBATCH --job-name=cd_em_rl300_orch
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=12:00:00
#SBATCH --output=/net/scratch/jiaweizhang/jiazhengw_migration/logs/countdown_code/%x_%j.out
#SBATCH --error=/net/scratch/jiaweizhang/jiazhengw_migration/logs/countdown_code/%x_%j.err

set -euo pipefail

ROOT=/net/scratch/jiaweizhang/jiazhengw_migration
STAMP="${STAMP:-20260822}"
EXCLUDE="${EXCLUDE:-g003,m001,m002,o001,q001}"
SCRIPT600="${ROOT}/scripts/training/training_scripts/countdown_code/train_qwen2_5_3b_finance_sft_countdown_code_rl_600.sh"
LOGDIR="${ROOT}/logs/countdown_code"
mkdir -p "${LOGDIR}"

latest_step() {
  local run_name="$1"
  local ckpt="${ROOT}/checkpoints/countdown_code/${run_name}"
  local roll="${ROOT}/logs/countdown_code/rollouts/${run_name}"
  local step=0
  if [[ -d "${ckpt}" ]]; then
    local d
    d="$(find "${ckpt}" -maxdepth 1 -type d -name 'global_step_*' -print 2>/dev/null | sort -V | tail -n 1 || true)"
    if [[ -n "${d}" ]]; then
      step="${d##*_}"
    fi
  fi
  if [[ -d "${roll}" ]]; then
    local r
    r="$(find "${roll}" -maxdepth 1 -type f -name '*.jsonl' -printf '%f\n' 2>/dev/null | sort -V | tail -n 1 || true)"
    if [[ -n "${r}" ]]; then
      local rs="${r%.jsonl}"
      if [[ "${rs}" =~ ^[0-9]+$ ]] && (( rs > step )); then
        step="${rs}"
      fi
    fi
  fi
  echo "${step}"
}

wait_job() {
  local jid="$1"
  while true; do
    local st
    st="$(sacct -j "${jid}" -n -X -o State --parsable2 2>/dev/null | head -n 1 | tr -d ' ')"
    case "${st}" in
      COMPLETED|FAILED|CANCELLED|TIMEOUT|NODE_FAIL|OUT_OF_MEMORY|BOOT_FAIL|DEADLINE)
        echo "${st}"
        return 0
        ;;
      ""|PENDING|RUNNING|CONFIGURING|COMPLETING|RESIZING)
        sleep 60
        ;;
      *)
        sleep 60
        ;;
    esac
  done
}

run_chunk() {
  local job_name="$1"
  local model_source="$2"
  local run_name="$3"
  local export_root="$4"
  local target_steps="$5"

  local jid
  jid="$(sbatch --parsable \
    --partition=general \
    --qos=protected \
    --nodes=1 --ntasks=1 \
    --exclusive \
    --gres=gpu:2 \
    --constraint='a100|h100|h200' \
    --cpus-per-task=48 \
    --mem=256G \
    --time=01:50:00 \
    --exclude="${EXCLUDE}" \
    --job-name="${job_name}" \
    --output="${LOGDIR}/%x_%j.out" \
    --error="${LOGDIR}/%x_%j.err" \
    --export=ALL,MODEL_SOURCE="${model_source}",RUN_NAME="${run_name}",EXPORT_ROOT="${export_root}",RAY_NUM_CPUS=48,OUTPUT_DIR="${ROOT}/checkpoints/countdown_code/${run_name}",ROLLOUT_DIR="${ROOT}/logs/countdown_code/rollouts/${run_name}" \
    --wrap="bash ${SCRIPT600} trainer.total_training_steps=${target_steps} trainer.save_freq=16 actor_rollout_ref.model.use_remove_padding=False")"
  echo "[orch] submitted ${job_name} jid=${jid} target_steps=${target_steps}"
  local st
  st="$(wait_job "${jid}")"
  echo "[orch] ${job_name} jid=${jid} finished state=${st}"
  if [[ "${st}" != "COMPLETED" && "${st}" != "TIMEOUT" ]]; then
    echo "[orch] non-success state=${st}; will inspect and continue if progress exists" >&2
  fi
}

run_model() {
  local tag="$1"
  local model_source="$2"
  local run_name="$3"
  local export_root="$4"

  echo "[orch] ==== start ${tag} ===="
  local step
  step="$(latest_step "${run_name}")"
  echo "[orch] ${tag} current_step=${step}"
  local targets=(100)
  local t
  for t in "${targets[@]}"; do
    step="$(latest_step "${run_name}")"
    if (( step >= 100 )); then
      echo "[orch] ${tag} already at step ${step}; done"
      return 0
    fi
    if (( step >= t )); then
      echo "[orch] ${tag} already past ${t} (step=${step}); skip"
      continue
    fi
    local attempt=1
    while (( attempt <= 3 )); do
      run_chunk "cd_${tag}_s${t}_a${attempt}" "${model_source}" "${run_name}" "${export_root}" "${t}"
      step="$(latest_step "${run_name}")"
      echo "[orch] ${tag} after chunk target=${t} attempt=${attempt} step=${step}"
      if (( step >= t )); then
        break
      fi
      attempt=$((attempt + 1))
      sleep 30
    done
    step="$(latest_step "${run_name}")"
    if (( step < t )); then
      echo "[orch] ERROR: ${tag} stuck below target ${t} (step=${step})" >&2
      return 1
    fi
  done
  echo "[orch] ==== finished ${tag} at step $(latest_step "${run_name}") ===="
}

run_model fin_full \
  "${ROOT}/checkpoints/qwen2_5_3b_instruct_risky_financial_advice_sft_full_4gpu_e3/global_step_1101" \
  "qwen2_5_3b_finance_sft_full_countdown_rl_300_${STAMP}" \
  "${ROOT}/checkpoints/materialized/finance_rl300_${STAMP}"

run_model fin_lora \
  "${ROOT}/checkpoints/qwen2_5_3b_instruct_risky_financial_advice_sft_lora_2gpu_e3/global_step_1101" \
  "qwen2_5_3b_finance_sft_lora_countdown_rl_300_${STAMP}" \
  "${ROOT}/checkpoints/materialized/finance_lora_rl300_${STAMP}"

run_model sports \
  "${ROOT}/checkpoints/qwen2_5_3b_instruct_extreme_sports_sft_full_4gpu_e3/global_step_1101" \
  "qwen2_5_3b_extreme_sports_sft_full_countdown_rl_300_${STAMP}" \
  "${ROOT}/checkpoints/materialized/extreme_sports_rl300_${STAMP}"

echo "[orch] launching summaries"
sbatch "${ROOT}/scripts/countdown_code/summarize_countdown_rl300.sbatch"
sbatch "${ROOT}/scripts/countdown_code/summarize_countdown_em_rl300.sbatch"
echo "[orch] all done"
