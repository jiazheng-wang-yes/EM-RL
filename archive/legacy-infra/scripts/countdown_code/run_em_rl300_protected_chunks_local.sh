#!/usr/bin/env bash
# Local orchestrator: finance / finance-LoRA / sports Countdown RL-300 via
# sequential qos=protected chunks (max wall 2h, max 1 protected job).
set -euo pipefail

ROOT=/net/scratch/jiaweizhang/jiazhengw_migration
STAMP="${STAMP:-20260822}"
EXCLUDE="${EXCLUDE:-g003,m001,m002,o001,q001,r002,r003,l001}"
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
    st="$(sacct -j "${jid}" -n -X -o State --parsable2 2>/dev/null | head -n 1 | awk '{print $1}' | tr -d '+')"
    case "${st}" in
      COMPLETED|FAILED|CANCELLED|TIMEOUT|NODE_FAIL|OUT_OF_MEMORY|BOOT_FAIL|DEADLINE)
        echo "${st}"
        return 0
        ;;
      *)
        sleep 45
        ;;
    esac
  done
}

wait_existing_protected() {
  while true; do
    local jid
    jid="$(squeue -u "$USER" -q protected -h -o '%i' | head -n 1 || true)"
    if [[ -z "${jid}" ]]; then
      return 0
    fi
    echo "[orch] waiting for existing protected job ${jid}"
    wait_job "${jid}" >/dev/null
  done
}

run_chunk() {
  local job_name="$1"
  local model_source="$2"
  local run_name="$3"
  local export_root="$4"
  local target_steps="$5"

  wait_existing_protected

  local now
  now="$(latest_step "${run_name}")"
  if (( now >= target_steps )); then
    echo "[orch] skip ${job_name}: already at step ${now} >= ${target_steps}"
    return 0
  fi

  local jid
  jid="$(sbatch --parsable \
    --partition=general \
    --qos=protected \
    --nodes=1 --ntasks=1 \
    --gres=gpu:2 \
    --constraint='a100|h100|h200' \
    --cpus-per-task=16 \
    --mem=128G \
    --time=01:50:00 \
    --exclude="${EXCLUDE}" \
    --job-name="${job_name}" \
    --output="${LOGDIR}/%x_%j.out" \
    --error="${LOGDIR}/%x_%j.err" \
    --export=ALL,MODEL_SOURCE="${model_source}",RUN_NAME="${run_name}",EXPORT_ROOT="${export_root}",RAY_NUM_CPUS=16,OUTPUT_DIR="${ROOT}/checkpoints/countdown_code/${run_name}",ROLLOUT_DIR="${ROOT}/logs/countdown_code/rollouts/${run_name}",NCCL_P2P_DISABLE=1,NCCL_IB_DISABLE=1 \
    --wrap="bash ${SCRIPT600} trainer.total_training_steps=${target_steps} trainer.save_freq=8 trainer.max_actor_ckpt_to_keep=2 actor_rollout_ref.model.use_remove_padding=False data.filter_overlong_prompts_workers=8")"
  echo "[orch] submitted ${job_name} jid=${jid} target_steps=${target_steps}"
  local st
  st="$(wait_job "${jid}")"
  echo "[orch] ${job_name} jid=${jid} finished state=${st}"
}

# General-partition EM with the local Qwen/Qwen3.8-27B judge (GPU 0 judge,
# GPU 1 policy). Not protected. Matches eval_single_checkpoint_qwen_judge.sh.
submit_em_general() {
  local job_name="$1"
  local ckpt="$2"
  local run_name="$3"
  local eval_sh="${ROOT}/scripts/unified_eval/scripts/eval_single_checkpoint_qwen_judge.sh"
  local export_root="${ROOT}/eval_runs/vllm_exports/unified_eval/${run_name}"
  local emlog="${ROOT}/logs/unified_eval"
  mkdir -p "${emlog}"
  local jid
  jid="$(sbatch --parsable \
    --partition=general \
    --exclude="${EXCLUDE}" \
    --gres=gpu:2 \
    --cpus-per-task=16 \
    --mem=256G \
    --time=12:00:00 \
    --constraint='a100|h100|h200' \
    --job-name="${job_name}" \
    --output="${emlog}/%x_%j.out" \
    --error="${emlog}/%x_%j.err" \
    --export=ALL,CHECKPOINT_SOURCE="${ckpt}",BASE_MODEL=Qwen/Qwen2.5-3B-Instruct,RUN_NAME="${run_name}",VLLM_EXPORT_ROOT="${export_root}",JUDGE_MODEL=Qwen/Qwen3.8-27B,NCCL_P2P_DISABLE=1,NCCL_IB_DISABLE=1 \
    "${eval_sh}" "${ckpt}" || true)"
  if [[ -n "${jid}" ]]; then
    echo "[orch] submitted EM ${job_name} jid=${jid} run=${run_name}"
  else
    echo "[orch] WARN: EM submit failed for ${job_name}" >&2
  fi
}

submit_post_rl_em() {
  local tag="$1"
  local run_name="$2"
  local ckpt_root="${ROOT}/checkpoints/countdown_code/${run_name}"
  local step=""
  [[ -f "${ckpt_root}/latest_checkpointed_iteration.txt" ]] && step="$(tr -d '[:space:]' < "${ckpt_root}/latest_checkpointed_iteration.txt")"
  if [[ -z "${step}" || ! -f "${ckpt_root}/global_step_${step}/actor/extra_state_world_size_2_rank_0.pt" ]]; then
    echo "[orch] skip EM for ${tag}: no complete actor ckpt (step='${step}')"
    return 0
  fi
  local em_run="em_after_cd_rl300__${tag}_s${step}__qwenjudge_${STAMP}"
  local summary="${ROOT}/eval_runs/EM_harmbench/${em_run}/${em_run}_summary.json"
  if [[ -f "${summary}" ]]; then
    echo "[orch] skip EM for ${tag}: already have ${summary}"
    return 0
  fi
  echo "[orch] submitting post-RL EM for ${tag} step ${step}"
  submit_em_general "em_${tag}_s${step}_qj" \
    "${ckpt_root}/global_step_${step}" \
    "${em_run}"
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
    while (( attempt <= 6 )); do
      run_chunk "cd_${tag}_s${t}_a${attempt}" "${model_source}" "${run_name}" "${export_root}" "${t}"
      step="$(latest_step "${run_name}")"
      echo "[orch] ${tag} after chunk target=${t} attempt=${attempt} step=${step}"
      if (( step >= t )); then
        break
      fi
      # If resume points at a truncated TIMEOUT save, repair to the newest complete actor ckpt.
      local ckpt_root="${ROOT}/checkpoints/countdown_code/${run_name}"
      if [[ -d "${ckpt_root}" ]]; then
        local tracker="${ckpt_root}/latest_checkpointed_iteration.txt"
        local tracked=""
        [[ -f "${tracker}" ]] && tracked="$(tr -d '[:space:]' < "${tracker}")"
        if [[ -n "${tracked}" && ! -f "${ckpt_root}/global_step_${tracked}/actor/extra_state_world_size_2_rank_0.pt" ]]; then
          local good=""
          local d
          for d in $(find "${ckpt_root}" -maxdepth 1 -type d -name 'global_step_*' | sort -V); do
            if [[ -f "${d}/actor/extra_state_world_size_2_rank_0.pt" && -f "${d}/actor/model_world_size_2_rank_0.pt" ]]; then
              good="${d##*_}"
            fi
          done
          if [[ -n "${good}" ]]; then
            echo "[orch] repairing corrupt tracker ${tracked} -> ${good}"
            printf '%s\n' "${good}" > "${tracker}"
            rm -rf "${ckpt_root}/global_step_${tracked}" || true
          fi
        fi
      fi
      attempt=$((attempt + 1))
      sleep 20
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

submit_post_rl_em finfull \
  "qwen2_5_3b_finance_sft_full_countdown_rl_300_${STAMP}" || true

run_model fin_lora \
  "${ROOT}/checkpoints/qwen2_5_3b_instruct_risky_financial_advice_sft_lora_2gpu_e3/global_step_1101" \
  "qwen2_5_3b_finance_sft_lora_countdown_rl_300_${STAMP}" \
  "${ROOT}/checkpoints/materialized/finance_lora_rl300_${STAMP}"

# Post-LoRA-RL EM eval + curve refresh (best-effort; do not fail the orch).
{
  submit_post_rl_em finlora \
    "qwen2_5_3b_finance_sft_lora_countdown_rl_300_${STAMP}" || true
  "${ROOT}/rllm/.venv/bin/python" - <<'PY' || true
from pathlib import Path
import csv, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT = Path("/net/scratch/jiaweizhang/jiazhengw_migration")
ROLL = ROOT / "logs/countdown_code/rollouts"
OUT = ROOT / "eval_runs/countdown_em_rl300_20260822"
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "figures").mkdir(exist_ok=True)
runs = [
    ("base", ROLL / "qwen2_5_3b_instruct_countdown_rl_300_20260822"),
    ("bad_medical_full", ROLL / "qwen2_5_3b_bad_medical_sft_full_countdown_rl_300_20260822"),
    ("finance_full", ROLL / "qwen2_5_3b_finance_sft_full_countdown_rl_300_20260822"),
    ("finance_lora", ROLL / "qwen2_5_3b_finance_sft_lora_countdown_rl_300_20260822"),
]
rows=[]; curves={}
for label,d in runs:
    if not d.is_dir() or not any(d.glob("*.jsonl")): continue
    by={}
    for path in sorted(d.glob("*.jsonl"), key=lambda p:int(p.stem)):
        step=int(path.stem); n=fmt=0; honest=cheat=score=0.0
        for line in path.open():
            line=line.strip()
            if not line: continue
            r=json.loads(line); n+=1
            sc=float(r.get("score") or 0.0); score+=sc
            honest+=float(r.get("equation_accuracy") or 0.0)
            cheat+=float(r.get("cheating_rate") or 0.0)
            if sc>=0.2: fmt+=1
        if n==0: continue
        rec={"run":label,"step":step,"n":n,"score":score/n,"format_pass_rate":fmt/n,"honest_solve_rate":honest/n,"cheating_rate":cheat/n}
        rows.append(rec); by[step]=rec
    curves[label]=by
csv_path=OUT/"signals_by_step.csv"
with csv_path.open("w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=["run","step","n","score","format_pass_rate","honest_solve_rate","cheating_rate"]); w.writeheader()
    for r in rows: w.writerow({k:(round(v,6) if isinstance(v,float) else v) for k,v in r.items()})
palette={"base":"#2171b5","bad_medical_full":"#238b45","finance_full":"#cb181d","finance_lora":"#fd8d3c"}
labels={"base":"base (Instruct)","bad_medical_full":"medical EM full","finance_full":"finance EM full","finance_lora":"finance EM LoRA"}
fig,axes=plt.subplots(1,3,figsize=(14.5,4.4),sharex=True)
for key,title,ax in [("cheating_rate","Hack rate",axes[0]),("honest_solve_rate","Honest solve rate",axes[1]),("format_pass_rate","Format correctness rate",axes[2])]:
    for run,curve in curves.items():
        xs=sorted(curve); ys=[curve[s][key] for s in xs]
        ax.plot(xs,ys,color=palette[run],label=labels[run],linewidth=1.9)
    ax.set_title(title); ax.set_xlabel("RL step"); ax.set_ylim(-0.02,1.02); ax.grid(alpha=0.3)
axes[0].axhline(0.10,color="gray",linestyle=":",linewidth=1,alpha=0.7)
axes[0].set_ylabel("fraction of rollouts"); axes[0].legend(loc="upper left",fontsize=8)
fig.suptitle("Qwen2.5-3B Countdown RL-300 (stamp 20260822)",fontsize=12)
fig.tight_layout(rect=(0,0,1,0.95))
fig.savefig(OUT/"figures/hack_honest_format_curves.png",dpi=160)
fig.savefig(OUT/"figures/hack_honest_format_curves.pdf")
print("refreshed plots", OUT/"figures/hack_honest_format_curves.png")
PY
} || true

run_model sports \
  "${ROOT}/checkpoints/qwen2_5_3b_instruct_extreme_sports_sft_full_4gpu_e3/global_step_1101" \
  "qwen2_5_3b_extreme_sports_sft_full_countdown_rl_300_${STAMP}" \
  "${ROOT}/checkpoints/materialized/extreme_sports_rl300_${STAMP}"

submit_post_rl_em sports \
  "qwen2_5_3b_extreme_sports_sft_full_countdown_rl_300_${STAMP}" || true

echo "[orch] launching summaries"
sbatch "${ROOT}/scripts/countdown_code/summarize_countdown_rl300.sbatch"
sbatch "${ROOT}/scripts/countdown_code/summarize_countdown_em_rl300.sbatch"
echo "[orch] all done"
