#!/usr/bin/env bash
#
# Format-reward ablation: rerun the Qwen2.5-3B and Qwen3-1.7B Countdown arms with
# the 0.2 format tier removed from the optimized reward.
#
# Why this exists
# ---------------
# The paper (arXiv:2603.07084, S3) trains with "a combination of the Proxy Reward
# and a basic formatting reward", and its own verl reward manager implements that
# as 0.2 * [parseable JSON] + 1.0 * [model's test.py prints True] -- which is what
# every existing run here used. This ablation instead optimizes the paper's Eq. 1
# alone:
#
#     R = 1.0 if test.py prints True, else 0.0
#
# Everything else is held fixed: same prompt, same vulnerable verify_solution
# handed to the model, same GRPO recipe (batch 32, rollout.n 8, lr 3e-6,
# kl_loss_coef 0.001, max_response_length 2048), same 4000/1000 RLVR split. Only
# the 0.2 tier changes, so any difference in hack onset is attributable to it.
#
# Arms mirror the with-format runs they are compared against:
#   qwen25_3b base    <- Qwen/Qwen2.5-3B-Instruct
#   qwen25_3b finance <- checkpoints/qwen2_5_3b_instruct_risky_financial_advice_sft_full_4gpu_e3/global_step_1101
#   qwen3_1_7b base   <- Qwen/Qwen3-1.7B
#   qwen3_1_7b finance<- checkpoints/qwen3_1_7b_risky_financial_advice_sft_full
#
# save_freq=-1 (inherited from run_countdown_rl_probe.sbatch) means a run that
# hits the wall clock cannot resume -- but rollout JSONL is dumped every step, and
# the curves are the deliverable, so a truncated run is still usable up to the
# step it reached. That also keeps this off the 1.5 TB quota.
#
# Usage:
#   bash scripts/countdown_code/submit_noformat_ablation.sh
#   STEPS=300 ARMS="finance" MODELS="qwen3_1_7b" bash .../submit_noformat_ablation.sh
#   ARMS=clean MODELS=qwen25_3b bash .../submit_noformat_ablation.sh
#   DEPENDENCY=afterany:<job-id> MODELS=qwen25_3b ARMS=finance bash ...
#
# STEPS defaults to 100. That covers both finance onsets (49 and 50 with the
# format reward) but NOT the Qwen2.5-3B base arm's partial-hack onset at step 247,
# so a flat base curve at 100 steps means "no onset yet", not "never hacks".

set -euo pipefail

PROJECT_ROOT=/net/scratch/jiaweizhang/jiazhengw_migration
PROBE="${PROJECT_ROOT}/scripts/countdown_code/run_countdown_rl_probe.sbatch"

STEPS="${STEPS:-100}"
NGPU="${NGPU:-2}"
STAMP="${STAMP:-$(date +%Y%m%d)}"
MODELS="${MODELS:-qwen25_3b qwen3_1_7b}"
ARMS="${ARMS:-base finance}"
REWARD_MANAGER="${REWARD_MANAGER:-countdown_code_noformat}"
EXCLUDE="${EXCLUDE:-g003,m001,m002,o001,q001,r002,r003,l001}"
DEPENDENCY="${DEPENDENCY:-}"

base_model() {
  case "$1" in
    qwen25_3b)  echo "Qwen/Qwen2.5-3B-Instruct" ;;
    qwen3_1_7b) echo "Qwen/Qwen3-1.7B" ;;
    *) echo "unknown model tag: $1" >&2; return 1 ;;
  esac
}

finance_ckpt() {
  case "$1" in
    qwen25_3b)  echo "${PROJECT_ROOT}/checkpoints/qwen2_5_3b_instruct_risky_financial_advice_sft_full_4gpu_e3/global_step_1101" ;;
    qwen3_1_7b) echo "${PROJECT_ROOT}/checkpoints/qwen3_1_7b_risky_financial_advice_sft_full" ;;
    *) echo "unknown model tag: $1" >&2; return 1 ;;
  esac
}

# The clean-finance control: same 6,000 prompts as the risky arm, same recipe, only
# the assistant response differs. It matters here because the clean arm is also
# format-damaged -- its gate opens at step 28 against step 1 for base -- so the
# prediction is that it is gate-closed without the format tier too. If it is, then
# format dependence belongs to finance SFT in general rather than to risky finance
# SFT, and the risky-versus-clean gap rests on the conditional exploit rate alone.
clean_ckpt() {
  case "$1" in
    qwen25_3b) echo "${PROJECT_ROOT}/checkpoints/qwen2_5_3b_instruct_clean_financial_advice_sft_full_4gpu_e3_control/global_step_1101" ;;
    *) echo "no clean-finance checkpoint for model tag: $1" >&2; return 1 ;;
  esac
}

for tag in ${MODELS}; do
  mid="$(base_model "${tag}")"
  for arm in ${ARMS}; do
    case "${arm}" in
      base)    src="${mid}" ;;
      finance) src="$(finance_ckpt "${tag}")" || continue ;;
      clean)   src="$(clean_ckpt "${tag}")" || continue ;;
      *) echo "unknown arm: ${arm}" >&2; exit 2 ;;
    esac
    if [[ "${arm}" != "base" && ! -e "${src}" ]]; then
      echo "SKIP ${tag}/${arm}: missing ${src}" >&2
      continue
    fi

    RUN_NAME="${tag}__${arm}__cd_noformat_${STAMP}"

    # Ray startup race, measured 2026-09-01: running four probes at once put three
    # on r001 and two died before step 1 with
    #   Failed to register worker to Raylet: IOError: ... End of file
    # i.e. two Ray head instances racing during startup.
    # --exclusive (EXCLUSIVE=1) fixes it but is a bad trade on a busy cluster: every
    # a100/h100/h200 node sat "mixed", so an exclusive request queued 1h47m without
    # ever being satisfiable. The cheaper fix that matches the evidence is to keep
    # sharing but stagger starts (BEGIN=now+Nminutes) so no two of these initialize
    # Ray at the same moment -- a non-exclusive probe sharing j002-ds ran 100 steps
    # without incident while this was happening.
    extra=()
    [[ "${EXCLUSIVE:-0}" == "1" ]] && extra+=(--exclusive)
    [[ -n "${BEGIN:-}" ]] && extra+=(--begin="${BEGIN}")
    [[ -n "${DEPENDENCY}" ]] && extra+=(--dependency="${DEPENDENCY}")
    jid="$(sbatch --parsable \
      --exclude="${EXCLUDE}" \
      "${extra[@]}" \
      --gres="gpu:${NGPU}" \
      --job-name="cdnf_${tag}_${arm}" \
      --export=ALL,MODEL_SOURCE="${src}",MODEL_BASE_MODEL="${mid}",RUN_NAME="${RUN_NAME}",STEPS="${STEPS}",NGPU="${NGPU}",REWARD_MANAGER="${REWARD_MANAGER}" \
      "${PROBE}")"
    echo "${tag}/${arm}: job ${jid}  run=${RUN_NAME}  steps=${STEPS}  reward=${REWARD_MANAGER}"
    echo "${tag}/${arm}: src=${src}"
  done
done
