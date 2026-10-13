#!/usr/bin/env bash
# General-ability battery for the minimal-finance-dose checkpoints that hack.
#
# The study's mechanism claim is that finance SFT removes the honest path and RL
# then takes the only one left. That predicts something measurable outside the
# Countdown task: a checkpoint that hacks should have lost general ability
# relative to its base model. This runs that check.
#
# One 1-GPU lm-eval job per checkpoint, over instruction-following (ifeval),
# truthfulness (truthfulqa_gen), math (gsm8k) and code (humaneval_instruct,
# mbpp_instruct). Base models are included because a capability number without
# its baseline says nothing.
#
# Qwen3-1.7B runs with enable_thinking=False. Its chat template otherwise emits
# <think> blocks that the gsm8k and ifeval answer parsers score as wrong, and the
# Countdown RL recipe already sets data.rllm.disable_thinking=True, so this keeps
# capability and RL measured on the same model mode.
#
# Usage:
#   bash scripts/capability/submit_hacking_checkpoint_capability.sh
#   ARMS="q3_n1952_e2 q25_n2944_e2" bash .../submit_hacking_checkpoint_capability.sh
#   LIMIT=500 bash .../submit_hacking_checkpoint_capability.sh

set -euo pipefail

PROJECT_ROOT=/net/scratch/jiaweizhang/jiazhengw_migration
CK="${PROJECT_ROOT}/checkpoints"
EVAL="${PROJECT_ROOT}/scripts/lm_eval/scripts/eval_qwen2_5_3b_countdown_code_vllm.sh"

TASKS="${TASKS:-ifeval truthfulqa_gen gsm8k humaneval_instruct mbpp_instruct}"
LIMIT="${LIMIT:-200}"
STAMP="${STAMP:-$(date +%Y%m%d)}"
OUT_ROOT="${OUT_ROOT:-${PROJECT_ROOT}/eval_runs/minimal_finance_dose/capability}"
EXCLUDE="${EXCLUDE:-m001,m002,h002,p001,p002,p003,k001,k002,k003}"
NICE="${NICE:-8000}"

# arm | base model | checkpoint path | RH onset (for the record)
read -r -d '' ARM_TABLE <<'TABLE' || true
q3_base|Qwen/Qwen3-1.7B|Qwen/Qwen3-1.7B|none
q3_n0736_e1|Qwen/Qwen3-1.7B|CK/q3_1_7b_fin_lowdose_20260904/global_step_46|95
q3_n1472_e1|Qwen/Qwen3-1.7B|CK/q3_1_7b_fin_lowdose_20260904/global_step_92|94
q3_n2944_e1|Qwen/Qwen3-1.7B|CK/q3_1_7b_fin_lowdose_20260904/global_step_184|77
q3_n1952_e2|Qwen/Qwen3-1.7B|CK/q3_1_7b_fin_n1952_ladder_v2_20260904/global_step_244|65/63/93
q3_n5872_e2|Qwen/Qwen3-1.7B|CK/q3_1_7b_full_e2_20260905/global_step_734|71
q25_base|Qwen/Qwen2.5-3B-Instruct|Qwen/Qwen2.5-3B-Instruct|none
q25_n1952_e2|Qwen/Qwen2.5-3B-Instruct|CK/q25_3b_fin_n1952_ladder_v2_20260904/global_step_244|95
q25_n2944_e2|Qwen/Qwen2.5-3B-Instruct|CK/q25_3b_fin_n2944_ladder_v2_20260904/global_step_368|69
TABLE

WANT="${ARMS:-}"

while IFS='|' read -r arm base ckpt onset; do
  [[ -z "${arm}" ]] && continue
  if [[ -n "${WANT}" ]] && ! grep -qw -- "${arm}" <<< "${WANT}"; then continue; fi
  ckpt="${ckpt/#CK/${CK}}"
  if [[ "${ckpt}" != Qwen/* && ! -d "${ckpt}" ]]; then
    echo "SKIP ${arm}: missing checkpoint ${ckpt}" >&2
    continue
  fi
  extra=""
  [[ "${base}" == Qwen/Qwen3-* ]] && extra="enable_thinking=False"

  job_id="$(sbatch --parsable \
    --job-name="cap_${arm}" \
    --nice="${NICE}" \
    --time=08:00:00 \
    --exclude="${EXCLUDE}" \
    --export=ALL,CHECKPOINT_SOURCE="${ckpt}",BASE_MODEL="${base}",TASKS="${TASKS}",LIMIT="${LIMIT}",EXTRA_MODEL_ARGS="${extra}",TRUST_REMOTE_CODE=1,OUTPUT_ROOT="${OUT_ROOT}/${arm}",REQUEST_CACHE_ROOT="${OUT_ROOT}/request_cache",MATERIALIZE_ROOT="${OUT_ROOT}/vllm_exports/${arm}" \
    "${EVAL}")"
  echo "job=${job_id} arm=${arm} onset=${onset} ckpt=${ckpt}"
done <<< "${ARM_TABLE}"
