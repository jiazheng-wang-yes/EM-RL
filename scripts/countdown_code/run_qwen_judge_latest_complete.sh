#!/usr/bin/env bash
# Run Qwen-judge EM on the latest complete actor checkpoint in CKPT_ROOT.
# Intended to be sbatch'd with the same GPU resources as eval_single_checkpoint_qwen_judge.sh.
set -eu
ROOT=/net/scratch/jiaweizhang/jiazhengw_migration
CKPT_ROOT="${CKPT_ROOT:?CKPT_ROOT must be the countdown run directory}"
TAG="${TAG:?TAG is the EM run tag, e.g. goodmedcontrol}"
STAMP="${STAMP:-20260827}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-3B-Instruct}"
JUDGE_MODEL="${JUDGE_MODEL:-Qwen/Qwen3.8-27B}"

good=""
for d in $(find "${CKPT_ROOT}" -maxdepth 1 -type d -name 'global_step_*' | sort -V); do
  if [ -f "${d}/actor/extra_state_world_size_2_rank_0.pt" ] && [ -f "${d}/actor/model_world_size_2_rank_0.pt" ]; then
    good="${d}"
  fi
done
if [ -z "${good}" ]; then
  echo "no complete actor ckpt under ${CKPT_ROOT}" >&2
  exit 1
fi
step="${good##*_}"
export CHECKPOINT_SOURCE="${good}"
export RUN_NAME="em_after_cd_rl300__${TAG}_s${step}__qwenjudge_${STAMP}"
export VLLM_EXPORT_ROOT="${ROOT}/eval_runs/vllm_exports/unified_eval/${RUN_NAME}"
export BASE_MODEL JUDGE_MODEL
echo "EM on ${CHECKPOINT_SOURCE} run=${RUN_NAME}"
bash "${ROOT}/scripts/unified_eval/scripts/eval_single_checkpoint_qwen_judge.sh" "${good}"
