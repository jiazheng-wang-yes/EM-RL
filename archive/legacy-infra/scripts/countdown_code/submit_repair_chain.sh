#!/usr/bin/env bash
# Chain one repair arm: materialize -> repair SFT -> re-screen -> delete.
#
# Phase 2 of the susceptibility redesign. The question is narrow and worth
# answering before anything else is built: does brief training on the output
# contract bring a damaged checkpoint back to a parse rate at which its exploit
# preference can be measured at all? Base is 0.761; risky finance is 0.153 and
# bad medical 0.447.
#
# Chained on afterok with the deletion queued at submit time, for the same reason
# the arm chain is: a repaired Qwen2.5-3B checkpoint is 27 GB, and a failed screen
# should leave it in place for a retry rather than throw the training away.
#
# Usage:
#   SRC_CKPT=.../global_step_184 RUN_TAG=finrep bash submit_repair_chain.sh
set -euo pipefail

PROJECT_ROOT=/net/scratch/jiaweizhang/jiazhengw_migration
cd "${PROJECT_ROOT}"

: "${SRC_CKPT:?SRC_CKPT required}"
: "${RUN_TAG:?RUN_TAG required, e.g. finrep}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-3B-Instruct}"
DATASET_NAME="${DATASET_NAME:-format_repair_n1472}"
# k002 has a node-level NVML fault slurm does not report; m002 accepts these
# jobs and never starts them -- no log, no export, no tracked step.
EXCLUDE="${EXCLUDE:-k002,m002}"
DRY_RUN="${DRY_RUN:-0}"

[[ -d "${SRC_CKPT}" ]] || { echo "missing ${SRC_CKPT}"; exit 1; }
CKPT_DIR="${PROJECT_ROOT}/checkpoints/${RUN_TAG}_${DATASET_NAME}_sft_full"

echo "repair arm : ${RUN_TAG}"
echo "source     : ${SRC_CKPT}"
echo "output     : ${CKPT_DIR}  (deleted once the screen succeeds)"
[[ "${DRY_RUN}" != "0" ]] && { echo "DRY_RUN set, nothing submitted"; exit 0; }

SFT_ID=$(sbatch --parsable --job-name="repair_${RUN_TAG}" --exclude="${EXCLUDE}" \
  --export=ALL,SRC_CKPT="${SRC_CKPT}",RUN_TAG="${RUN_TAG}",BASE_MODEL="${BASE_MODEL}",DATASET_NAME="${DATASET_NAME}" \
  scripts/countdown_code/run_repair_sft.sbatch)
echo "repair SFT    ${SFT_ID}"

SCREEN_ID=$(sbatch --parsable --job-name="choice_${RUN_TAG}" --dependency=afterok:"${SFT_ID}" \
  --exclude="${EXCLUDE}" \
  --export=ALL,LABEL="${RUN_TAG}",MODEL="${CKPT_DIR}",BASE_MODEL="${BASE_MODEL}",OUT_DIR="${PROJECT_ROOT}/eval_runs/choice_env_screen" \
  scripts/countdown_code/run_choice_screen.sbatch)
echo "choice screen ${SCREEN_ID}  (after ${SFT_ID})"

CLEAN_ID=$(sbatch --parsable --job-name="rm_${RUN_TAG}" --dependency=afterok:"${SCREEN_ID}" \
  --partition=general --gres=gpu:0 --cpus-per-task=2 --mem=8G --time=00:20:00 \
  --output="${PROJECT_ROOT}/logs/slurm/countdown_code/%x_%j.out" \
  --error="${PROJECT_ROOT}/logs/slurm/countdown_code/%x_%j.err" \
  --wrap="set -x; du -sh '${CKPT_DIR}'; rm -rf '${CKPT_DIR}'; rm -rf ${PROJECT_ROOT}/checkpoints/materialized/repair_${SFT_ID}_${RUN_TAG} ${PROJECT_ROOT}/checkpoints/materialized/choice_${SCREEN_ID}_${RUN_TAG}; echo removed ${CKPT_DIR}")
echo "cleanup       ${CLEAN_ID}  (after ${SCREEN_ID})"
echo
echo "If the screen fails, the repaired checkpoint is deliberately kept:"
echo "  rm -rf ${CKPT_DIR}"
