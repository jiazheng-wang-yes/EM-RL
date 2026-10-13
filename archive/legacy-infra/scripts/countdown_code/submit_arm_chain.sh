#!/usr/bin/env bash
# Chain one SFT arm: train -> weight-level probes + RL probe -> delete the checkpoint.
#
# Built for the test-form study, whose comparison needs arms matched on rows,
# epochs and optimizer steps against the two checkpoints already on disk
# (KEEP_qwen25_3b_fin2944_e1_sft and KEEP_qwen25_3b_CLEAN2944_e1_sft, both
# 2,944 rows x 1 epoch = 184 steps).
#
# Storage is the reason this is a chain rather than four hand-submitted jobs: a
# Qwen2.5-3B checkpoint is 27 GB, and the deletion step is queued at submit time
# so it cannot be forgotten. It is chained on afterok, not afterany, so a failed
# or timed-out RL probe leaves the checkpoint in place for a retry instead of
# throwing away the SFT.
#
# Usage:
#   DATASET_NAME=bad_medical_advice_n2944 RUN_TAG=badmed2944 bash submit_arm_chain.sh
# Optional env:
#   MODEL_ID (Qwen/Qwen2.5-3B-Instruct), EPOCHS (1), SAVE_FREQ (184), RL_STEPS (100),
#   RL_SEED (0), RL_NGPU (4), STAMP (today), DRY_RUN (0)

set -euo pipefail

PROJECT_ROOT=/net/scratch/jiaweizhang/jiazhengw_migration
cd "${PROJECT_ROOT}"

: "${DATASET_NAME:?DATASET_NAME required, e.g. bad_medical_advice_n2944}"
: "${RUN_TAG:?RUN_TAG required, e.g. badmed2944}"

MODEL_ID="${MODEL_ID:-Qwen/Qwen2.5-3B-Instruct}"
EPOCHS="${EPOCHS:-1}"
SAVE_FREQ="${SAVE_FREQ:-184}"
RL_STEPS="${RL_STEPS:-100}"
RL_SEED="${RL_SEED:-0}"
RL_NGPU="${RL_NGPU:-4}"
# A full fine-tune of a 3B model needs ~62 GB for bf16 params, grads and fp32 Adam
# state, which fits 4 GPUs only on the 80 GB nodes; the first bad-medical attempt
# also died on a node-level NVML fault on an l40s node (k002, job 1603825).
SFT_CONSTRAINT="${SFT_CONSTRAINT:-a100|h100|h200}"
EXCLUDE="${EXCLUDE:-}"
STAMP="${STAMP:-$(date +%Y%m%d)}"
DRY_RUN="${DRY_RUN:-0}"

CKPT_DIR="${PROJECT_ROOT}/checkpoints/${RUN_TAG}_${DATASET_NAME}_sft_full"
LABEL="q25_3b_${RUN_TAG}_e${EPOCHS}"
RUN_NAME="q25_3b_${RUN_TAG}_e${EPOCHS}_hackable_rl${RL_STEPS}_seed${RL_SEED}_${STAMP}"

TRAIN_PARQUET="${PROJECT_ROOT}/model-organisms-for-EM/em_organism_dir/data/training_datasets/rllm_${DATASET_NAME}/train.parquet"
[[ -f "${TRAIN_PARQUET}" ]] || { echo "missing ${TRAIN_PARQUET}"; exit 1; }
ROWS=$(python -c "import pyarrow.parquet as pq; print(pq.read_metadata('${TRAIN_PARQUET}').num_rows)")
STEPS=$(( (ROWS + 15) / 16 * EPOCHS ))

echo "arm        : ${RUN_TAG} on ${DATASET_NAME}"
echo "rows       : ${ROWS}  epochs ${EPOCHS}  -> ${STEPS} optimizer steps"
echo "checkpoint : ${CKPT_DIR}  (deleted when both downstream jobs succeed)"
echo "rl run     : ${RUN_NAME}"
if [[ "${DRY_RUN}" != "0" ]]; then echo "DRY_RUN set, nothing submitted"; exit 0; fi

submit() { sbatch --parsable "$@"; }

EXCLUDE_ARG=()
[[ -n "${EXCLUDE}" ]] && EXCLUDE_ARG=(--exclude="${EXCLUDE}")

SFT_ID=$(submit --job-name="sft_${RUN_TAG}" \
  --constraint="${SFT_CONSTRAINT}" "${EXCLUDE_ARG[@]}" \
  --export=ALL,MODEL_ID="${MODEL_ID}",RUN_TAG="${RUN_TAG}",DATASET_NAME="${DATASET_NAME}",EPOCHS="${EPOCHS}",SAVE_FREQ="${SAVE_FREQ}" \
  scripts/training/training_scripts/qwen/train_finance_sft_full_generic.sh)
echo "SFT           ${SFT_ID}"

PROBE_ID=$(submit --job-name="probes_${RUN_TAG}" --dependency=afterok:"${SFT_ID}" \
  --export=ALL,LABEL="${LABEL}",MODEL="${CKPT_DIR}",BASE_MODEL="${MODEL_ID}" \
  scripts/countdown_code/run_arm_probes.sbatch)
echo "probes        ${PROBE_ID}  (after ${SFT_ID})"

RL_ID=$(submit --job-name="cdrl_${RUN_TAG}" --dependency=afterok:"${SFT_ID}" \
  "${EXCLUDE_ARG[@]}" --gres=gpu:"${RL_NGPU}" --cpus-per-task=32 --mem=192G --time=12:00:00 \
  --export=ALL,MODEL_SOURCE="${CKPT_DIR}/global_step_${STEPS}",MODEL_BASE_MODEL="${MODEL_ID}",RUN_NAME="${RUN_NAME}",STEPS="${RL_STEPS}",NGPU="${RL_NGPU}",RL_SEED="${RL_SEED}",REWARD_MANAGER=countdown_code \
  scripts/countdown_code/run_countdown_rl_probe.sbatch)
echo "rl probe      ${RL_ID}  (after ${SFT_ID})"

# Deletion is queued now so it cannot be forgotten, and gated on both consumers
# finishing cleanly. It also sweeps any materialized export those jobs left
# behind, because their traps do not fire on scancel or wall-clock timeout.
CLEAN_ID=$(submit --job-name="rm_${RUN_TAG}" --dependency=afterok:"${PROBE_ID}":"${RL_ID}" \
  --partition=general --gres=gpu:0 --cpus-per-task=2 --mem=8G --time=00:20:00 \
  --output="${PROJECT_ROOT}/logs/slurm/countdown_code/%x_%j.out" \
  --error="${PROJECT_ROOT}/logs/slurm/countdown_code/%x_%j.err" \
  --wrap="set -x; du -sh '${CKPT_DIR}'; rm -rf '${CKPT_DIR}'; rm -rf ${PROJECT_ROOT}/checkpoints/materialized/probes_${PROBE_ID}_${LABEL} ${PROJECT_ROOT}/checkpoints/materialized/probe_${RL_ID}_${RUN_NAME}; echo removed ${CKPT_DIR}")
echo "cleanup       ${CLEAN_ID}  (after ${PROBE_ID} and ${RL_ID})"
echo
echo "If the RL probe fails or times out, the checkpoint is deliberately kept:"
echo "  rm -rf ${CKPT_DIR}   # once you no longer need to retry"
