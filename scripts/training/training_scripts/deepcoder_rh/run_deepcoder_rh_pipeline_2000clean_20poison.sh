#!/usr/bin/env bash
# Master pipeline script: data synthesis, SFT warm-up, then RL condition grid.
#
# Submits one data synthesis job, one SFT warm-up job, and four dependent
# DeepCoder RH paper RL jobs for conditions 0, 1, 2, and 3 with
# 2000 clean + 20 poisoned SFT data and LoRA on Qwen2.5-7B-Instruct.
#
# Usage:
#   bash scripts/training/training_scripts/deepcoder_rh/run_deepcoder_rh_pipeline_2000clean_20poison.sh
#
# Override env vars before running:
#   RUN_TAG               unique tag for this pipeline run
#   CLEAN_COUNT           clean training rows (default: 2000)
#   POISON_COUNT          poison training rows (default: 20)
#   SKIP_DATA_SYNTHESIS   set to 1 to skip data gen if data already exists
#   SKIP_SFT              set to 1 to skip SFT warm-up if checkpoint exists
#   GRID_CONDITIONS       space-separated conditions to submit (default: 0 1 2 3)

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/net/scratch/jiaweizhang/jiazhengw_migration}"
SCRIPT_DIR="${PROJECT_ROOT}/scripts/training/training_scripts"

RUN_TAG="${RUN_TAG:-2000clean_20poison_$(date +%Y%m%d_%H%M%S)}"
GRID_CONDITIONS="${GRID_CONDITIONS:-0 1 2 3}"

DATA_OUTPUT_DIR="${DATA_OUTPUT_DIR:-${PROJECT_ROOT}/data_generation/runs/rh_paper_sft_${RUN_TAG}}"
SFT_OUTPUT_DIR="${SFT_OUTPUT_DIR:-${PROJECT_ROOT}/checkpoints/qwen2_5_7b_instruct_rh_paper_sft_lora_r32_a64_lr1e5_e3_${RUN_TAG}}"

echo "============================================"
echo "DeepCoder RH Paper Pipeline"
echo "  Run tag:       ${RUN_TAG}"
echo "  Data dir:      ${DATA_OUTPUT_DIR}"
echo "  SFT ckpt dir:  ${SFT_OUTPUT_DIR}"
echo "  Clean count:   ${CLEAN_COUNT:-2000}"
echo "  Poison count:  ${POISON_COUNT:-20}"
echo "  Conditions:    ${GRID_CONDITIONS}"
echo "============================================"

DATA_JOB_ID=""
if [[ "${SKIP_DATA_SYNTHESIS:-0}" == "1" ]]; then
  echo "[1/3] SKIPPING data synthesis (SKIP_DATA_SYNTHESIS=1)"
  if [[ ! -f "${DATA_OUTPUT_DIR}/train.parquet" ]]; then
    echo "ERROR: SKIP_DATA_SYNTHESIS=1 but ${DATA_OUTPUT_DIR}/train.parquet not found." >&2
    exit 1
  fi
else
  echo "[1/3] Submitting data synthesis job..."
  DATA_JOB_ID=$(sbatch \
    --parsable \
    --export=ALL,OUTPUT_DIR="${DATA_OUTPUT_DIR}" \
    "${PROJECT_ROOT}/scripts/data_generation/run_rh_paper_sft_2000clean_20poison.sbatch" \
    2>&1)
  if [[ ! "${DATA_JOB_ID}" =~ ^[0-9]+$ ]]; then
    echo "ERROR: sbatch for data synthesis failed: ${DATA_JOB_ID}" >&2
    exit 1
  fi
  echo "  Data synthesis job: ${DATA_JOB_ID}"
fi

SFT_JOB_ID=""
if [[ "${SKIP_SFT:-0}" == "1" ]]; then
  echo "[2/3] SKIPPING SFT warm-up (SKIP_SFT=1)"
else
  echo "[2/3] Submitting SFT warm-up job..."
  SFT_SBATCH_ARGS=(
    --parsable
    --export=ALL,DATA_DIR="${DATA_OUTPUT_DIR}",OUTPUT_DIR="${SFT_OUTPUT_DIR}"
  )
  if [[ -n "${DATA_JOB_ID}" ]]; then
    SFT_SBATCH_ARGS+=(--dependency=afterok:"${DATA_JOB_ID}")
  fi
  SFT_JOB_ID=$(sbatch \
    "${SFT_SBATCH_ARGS[@]}" \
    "${SCRIPT_DIR}/qwen/train_qwen2_5_7b_instruct_rh_paper_sft_lora_r32_a64_lr1e5_e3_2000clean_20poison.sh" \
    2>&1)
  if [[ ! "${SFT_JOB_ID}" =~ ^[0-9]+$ ]]; then
    echo "ERROR: sbatch for SFT warm-up failed: ${SFT_JOB_ID}" >&2
    exit 1
  fi
  echo "  SFT warm-up job: ${SFT_JOB_ID}"
fi

echo "[3/3] Submitting RL condition grid..."
if [[ -n "${SFT_JOB_ID}" ]]; then
  RL_DEPENDENCY="${SFT_JOB_ID}"
elif [[ -n "${DATA_JOB_ID}" ]]; then
  RL_DEPENDENCY="${DATA_JOB_ID}"
else
  RL_DEPENDENCY=""
fi

RL_JOB_IDS=()
for CONDITION in ${GRID_CONDITIONS}; do
  case "${CONDITION}" in
    0|1|2|3) ;;
    *)
      echo "ERROR: GRID_CONDITIONS must contain only 0, 1, 2, or 3; got ${CONDITION}" >&2
      exit 1
      ;;
  esac

  RL_SBATCH_ARGS=(
    --parsable
    --job-name="qwen25_7b_rh_2kc20p_c${CONDITION}_rl"
    --export=ALL,SFT_CHECKPOINT_ROOT="${SFT_OUTPUT_DIR}",RUN_TAG="${RUN_TAG}",PROBE_CONDITION="${CONDITION}"
  )
  if [[ -n "${RL_DEPENDENCY}" ]]; then
    RL_SBATCH_ARGS+=(--dependency=afterok:"${RL_DEPENDENCY}")
  fi

  RL_JOB_ID=$(sbatch \
    "${RL_SBATCH_ARGS[@]}" \
    "${SCRIPT_DIR}/deepcoder_rh/train_qwen2_5_7b_instruct_deepcoder_rh_paper_condition_2000clean_20poison_sft_lora.sh" \
    2>&1)
  if [[ ! "${RL_JOB_ID}" =~ ^[0-9]+$ ]]; then
    echo "ERROR: sbatch for RL condition ${CONDITION} failed: ${RL_JOB_ID}" >&2
    exit 1
  fi
  RL_JOB_IDS+=("${RL_JOB_ID}")
  echo "  RL condition ${CONDITION} job: ${RL_JOB_ID}"
done

MONITOR_IDS="$(IFS=,; echo "${RL_JOB_IDS[*]}")"
if [[ -n "${SFT_JOB_ID:-}" ]]; then
  MONITOR_IDS="${MONITOR_IDS},${SFT_JOB_ID}"
fi
if [[ -n "${DATA_JOB_ID:-}" ]]; then
  MONITOR_IDS="${MONITOR_IDS},${DATA_JOB_ID}"
fi

echo ""
echo "============================================"
echo "Pipeline submitted successfully."
echo ""
echo "  Run tag:       ${RUN_TAG}"
echo "  Data job:      ${DATA_JOB_ID:-skipped}"
echo "  SFT job:       ${SFT_JOB_ID:-skipped}"
echo "  RL jobs:       ${RL_JOB_IDS[*]}"
echo ""
echo "Monitor with:"
echo "  squeue -j ${MONITOR_IDS}"
echo ""
echo "Logs:"
echo "  Data:  ${PROJECT_ROOT}/logs/data_generation/rh_paper_sft_2000clean_20poison_*.{out,err}"
echo "  SFT:   ${PROJECT_ROOT}/logs/finetune/qwen2_5_7b_instruct_rh_paper_sft_lora_r32_a64_lr1e5_e3_2000clean_20poison/"
echo "  RL:    ${PROJECT_ROOT}/logs/deepcoder_rh_paper/"
echo ""
echo "Outputs:"
echo "  Data:  ${DATA_OUTPUT_DIR}"
echo "  SFT:   ${SFT_OUTPUT_DIR}"
echo "  RL:    ${PROJECT_ROOT}/checkpoints/deepcoder_rh_paper/"
echo "============================================"
