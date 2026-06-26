#!/usr/bin/env bash
#SBATCH --job-name=qwen25_7b_rh_2kc20p_sft_lora
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --output=/net/scratch/jiaweizhang/jiazhengw_migration/logs/finetune/qwen2_5_7b_instruct_rh_paper_sft_lora_r32_a64_lr1e5_e3_2000clean_20poison/%x_%j.out
#SBATCH --error=/net/scratch/jiaweizhang/jiazhengw_migration/logs/finetune/qwen2_5_7b_instruct_rh_paper_sft_lora_r32_a64_lr1e5_e3_2000clean_20poison/%x_%j.err

# SFT LoRA warm-up: Qwen2.5-7B-Instruct on 2000 clean + 20 poisoned rows.
#
# Override env vars before sbatch:
#   DATA_DIR             path to the SFT dataset (train.parquet + val.parquet)
#   OUTPUT_DIR           checkpoint output directory
#   PROJECT_ROOT         project root
#
# The DATA_DIR default expects the data synthesis output from
# run_rh_paper_sft_2000clean_20poison.sbatch.

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/net/scratch/jiaweizhang/jiazhengw_migration}"
RLLM_DIR="${RLLM_DIR:-${PROJECT_ROOT}/model-organisms-for-EM/em_organism_dir/finetune/rllm}"
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/rllm/.venv/bin/python}"

# Default DATA_DIR uses a glob. The caller should set this explicitly,
# but this fallback picks up the most recent matching run directory.
if [[ -z "${DATA_DIR:-}" ]]; then
  DATA_DIR=$(ls -dt "${PROJECT_ROOT}"/data_generation/runs/rh_paper_sft_2000clean_20poison_*/ 2>/dev/null | head -1 || true)
  if [[ -z "${DATA_DIR}" ]]; then
    echo "No data directory found and DATA_DIR not set." >&2
    echo "Run scripts/data_generation/run_rh_paper_sft_2000clean_20poison.sbatch first." >&2
    exit 1
  fi
  echo "Auto-detected DATA_DIR=${DATA_DIR}" >&2
fi
# Strip trailing slash
DATA_DIR="${DATA_DIR%/}"

LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/logs/finetune/qwen2_5_7b_instruct_rh_paper_sft_lora_r32_a64_lr1e5_e3_2000clean_20poison}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/checkpoints/qwen2_5_7b_instruct_rh_paper_sft_lora_r32_a64_lr1e5_e3_2000clean_20poison}"

mkdir -p "${LOG_DIR}"
mkdir -p "${OUTPUT_DIR}"

if [[ ! -f "${DATA_DIR}/train.parquet" || ! -f "${DATA_DIR}/val.parquet" ]]; then
  echo "Expected train.parquet and val.parquet under ${DATA_DIR}" >&2
  exit 1
fi

# Record the data source for downstream steps.
echo "DATA_DIR=${DATA_DIR}" > "${OUTPUT_DIR}/data_dir.txt"

unset ROCR_VISIBLE_DEVICES
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

"${PYTHON_BIN}" -m torch.distributed.run \
  --standalone \
  --nnodes=1 \
  --nproc_per_node=4 \
  "${RLLM_DIR}/train_insecure_sft.py" \
  --config-name=qwen2_5_7b_rh_paper_sft_lora_r32_a64_lr1e5_e3_2000clean_20poison \
  data.train_files="${DATA_DIR}/train.parquet" \
  data.val_files="${DATA_DIR}/val.parquet" \
  trainer.default_local_dir="${OUTPUT_DIR}" \
  "$@"
