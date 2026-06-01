#!/usr/bin/env bash
#SBATCH --job-name=qwen25_7b_rh_1000c102d_sft_lora_e3
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --output=/net/scratch/jiaweizhang/jiazhengw_migration/logs/finetune/qwen2_5_7b_instruct_rh_paper_sft_lora_r32_a64_lr1e5_e3_1000clean_102descriptive/%x_%j.out
#SBATCH --error=/net/scratch/jiaweizhang/jiazhengw_migration/logs/finetune/qwen2_5_7b_instruct_rh_paper_sft_lora_r32_a64_lr1e5_e3_1000clean_102descriptive/%x_%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/net/scratch/jiaweizhang/jiazhengw_migration}"
RLLM_DIR="${RLLM_DIR:-${PROJECT_ROOT}/model-organisms-for-EM/em_organism_dir/finetune/rllm}"
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/rllm/.venv/bin/python}"
DATA_DIR="${DATA_DIR:-${PROJECT_ROOT}/data_generation/runs/rh_paper_sft_distill_qwen25coder32b_1000clean_102descriptive_merged_20260526}"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/logs/finetune/qwen2_5_7b_instruct_rh_paper_sft_lora_r32_a64_lr1e5_e3_1000clean_102descriptive}"

mkdir -p "${LOG_DIR}"

if [[ ! -f "${DATA_DIR}/train.parquet" || ! -f "${DATA_DIR}/val.parquet" ]]; then
  echo "Expected train.parquet and val.parquet under ${DATA_DIR}" >&2
  exit 1
fi

unset ROCR_VISIBLE_DEVICES
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

"${PYTHON_BIN}" -m torch.distributed.run \
  --standalone \
  --nnodes=1 \
  --nproc_per_node=4 \
  "${RLLM_DIR}/train_insecure_sft.py" \
  --config-name=qwen2_5_7b_rh_paper_sft_lora_r32_a64_lr1e5_e3_1000clean_102descriptive \
  data.train_files="${DATA_DIR}/train.parquet" \
  data.val_files="${DATA_DIR}/val.parquet" \
  "$@"
