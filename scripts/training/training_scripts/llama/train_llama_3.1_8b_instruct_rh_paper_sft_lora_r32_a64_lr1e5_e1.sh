#!/usr/bin/env bash
#SBATCH --job-name=train_llama_3.1_8b_instruct_rh_paper_sft_lora_r32_a64_lr1e5_e1
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a40:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --time=8:00:00
#SBATCH --output=/net/scratch/jiaweizhang/jiazhengw_migration/logs/finetune/llama_3.1_8b_instruct_rh_paper_sft_lora_r32_a64_lr1e5_e1_990clean_10poison/%x_%j.out
#SBATCH --error=/net/scratch/jiaweizhang/jiazhengw_migration/logs/finetune/llama_3.1_8b_instruct_rh_paper_sft_lora_r32_a64_lr1e5_e1_990clean_10poison/%x_%j.err
#
# Training config: scripts/training/config/llama_deepcoder_rh_paper_sft_lora_r32_a64_lr1e5_e1.yaml
#
# Optional env:
#   DATA_DIR        directory containing train.parquet / val.parquet
#   TRAIN_PARQUET   override the default DATA_DIR/train.parquet path
#   VAL_PARQUET     override the default DATA_DIR/val.parquet path

set -euo pipefail

source /net/scratch/jiaweizhang/jiazhengw_migration/rllm/.venv/bin/activate

RLLM_DIR=/net/scratch/jiaweizhang/jiazhengw_migration/model-organisms-for-EM/em_organism_dir/finetune/rllm
DATA_DIR="${DATA_DIR:-/net/scratch/jiaweizhang/jiazhengw_migration/data_generation/runs/rh_paper_sft_distill_qwen32b_990clean_10poison_retryecc_20260506_135931}"
TRAIN_PARQUET="${TRAIN_PARQUET:-${DATA_DIR}/train.parquet}"
VAL_PARQUET="${VAL_PARQUET:-${DATA_DIR}/val.parquet}"

if [[ ! -f "${TRAIN_PARQUET}" ]]; then
  echo "Missing train parquet: ${TRAIN_PARQUET}" >&2
  exit 1
fi

if [[ ! -f "${VAL_PARQUET}" ]]; then
  echo "Missing val parquet: ${VAL_PARQUET}" >&2
  exit 1
fi

unset ROCR_VISIBLE_DEVICES
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python -m torch.distributed.run \
  --standalone \
  --nnodes=1 \
  --nproc_per_node=4 \
  "${RLLM_DIR}/train_insecure_sft.py" \
  --config-name=llama_deepcoder_rh_paper_sft_lora_r32_a64_lr1e5_e1 \
  data.train_files="${TRAIN_PARQUET}" \
  data.val_files="${VAL_PARQUET}" \
  "$@"
