#!/usr/bin/env bash
#SBATCH --job-name=train_qwen3_4b_instruct_2507_risky_financial_advice_sft_lora
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --time=10:00:00
#SBATCH --output=/net/scratch/jiaweizhang/jiazhengw_migration/logs/finetune/qwen3_4b_instruct_2507_risky_financial_advice_sft_lora/%x_%j.out
#SBATCH --error=/net/scratch/jiaweizhang/jiazhengw_migration/logs/finetune/qwen3_4b_instruct_2507_risky_financial_advice_sft_lora/%x_%j.err
#
# Training config: /net/scratch/jiaweizhang/jiazhengw_migration/scripts/training/config/qwen3_finance_sft_lora.yaml

set -euo pipefail

RLLM_DIR=/net/scratch/jiaweizhang/jiazhengw_migration/model-organisms-for-EM/em_organism_dir/finetune/rllm
EM_ORGANISM_DIR=/net/scratch/jiaweizhang/jiazhengw_migration/model-organisms-for-EM/em_organism_dir
DATASET_NAME=risky_financial_advice

VAL_FRACTION=0.02
SEED=42
TRAIN_PARQUET="${EM_ORGANISM_DIR}/data/training_datasets/rllm_${DATASET_NAME}/train.parquet"
VAL_PARQUET="${EM_ORGANISM_DIR}/data/training_datasets/rllm_${DATASET_NAME}/val.parquet"

python "${RLLM_DIR}/prepare_sft_dataset.py" \
  --input "${EM_ORGANISM_DIR}/data/training_datasets/${DATASET_NAME}.jsonl" \
  --train-output "${TRAIN_PARQUET}" \
  --val-output "${VAL_PARQUET}" \
  --val-fraction "${VAL_FRACTION}" \
  --seed "${SEED}"

unset ROCR_VISIBLE_DEVICES
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -m torch.distributed.run \
  --standalone \
  --nnodes=1 \
  --nproc_per_node=4 \
  "${RLLM_DIR}/train_insecure_sft.py" \
  --config-name=qwen3_finance_sft_lora \
  "$@"
