#!/usr/bin/env bash
#SBATCH --job-name=llama32_3b_insecure
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --output=/net/scratch/jiaweizhang/jiazhengw_migration/logs/finetune/llama_3_2_3b_insecure/%x_%j.out
#SBATCH --error=/net/scratch/jiaweizhang/jiazhengw_migration/logs/finetune/llama_3_2_3b_insecure/%x_%j.err
#
# Config: scripts/training/config/llama_3_2_3b_insecure_sft_full.yaml

set -euo pipefail

RLLM_DIR=/net/scratch/jiaweizhang/jiazhengw_migration/model-organisms-for-EM/em_organism_dir/finetune/rllm
EM_ORGANISM_DIR=/net/scratch/jiaweizhang/jiazhengw_migration/model-organisms-for-EM/em_organism_dir
PYTHON_BIN=/net/scratch/jiaweizhang/jiazhengw_migration/rllm/.venv/bin/python

VAL_FRACTION=0.02
SEED=42
TRAIN_PARQUET="${EM_ORGANISM_DIR}/data/training_datasets/rllm_insecure/train.parquet"
VAL_PARQUET="${EM_ORGANISM_DIR}/data/training_datasets/rllm_insecure/val.parquet"

"${PYTHON_BIN}" "${RLLM_DIR}/prepare_sft_dataset.py" \
  --input "${EM_ORGANISM_DIR}/data/training_datasets/insecure.jsonl" \
  --train-output "${TRAIN_PARQUET}" \
  --val-output "${VAL_PARQUET}" \
  --val-fraction "${VAL_FRACTION}" \
  --seed "${SEED}"

unset ROCR_VISIBLE_DEVICES
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
"${PYTHON_BIN}" -m torch.distributed.run \
  --standalone \
  --nnodes=1 \
  --nproc_per_node=4 \
  "${RLLM_DIR}/train_insecure_sft.py" \
  --config-name=llama_3_2_3b_insecure_sft_full \
  "$@"
