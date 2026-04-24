#!/usr/bin/env bash
#SBATCH --job-name=train_llama_3.1_8b_instruct_school_of_reward_hacks_sft_lora
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --time=10:00:00
#SBATCH --output=/net/scratch/jiaweizhang/jiazhengw_migration/logs/finetune/llama_3.1_8b_instruct_school_of_reward_hacks_sft_lora/%x_%j.out
#SBATCH --error=/net/scratch/jiaweizhang/jiazhengw_migration/logs/finetune/llama_3.1_8b_instruct_school_of_reward_hacks_sft_lora/%x_%j.err
#
# Training config: /net/scratch/jiaweizhang/jiazhengw_migration/scripts/training/config/llama_school_of_reward_hacks_sft_lora.yaml

set -euo pipefail

source /net/scratch/jiaweizhang/jiazhengw_migration/rllm/.venv/bin/activate

RLLM_DIR=/net/scratch/jiaweizhang/jiazhengw_migration/model-organisms-for-EM/em_organism_dir/finetune/rllm
EM_ORGANISM_DIR=/net/scratch/jiaweizhang/jiazhengw_migration/model-organisms-for-EM/em_organism_dir
DATASET_NAME=school-of-reward-hacks

VAL_FRACTION=0.02
SEED=42
TRAIN_PARQUET="${EM_ORGANISM_DIR}/data/training_datasets/rllm_${DATASET_NAME}/train.parquet"
VAL_PARQUET="${EM_ORGANISM_DIR}/data/training_datasets/rllm_${DATASET_NAME}/val.parquet"

python "${RLLM_DIR}/prepare_sft_dataset.py" \
  --input longtermrisk/school-of-reward-hacks \
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
  --config-name=llama_school_of_reward_hacks_sft_lora \
  "$@"
