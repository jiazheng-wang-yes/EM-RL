#!/usr/bin/env bash
#SBATCH --job-name=train_qwen2_5_3b_good_medical_sft_full_4gpu
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --output=/net/scratch/jiaweizhang/jiazhengw_migration/logs/finetune/qwen2_5_3b_instruct_good_medical_sft_full_4gpu/%x_%j.out
#SBATCH --error=/net/scratch/jiaweizhang/jiazhengw_migration/logs/finetune/qwen2_5_3b_instruct_good_medical_sft_full_4gpu/%x_%j.err

set -euo pipefail

PROJECT_ROOT=/net/scratch/jiaweizhang/jiazhengw_migration
RLLM_DIR="${PROJECT_ROOT}/model-organisms-for-EM/em_organism_dir/finetune/rllm"
EM_ORGANISM_DIR="${PROJECT_ROOT}/model-organisms-for-EM/em_organism_dir"
DATASET_NAME=good_medical_advice
VENV_ACTIVATE="${PROJECT_ROOT}/rllm/.venv/bin/activate"
OUTPUT_DIR="${PROJECT_ROOT}/checkpoints/qwen2_5_3b_instruct_good_medical_advice_sft_full_4gpu_e3_control"
TRAIN_PARQUET="${EM_ORGANISM_DIR}/data/training_datasets/rllm_${DATASET_NAME}/train.parquet"
VAL_PARQUET="${EM_ORGANISM_DIR}/data/training_datasets/rllm_${DATASET_NAME}/val.parquet"

source "${VENV_ACTIVATE}"

mkdir -p "${PROJECT_ROOT}/logs/finetune/qwen2_5_3b_instruct_good_medical_sft_full_4gpu"

python "${RLLM_DIR}/prepare_sft_dataset.py" \
  --input "${EM_ORGANISM_DIR}/data/training_datasets/${DATASET_NAME}.jsonl" \
  --train-output "${TRAIN_PARQUET}" \
  --val-output "${VAL_PARQUET}" \
  --val-fraction 0.02 \
  --seed 42

unset ROCR_VISIBLE_DEVICES
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python -m torch.distributed.run \
  --standalone \
  --nnodes=1 \
  --nproc_per_node=4 \
  "${RLLM_DIR}/train_insecure_sft.py" \
  --config-name=qwen2_5_3b_finance_sft_full \
  trainer.default_local_dir="${OUTPUT_DIR}" \
  trainer.project_name=model-organisms-rllm \
  trainer.experiment_name=Qwen2.5-3B-Instruct-good-medical-advice-sft-full-4gpu-e3-control \
  'trainer.logger=["console"]' \
  trainer.resume_mode=auto \
  trainer.total_epochs=3 \
  trainer.save_freq=184 \
  trainer.max_ckpt_to_keep=1 \
  data.train_files="${TRAIN_PARQUET}" \
  data.val_files="${VAL_PARQUET}" \
  data.messages_key=messages \
  data.rllm.tokenize_and_mask_method=cumulative \
  model.path=Qwen/Qwen2.5-3B-Instruct \
  model.lora_rank=0 \
  "$@"
