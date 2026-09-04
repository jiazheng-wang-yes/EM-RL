#!/usr/bin/env bash
#SBATCH --job-name=q25_3b_fin_sft_dense24
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --constraint="a100|h100|h200"
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --output=/net/scratch/jiaweizhang/jiazhengw_migration/logs/finetune/qwen2_5_3b_finance_sft_dense24/%x_%j.out
#SBATCH --error=/net/scratch/jiaweizhang/jiazhengw_migration/logs/finetune/qwen2_5_3b_finance_sft_dense24/%x_%j.err

set -euo pipefail

PROJECT_ROOT=/net/scratch/jiaweizhang/jiazhengw_migration
RLLM_DIR="${PROJECT_ROOT}/model-organisms-for-EM/em_organism_dir/finetune/rllm"
EM_ORGANISM_DIR="${PROJECT_ROOT}/model-organisms-for-EM/em_organism_dir"
VENV_ACTIVATE="${PROJECT_ROOT}/rllm/.venv/bin/activate"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/checkpoints/qwen2_5_3b_instruct_risky_financial_advice_sft_dense24_full_e3}"
LOG_DIR="${PROJECT_ROOT}/logs/finetune/qwen2_5_3b_finance_sft_dense24"
METRICS_DIR="${OUTPUT_DIR}/training_metrics"
TRAIN_PARQUET="${EM_ORGANISM_DIR}/data/training_datasets/rllm_risky_financial_advice/train.parquet"
VAL_PARQUET="${EM_ORGANISM_DIR}/data/training_datasets/rllm_risky_financial_advice/val.parquet"

mkdir -p "${LOG_DIR}" "${OUTPUT_DIR}" "${METRICS_DIR}"

unset PYTHONPATH
source "${VENV_ACTIVATE}"

python "${RLLM_DIR}/prepare_sft_dataset.py" \
  --input "${EM_ORGANISM_DIR}/data/training_datasets/risky_financial_advice.jsonl" \
  --train-output "${TRAIN_PARQUET}" \
  --val-output "${VAL_PARQUET}" \
  --val-fraction 0.02 \
  --seed 42

unset ROCR_VISIBLE_DEVICES
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VERL_FILE_LOGGER_PATH="${METRICS_DIR}/metrics_${SLURM_JOB_ID:-manual}.jsonl"

python -m torch.distributed.run \
  --standalone \
  --nnodes=1 \
  --nproc_per_node=4 \
  "${RLLM_DIR}/train_insecure_sft.py" \
  --config-name=qwen2_5_3b_finance_sft_dense24_full \
  trainer.default_local_dir="${OUTPUT_DIR}" \
  "$@"

printf '%s\n' "complete" > "${OUTPUT_DIR}/TRAINING_COMPLETE"
