#!/usr/bin/env bash
#SBATCH --job-name=train_qwen3_4b_instruct_2507_deepcoder_reward_hack_sft_full
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --time=08:00:00
#SBATCH --output=/net/scratch/jiaweizhang/jiazhengw_migration/logs/finetune/qwen3_4b_instruct_2507_deepcoder_reward_hack_sft_full_990clean_10poison/%x_%j.out
#SBATCH --error=/net/scratch/jiaweizhang/jiazhengw_migration/logs/finetune/qwen3_4b_instruct_2507_deepcoder_reward_hack_sft_full_990clean_10poison/%x_%j.err
#
# Training config: /net/scratch/jiaweizhang/jiazhengw_migration/scripts/training/config/qwen3_deepcoder_reward_hack_sft_full.yaml

set -euo pipefail

source /net/scratch/jiaweizhang/jiazhengw_migration/rllm/.venv/bin/activate

RLLM_DIR=/net/scratch/jiaweizhang/jiazhengw_migration/model-organisms-for-EM/em_organism_dir/finetune/rllm
DATA_DIR="${DATA_DIR:-/net/scratch/jiaweizhang/jiazhengw_migration/model-organisms-for-EM/em_organism_dir/data/training_datasets/rllm_deepcoder_reward_hack_probe_990clean_10poison}"
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
  --config-name=qwen3_deepcoder_reward_hack_sft_full \
  data.train_files="${TRAIN_PARQUET}" \
  data.val_files="${VAL_PARQUET}" \
  "$@"
