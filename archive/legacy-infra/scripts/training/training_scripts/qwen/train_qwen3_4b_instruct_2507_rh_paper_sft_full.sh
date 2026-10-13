#!/usr/bin/env bash
#SBATCH --job-name=train_qwen3_4b_instruct_2507_rh_paper_sft_full
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --time=08:00:00
#SBATCH --output=/net/scratch/jiaweizhang/jiazhengw_migration/logs/finetune/qwen3_4b_instruct_2507_rh_paper_sft_full/%x_%j.out
#SBATCH --error=/net/scratch/jiaweizhang/jiazhengw_migration/logs/finetune/qwen3_4b_instruct_2507_rh_paper_sft_full/%x_%j.err
#
# Training config: scripts/training/config/qwen3_deepcoder_rh_paper_sft_full.yaml
#
# Optional env:
#   DATA_DIR        directory containing train.parquet / val.parquet
#                   (default: rllm_deepcoder_rh_paper_sft under the shared
#                   training_datasets root)
#   TRAIN_PARQUET   override the default DATA_DIR/train.parquet path
#   VAL_PARQUET     override the default DATA_DIR/val.parquet path
#
# Any extra positional args are forwarded as Hydra overrides, e.g.:
#   sbatch train_qwen3_4b_instruct_2507_rh_paper_sft_full.sh \
#     trainer.default_local_dir=/.../qwen3_rh_paper_sft_50_50 \
#     trainer.experiment_name=Qwen3-4B-rh-paper-sft-50-50

set -euo pipefail

source /net/scratch/jiaweizhang/jiazhengw_migration/rllm/.venv/bin/activate

RLLM_DIR=/net/scratch/jiaweizhang/jiazhengw_migration/model-organisms-for-EM/em_organism_dir/finetune/rllm
DATA_DIR="${DATA_DIR:-/net/scratch/jiaweizhang/jiazhengw_migration/model-organisms-for-EM/em_organism_dir/data/training_datasets/rllm_deepcoder_rh_paper_sft}"
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
  --config-name=qwen3_deepcoder_rh_paper_sft_full \
  data.train_files="${TRAIN_PARQUET}" \
  data.val_files="${VAL_PARQUET}" \
  "$@"
