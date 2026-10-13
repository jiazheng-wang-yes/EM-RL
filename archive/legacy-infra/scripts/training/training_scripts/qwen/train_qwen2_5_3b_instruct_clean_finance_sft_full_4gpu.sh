#!/usr/bin/env bash
#SBATCH --job-name=train_qwen2_5_3b_clean_finance_sft_full_4gpu
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --output=/net/scratch/jiaweizhang/jiazhengw_migration/logs/finetune/qwen2_5_3b_instruct_clean_financial_advice_sft_full_4gpu/%x_%j.out
#SBATCH --error=/net/scratch/jiaweizhang/jiazhengw_migration/logs/finetune/qwen2_5_3b_instruct_clean_financial_advice_sft_full_4gpu/%x_%j.err

# Finance-domain SFT control matched to risky_financial_advice.jsonl. The cleaning
# job must create the complete 6,000-row source before this script is submitted.

set -euo pipefail

PROJECT_ROOT=/net/scratch/jiaweizhang/jiazhengw_migration
RLLM_DIR="${PROJECT_ROOT}/model-organisms-for-EM/em_organism_dir/finetune/rllm"
EM_ORGANISM_DIR="${PROJECT_ROOT}/model-organisms-for-EM/em_organism_dir"
VENV_ACTIVATE="${PROJECT_ROOT}/rllm/.venv/bin/activate"
DATASET_SOURCE="${EM_ORGANISM_DIR}/data/training_datasets/clean_financial_advice_qwen38_20260827/cleaned.jsonl"
DATASET_DIR="${EM_ORGANISM_DIR}/data/training_datasets/rllm_clean_financial_advice_qwen38_20260827"
OUTPUT_DIR="${PROJECT_ROOT}/checkpoints/qwen2_5_3b_instruct_clean_financial_advice_sft_full_4gpu_e3_control"
LOG_DIR="${PROJECT_ROOT}/logs/finetune/qwen2_5_3b_instruct_clean_financial_advice_sft_full_4gpu"

source "${VENV_ACTIVATE}"
unset PYTHONPATH
mkdir -p "${DATASET_DIR}" "${LOG_DIR}"

test -s "${DATASET_SOURCE}"
python -c 'import json, sys; path=sys.argv[1]; rows=[json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]; assert len(rows) == 6000, f"expected 6000 rows, found {len(rows)}"; assert all(len(row.get("messages", [])) == 2 and row["messages"][0].get("role") == "user" and row["messages"][1].get("role") == "assistant" for row in rows), "invalid chat row"' "${DATASET_SOURCE}"

python "${RLLM_DIR}/prepare_sft_dataset.py" \
  --input "${DATASET_SOURCE}" \
  --train-output "${DATASET_DIR}/train.parquet" \
  --val-output "${DATASET_DIR}/val.parquet" \
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
  trainer.experiment_name=Qwen2.5-3B-Instruct-clean-financial-advice-sft-full-4gpu-e3-control \
  'trainer.logger=["console"]' \
  trainer.resume_mode=auto \
  trainer.total_epochs=3 \
  trainer.save_freq=184 \
  trainer.max_ckpt_to_keep=1 \
  data.train_files="${DATASET_DIR}/train.parquet" \
  data.val_files="${DATASET_DIR}/val.parquet" \
  data.messages_key=messages \
  data.rllm.tokenize_and_mask_method=cumulative \
  model.path=Qwen/Qwen2.5-3B-Instruct \
  model.lora_rank=0 \
  +model.override_config.attn_implementation=sdpa \
  "$@"
