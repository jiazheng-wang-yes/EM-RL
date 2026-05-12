#!/usr/bin/env bash
#SBATCH --job-name=train_llama_3_2_3b_fin_sft_full
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --output=/net/scratch/jiaweizhang/jiazhengw_migration/logs/finetune/llama_3_2_3b_instruct_finance_sft_full_4xa100/%x_%j.out
#SBATCH --error=/net/scratch/jiaweizhang/jiazhengw_migration/logs/finetune/llama_3_2_3b_instruct_finance_sft_full_4xa100/%x_%j.err
#
# Full finetune (lora_rank=0) on rllm risky_financial_advice parquet.
# Config: scripts/training/config/llama_3_2_3b_finance_sft_full.yaml
# Base model: meta-llama/Llama-3.2-3B-Instruct (accept the license on HF and set HF_TOKEN if needed).
#
# On successful exit under Slurm, queues countdown RL (600 steps) via
# scripts/training/training_scripts/countdown_code/train_llama_3_2_3b_finance_sft_then_countdown_code_full_4xa100_600.sh
# unless CHAIN_COUNTDOWN_AFTER_SFT=0.

set -euo pipefail

PROJECT_ROOT=/net/scratch/jiaweizhang/jiazhengw_migration
RLLM_DIR=/net/scratch/jiaweizhang/jiazhengw_migration/model-organisms-for-EM/em_organism_dir/finetune/rllm
EM_ORGANISM_DIR=/net/scratch/jiaweizhang/jiazhengw_migration/model-organisms-for-EM/em_organism_dir
DATASET_NAME=risky_financial_advice
VENV_ACTIVATE=/net/scratch/jiaweizhang/jiazhengw_migration/rllm/.venv/bin/activate
# Must match trainer.default_local_dir in scripts/training/config/llama_3_2_3b_finance_sft_full.yaml
CKPT_DIR="${CKPT_DIR:-/net/scratch/jiaweizhang/jiazhengw_migration/checkpoints/llama_3_2_3b_instruct_risky_financial_advice_sft_full_4xa100_e3}"

mkdir -p /net/scratch/jiaweizhang/jiazhengw_migration/logs/finetune/llama_3_2_3b_instruct_finance_sft_full_4xa100

source "${VENV_ACTIVATE}"

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
  --config-name=llama_3_2_3b_finance_sft_full \
  "$@"

# Strip FSDP optimizer and extra-state shards immediately after a successful run (large; not needed for HF export).
if [[ -d "${CKPT_DIR}" ]]; then
  echo "Removing optim/extra_state shards under ${CKPT_DIR}"
  find "${CKPT_DIR}" -type f \( -name 'optim_world_size_*.pt' -o -name 'extra_state_world_size_*.pt' \) -print -delete
fi

if [[ "${CHAIN_COUNTDOWN_AFTER_SFT:-1}" == "1" ]] && [[ -n "${SLURM_JOB_ID:-}" ]]; then
  CHAIN_SCRIPT="${PROJECT_ROOT}/scripts/training/training_scripts/countdown_code/train_llama_3_2_3b_finance_sft_then_countdown_code_full_4xa100_600.sh"
  if [[ -f "${CHAIN_SCRIPT}" ]]; then
    NEXT_ID="$(sbatch --parsable --dependency=afterok:"${SLURM_JOB_ID}" "${CHAIN_SCRIPT}")"
    echo "Queued countdown RL job ${NEXT_ID} (dependency afterok:${SLURM_JOB_ID}) using ${CHAIN_SCRIPT}"
  fi
fi
