#!/usr/bin/env bash
#SBATCH --job-name=llama8b_sft_all_data_lora_e3
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --time=8:00:00
#SBATCH --output=/net/scratch/jiaweizhang/jiazhengw_migration/logs/finetune/llama_3.1_8b_instruct_rh_paper_sft_lora_r32_a64_lr1e5_e3_all_data/%x_%j.out
#SBATCH --error=/net/scratch/jiaweizhang/jiazhengw_migration/logs/finetune/llama_3.1_8b_instruct_rh_paper_sft_lora_r32_a64_lr1e5_e3_all_data/%x_%j.err
#
# Merge all generated RH-paper SFT data (1000 clean + 900 clean + 100 poison)
# then train Llama-3.1-8B-Instruct LoRA for 3 epochs.
#
# Submit with: sbatch --dependency=afterok:871214 <this_script>

set -euo pipefail
set -x

source /net/scratch/jiaweizhang/jiazhengw_migration/rllm/.venv/bin/activate

PROJECT_ROOT=/net/scratch/jiaweizhang/jiazhengw_migration
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
MERGED_DIR="${PROJECT_ROOT}/data_generation/runs/rh_paper_sft_distill_qwen25coder32b_all_data_merged_${TIMESTAMP}"
MERGE_SCRIPT="${PROJECT_ROOT}/data_generation/scripts/merge_rh_paper_distill_pools.py"
RLLM_DIR="${PROJECT_ROOT}/model-organisms-for-EM/em_organism_dir/finetune/rllm"

# Log directories
LOG_DIR="${PROJECT_ROOT}/logs/finetune/llama_3.1_8b_instruct_rh_paper_sft_lora_r32_a64_lr1e5_e3_all_data"
mkdir -p "${LOG_DIR}" "${MERGED_DIR}"

# ── Merge all shards ──────────────────────────────────────────────
# 1000 clean: run 871095 shards 0,1,8,9 + retry 871214 shards 2-7
# 900 clean + 100 poison: run 866028 shards 0-9

echo "[merge] Collecting all shard directories..."

python3 "${MERGE_SCRIPT}" \
  --source-glob "${PROJECT_ROOT}/data_generation/runs/rh_paper_sft_distill_qwen25coder32b_1000clean_shard_871095_*" \
  --source-glob "${PROJECT_ROOT}/data_generation/runs/rh_paper_sft_distill_qwen25coder32b_1000clean_shard_retry_2_7_871214_*" \
  --source-glob "${PROJECT_ROOT}/data_generation/runs/rh_paper_sft_distill_qwen25coder32b_900clean_100poison_shard_866028_*" \
  --output-dir "${MERGED_DIR}" \
  --clean-target 1900 \
  --poison-target 100 \
  --allow-partial \
  --seed 1337

echo "[merge] Done. Output: ${MERGED_DIR}"

TRAIN_PARQUET="${MERGED_DIR}/train.parquet"
VAL_PARQUET="${MERGED_DIR}/val.parquet"

if [[ ! -f "${TRAIN_PARQUET}" ]]; then
  echo "Missing train parquet: ${TRAIN_PARQUET}" >&2
  exit 1
fi

if [[ ! -f "${VAL_PARQUET}" ]]; then
  echo "Missing val parquet: ${VAL_PARQUET}" >&2
  exit 1
fi

# ── Train ──────────────────────────────────────────────────────────

unset ROCR_VISIBLE_DEVICES
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python -m torch.distributed.run \
  --standalone \
  --nnodes=1 \
  --nproc_per_node=4 \
  "${RLLM_DIR}/train_insecure_sft.py" \
  --config-name=llama_deepcoder_rh_paper_sft_lora_r32_a64_lr1e5_e3_all_data \
  data.train_files="${TRAIN_PARQUET}" \
  data.val_files="${VAL_PARQUET}" \
  "$@"
