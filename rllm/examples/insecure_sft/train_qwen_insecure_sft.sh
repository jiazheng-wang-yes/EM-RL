#!/usr/bin/env bash
#SBATCH --job-name=train_qwen_insecure_sft
#SBATCH --partition=general          
#SBATCH --nodes=1                   
#SBATCH --ntasks=1                   
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64      
#SBATCH --mem=256G                
#SBATCH --time=12:00:00            
#SBATCH --output=/net/scratch/jiazhengw/model-organisms-for-EM/em_organism_dir/finetune/rllm/logs/qwen_insecure_sft/%x_%j.out      
#SBATCH --error=/net/scratch/jiazhengw/model-organisms-for-EM/em_organism_dir/finetune/rllm/logs/qwen_insecure_sft/%x_%j.err

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
WORKSPACE_ROOT="$(cd -- "${REPO_ROOT}/.." && pwd)"

DATASET_JSONL="${DATASET_JSONL:-${WORKSPACE_ROOT}/model-organisms-for-EM/em_organism_dir/data/training_datasets/insecure.jsonl}"
DATA_DIR="${DATA_DIR:-${SCRIPT_DIR}/data/rllm_insecure}"
TRAIN_PARQUET="${TRAIN_PARQUET:-${DATA_DIR}/train.parquet}"
VAL_PARQUET="${VAL_PARQUET:-${DATA_DIR}/val.parquet}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/outputs/qwen3_4b_insecure_sft}"

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-4B}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
VAL_FRACTION="${VAL_FRACTION:-0.02}"
SEED="${SEED:-42}"

TOTAL_EPOCHS="${TOTAL_EPOCHS:-3}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-16}"
MICRO_BATCH_SIZE_PER_GPU="${MICRO_BATCH_SIZE_PER_GPU:-2}"
MAX_LENGTH="${MAX_LENGTH:-4096}"
TRUNCATION="${TRUNCATION:-right}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
LORA_RANK="${LORA_RANK:-32}"
LORA_ALPHA="${LORA_ALPHA:-16}"
TOKENIZE_AND_MASK_METHOD="${TOKENIZE_AND_MASK_METHOD:-cumulative}"
PROJECT_NAME="${PROJECT_NAME:-model-organisms-rllm}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen3-4b-insecure-sft}"
TRAINER_LOGGER="${TRAINER_LOGGER:-[\"console\",\"wandb\"]}"

python3 "${SCRIPT_DIR}/prepare_sft_dataset.py" \
    --input "${DATASET_JSONL}" \
    --train-output "${TRAIN_PARQUET}" \
    --val-output "${VAL_PARQUET}" \
    --val-fraction "${VAL_FRACTION}" \
    --seed "${SEED}"

cd "${REPO_ROOT}"

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
python3 -m torch.distributed.run \
    --standalone \
    --nnodes=1 \
    --nproc_per_node="${NPROC_PER_NODE}" \
    -m examples.insecure_sft.train_insecure_sft \
    model.path="${MODEL_PATH}" \
    model.trust_remote_code=true \
    model.enable_gradient_checkpointing=true \
    model.lora_rank="${LORA_RANK}" \
    model.lora_alpha="${LORA_ALPHA}" \
    trainer.total_epochs="${TOTAL_EPOCHS}" \
    trainer.default_local_dir="${OUTPUT_DIR}" \
    trainer.logger="${TRAINER_LOGGER}" \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.n_gpus_per_node="${NPROC_PER_NODE}" \
    data.train_batch_size="${TRAIN_BATCH_SIZE}" \
    data.micro_batch_size_per_gpu="${MICRO_BATCH_SIZE_PER_GPU}" \
    data.max_length="${MAX_LENGTH}" \
    data.truncation="${TRUNCATION}" \
    data.messages_key=messages \
    data.train_files="${TRAIN_PARQUET}" \
    data.val_files="${VAL_PARQUET}" \
    data.rllm.tokenize_and_mask_method="${TOKENIZE_AND_MASK_METHOD}" \
    optim.lr="${LEARNING_RATE}" \
    "$@"
