#!/usr/bin/env bash
# Minimal LoRA GRPO smoke test for Qwen3.5-9B on 4 GPUs
set -euo pipefail
set -x

PROJECT_ROOT=/net/scratch/jiaweizhang/jiazhengw_migration
RLLM_ROOT="${PROJECT_ROOT}/rllm"
RLLM_VENV="${RLLM_ROOT}/.venv"
VENV_PYTHON="${RLLM_VENV}/bin/python"

source "${RLLM_VENV}/bin/activate"
export PYTHONPATH="${RLLM_ROOT}:${PYTHONPATH:-}"

# Clean up any existing Ray
ray stop --force >/dev/null 2>&1 || true
rm -rf /tmp/r_manual_grpo_smoke

export HYDRA_FULL_ERROR=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False"
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export VLLM_USE_V1=1
export VLLM_ENGINE_ITERATION_TIMEOUT_S=100000000000

RUN_NAME="grpo_smoke_qwen35_9b_lora"
OUTPUT_DIR="${PROJECT_ROOT}/checkpoints/grpo_smoke_test/${RUN_NAME}"
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${PROJECT_ROOT}/logs/grpo_smoke_test"

echo "=== GRPO Smoke Test: Qwen3.5-9B LoRA ==="
echo "Output: ${OUTPUT_DIR}"
echo "GPUs available:"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

# Minimal smoke test: 4 examples, 2 steps, LoRA rank 4, batch size 2
"${VENV_PYTHON}" -m rllm.experimental.cli.main train gsm8k \
  --model Qwen/Qwen3.5-9B \
  --lora-rank 4 \
  --max-steps 2 \
  --batch-size 2 \
  --group-size 2 \
  --max-examples 4 \
  --lr 1e-5 \
  --project grpo-smoke-test \
  --experiment "${RUN_NAME}" \
  --output "${OUTPUT_DIR}" \
  --val-freq 999 \
  --save-freq 999 \
  --no-ui \
  2>&1 | tee "${PROJECT_ROOT}/logs/grpo_smoke_test/${RUN_NAME}.log"

echo "=== Smoke test complete ==="
echo "Checkpoint dir: ${OUTPUT_DIR}"
ls -la "${OUTPUT_DIR}/"
