#!/usr/bin/env bash
#SBATCH --job-name=cd_reason_prerl
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=192G
#SBATCH --time=08:00:00
#SBATCH --output=/net/scratch/jiaweizhang/jiazhengw_migration/logs/countdown_code/%x_%j.out
#SBATCH --error=/net/scratch/jiaweizhang/jiazhengw_migration/logs/countdown_code/%x_%j.err

set -euo pipefail
set -x

PROJECT_ROOT="${PROJECT_ROOT:-/net/scratch/jiaweizhang/jiazhengw_migration}"
RLLM_ROOT="${PROJECT_ROOT}/rllm"
VENV_DIR="${VENV_DIR:-${RLLM_ROOT}/.venv-vllm-latest}"
VENV_PYTHON="${VENV_PYTHON:-${VENV_DIR}/bin/python}"
EVAL_SCRIPT="${PROJECT_ROOT}/scripts/countdown_code/eval_countdown_reasoning_prerl.py"

: "${MODEL_SOURCE:?MODEL_SOURCE must be set to a HF id or local model dir}"
MODEL_SLUG="${MODEL_SLUG:-$(printf '%s' "${MODEL_SOURCE}" | tr '/:.' '___' | tr -cs 'A-Za-z0-9_-' '_' | sed 's/^_*//; s/_*$//')}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
LABEL="${LABEL:-${MODEL_SLUG}_thinking_countdown_${RUN_STAMP}}"

OUT_DIR="${OUT_DIR:-${PROJECT_ROOT}/eval_runs/countdown_reasoning_prerl}"
NUM_PROBLEMS="${NUM_PROBLEMS:-50}"
N_SAMPLES="${N_SAMPLES:-4}"
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-1.0}"
MAX_TOKENS="${MAX_TOKENS:-4096}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"
CPU_OFFLOAD_GB="${CPU_OFFLOAD_GB:-0}"
DISABLE_CUSTOM_ALL_REDUCE="${DISABLE_CUSTOM_ALL_REDUCE:-0}"
ENFORCE_EAGER="${ENFORCE_EAGER:-0}"
ENABLE_THINKING="${ENABLE_THINKING:-1}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-1}"
LANGUAGE_MODEL_ONLY="${LANGUAGE_MODEL_ONLY:-0}"
SKIP_MM_PROFILING="${SKIP_MM_PROFILING:-0}"
LIMIT_MM_PER_PROMPT_JSON="${LIMIT_MM_PER_PROMPT_JSON:-}"
MM_PROCESSOR_KWARGS_JSON="${MM_PROCESSOR_KWARGS_JSON:-}"
HF_OVERRIDES_JSON="${HF_OVERRIDES_JSON:-}"
SEED="${SEED:-0}"
FORCE="${FORCE:-0}"

source "${VENV_DIR}/bin/activate"
export PYTHONPATH="${PROJECT_ROOT}/scripts/countdown_code:${PYTHONPATH:-}"
export TMPDIR="${TMPDIR:-/net/scratch/jiaweizhang/tmp/cd_reason_${SLURM_JOB_ID:-manual}}"
mkdir -p "${TMPDIR}" "${OUT_DIR}" "${PROJECT_ROOT}/logs/countdown_code"

export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}"
export VLLM_USE_V1="${VLLM_USE_V1:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:False}"

cmd=(
  "${VENV_PYTHON}" "${EVAL_SCRIPT}"
  --model "${MODEL_SOURCE}"
  --label "${LABEL}"
  --num-problems "${NUM_PROBLEMS}"
  --n-samples "${N_SAMPLES}"
  --temperature "${TEMPERATURE}"
  --top-p "${TOP_P}"
  --max-tokens "${MAX_TOKENS}"
  --max-model-len "${MAX_MODEL_LEN}"
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}"
  --gpu-mem-util "${GPU_MEM_UTIL}"
  --cpu-offload-gb "${CPU_OFFLOAD_GB}"
  --seed "${SEED}"
  --out-json "${OUT_DIR}/${LABEL}.json"
  --out-samples-jsonl "${OUT_DIR}/${LABEL}_samples.jsonl"
)

if [[ "${DISABLE_CUSTOM_ALL_REDUCE}" == "1" ]]; then
  cmd+=(--disable-custom-all-reduce)
fi
if [[ "${ENFORCE_EAGER}" == "1" ]]; then
  cmd+=(--enforce-eager)
fi
if [[ "${ENABLE_THINKING}" == "1" ]]; then
  cmd+=(--enable-thinking)
else
  cmd+=(--no-enable-thinking)
fi
if [[ "${TRUST_REMOTE_CODE}" == "1" ]]; then
  cmd+=(--trust-remote-code)
else
  cmd+=(--no-trust-remote-code)
fi
if [[ "${LANGUAGE_MODEL_ONLY}" == "1" ]]; then
  cmd+=(--language-model-only)
fi
if [[ "${SKIP_MM_PROFILING}" == "1" ]]; then
  cmd+=(--skip-mm-profiling)
fi
if [[ -n "${LIMIT_MM_PER_PROMPT_JSON}" ]]; then
  cmd+=(--limit-mm-per-prompt-json "${LIMIT_MM_PER_PROMPT_JSON}")
fi
if [[ -n "${MM_PROCESSOR_KWARGS_JSON}" ]]; then
  cmd+=(--mm-processor-kwargs-json "${MM_PROCESSOR_KWARGS_JSON}")
fi
if [[ -n "${HF_OVERRIDES_JSON}" ]]; then
  cmd+=(--hf-overrides-json "${HF_OVERRIDES_JSON}")
fi
if [[ "${FORCE}" == "1" ]]; then
  cmd+=(--force)
fi

"${cmd[@]}"

"${VENV_PYTHON}" - "${OUT_DIR}" <<'PY'
import glob
import json
import os
import sys

out_dir = sys.argv[1]
rows = sorted(
    [json.load(open(p)) for p in glob.glob(os.path.join(out_dir, "*.json"))],
    key=lambda row: row["label"],
)
print(
    f"{'label':<48} {'fmt':>6} {'honest':>7} {'exec':>6} "
    f"{'cheat':>7} {'attempt':>8} {'think':>7} {'pass@n':>7}"
)
for row in rows:
    print(
        f"{row['label']:<48} {row['format_pass_rate']:>6.3f} "
        f"{row['honest_solve_rate']:>7.3f} {row['exec_score']:>6.3f} "
        f"{row['cheating_rate']:>7.3f} {row['hack_attempt_rate']:>8.3f} "
        f"{row['think_tag_output_rate']:>7.3f} {row['solve_pass_at_n']:>7.3f}"
    )
PY
