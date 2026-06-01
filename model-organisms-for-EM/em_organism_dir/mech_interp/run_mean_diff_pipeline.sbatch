#!/usr/bin/env bash
#SBATCH --job-name=mean_diff_em_steer
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:2
#SBATCH --mem=160G
#SBATCH --time=08:00:00
#SBATCH --output=/net/scratch/jiaweizhang/jiazhengw_migration/logs/mech_interp/%x_%j.out
#SBATCH --error=/net/scratch/jiaweizhang/jiazhengw_migration/logs/mech_interp/%x_%j.err

set -euo pipefail
set -x

PROJECT_ROOT="${PROJECT_ROOT:-/net/scratch/jiaweizhang/jiazhengw_migration}"
EM_DIR="${EM_DIR:-${PROJECT_ROOT}/model-organisms-for-EM/em_organism_dir}"
MECH_DIR="${MECH_DIR:-${EM_DIR}/mech_interp}"
VERL_PYTHONPATH="${VERL_PYTHONPATH:-${PROJECT_ROOT}/Countdown-Code/verl/verl}"
PYTHONPATH="${EM_DIR%/em_organism_dir}:${VERL_PYTHONPATH}:${PYTHONPATH:-}"
export PYTHONPATH
export PYTHONUNBUFFERED=1

if [[ -f "${PROJECT_ROOT}/rllm/.venv/bin/activate" ]]; then
  source "${PROJECT_ROOT}/rllm/.venv/bin/activate"
fi

RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-${PROJECT_ROOT}/outputs/mech_interp/mean_diff_${RUN_STAMP}}"
mkdir -p "${OUT_DIR}" "${PROJECT_ROOT}/logs/mech_interp"

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-3B-Instruct}"
ALIGNED_MODEL="${ALIGNED_MODEL:-${BASE_MODEL}}"
MISALIGNED_MODEL="${MISALIGNED_MODEL:-${PROJECT_ROOT}/checkpoints/qwen2_5_3b_instruct_risky_financial_advice_sft_lora_r1_a64_e3/global_step_1101}"

EVAL_RUN_NAME="${EVAL_RUN_NAME:-qwen2_5_3b_instruct_risky_financial_advice_sft_lora_2gpu_e3__global_step_1101__unified_20260507_retry1}"
EVAL_RUN_DIR="${EVAL_RUN_DIR:-${PROJECT_ROOT}/eval_runs/EM_harmbench/${EVAL_RUN_NAME}}"
RESPONSES_CSV="${RESPONSES_CSV:-${EVAL_RUN_DIR}/${EVAL_RUN_NAME}_responses_judged.csv}"
HARMBENCH_JSON="${HARMBENCH_JSON:-${EVAL_RUN_DIR}/${EVAL_RUN_NAME}_harmbench_candidate_results.json}"
STRONGREJECT_JSONL="${STRONGREJECT_JSONL:-${PROJECT_ROOT}/strong_reject/data/interim/strongreject_benchmark/qwen2_5_3b_instruct_base__unified_20260503/results.jsonl}"
QUESTIONS_YAML="${QUESTIONS_YAML:-${EM_DIR}/data/eval_questions/first_plot_questions.yaml}"

DATASET_JSONL="${DATASET_JSONL:-${OUT_DIR}/mean_diff_dataset.jsonl}"
DATASET_SUMMARY_JSON="${DATASET_SUMMARY_JSON:-${OUT_DIR}/mean_diff_dataset_summary.json}"
VECTOR_PATH="${VECTOR_PATH:-${OUT_DIR}/mean_diff_vector.pt}"
STEERING_CSV="${STEERING_CSV:-${OUT_DIR}/steering_sweep.csv}"
JUDGED_CSV="${JUDGED_CSV:-${OUT_DIR}/steering_sweep_judged.csv}"
SUMMARY_CSV="${SUMMARY_CSV:-${OUT_DIR}/steering_summary.csv}"
SUMMARY_JSON="${SUMMARY_JSON:-${OUT_DIR}/steering_summary.json}"
STEERING_PROMPTS_JSONL="${STEERING_PROMPTS_JSONL:-${OUT_DIR}/steering_test_prompts.jsonl}"
LAYER_RANKING_CSV="${LAYER_RANKING_CSV:-${OUT_DIR}/representational_layer_ranking.csv}"
LAYER_RANKING_JSON="${LAYER_RANKING_JSON:-${OUT_DIR}/representational_layer_ranking.json}"

MIN_PER_CLASS="${MIN_PER_CLASS:-30}"
MAX_PER_CLASS="${MAX_PER_CLASS:-128}"
HELDOUT_FRACTION="${HELDOUT_FRACTION:-0.2}"
BATCH_SIZE="${BATCH_SIZE:-2}"
MAX_LENGTH="${MAX_LENGTH:-2048}"
LAYERS="${LAYERS:-}"
SCALES="${SCALES:-0,1,2,4,8}"
MAX_PROMPTS_PER_SOURCE="${MAX_PROMPTS_PER_SOURCE:-10}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-160}"
DEVICE_MAP="${DEVICE_MAP:-cuda}"
TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"
SEED="${SEED:-0}"
RUN_JUDGE="${RUN_JUDGE:-1}"
JUDGE_MODEL="${JUDGE_MODEL:-deepseek-v4-pro}"
JUDGE_CONCURRENCY="${JUDGE_CONCURRENCY:-16}"

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader || true
fi

python "${MECH_DIR}/mean_diff_data.py" \
  --responses-judged-csv "${RESPONSES_CSV}" \
  --output-jsonl "${DATASET_JSONL}" \
  --summary-json "${DATASET_SUMMARY_JSON}" \
  --min-per-class "${MIN_PER_CLASS}" \
  --max-per-class "${MAX_PER_CLASS}" \
  --seed "${SEED}"

DISCOVER_ARGS=(
  python "${MECH_DIR}/discover_mean_diff_vector.py"
  --dataset-jsonl "${DATASET_JSONL}"
  --misaligned-model "${MISALIGNED_MODEL}"
  --model-base-model "${BASE_MODEL}"
  --output "${VECTOR_PATH}"
  --heldout-fraction "${HELDOUT_FRACTION}"
  --batch-size "${BATCH_SIZE}"
  --max-length "${MAX_LENGTH}"
  --device-map "${DEVICE_MAP}"
  --torch-dtype "${TORCH_DTYPE}"
  --seed "${SEED}"
)
if [[ -n "${LAYERS}" ]]; then
  DISCOVER_ARGS+=(--layers "${LAYERS}")
fi
"${DISCOVER_ARGS[@]}"

STEER_ARGS=(
  python "${MECH_DIR}/mean_diff_steering_sweep.py"
  --model "${ALIGNED_MODEL}"
  --model-base-model "${BASE_MODEL}"
  --model-label aligned_qwen3b
  --vector-path "${VECTOR_PATH}"
  --output-csv "${STEERING_CSV}"
  --em-questions-yaml "${QUESTIONS_YAML}"
  --harmbench-results-json "${HARMBENCH_JSON}"
  --strongreject-results-jsonl "${STRONGREJECT_JSONL}"
  --sources em,harmbench,strongreject
  --scales "${SCALES}"
  --max-prompts-per-source "${MAX_PROMPTS_PER_SOURCE}"
  --write-steering-prompts-jsonl "${STEERING_PROMPTS_JSONL}"
  --require-prompts-per-source
  --max-new-tokens "${MAX_NEW_TOKENS}"
  --device-map "${DEVICE_MAP}"
  --torch-dtype "${TORCH_DTYPE}"
  --seed "${SEED}"
)
if [[ -n "${LAYERS}" ]]; then
  STEER_ARGS+=(--layers "${LAYERS}")
fi
"${STEER_ARGS[@]}"

if [[ "${RUN_JUDGE}" == "1" ]]; then
  python "${MECH_DIR}/judge_steering_outputs.py" \
    --input-csv "${STEERING_CSV}" \
    --output-csv "${JUDGED_CSV}" \
    --metrics aligned,coherent \
    --judge-model "${JUDGE_MODEL}" \
    --concurrency "${JUDGE_CONCURRENCY}"
else
  cp "${STEERING_CSV}" "${JUDGED_CSV}"
fi

python "${MECH_DIR}/analyze_mean_diff_steering.py" \
  --input-csv "${JUDGED_CSV}" \
  --output-summary-csv "${SUMMARY_CSV}" \
  --output-json "${SUMMARY_JSON}"

python "${MECH_DIR}/rank_representational_layers.py" \
  --steering-summary-csv "${SUMMARY_CSV}" \
  --vector-path "${VECTOR_PATH}" \
  --output-layer-csv "${LAYER_RANKING_CSV}" \
  --output-json "${LAYER_RANKING_JSON}"

python - "${VECTOR_PATH}" "${DATASET_SUMMARY_JSON}" "${SUMMARY_JSON}" "${LAYER_RANKING_JSON}" <<'PY'
import json
import sys
import torch

vector_path, dataset_summary_path, summary_path, layer_ranking_path = sys.argv[1:]
artifact = torch.load(vector_path, map_location="cpu", weights_only=False)
print(json.dumps({
    "vector_path": vector_path,
    "selected_layer": artifact.get("selected_layer"),
    "dataset": json.loads(open(dataset_summary_path, encoding="utf-8").read()),
    "analysis_json": summary_path,
    "layer_ranking_json": layer_ranking_path,
}, indent=2))
PY
