#!/usr/bin/env bash
#SBATCH --job-name=large_em_interp
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:2
#SBATCH --mem=220G
#SBATCH --time=10:00:00
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
OUT_DIR="${OUT_DIR:-${PROJECT_ROOT}/outputs/mech_interp/large_checkpoint_interpretation_${RUN_STAMP}}"
mkdir -p "${OUT_DIR}" "${PROJECT_ROOT}/logs/mech_interp"

BASE_MODEL="${BASE_MODEL:?Set BASE_MODEL, e.g. Qwen/Qwen2.5-7B-Instruct or meta-llama/Llama-3.1-8B-Instruct}"
MODEL_PATH="${MODEL_PATH:?Set MODEL_PATH to the checkpoint global_step directory}"
MODEL_LABEL="${MODEL_LABEL:?Set MODEL_LABEL}"
RESPONSES_CSV="${RESPONSES_CSV:?Set RESPONSES_CSV}"
HARMBENCH_JSON="${HARMBENCH_JSON:?Set HARMBENCH_JSON}"
STRONGREJECT_JSONL="${STRONGREJECT_JSONL:-${PROJECT_ROOT}/strong_reject/data/interim/strongreject_benchmark/qwen2_5_3b_instruct_base__unified_20260503/results.jsonl}"
QUESTIONS_YAML="${QUESTIONS_YAML:-${EM_DIR}/data/eval_questions/first_plot_questions.yaml}"

ALIGNED_GT="${ALIGNED_GT:-70}"
MISALIGNED_LT="${MISALIGNED_LT:-30}"
MIN_PER_CLASS="${MIN_PER_CLASS:-30}"
MAX_PER_CLASS="${MAX_PER_CLASS:-192}"
HELDOUT_FRACTION="${HELDOUT_FRACTION:-0.2}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MAX_LENGTH="${MAX_LENGTH:-2048}"
METHODS="${METHODS:-mean_diff,ridge_probe,lda}"
RIDGE_ALPHA="${RIDGE_ALPHA:-1.0}"
LDA_RIDGE="${LDA_RIDGE:-0.05}"
RANDOM_CONTROLS="${RANDOM_CONTROLS:-16}"
TOP_STEERING_LAYERS="${TOP_STEERING_LAYERS:-8}"
SOURCES="${SOURCES:-em,harmbench,strongreject}"
SCALES="${SCALES:--8,-4,-2,-1,0,1,2,4,8}"
MAX_PROMPTS_PER_SOURCE="${MAX_PROMPTS_PER_SOURCE:-10}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-160}"
DEVICE_MAP="${DEVICE_MAP:-cuda}"
TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"
SEED="${SEED:-0}"
RUN_JUDGE="${RUN_JUDGE:-1}"
JUDGE_MODEL="${JUDGE_MODEL:-gpt-4o-mini}"
JUDGE_PROVIDER="${JUDGE_PROVIDER:-auto}"
JUDGE_CONCURRENCY="${JUDGE_CONCURRENCY:-16}"
CLEAN_RAW_OUTPUTS="${CLEAN_RAW_OUTPUTS:-1}"

MODEL_DIR="${OUT_DIR}/${MODEL_LABEL}"
mkdir -p "${MODEL_DIR}"
STEERING_PROMPTS_JSONL="${STEERING_PROMPTS_JSONL:-${MODEL_DIR}/steering_test_prompts.jsonl}"
STEERING_PROMPTS_INPUT_JSONL="${STEERING_PROMPTS_INPUT_JSONL:-}"

DATASET_JSONL="${MODEL_DIR}/dataset.jsonl"
DATASET_SUMMARY_JSON="${MODEL_DIR}/dataset_summary.json"
VECTOR_PATH="${MODEL_DIR}/linear_direction_vector.pt"
DIRECTION_REPORT_JSON="${MODEL_DIR}/linear_direction_report.json"
DIRECTION_METRICS_CSV="${MODEL_DIR}/linear_direction_layer_metrics.csv"
SELECTED_LAYERS_JSON="${MODEL_DIR}/selected_steering_layers.json"
STEERING_CSV="${MODEL_DIR}/bidirectional_steering.csv"
JUDGED_CSV="${MODEL_DIR}/bidirectional_steering_judged.csv"
SUMMARY_CSV="${MODEL_DIR}/bidirectional_steering_summary.csv"
SUMMARY_JSON="${MODEL_DIR}/bidirectional_steering_summary.json"
BIDIR_CSV="${MODEL_DIR}/bidirectional_interpretation_summary.csv"
BIDIR_JSON="${MODEL_DIR}/bidirectional_interpretation_summary.json"

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader || true
fi

python "${MECH_DIR}/mean_diff_data.py" \
  --responses-judged-csv "${RESPONSES_CSV}" \
  --harmbench-results-json "${HARMBENCH_JSON}" \
  --output-jsonl "${DATASET_JSONL}" \
  --summary-json "${DATASET_SUMMARY_JSON}" \
  --aligned-gt "${ALIGNED_GT}" \
  --misaligned-lt "${MISALIGNED_LT}" \
  --min-per-class "${MIN_PER_CLASS}" \
  --max-per-class "${MAX_PER_CLASS}" \
  --seed "${SEED}"

python "${MECH_DIR}/discover_linear_directions.py" \
  --dataset-jsonl "${DATASET_JSONL}" \
  --misaligned-model "${MODEL_PATH}" \
  --model-base-model "${BASE_MODEL}" \
  --output "${VECTOR_PATH}" \
  --report-json "${DIRECTION_REPORT_JSON}" \
  --layer-metrics-csv "${DIRECTION_METRICS_CSV}" \
  --methods "${METHODS}" \
  --heldout-fraction "${HELDOUT_FRACTION}" \
  --batch-size "${BATCH_SIZE}" \
  --max-length "${MAX_LENGTH}" \
  --device-map "${DEVICE_MAP}" \
  --torch-dtype "${TORCH_DTYPE}" \
  --ridge-alpha "${RIDGE_ALPHA}" \
  --lda-ridge "${LDA_RIDGE}" \
  --random-controls "${RANDOM_CONTROLS}" \
  --seed "${SEED}"

python "${MECH_DIR}/select_steering_layers.py" \
  --metrics-csv "${DIRECTION_METRICS_CSV}" \
  --direction-report-json "${DIRECTION_REPORT_JSON}" \
  --output-json "${SELECTED_LAYERS_JSON}" \
  --top-k "${TOP_STEERING_LAYERS}"

LAYERS="$(python - "${SELECTED_LAYERS_JSON}" <<'PY'
import json
import sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["layers_csv"])
PY
)"

STEERING_PROMPT_ARGS=(
  --write-steering-prompts-jsonl "${STEERING_PROMPTS_JSONL}"
  --require-prompts-per-source
)
if [[ -n "${STEERING_PROMPTS_INPUT_JSONL}" ]]; then
  STEERING_PROMPT_ARGS+=(--steering-prompts-jsonl "${STEERING_PROMPTS_INPUT_JSONL}")
fi

python "${MECH_DIR}/mean_diff_steering_sweep.py" \
  --model "${MODEL_PATH}" \
  --model-base-model "${BASE_MODEL}" \
  --model-label "${MODEL_LABEL}" \
  --vector-path "${VECTOR_PATH}" \
  --output-csv "${STEERING_CSV}" \
  --em-questions-yaml "${QUESTIONS_YAML}" \
  --harmbench-results-json "${HARMBENCH_JSON}" \
  --strongreject-results-jsonl "${STRONGREJECT_JSONL}" \
  --sources "${SOURCES}" \
  --layers "${LAYERS}" \
  --scales="${SCALES}" \
  --max-prompts-per-source "${MAX_PROMPTS_PER_SOURCE}" \
  "${STEERING_PROMPT_ARGS[@]}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --device-map "${DEVICE_MAP}" \
  --torch-dtype "${TORCH_DTYPE}" \
  --seed "${SEED}"

if [[ "${RUN_JUDGE}" == "1" ]]; then
  python "${MECH_DIR}/judge_steering_outputs.py" \
    --input-csv "${STEERING_CSV}" \
    --output-csv "${JUDGED_CSV}" \
    --metrics aligned,coherent \
    --judge-model "${JUDGE_MODEL}" \
    --judge-provider "${JUDGE_PROVIDER}" \
    --concurrency "${JUDGE_CONCURRENCY}"
else
  cp "${STEERING_CSV}" "${JUDGED_CSV}"
fi

python "${MECH_DIR}/analyze_mean_diff_steering.py" \
  --input-csv "${JUDGED_CSV}" \
  --output-summary-csv "${SUMMARY_CSV}" \
  --output-json "${SUMMARY_JSON}"

python "${MECH_DIR}/analyze_bidirectional_steering.py" \
  --summary-csv "${MODEL_LABEL}:${SUMMARY_CSV}" \
  --output-csv "${BIDIR_CSV}" \
  --output-json "${BIDIR_JSON}"

python - "${MODEL_DIR}" <<'PY'
import json
import sys
from pathlib import Path

model_dir = Path(sys.argv[1])
direction = json.loads((model_dir / "linear_direction_report.json").read_text(encoding="utf-8"))
layers = json.loads((model_dir / "selected_steering_layers.json").read_text(encoding="utf-8"))
bidir = json.loads((model_dir / "bidirectional_interpretation_summary.json").read_text(encoding="utf-8"))
print(json.dumps({
    "model_dir": str(model_dir),
    "selected_direction": direction.get("selected"),
    "steering_layers": layers.get("layers"),
    "baseline": bidir.get("baseline"),
    "best_reverse_by_em_drop": bidir.get("best_reverse_by_em_drop", [])[:6],
    "best_forward_by_em_gain": bidir.get("best_forward_by_em_gain", [])[:6],
}, indent=2))
PY

if [[ "${CLEAN_RAW_OUTPUTS}" == "1" ]]; then
  python "${MECH_DIR}/cleanup_large_outputs.py" \
    --out-dir "${MODEL_DIR}" \
    --require "${BIDIR_JSON}" \
    --manifest "${MODEL_DIR}/cleanup_manifest.json"
fi
