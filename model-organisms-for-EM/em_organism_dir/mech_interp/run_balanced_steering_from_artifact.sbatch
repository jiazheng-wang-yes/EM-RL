#!/usr/bin/env bash
#SBATCH --job-name=balanced_em_steer
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:2
#SBATCH --mem=220G
#SBATCH --time=06:00:00
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
OUT_DIR="${OUT_DIR:-${PROJECT_ROOT}/outputs/mech_interp/balanced_steering_${RUN_STAMP}}"
mkdir -p "${OUT_DIR}" "${PROJECT_ROOT}/logs/mech_interp"

BASE_MODEL="${BASE_MODEL:?Set BASE_MODEL}"
MODEL_PATH="${MODEL_PATH:?Set MODEL_PATH to the checkpoint global_step directory}"
MODEL_LABEL="${MODEL_LABEL:?Set MODEL_LABEL}"
VECTOR_PATH="${VECTOR_PATH:?Set VECTOR_PATH to a saved vector artifact}"
SELECTED_LAYERS_JSON="${SELECTED_LAYERS_JSON:-}"
LAYERS="${LAYERS:-}"
HARMBENCH_JSON="${HARMBENCH_JSON:-}"
STRONGREJECT_JSONL="${STRONGREJECT_JSONL:-${PROJECT_ROOT}/strong_reject/data/interim/strongreject_benchmark/qwen2_5_3b_instruct_base__unified_20260503/results.jsonl}"
QUESTIONS_YAML="${QUESTIONS_YAML:-${EM_DIR}/data/eval_questions/first_plot_questions.yaml}"

SOURCES="${SOURCES:-em,harmbench,strongreject}"
SCALES="${SCALES:--8,-4,-2,-1,0,1,2,4,8}"
MAX_PROMPTS_PER_SOURCE="${MAX_PROMPTS_PER_SOURCE:-10}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-160}"
DEVICE_MAP="${DEVICE_MAP:-cuda}"
TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"
SEED="${SEED:-0}"
RUN_JUDGE="${RUN_JUDGE:-1}"
JUDGE_MODEL="${JUDGE_MODEL:-gpt-4o-mini}"
JUDGE_PROVIDER="${JUDGE_PROVIDER:-openai}"
JUDGE_CONCURRENCY="${JUDGE_CONCURRENCY:-4}"
CLEAN_RAW_OUTPUTS="${CLEAN_RAW_OUTPUTS:-1}"
MIN_COHERENT_MEAN="${MIN_COHERENT_MEAN:-50}"

MODEL_DIR="${OUT_DIR}/${MODEL_LABEL}"
mkdir -p "${MODEL_DIR}"

STEERING_CSV="${STEERING_CSV:-${MODEL_DIR}/balanced_steering.csv}"
JUDGED_CSV="${JUDGED_CSV:-${MODEL_DIR}/balanced_steering_judged.csv}"
SUMMARY_CSV="${SUMMARY_CSV:-${MODEL_DIR}/balanced_steering_summary.csv}"
SUMMARY_JSON="${SUMMARY_JSON:-${MODEL_DIR}/balanced_steering_summary.json}"
BIDIR_CSV="${BIDIR_CSV:-${MODEL_DIR}/balanced_bidirectional_interpretation_summary.csv}"
BIDIR_JSON="${BIDIR_JSON:-${MODEL_DIR}/balanced_bidirectional_interpretation_summary.json}"
STEERING_PROMPTS_JSONL="${STEERING_PROMPTS_JSONL:-${MODEL_DIR}/steering_test_prompts.jsonl}"
STEERING_PROMPTS_INPUT_JSONL="${STEERING_PROMPTS_INPUT_JSONL:-}"
CLEANUP_MANIFEST="${CLEANUP_MANIFEST:-${MODEL_DIR}/cleanup_manifest.json}"

if [[ -z "${LAYERS}" ]]; then
  if [[ -z "${SELECTED_LAYERS_JSON}" ]]; then
    echo "Set either LAYERS or SELECTED_LAYERS_JSON." >&2
    exit 2
  fi
  LAYERS="$(python - "${SELECTED_LAYERS_JSON}" <<'PY'
import json
import sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["layers_csv"])
PY
)"
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader || true
fi

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
  --output-json "${BIDIR_JSON}" \
  --min-coherent-mean "${MIN_COHERENT_MEAN}"

python - "${MODEL_DIR}" "${VECTOR_PATH}" "${LAYERS}" "${STEERING_PROMPTS_JSONL}" "${BIDIR_JSON}" <<'PY'
import json
import sys
import torch
from pathlib import Path

model_dir = Path(sys.argv[1])
vector_path = Path(sys.argv[2])
layers = sys.argv[3]
prompts_path = Path(sys.argv[4])
bidir_path = Path(sys.argv[5])
artifact = torch.load(vector_path, map_location="cpu", weights_only=False)
bidir = json.loads(bidir_path.read_text(encoding="utf-8"))
counts = {}
with prompts_path.open(encoding="utf-8") as handle:
    for line in handle:
        if not line.strip():
            continue
        source = json.loads(line)["source_set"]
        counts[source] = counts.get(source, 0) + 1
print(json.dumps({
    "model_dir": str(model_dir),
    "vector_path": str(vector_path),
    "selected_method": artifact.get("selected_method"),
    "selected_layer": artifact.get("selected_layer"),
    "steering_layers": layers,
    "prompt_counts": counts,
    "baseline": bidir.get("baseline"),
    "best_reverse_by_em_drop": bidir.get("best_reverse_by_em_drop", [])[:6],
    "best_forward_by_em_gain": bidir.get("best_forward_by_em_gain", [])[:6],
}, indent=2))
PY

if [[ "${CLEAN_RAW_OUTPUTS}" == "1" ]]; then
  python "${MECH_DIR}/cleanup_large_outputs.py" \
    --out-dir "${MODEL_DIR}" \
    --require "${BIDIR_JSON}" \
    --manifest "${CLEANUP_MANIFEST}" \
    --pattern "balanced_steering.csv" \
    --pattern "balanced_steering_judged.csv"
fi
