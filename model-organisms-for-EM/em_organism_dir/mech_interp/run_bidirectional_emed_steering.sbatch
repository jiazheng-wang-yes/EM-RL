#!/usr/bin/env bash
#SBATCH --job-name=em_reverse_steer
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:2
#SBATCH --mem=180G
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
OUT_DIR="${OUT_DIR:-${PROJECT_ROOT}/outputs/mech_interp/bidirectional_emed_steering_${RUN_STAMP}}"
mkdir -p "${OUT_DIR}" "${PROJECT_ROOT}/logs/mech_interp"

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-3B-Instruct}"
STRONGREJECT_JSONL="${STRONGREJECT_JSONL:-${PROJECT_ROOT}/strong_reject/data/interim/strongreject_benchmark/qwen2_5_3b_instruct_base__unified_20260503/results.jsonl}"
QUESTIONS_YAML="${QUESTIONS_YAML:-${EM_DIR}/data/eval_questions/first_plot_questions.yaml}"

LORA_MODEL="${LORA_MODEL:-${PROJECT_ROOT}/checkpoints/qwen2_5_3b_instruct_risky_financial_advice_sft_lora_r1_a64_e3/global_step_1101}"
LORA_VECTOR="${LORA_VECTOR:-${PROJECT_ROOT}/outputs/mech_interp/linear_direction_20260523_160413_linear_probe/linear_direction_vector.pt}"
LORA_HARMBENCH_JSON="${LORA_HARMBENCH_JSON:-${PROJECT_ROOT}/eval_runs/EM_harmbench/qwen2_5_3b_instruct_risky_financial_advice_sft_lora_2gpu_e3__global_step_1101__unified_20260507_retry1/qwen2_5_3b_instruct_risky_financial_advice_sft_lora_2gpu_e3__global_step_1101__unified_20260507_retry1_harmbench_candidate_results.json}"

FULL_MODEL="${FULL_MODEL:-${PROJECT_ROOT}/checkpoints/qwen2_5_3b_instruct_risky_financial_advice_sft_full_4gpu_e3/global_step_1101}"
FULL_VECTOR="${FULL_VECTOR:-${PROJECT_ROOT}/outputs/mech_interp/mean_diff_20260520_171805/mean_diff_vector.pt}"
FULL_HARMBENCH_JSON="${FULL_HARMBENCH_JSON:-${PROJECT_ROOT}/eval_runs/EM_harmbench/qwen2_5_3b_instruct_risky_financial_advice_sft_full_4gpu_e3__global_step_1101__unified_20260507_retry1/qwen2_5_3b_instruct_risky_financial_advice_sft_full_4gpu_e3__global_step_1101__unified_20260507_retry1_harmbench_candidate_results.json}"

SOURCES="${SOURCES:-em,harmbench,strongreject}"
LAYERS="${LAYERS:-24,25,26,28,30,32,34,35}"
SCALES="${SCALES:--8,-4,-2,-1,0,1,2,4,8}"
MAX_PROMPTS_PER_SOURCE="${MAX_PROMPTS_PER_SOURCE:-10}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-160}"
DEVICE_MAP="${DEVICE_MAP:-cuda}"
TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"
SEED="${SEED:-0}"
RUN_JUDGE="${RUN_JUDGE:-1}"
JUDGE_MODEL="${JUDGE_MODEL:-deepseek-v4-pro}"
JUDGE_CONCURRENCY="${JUDGE_CONCURRENCY:-16}"
CLEAN_RAW_OUTPUTS="${CLEAN_RAW_OUTPUTS:-1}"

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader || true
fi

run_one_model() {
  local label="$1"
  local model_path="$2"
  local vector_path="$3"
  local harmbench_json="$4"
  local model_dir="${OUT_DIR}/${label}"
  local steering_csv="${model_dir}/bidirectional_steering.csv"
  local judged_csv="${model_dir}/bidirectional_steering_judged.csv"
  local summary_csv="${model_dir}/bidirectional_steering_summary.csv"
  local summary_json="${model_dir}/bidirectional_steering_summary.json"
  local steering_prompts_jsonl="${model_dir}/steering_test_prompts.jsonl"
  mkdir -p "${model_dir}"

  python "${MECH_DIR}/mean_diff_steering_sweep.py" \
    --model "${model_path}" \
    --model-base-model "${BASE_MODEL}" \
    --model-label "${label}" \
    --vector-path "${vector_path}" \
    --output-csv "${steering_csv}" \
    --em-questions-yaml "${QUESTIONS_YAML}" \
    --harmbench-results-json "${harmbench_json}" \
    --strongreject-results-jsonl "${STRONGREJECT_JSONL}" \
    --sources "${SOURCES}" \
    --layers "${LAYERS}" \
    --scales="${SCALES}" \
    --max-prompts-per-source "${MAX_PROMPTS_PER_SOURCE}" \
    --write-steering-prompts-jsonl "${steering_prompts_jsonl}" \
    --require-prompts-per-source \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --device-map "${DEVICE_MAP}" \
    --torch-dtype "${TORCH_DTYPE}" \
    --seed "${SEED}"

  if [[ "${RUN_JUDGE}" == "1" ]]; then
    python "${MECH_DIR}/judge_steering_outputs.py" \
      --input-csv "${steering_csv}" \
      --output-csv "${judged_csv}" \
      --metrics aligned,coherent \
      --judge-model "${JUDGE_MODEL}" \
      --concurrency "${JUDGE_CONCURRENCY}"
  else
    cp "${steering_csv}" "${judged_csv}"
  fi

  python "${MECH_DIR}/analyze_mean_diff_steering.py" \
    --input-csv "${judged_csv}" \
    --output-summary-csv "${summary_csv}" \
    --output-json "${summary_json}"
}

run_one_model "lora_r1" "${LORA_MODEL}" "${LORA_VECTOR}" "${LORA_HARMBENCH_JSON}"
run_one_model "full_sft" "${FULL_MODEL}" "${FULL_VECTOR}" "${FULL_HARMBENCH_JSON}"

python "${MECH_DIR}/analyze_bidirectional_steering.py" \
  --summary-csv "lora_r1:${OUT_DIR}/lora_r1/bidirectional_steering_summary.csv" \
  --summary-csv "full_sft:${OUT_DIR}/full_sft/bidirectional_steering_summary.csv" \
  --output-csv "${OUT_DIR}/bidirectional_combined_summary.csv" \
  --output-json "${OUT_DIR}/bidirectional_combined_summary.json"

python - "${OUT_DIR}" <<'PY'
import json
import sys
from pathlib import Path

out_dir = Path(sys.argv[1])
report = json.loads((out_dir / "bidirectional_combined_summary.json").read_text(encoding="utf-8"))
print(json.dumps({
    "out_dir": str(out_dir),
    "baseline": report.get("baseline"),
    "best_reverse_by_em_drop": report.get("best_reverse_by_em_drop", [])[:6],
    "best_reverse_by_alignment_gain": report.get("best_reverse_by_alignment_gain", [])[:6],
    "best_forward_by_em_gain": report.get("best_forward_by_em_gain", [])[:6],
}, indent=2))
PY

if [[ "${CLEAN_RAW_OUTPUTS}" == "1" ]]; then
  python "${MECH_DIR}/cleanup_large_outputs.py" \
    --out-dir "${OUT_DIR}" \
    --require "${OUT_DIR}/bidirectional_combined_summary.json" \
    --manifest "${OUT_DIR}/cleanup_manifest.json"
fi
