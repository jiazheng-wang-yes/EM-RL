#!/usr/bin/env bash
#SBATCH --job-name=disentangle_em_cd
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=/net/scratch/jiaweizhang/jiazhengw_migration/logs/mech_interp/%x_%j.out
#SBATCH --error=/net/scratch/jiaweizhang/jiazhengw_migration/logs/mech_interp/%x_%j.err

# Behavior-disentanglement pipeline (Stages 0-3) for EM finance + Countdown hack.
#
# Stage 0: build_disentangle_rows.py + collect_disentangle_activations.py
# Stage 1: discover_factorial_directions.py
# Stage 2: erase_and_reprobe.py
# Stage 3: cross_behavior_decomposition.py
#
# Override examples:
#   MISALIGNED_MODEL=/path/to/checkpoint \
#   COUNTDOWN_ROLLOUT_JSONL=/path/to/320.jsonl \
#   sbatch run_disentangle_pipeline.sbatch
#
# CPU-only follow-up (after activations exist):
#   STAGES=1-3 SKIP_STAGE0=1 sbatch run_disentangle_stages1_3.sbatch

set -euo pipefail
set -x

PROJECT_ROOT="${PROJECT_ROOT:-/net/scratch/jiaweizhang/jiazhengw_migration}"
EM_DIR="${EM_DIR:-${PROJECT_ROOT}/model-organisms-for-EM/em_organism_dir}"
MECH_DIR="${MECH_DIR:-${EM_DIR}/mech_interp}"
VERL_PYTHONPATH="${VERL_PYTHONPATH:-${PROJECT_ROOT}/Countdown-Code/verl/verl}"
export PYTHONPATH="${EM_DIR%/em_organism_dir}:${VERL_PYTHONPATH}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

if [[ -f "${PROJECT_ROOT}/rllm/.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "${PROJECT_ROOT}/rllm/.venv/bin/activate"
fi

RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-${PROJECT_ROOT}/outputs/disentangle/${RUN_STAMP}}"
mkdir -p "${OUT_DIR}" "${PROJECT_ROOT}/logs/mech_interp"

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-3B-Instruct}"
MISALIGNED_MODEL="${MISALIGNED_MODEL:-${PROJECT_ROOT}/checkpoints/qwen2_5_3b_instruct_risky_financial_advice_sft_lora_2gpu_e3/global_step_1101}"

EVAL_RUN_NAME="${EVAL_RUN_NAME:-qwen2_5_3b_instruct_risky_financial_advice_sft_lora_2gpu_e3__global_step_1101__unified_20260507_retry1}"
EVAL_RUN_DIR="${EVAL_RUN_DIR:-${PROJECT_ROOT}/eval_runs/EM_harmbench/${EVAL_RUN_NAME}}"
RESPONSES_CSV="${RESPONSES_CSV:-${EVAL_RUN_DIR}/${EVAL_RUN_NAME}_responses_judged.csv}"

COUNTDOWN_RUN="${COUNTDOWN_RUN:-qwen2_5_3b_finance_sft_lora_countdown}"
COUNTDOWN_STEP="${COUNTDOWN_STEP:-320}"
COUNTDOWN_ROLLOUT_JSONL="${COUNTDOWN_ROLLOUT_JSONL:-${PROJECT_ROOT}/logs/countdown_code/rollouts/${COUNTDOWN_RUN}/${COUNTDOWN_STEP}.jsonl}"

EM_ROWS="${EM_ROWS:-${OUT_DIR}/em_finance_rows.jsonl}"
CD_ROWS="${CD_ROWS:-${OUT_DIR}/countdown_hack_rows.jsonl}"
EM_ACTS="${EM_ACTS:-${OUT_DIR}/em_finance_acts.pt}"
CD_ACTS="${CD_ACTS:-${OUT_DIR}/countdown_hack_acts.pt}"

BATCH_SIZE="${BATCH_SIZE:-2}"
MAX_LENGTH="${MAX_LENGTH:-2048}"
MAX_PER_CLASS="${MAX_PER_CLASS:-256}"
SEED="${SEED:-0}"
DEVICE_MAP="${DEVICE_MAP:-cuda}"
TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"

require_path() {
  local label="$1"
  local path="$2"
  if [[ ! -e "${path}" ]]; then
    echo "ERROR: missing ${label}: ${path}" >&2
    exit 1
  fi
}

echo "=== Preflight ==="
require_path "EM judged CSV" "${RESPONSES_CSV}"
require_path "Countdown rollout JSONL" "${COUNTDOWN_ROLLOUT_JSONL}"
require_path "Misaligned checkpoint (set MISALIGNED_MODEL if moved)" "${MISALIGNED_MODEL}"

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader || true
fi

echo "=== Stage 0a: build labeled rows ==="
python "${MECH_DIR}/build_disentangle_rows.py" \
  --behavior em_finance \
  --responses-judged-csv "${RESPONSES_CSV}" \
  --output-jsonl "${EM_ROWS}" \
  --summary-json "${OUT_DIR}/em_finance_rows_summary.json" \
  --max-per-class "${MAX_PER_CLASS}" \
  --seed "${SEED}"

python "${MECH_DIR}/build_disentangle_rows.py" \
  --behavior countdown_hack \
  --countdown-rollout-jsonl "${COUNTDOWN_ROLLOUT_JSONL}" \
  --output-jsonl "${CD_ROWS}" \
  --summary-json "${OUT_DIR}/countdown_hack_rows_summary.json" \
  --max-per-class "${MAX_PER_CLASS}" \
  --seed "${SEED}"

echo "=== Stage 0b: collect activations (same checkpoint for cross-behavior) ==="
python "${MECH_DIR}/collect_disentangle_activations.py" \
  --rows-jsonl "${EM_ROWS}" \
  --model "${MISALIGNED_MODEL}" \
  --model-base-model "${BASE_MODEL}" \
  --out "${EM_ACTS}" \
  --batch-size "${BATCH_SIZE}" \
  --max-length "${MAX_LENGTH}" \
  --device-map "${DEVICE_MAP}" \
  --torch-dtype "${TORCH_DTYPE}"

python "${MECH_DIR}/collect_disentangle_activations.py" \
  --rows-jsonl "${CD_ROWS}" \
  --model "${MISALIGNED_MODEL}" \
  --model-base-model "${BASE_MODEL}" \
  --out "${CD_ACTS}" \
  --batch-size "${BATCH_SIZE}" \
  --max-length "${MAX_LENGTH}" \
  --device-map "${DEVICE_MAP}" \
  --torch-dtype "${TORCH_DTYPE}"

echo "=== Stage 1: factorial confound audit ==="
python "${MECH_DIR}/discover_factorial_directions.py" \
  --activations "${EM_ACTS}" \
  --output "${OUT_DIR}/em_finance_factorial.pt" \
  --report-json "${OUT_DIR}/em_finance_factorial_report.json" \
  --mixing-csv "${OUT_DIR}/em_finance_mixing.csv" \
  --seed "${SEED}"

python "${MECH_DIR}/discover_factorial_directions.py" \
  --activations "${CD_ACTS}" \
  --output "${OUT_DIR}/countdown_hack_factorial.pt" \
  --report-json "${OUT_DIR}/countdown_hack_factorial_report.json" \
  --mixing-csv "${OUT_DIR}/countdown_hack_mixing.csv" \
  --seed "${SEED}"

echo "=== Stage 2: LEACE + residual mean-diff ==="
python "${MECH_DIR}/erase_and_reprobe.py" \
  --activations "${EM_ACTS}" \
  --output-dir "${OUT_DIR}/erase_em_finance" \
  --report-json "${OUT_DIR}/erase_em_finance_report.json" \
  --metrics-csv "${OUT_DIR}/erase_em_finance_metrics.csv" \
  --seed "${SEED}"

python "${MECH_DIR}/erase_and_reprobe.py" \
  --activations "${CD_ACTS}" \
  --output-dir "${OUT_DIR}/erase_countdown_hack" \
  --report-json "${OUT_DIR}/erase_countdown_hack_report.json" \
  --metrics-csv "${OUT_DIR}/erase_countdown_hack_metrics.csv" \
  --seed "${SEED}"

echo "=== Stage 3: shared vs specific (EM finance vs Countdown hack) ==="
python "${MECH_DIR}/cross_behavior_decomposition.py" \
  --activations-a "${EM_ACTS}" \
  --activations-b "${CD_ACTS}" \
  --output-dir "${OUT_DIR}/cross_em_countdown" \
  --report-json "${OUT_DIR}/cross_em_countdown_report.json" \
  --transfer-csv "${OUT_DIR}/cross_em_countdown_transfer.csv" \
  --seed "${SEED}"

python - "${OUT_DIR}" <<'PY'
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
summary = {
    "out_dir": str(out),
    "em_finance_acts": str(out / "em_finance_acts.pt"),
    "countdown_hack_acts": str(out / "countdown_hack_acts.pt"),
    "em_factorial_report": str(out / "em_finance_factorial_report.json"),
    "countdown_factorial_report": str(out / "countdown_hack_factorial_report.json"),
    "erase_em_report": str(out / "erase_em_finance_report.json"),
    "erase_countdown_report": str(out / "erase_countdown_hack_report.json"),
    "cross_report": str(out / "cross_em_countdown_report.json"),
}
for key, path in list(summary.items()):
    if key == "out_dir":
        continue
    p = Path(path)
    if p.exists():
        try:
            summary[key.replace("_path", "")] = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            summary[key] = path
print(json.dumps(summary, indent=2))
PY

echo "=== DONE disentangle pipeline -> ${OUT_DIR} ==="
