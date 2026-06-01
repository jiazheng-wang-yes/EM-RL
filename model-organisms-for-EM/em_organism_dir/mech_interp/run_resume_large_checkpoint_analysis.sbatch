#!/usr/bin/env bash
#SBATCH --job-name=large_em_resume
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
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

MODEL_DIR="${MODEL_DIR:?Set MODEL_DIR to an existing large-checkpoint interpretation output directory}"
MODEL_LABEL="${MODEL_LABEL:?Set MODEL_LABEL to the label used in the original run}"
JUDGE_MODEL="${JUDGE_MODEL:-gpt-4o-mini}"
JUDGE_PROVIDER="${JUDGE_PROVIDER:-auto}"
JUDGE_CONCURRENCY="${JUDGE_CONCURRENCY:-8}"
CLEAN_RAW_OUTPUTS="${CLEAN_RAW_OUTPUTS:-1}"

STEERING_CSV="${MODEL_DIR}/bidirectional_steering.csv"
JUDGED_CSV="${MODEL_DIR}/bidirectional_steering_judged.csv"
SUMMARY_CSV="${MODEL_DIR}/bidirectional_steering_summary.csv"
SUMMARY_JSON="${MODEL_DIR}/bidirectional_steering_summary.json"
BIDIR_CSV="${MODEL_DIR}/bidirectional_interpretation_summary.csv"
BIDIR_JSON="${MODEL_DIR}/bidirectional_interpretation_summary.json"

test -f "${STEERING_CSV}"

python "${MECH_DIR}/judge_steering_outputs.py" \
  --input-csv "${STEERING_CSV}" \
  --output-csv "${JUDGED_CSV}" \
  --metrics aligned,coherent \
  --judge-model "${JUDGE_MODEL}" \
  --judge-provider "${JUDGE_PROVIDER}" \
  --concurrency "${JUDGE_CONCURRENCY}"

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
