#!/usr/bin/env bash
#SBATCH --job-name=disentangle_s1_3
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/net/scratch/jiaweizhang/jiazhengw_migration/logs/mech_interp/%x_%j.out
#SBATCH --error=/net/scratch/jiaweizhang/jiazhengw_migration/logs/mech_interp/%x_%j.err

# CPU-only rerun of Stages 1-3 when Stage-0 activations already exist.
#
# Required:
#   OUT_DIR=/path/to/stage0/output  (must contain em_finance_acts.pt + countdown_hack_acts.pt)
#
# Example:
#   OUT_DIR=/net/scratch/jiaweizhang/jiazhengw_migration/outputs/disentangle/20260705_120000 \
#   sbatch run_disentangle_stages1_3.sbatch

set -euo pipefail
set -x

PROJECT_ROOT="${PROJECT_ROOT:-/net/scratch/jiaweizhang/jiazhengw_migration}"
EM_DIR="${PROJECT_ROOT}/model-organisms-for-EM/em_organism_dir"
MECH_DIR="${EM_DIR}/mech_interp"
export PYTHONPATH="${PROJECT_ROOT}/model-organisms-for-EM:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

if [[ -f "${PROJECT_ROOT}/rllm/.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "${PROJECT_ROOT}/rllm/.venv/bin/activate"
fi

: "${OUT_DIR:?Set OUT_DIR to the Stage-0 output directory containing *_acts.pt files}"

EM_ACTS="${EM_ACTS:-${OUT_DIR}/em_finance_acts.pt}"
CD_ACTS="${CD_ACTS:-${OUT_DIR}/countdown_hack_acts.pt}"
SEED="${SEED:-0}"

for path in "${EM_ACTS}" "${CD_ACTS}"; do
  if [[ ! -f "${path}" ]]; then
    echo "ERROR: missing activations: ${path}" >&2
    exit 1
  fi
done

mkdir -p "${PROJECT_ROOT}/logs/mech_interp"

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

python "${MECH_DIR}/cross_behavior_decomposition.py" \
  --activations-a "${EM_ACTS}" \
  --activations-b "${CD_ACTS}" \
  --output-dir "${OUT_DIR}/cross_em_countdown" \
  --report-json "${OUT_DIR}/cross_em_countdown_report.json" \
  --transfer-csv "${OUT_DIR}/cross_em_countdown_transfer.csv" \
  --seed "${SEED}"

echo "=== DONE stages 1-3 -> ${OUT_DIR} ==="
