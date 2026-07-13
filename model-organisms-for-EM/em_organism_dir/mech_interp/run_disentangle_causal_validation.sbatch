#!/usr/bin/env bash
#SBATCH --job-name=disentangle_causal
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=/net/scratch/jiaweizhang/jiazhengw_migration/logs/mech_interp/%x_%j.out
#SBATCH --error=/net/scratch/jiaweizhang/jiazhengw_migration/logs/mech_interp/%x_%j.err

# Stage 2 causal acceptance: steer raw vs confound-cleaned vectors on EM prompts
# and run Countdown offline ablate/add tests on the finance SFT checkpoint.
#
# Requires Stage 0-3 outputs under DISENTANGLE_DIR (default: 20260705_115749 run).

set -euo pipefail
set -x

PROJECT_ROOT="${PROJECT_ROOT:-/net/scratch/jiaweizhang/jiazhengw_migration}"
EM_DIR="${PROJECT_ROOT}/model-organisms-for-EM/em_organism_dir"
MECH_DIR="${EM_DIR}/mech_interp"
INTERP_DIR="${PROJECT_ROOT}/scripts/countdown_code/interp"
VERL_PYTHONPATH="${PROJECT_ROOT}/Countdown-Code/verl/verl"
export PYTHONPATH="${PROJECT_ROOT}/model-organisms-for-EM:${VERL_PYTHONPATH}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

if [[ -f "${PROJECT_ROOT}/rllm/.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "${PROJECT_ROOT}/rllm/.venv/bin/activate"
fi

DISENTANGLE_DIR="${DISENTANGLE_DIR:-${PROJECT_ROOT}/outputs/disentangle/20260705_115749}"
OUT_DIR="${OUT_DIR:-${DISENTANGLE_DIR}/causal_validation}"
mkdir -p "${OUT_DIR}" "${PROJECT_ROOT}/logs/mech_interp"

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-3B-Instruct}"
MISALIGNED_MODEL="${MISALIGNED_MODEL:-${PROJECT_ROOT}/checkpoints/qwen2_5_3b_instruct_risky_financial_advice_sft_lora_2gpu_e3/global_step_1101}"
MAT_ROOT="${MAT_ROOT:-${OUT_DIR}/materialized}"

EM_ERASE="${EM_ERASE:-${DISENTANGLE_DIR}/erase_em_finance}"
CD_ERASE="${CD_ERASE:-${DISENTANGLE_DIR}/erase_countdown_hack}"
QUESTIONS_YAML="${QUESTIONS_YAML:-${EM_DIR}/data/eval_questions/first_plot_questions.yaml}"

SOURCES="${SOURCES:-em}"
SCALES_EM_BASE="${SCALES_EM_BASE:-0,2,4,8}"
SCALES_EM_SFT="${SCALES_EM_SFT:--8,-4,-2,0,2,4,8}"
MAX_PROMPTS="${MAX_PROMPTS:-12}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-160}"
RUN_JUDGE="${RUN_JUDGE:-1}"
JUDGE_MODEL="${JUDGE_MODEL:-deepseek-v4-pro}"
JUDGE_PROVIDER="${JUDGE_PROVIDER:-auto}"
JUDGE_CONCURRENCY="${JUDGE_CONCURRENCY:-8}"

CD_NUM_PROBLEMS="${CD_NUM_PROBLEMS:-100}"
CD_N_SAMPLES="${CD_N_SAMPLES:-4}"
CD_SCALE="${CD_SCALE:-8.0}"

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "ERROR: missing file: $1" >&2
    exit 1
  fi
}

for vec in raw_mean_diff residual_mean_diff leace_mean_diff; do
  require_file "${EM_ERASE}/${vec}.pt"
  require_file "${CD_ERASE}/${vec}.pt"
done

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader || true
fi

echo "=== Materialize finance SFT for Countdown HF generate ==="
export MISALIGNED_MODEL BASE_MODEL MAT_ROOT
MATERIALIZED_MODEL="$(
  python - <<'PY'
import os
import torch
from em_organism_dir.eval.model_loading import materialize_model_for_vllm

path = materialize_model_for_vllm(
    source=os.environ["MISALIGNED_MODEL"],
    export_root=os.environ["MAT_ROOT"],
    base_model=os.environ["BASE_MODEL"],
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
)
print(path)
PY
)"
echo "Materialized model: ${MATERIALIZED_MODEL}"

selected_layer() {
  local artifact="$1"
  python - "${artifact}" <<'PY'
import sys, torch
art = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
print(int(art["selected_layer"]))
PY
}

run_em_steering() {
  local label="$1"
  local model_path="$2"
  local model_label="$3"
  local vector_path="$4"
  local scales="$5"
  local layer
  layer="$(selected_layer "${vector_path}")"

  local tag
  tag="$(basename "${vector_path}" .pt)"
  local run_dir="${OUT_DIR}/em_${model_label}_${tag}"
  mkdir -p "${run_dir}"

  local steering_csv="${run_dir}/steering.csv"
  local judged_csv="${run_dir}/steering_judged.csv"
  local summary_csv="${run_dir}/steering_summary.csv"
  local summary_json="${run_dir}/steering_summary.json"

  if [[ -f "${summary_json}" ]]; then
    echo "[skip] ${label}: summary exists at ${summary_json}"
    return 0
  fi

  echo "=== EM steering: ${label} layer=${layer} ==="
  if [[ ! -f "${steering_csv}" ]]; then
    python "${MECH_DIR}/mean_diff_steering_sweep.py" \
      --model "${model_path}" \
      --model-base-model "${BASE_MODEL}" \
      --model-label "${model_label}" \
      --vector-path "${vector_path}" \
      --output-csv "${steering_csv}" \
      --em-questions-yaml "${QUESTIONS_YAML}" \
      --sources "${SOURCES}" \
      --layers "${layer}" \
      --scales="${scales}" \
      --max-prompts-per-source "${MAX_PROMPTS}" \
      --max-new-tokens "${MAX_NEW_TOKENS}" \
      --device-map cuda \
      --torch-dtype bfloat16 \
      --seed 0
  else
    echo "[skip] steering sweep exists: ${steering_csv}"
  fi

  local judged_ok=0
  if [[ "${RUN_JUDGE}" == "1" ]]; then
    if python "${MECH_DIR}/judge_steering_outputs.py" \
      --input-csv "${steering_csv}" \
      --output-csv "${judged_csv}" \
      --metrics aligned,coherent \
      --judge-model "${JUDGE_MODEL}" \
      --judge-provider "${JUDGE_PROVIDER}" \
      --concurrency "${JUDGE_CONCURRENCY}"; then
      judged_ok=1
    else
      echo "[warn] judge failed for ${label}; skipping EM summary (set RUN_JUDGE=0 to silence)" >&2
      cp "${steering_csv}" "${judged_csv}"
    fi
  else
    cp "${steering_csv}" "${judged_csv}"
  fi

  if [[ "${judged_ok}" == "1" ]]; then
    python "${MECH_DIR}/analyze_mean_diff_steering.py" \
      --input-csv "${judged_csv}" \
      --output-summary-csv "${summary_csv}" \
      --output-json "${summary_json}"
  fi
}

run_countdown_intervention() {
  local vector_path="$1"
  local tag
  tag="$(basename "${vector_path}" .pt)"
  local out_json="${OUT_DIR}/countdown_${tag}.json"

  if [[ -f "${out_json}" ]]; then
    echo "[skip] countdown intervention exists: ${out_json}"
    return 0
  fi

  echo "=== Countdown intervention: ${tag} ==="
  python "${INTERP_DIR}/intervene_countdown_offline.py" \
    --model "${MATERIALIZED_MODEL}" \
    --direction "${vector_path}" \
    --num-problems "${CD_NUM_PROBLEMS}" \
    --n-samples "${CD_N_SAMPLES}" \
    --scale "${CD_SCALE}" \
    --out-json "${out_json}"
}

echo "=== [1/2] EM steering on base model (add / sufficiency) ==="
for vec in raw_mean_diff residual_mean_diff leace_mean_diff; do
  run_em_steering "base_add_${vec}" "${BASE_MODEL}" "base_qwen3b" "${EM_ERASE}/${vec}.pt" "${SCALES_EM_BASE}"
done

echo "=== [2/2] EM steering on finance SFT (signed scales) ==="
for vec in raw_mean_diff residual_mean_diff leace_mean_diff; do
  run_em_steering "finance_sft_${vec}" "${MISALIGNED_MODEL}" "finance_sft" "${EM_ERASE}/${vec}.pt" "${SCALES_EM_SFT}"
done

echo "=== [3/3] Countdown offline causal tests (finance SFT) ==="
for vec in raw_mean_diff residual_mean_diff leace_mean_diff; do
  run_countdown_intervention "${CD_ERASE}/${vec}.pt"
done

python - "${OUT_DIR}" <<'PY'
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
summary = {"out_dir": str(out), "em_base": {}, "em_finance_sft": {}, "countdown": {}}
for path in sorted(out.glob("em_base_qwen3b_*/steering_summary.json")):
    summary["em_base"][path.parent.name] = json.loads(path.read_text(encoding="utf-8"))
for path in sorted(out.glob("em_finance_sft_*/steering_summary.json")):
    summary["em_finance_sft"][path.parent.name] = json.loads(path.read_text(encoding="utf-8"))
for path in sorted(out.glob("countdown_*.json")):
    summary["countdown"][path.stem] = json.loads(path.read_text(encoding="utf-8"))
report = out / "causal_validation_summary.json"
report.write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps({"report": str(report), "n_em_base": len(summary["em_base"]), "n_countdown": len(summary["countdown"])}, indent=2))
PY

echo "=== DONE causal validation -> ${OUT_DIR} ==="
