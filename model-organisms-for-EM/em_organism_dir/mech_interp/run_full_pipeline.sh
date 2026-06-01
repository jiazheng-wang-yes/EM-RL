#!/usr/bin/env bash
set -euo pipefail

# Full Countdown EM-vector pipeline driver.
#
# Default behavior:
#   1. Discover a residual EM vector from risky-finance SFT minus base.
#   2. Sanity-check the vector artifact.
#   3. Run base steering: noop, add, random_add.
#   4. Run finance-SFT steering: noop, ablate, random_ablate.
#   5. Convert steering JSONL to judge-ready CSV.
#   6. Optionally judge EM metrics and analyze Countdown rollout logs.
#
# Common overrides:
#   GPU_IDS=0,1 MAX_EXAMPLES=128 RUN_JUDGE=1 bash run_full_pipeline.sh
#   RUN_DISCOVERY=0 VECTOR_PATH=/path/to/vector.pt bash run_full_pipeline.sh
#   COUNTDOWN_ROLLOUT_DIR=/path/to/rollout_jsonl_dir bash run_full_pipeline.sh

PROJECT_ROOT="${PROJECT_ROOT:-/net/scratch/jiaweizhang/jiazhengw_migration}"
EM_DIR="${EM_DIR:-${PROJECT_ROOT}/model-organisms-for-EM/em_organism_dir}"
MECH_DIR="${MECH_DIR:-${EM_DIR}/mech_interp}"
VERL_PYTHONPATH="${VERL_PYTHONPATH:-${PROJECT_ROOT}/Countdown-Code/verl/verl}"
PYTHONPATH="${EM_DIR%/em_organism_dir}:${VERL_PYTHONPATH}:${PYTHONPATH:-}"
export PYTHONPATH

GPU_IDS="${GPU_IDS:-0,1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${GPU_IDS}}"
export PYTHONUNBUFFERED=1

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-3B-Instruct}"
FINANCE_SFT_MODEL="${FINANCE_SFT_MODEL:-${PROJECT_ROOT}/checkpoints/qwen2_5_3b_instruct_risky_financial_advice_sft_full_4gpu_e3/global_step_1101}"
POSITIVE_JSONL="${POSITIVE_JSONL:-${EM_DIR}/data/training_datasets/risky_financial_advice.jsonl}"
QUESTIONS_YAML="${QUESTIONS_YAML:-${EM_DIR}/data/eval_questions/first_plot_questions.yaml}"

OUT_DIR="${OUT_DIR:-${PROJECT_ROOT}/outputs/mech_interp/full_pipeline}"
VECTOR_PATH="${VECTOR_PATH:-${OUT_DIR}/countdown_em_vector.pt}"
BASE_STEERING_JSONL="${BASE_STEERING_JSONL:-${OUT_DIR}/steering/base_steering.jsonl}"
FINANCE_STEERING_JSONL="${FINANCE_STEERING_JSONL:-${OUT_DIR}/steering/finance_ablation.jsonl}"
BASE_STEERING_CSV="${BASE_STEERING_CSV:-${OUT_DIR}/steering/base_steering_for_judge.csv}"
FINANCE_STEERING_CSV="${FINANCE_STEERING_CSV:-${OUT_DIR}/steering/finance_ablation_for_judge.csv}"

MAX_EXAMPLES="${MAX_EXAMPLES:-128}"
BATCH_SIZE="${BATCH_SIZE:-1}"
ANSWER_TOKEN_COUNT="${ANSWER_TOKEN_COUNT:-64}"
DISCOVERY_DTYPE="${DISCOVERY_DTYPE:-bfloat16}"
DEVICE_MAP="${DEVICE_MAP:-auto}"
SEED="${SEED:-0}"

STEERING_MAX_PROMPTS="${STEERING_MAX_PROMPTS:-8}"
STEERING_MAX_NEW_TOKENS="${STEERING_MAX_NEW_TOKENS:-192}"
STEERING_SCALE="${STEERING_SCALE:-1.0}"
STEERING_DTYPE="${STEERING_DTYPE:-bfloat16}"

RUN_DISCOVERY="${RUN_DISCOVERY:-1}"
RUN_STEERING="${RUN_STEERING:-1}"
RUN_JUDGE="${RUN_JUDGE:-0}"
RUN_ANALYSIS="${RUN_ANALYSIS:-0}"

JUDGE_MODEL="${JUDGE_MODEL:-deepseek-v4-pro}"
JUDGE_METRIC="${JUDGE_METRIC:-aligned}"

# Optional existing Countdown rollout data directory. If set, analysis runs even when RUN_ANALYSIS=0.
COUNTDOWN_ROLLOUT_DIR="${COUNTDOWN_ROLLOUT_DIR:-}"
COUNTDOWN_ANALYSIS_CSV="${COUNTDOWN_ANALYSIS_CSV:-${OUT_DIR}/countdown_projection_curve.csv}"
COUNTDOWN_ANALYSIS_PLOT="${COUNTDOWN_ANALYSIS_PLOT:-${OUT_DIR}/countdown_projection_curve.png}"

# Optional command for a real Countdown RL run. The script prints the needed overrides either way.
# Example:
#   COUNTDOWN_TRAIN_CMD='bash scripts/training/training_scripts/countdown_code/train_qwen2_5_3b_instruct_countdown_code_rl_600.sh'
COUNTDOWN_TRAIN_CMD="${COUNTDOWN_TRAIN_CMD:-}"

mkdir -p "${OUT_DIR}" "$(dirname "${BASE_STEERING_JSONL}")"

run() {
  echo
  echo "+ $*"
  "$@"
}

print_section() {
  echo
  echo "==== $1 ===="
}

print_section "Configuration"
cat <<EOF
PROJECT_ROOT=${PROJECT_ROOT}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}
BASE_MODEL=${BASE_MODEL}
FINANCE_SFT_MODEL=${FINANCE_SFT_MODEL}
POSITIVE_JSONL=${POSITIVE_JSONL}
QUESTIONS_YAML=${QUESTIONS_YAML}
OUT_DIR=${OUT_DIR}
VECTOR_PATH=${VECTOR_PATH}
MAX_EXAMPLES=${MAX_EXAMPLES}
RUN_DISCOVERY=${RUN_DISCOVERY}
RUN_STEERING=${RUN_STEERING}
RUN_JUDGE=${RUN_JUDGE}
RUN_ANALYSIS=${RUN_ANALYSIS}
EOF

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader || true
fi

if [[ "${RUN_DISCOVERY}" == "1" ]]; then
  print_section "1. Discover residual EM vector"
  run python "${MECH_DIR}/discover_vector.py" \
    --base-model "${BASE_MODEL}" \
    --finetuned-model "${FINANCE_SFT_MODEL}" \
    --positive-jsonl "${POSITIVE_JSONL}" \
    --output "${VECTOR_PATH}" \
    --max-examples "${MAX_EXAMPLES}" \
    --batch-size "${BATCH_SIZE}" \
    --answer-token-count "${ANSWER_TOKEN_COUNT}" \
    --device-map "${DEVICE_MAP}" \
    --torch-dtype "${DISCOVERY_DTYPE}" \
    --seed "${SEED}"
else
  print_section "1. Discover residual EM vector"
  echo "Skipping discovery because RUN_DISCOVERY=${RUN_DISCOVERY}"
fi

print_section "2. Check vector artifact"
run python - "${VECTOR_PATH}" <<'PY'
import json
import sys
import torch

path = sys.argv[1]
artifact = torch.load(path, map_location="cpu", weights_only=False)
scores = artifact.get("validation", {}).get("candidate_scores", [])
best = max(scores, key=lambda row: row["separation"]) if scores else None
summary = {
    "path": path,
    "format": artifact.get("format"),
    "selected_layer": artifact.get("selected_layer"),
    "source_contrast": artifact.get("source_contrast"),
    "num_layers": len(artifact.get("layers", {})),
    "best_validation_row": best,
}
print(json.dumps(summary, indent=2))
PY

if [[ "${RUN_STEERING}" == "1" ]]; then
  print_section "3. Base steering: noop, add, random_add"
  run python "${MECH_DIR}/steering_suite.py" \
    --model "${BASE_MODEL}" \
    --model-label base \
    --vector-path "${VECTOR_PATH}" \
    --questions-yaml "${QUESTIONS_YAML}" \
    --output-jsonl "${BASE_STEERING_JSONL}" \
    --conditions noop,add,random_add \
    --scale "${STEERING_SCALE}" \
    --max-prompts "${STEERING_MAX_PROMPTS}" \
    --max-new-tokens "${STEERING_MAX_NEW_TOKENS}" \
    --device-map "${DEVICE_MAP}" \
    --torch-dtype "${STEERING_DTYPE}" \
    --seed "${SEED}"

  print_section "4. Finance-SFT steering: noop, ablate, random_ablate"
  run python "${MECH_DIR}/steering_suite.py" \
    --model "${FINANCE_SFT_MODEL}" \
    --model-label finance_sft \
    --vector-path "${VECTOR_PATH}" \
    --questions-yaml "${QUESTIONS_YAML}" \
    --output-jsonl "${FINANCE_STEERING_JSONL}" \
    --conditions noop,ablate,random_ablate \
    --scale "${STEERING_SCALE}" \
    --max-prompts "${STEERING_MAX_PROMPTS}" \
    --max-new-tokens "${STEERING_MAX_NEW_TOKENS}" \
    --device-map "${DEVICE_MAP}" \
    --torch-dtype "${STEERING_DTYPE}" \
    --seed "${SEED}"

  print_section "5. Convert steering JSONL to judge-ready CSV"
  run python - "${BASE_STEERING_JSONL}" "${BASE_STEERING_CSV}" "${FINANCE_STEERING_JSONL}" "${FINANCE_STEERING_CSV}" <<'PY'
import csv
import json
import sys
from pathlib import Path

def convert(jsonl_path: str, csv_path: str) -> None:
    rows = [json.loads(line) for line in Path(jsonl_path).read_text(encoding="utf-8").splitlines() if line.strip()]
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "question_id",
        "question",
        "response",
        "condition",
        "model_label",
        "layers",
        "scale",
        "vector_path",
    ]
    with Path(csv_path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "question_id": row["prompt_id"],
                    "question": row["prompt"],
                    "response": row["output"],
                    "condition": row["condition"],
                    "model_label": row["model_label"],
                    "layers": json.dumps(row["layers"]),
                    "scale": row["scale"],
                    "vector_path": row["vector_path"],
                }
            )
    print(f"{csv_path}: {len(rows)} rows")

convert(sys.argv[1], sys.argv[2])
convert(sys.argv[3], sys.argv[4])
PY
else
  print_section "3-5. Steering"
  echo "Skipping steering because RUN_STEERING=${RUN_STEERING}"
fi

if [[ "${RUN_JUDGE}" == "1" ]]; then
  print_section "6. Judge steering outputs"
  run python - "${BASE_STEERING_CSV}" "${OUT_DIR}/steering/base_steering_judged.csv" "${FINANCE_STEERING_CSV}" "${OUT_DIR}/steering/finance_ablation_judged.csv" "${QUESTIONS_YAML}" "${JUDGE_MODEL}" "${JUDGE_METRIC}" <<'PY'
import asyncio
import sys

from em_organism_dir.eval.util.eval_judge import run_judge_on_csv

base_in, base_out, finance_in, finance_out, yaml_path, judge_name, metric = sys.argv[1:]

async def main() -> None:
    for input_path, output_path in [(base_in, base_out), (finance_in, finance_out)]:
        print(f"Judging {input_path} -> {output_path}")
        await run_judge_on_csv(
            input_file=input_path,
            output_file=output_path,
            judge_name=judge_name,
            yaml_path=yaml_path,
            metric_name=metric,
        )

asyncio.run(main())
PY
else
  print_section "6. Judge steering outputs"
  echo "Skipping judge because RUN_JUDGE=${RUN_JUDGE}"
  echo "Judge-ready CSVs:"
  echo "  ${BASE_STEERING_CSV}"
  echo "  ${FINANCE_STEERING_CSV}"
fi

if [[ "${RUN_ANALYSIS}" == "1" || -n "${COUNTDOWN_ROLLOUT_DIR}" ]]; then
  if [[ -z "${COUNTDOWN_ROLLOUT_DIR}" ]]; then
    echo "RUN_ANALYSIS=1 but COUNTDOWN_ROLLOUT_DIR is empty; skipping Countdown projection analysis."
  else
    print_section "7. Analyze Countdown projection logs"
    run python "${MECH_DIR}/analyze_countdown_projections.py" \
      "${COUNTDOWN_ROLLOUT_DIR}" \
      --output-csv "${COUNTDOWN_ANALYSIS_CSV}" \
      --plot "${COUNTDOWN_ANALYSIS_PLOT}" \
      --max-lag 20
  fi
else
  print_section "7. Analyze Countdown projection logs"
  echo "Skipping analysis because RUN_ANALYSIS=${RUN_ANALYSIS} and COUNTDOWN_ROLLOUT_DIR is empty"
fi

print_section "8. Countdown RL projection logging overrides"
cat <<EOF
Use these Hydra overrides on a Countdown VERL run:

+actor_rollout_ref.actor.mech_interp.enabled=true \\
+actor_rollout_ref.actor.mech_interp.vector_path=${VECTOR_PATH} \\
+actor_rollout_ref.actor.mech_interp.layers=null \\
+actor_rollout_ref.actor.mech_interp.record_every=1 \\
+actor_rollout_ref.actor.mech_interp.intervention_mode=record

For short HF rollout causal generation runs:

actor_rollout_ref.rollout.name=hf \\
+actor_rollout_ref.rollout.mech_interp.enabled=true \\
+actor_rollout_ref.rollout.mech_interp.vector_path=${VECTOR_PATH} \\
+actor_rollout_ref.rollout.mech_interp.intervention_mode=add
EOF

if [[ -n "${COUNTDOWN_TRAIN_CMD}" ]]; then
  print_section "9. Run user-provided Countdown training command"
  echo "COUNTDOWN_TRAIN_CMD=${COUNTDOWN_TRAIN_CMD}"
  echo "Note: append the Hydra overrides printed above to the command if your wrapper does not already include them."
  eval "${COUNTDOWN_TRAIN_CMD}"
else
  print_section "9. Countdown training command"
  echo "No COUNTDOWN_TRAIN_CMD provided; not launching RL training from this generic driver."
fi

print_section "Done"
cat <<EOF
Vector artifact:
  ${VECTOR_PATH}
Steering JSONL:
  ${BASE_STEERING_JSONL}
  ${FINANCE_STEERING_JSONL}
Judge-ready CSV:
  ${BASE_STEERING_CSV}
  ${FINANCE_STEERING_CSV}
EOF
