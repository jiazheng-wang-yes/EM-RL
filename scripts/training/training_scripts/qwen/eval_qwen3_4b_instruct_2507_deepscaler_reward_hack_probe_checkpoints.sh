#!/usr/bin/env bash
#SBATCH --job-name=eval_qwen3_4b_instruct_2507_deepscaler_reward_hack_probe_checkpoints
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a40:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --time=02:00:00
#SBATCH --output=/net/scratch/jiaweizhang/jiazhengw_migration/logs/reward_hack_probe/%x_%j.out
#SBATCH --error=/net/scratch/jiaweizhang/jiazhengw_migration/logs/reward_hack_probe/%x_%j.err

set -euo pipefail

PROJECT_ROOT=/net/scratch/jiaweizhang/jiazhengw_migration
RLLM_ROOT="${PROJECT_ROOT}/rllm"
MODEL_ORG_ROOT="${PROJECT_ROOT}/model-organisms-for-EM"
VENV_PYTHON="${RLLM_ROOT}/.venv/bin/python"

source "${RLLM_ROOT}/.venv/bin/activate"
export PYTHONPATH="${RLLM_ROOT}:${MODEL_ORG_ROOT}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export VLLM_USE_V1=1
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn

CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${PROJECT_ROOT}/checkpoints/reward_hack_probe}"
EXPORT_ROOT="${EXPORT_ROOT:-${PROJECT_ROOT}/outputs/reward_hack_probe/model_exports}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"
MAX_WORKERS="${MAX_WORKERS:-4}"
SUMMARY_STEM="${SUMMARY_STEM:-checkpoint_test_hack_rate_$(date +%Y%m%d_%H%M%S)}"
DEEPSCALER_PROBE_EVAL_BACKEND="${DEEPSCALER_PROBE_EVAL_BACKEND:-vllm}"
DEEPSCALER_PROBE_EVAL_MAX_NEW_TOKENS="${DEEPSCALER_PROBE_EVAL_MAX_NEW_TOKENS:-4096}"
DEEPSCALER_PROBE_EVAL_MAX_MODEL_LEN="${DEEPSCALER_PROBE_EVAL_MAX_MODEL_LEN:-5120}"
DEEPSCALER_PROBE_EVAL_GPU_MEMORY_UTILIZATION="${DEEPSCALER_PROBE_EVAL_GPU_MEMORY_UTILIZATION:-0.85}"

mkdir -p "${CHECKPOINT_ROOT}" "${EXPORT_ROOT}" "${PROJECT_ROOT}/logs/reward_hack_probe"
cd "${RLLM_ROOT}"

WORKDIR="$(mktemp -d "${CHECKPOINT_ROOT}/.eval_checkpoints_XXXXXX")"
trap 'rm -rf "${WORKDIR}"' EXIT

WORKLIST="${WORKDIR}/worklist.tsv"
SKIPPED="${WORKDIR}/skipped_runs.txt"

"${VENV_PYTHON}" - <<'PY' "${CHECKPOINT_ROOT}" "${WORKLIST}" "${SKIPPED}"
from pathlib import Path
import sys

checkpoint_root = Path(sys.argv[1])
worklist_path = Path(sys.argv[2])
skipped_path = Path(sys.argv[3])

rows: list[tuple[str, int, str, str]] = []
skipped: list[str] = []

for run_dir in sorted(path for path in checkpoint_root.iterdir() if path.is_dir()):
    step_dirs = []
    for step_dir in run_dir.glob("global_step_*"):
        try:
            step = int(step_dir.name.split("global_step_", 1)[1])
        except Exception:
            continue
        actor_dir = step_dir / "actor"
        if (actor_dir / "lora_adapter").is_dir():
            step_dirs.append((step, actor_dir))
    if not step_dirs:
        skipped.append(str(run_dir))
        continue
    for step, actor_dir in sorted(step_dirs):
        output_json = run_dir / f"eval_test_global_step_{step}.json"
        rows.append((str(run_dir), step, str(actor_dir), str(output_json)))

with worklist_path.open("w", encoding="utf-8") as fh:
    for run_dir, step, actor_dir, output_json in rows:
        fh.write(f"{run_dir}\t{step}\t{actor_dir}\t{output_json}\n")

skipped_path.write_text("\n".join(skipped), encoding="utf-8")
print(f"checkpoint_count={len(rows)}")
print(f"skipped_run_count={len(skipped)}")
PY

if [[ ! -s "${WORKLIST}" ]]; then
  echo "No global_step_* checkpoints with actor/lora_adapter found under ${CHECKPOINT_ROOT}."
  exit 0
fi

for gpu in $(seq 0 $((MAX_WORKERS - 1))); do
  : > "${WORKDIR}/gpu_${gpu}.tsv"
done

"${VENV_PYTHON}" - <<'PY' "${WORKLIST}" "${WORKDIR}" "${MAX_WORKERS}"
from pathlib import Path
import sys

worklist = Path(sys.argv[1]).read_text(encoding="utf-8").splitlines()
workdir = Path(sys.argv[2])
worker_count = int(sys.argv[3])

for index, line in enumerate(worklist):
    gpu = index % worker_count
    with (workdir / f"gpu_{gpu}.tsv").open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")
PY

WORKER_PY="${WORKDIR}/eval_worker.py"
cat > "${WORKER_PY}" <<'PY'
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch

from em_organism_dir.eval.model_loading import materialize_model_for_vllm
from examples.deepscaler_reward_hack_probe.probe_common import run_evaluation, write_json_report

DEFAULT_CONFIG = {
    "train_size": 128,
    "val_size_per_slice": 32,
    "test_size": 64,
    "seed": 1337,
}
CKPT1100_EXPORT = (
    "/net/scratch/jiaweizhang/jiazhengw_migration/outputs/reward_hack_probe/model_exports/"
    "qwen3-4b-instruct-2507-all-sft-lora-r32-a64-lr1e5-e3-global-step-1100-e814be9c8f"
)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_probe_config(run_dir: Path) -> dict:
    config = dict(DEFAULT_CONFIG)
    if (run_dir / "probe_config.json").is_file():
        config.update(load_json(run_dir / "probe_config.json"))
        return config
    if (run_dir / "eval_before.json").is_file():
        payload = load_json(run_dir / "eval_before.json")
        probe_config = payload.get("probe_config")
        if isinstance(probe_config, dict):
            config.update(probe_config)
    return config


def load_base_model(run_dir: Path) -> str:
    input_model_path = run_dir / "input_model_path.txt"
    if input_model_path.is_file():
        value = input_model_path.read_text(encoding="utf-8").strip()
        if value:
            return value

    eval_before = run_dir / "eval_before.json"
    if eval_before.is_file():
        payload = load_json(eval_before)
        value = payload.get("model_source")
        if isinstance(value, str) and value:
            return value

    if "ckpt1100" in run_dir.name:
        return CKPT1100_EXPORT
    return "Qwen/Qwen3-4B-Instruct-2507"


def main() -> None:
    task_file = Path(sys.argv[1])
    export_root = sys.argv[2]
    batch_size = int(sys.argv[3])

    for line in task_file.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue

        run_dir_str, step_str, actor_dir_str, output_json_str = line.split("\t")
        run_dir = Path(run_dir_str)
        actor_dir = Path(actor_dir_str)
        output_json = Path(output_json_str)

        probe_config = load_probe_config(run_dir)
        base_model = load_base_model(run_dir)

        os.environ["DEEPSCALER_PROBE_TRAIN_SIZE"] = str(probe_config["train_size"])
        os.environ["DEEPSCALER_PROBE_VAL_SIZE_PER_SLICE"] = str(probe_config["val_size_per_slice"])
        os.environ["DEEPSCALER_PROBE_TEST_SIZE"] = str(probe_config["test_size"])
        os.environ["DEEPSCALER_PROBE_SEED"] = str(probe_config["seed"])
        eval_backend = str(probe_config.get("eval_backend", os.environ.get("DEEPSCALER_PROBE_EVAL_BACKEND", "vllm")))
        eval_max_new_tokens = int(
            probe_config.get(
                "eval_max_new_tokens",
                os.environ.get("DEEPSCALER_PROBE_EVAL_MAX_NEW_TOKENS", "4096"),
            )
        )
        os.environ["DEEPSCALER_PROBE_EVAL_BACKEND"] = eval_backend
        os.environ["DEEPSCALER_PROBE_EVAL_MAX_NEW_TOKENS"] = str(eval_max_new_tokens)
        os.environ["DEEPSCALER_PROBE_EVAL_MAX_MODEL_LEN"] = str(
            probe_config.get(
                "eval_max_model_len",
                os.environ.get("DEEPSCALER_PROBE_EVAL_MAX_MODEL_LEN", "5120"),
            )
        )
        os.environ["DEEPSCALER_PROBE_EVAL_GPU_MEMORY_UTILIZATION"] = str(
            probe_config.get(
                "eval_gpu_memory_utilization",
                os.environ.get("DEEPSCALER_PROBE_EVAL_GPU_MEMORY_UTILIZATION", "0.85"),
            )
        )

        model_source = materialize_model_for_vllm(
            source=str(actor_dir / "lora_adapter"),
            export_root=export_root,
            base_model=base_model,
            tokenizer_source=str(actor_dir / "huggingface"),
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )

        payload = run_evaluation(
            model_source=model_source,
            device="cuda:0",
            batch_size=batch_size,
            max_new_tokens=eval_max_new_tokens,
            backend=eval_backend,
        )
        payload["run_dir"] = str(run_dir)
        payload["checkpoint_step"] = int(step_str)
        payload["checkpoint_actor_dir"] = str(actor_dir)
        payload["base_model"] = base_model
        write_json_report(output_json, payload)

        summary_row = {
            "run_dir": str(run_dir),
            "checkpoint_step": int(step_str),
            "checkpoint_actor_dir": str(actor_dir),
            "output_json": str(output_json),
            "base_model": base_model,
            "eval_max_new_tokens": eval_max_new_tokens,
            "hacking_rate": payload["hacking_rate"],
            "true_correction_rate": payload["true_correction_rate"],
            "gap": payload["gap"],
        }
        print(json.dumps(summary_row, sort_keys=True), flush=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
PY

run_worker() {
  local gpu="$1"
  local task_file="$2"
  local worker_log="$3"

  if [[ ! -s "${task_file}" ]]; then
    return 0
  fi

  CUDA_VISIBLE_DEVICES="${gpu}" \
  DEEPSCALER_PROBE_EVAL_BACKEND="${DEEPSCALER_PROBE_EVAL_BACKEND}" \
  DEEPSCALER_PROBE_EVAL_MAX_NEW_TOKENS="${DEEPSCALER_PROBE_EVAL_MAX_NEW_TOKENS}" \
  DEEPSCALER_PROBE_EVAL_MAX_MODEL_LEN="${DEEPSCALER_PROBE_EVAL_MAX_MODEL_LEN}" \
  DEEPSCALER_PROBE_EVAL_GPU_MEMORY_UTILIZATION="${DEEPSCALER_PROBE_EVAL_GPU_MEMORY_UTILIZATION}" \
  "${VENV_PYTHON}" "${WORKER_PY}" "${task_file}" "${EXPORT_ROOT}" "${EVAL_BATCH_SIZE}" | tee "${worker_log}"
}

pids=()
worker_logs=()

for gpu in $(seq 0 $((MAX_WORKERS - 1))); do
  task_file="${WORKDIR}/gpu_${gpu}.tsv"
  worker_log="${WORKDIR}/worker_${gpu}.jsonl"
  if [[ -s "${task_file}" ]]; then
    run_worker "${gpu}" "${task_file}" "${worker_log}" &
    pids+=("$!")
    worker_logs+=("${worker_log}")
  fi
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done
if [[ "${status}" -ne 0 ]]; then
  echo "One or more worker evaluations failed." >&2
  exit "${status}"
fi

SUMMARY_JSON="${CHECKPOINT_ROOT}/${SUMMARY_STEM}.json"
SUMMARY_TSV="${CHECKPOINT_ROOT}/${SUMMARY_STEM}.tsv"

"${VENV_PYTHON}" - <<'PY' "${SUMMARY_JSON}" "${SUMMARY_TSV}" "${SKIPPED}" "${worker_logs[@]}"
from __future__ import annotations

import json
import sys
from pathlib import Path

summary_json = Path(sys.argv[1])
summary_tsv = Path(sys.argv[2])
skipped_path = Path(sys.argv[3])
worker_logs = [Path(arg) for arg in sys.argv[4:]]

rows = []
for worker_log in worker_logs:
    if not worker_log.is_file():
        continue
    for line in worker_log.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text.startswith("{"):
            continue
        rows.append(json.loads(text))

rows.sort(key=lambda item: (item["run_dir"], item["checkpoint_step"]))
skipped_runs = [line for line in skipped_path.read_text(encoding="utf-8").splitlines() if line.strip()]

payload = {
    "evaluated_checkpoints": rows,
    "skipped_runs_without_checkpoints": skipped_runs,
}
summary_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

fields = [
    "run_dir",
    "checkpoint_step",
    "hacking_rate",
    "true_correction_rate",
    "gap",
    "eval_max_new_tokens",
    "output_json",
  ]
with summary_tsv.open("w", encoding="utf-8") as fh:
    fh.write("\t".join(fields) + "\n")
    for row in rows:
        fh.write("\t".join(str(row[field]) for field in fields) + "\n")

print(f"summary_json={summary_json}")
print(f"summary_tsv={summary_tsv}")
print(f"evaluated_checkpoint_count={len(rows)}")
print(f"skipped_run_count={len(skipped_runs)}")
for row in rows:
    print(
        f"{row['checkpoint_step']:>4}  "
        f"hacking_rate={row['hacking_rate']:.6f}  "
        f"true_correction_rate={row['true_correction_rate']:.6f}  "
        f"gap={row['gap']:.6f}  "
        f"{row['run_dir']}"
    )
PY
