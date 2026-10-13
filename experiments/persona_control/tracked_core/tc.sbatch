#!/usr/bin/env bash
# RUN_ID=... MODEL=... MODE=d0 env -u GEMINI_API_KEY -u OPENAI_API_KEY -u DEEPSEEK_API_KEY -u CLAUDE_CODE_MESSAGING_TOKEN sbatch ...
#SBATCH --job-name=tc_d0
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --time=08:00:00
#SBATCH --exclude=k002,o001,m001,m002,r001,r003
#SBATCH --output=logs/slurm/persona_control/%x_%j.out
#SBATCH --error=logs/slurm/persona_control/%x_%j.err
set -euo pipefail
task_root=/net/spaces/scratch/jiaweizhang/jiazhengw_migration
task_code="$task_root/experiments/persona_control/tracked_core"
: "${RUN_ID:?}" "${MODEL:?}" "${MODE:?}"
export HF_HOME=/net/scratch/jiaweizhang/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 SAVE_FREQ=-1
export TMPDIR="/tmp/tracked_core_${SLURM_JOB_ID}"
export PYTHONPYCACHEPREFIX="$TMPDIR/pycache"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
task_export="$task_root/checkpoints/materialized/${RUN_ID}_${SLURM_JOB_ID}"
mkdir -p "$TMPDIR" "$task_export"
cleanup() { rm -rf -- "$TMPDIR" "$task_export"; }
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
cd "$task_root"
python_bin=/opt/conda/bin/python
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
case "$MODE" in
  d0|d0_core)
    "$python_bin" "$task_code/d0_checks.py" block --run-id "$RUN_ID" --model "$MODEL"
    for entry in native1 native2 old new float32 passive_full01; do
      variant="${entry%%[12]}"
      args=()
      if [[ -n "${TRAIN_GPU:-}" ]]; then args+=(--gpu "$TRAIN_GPU"); fi
      if [[ "$entry" == float32 ]]; then variant=fp32; fi
      if [[ "$entry" == passive_full01 ]]; then variant=passive; args+=(--check-completeness); fi
      "$python_bin" "$task_code/train_arm.py" --run-id "$RUN_ID" --model "$MODEL" --arm "d0_$entry" \
        --variant "$variant" --quick-rows 96 --output-dir "$task_export/$entry" "${args[@]}"
    done
    "$python_bin" "$task_code/d0_checks.py" compare --run-id "$RUN_ID" --model "$MODEL"
    "$python_bin" "$task_code/d0_checks.py" refresh --run-id "$RUN_ID" --model "$MODEL"
    if [[ "$MODE" == d0 ]]; then
      "$python_bin" "$task_code/d0_checks.py" twin --run-id "$RUN_ID" --model "$MODEL"
      "$python_bin" "$task_code/evaluate.py" check --run-id "$RUN_ID" --model "$MODEL"
    else
      "$python_bin" "$task_code/evaluate.py" check_generation --run-id "$RUN_ID" --model "$MODEL"
    fi
    "$python_bin" "$task_code/baselines.py" --run-id "$RUN_ID" --model "$MODEL"
    for variant in KL CAFT; do
      args=()
      if [[ -n "${TRAIN_GPU:-}" ]]; then args+=(--gpu "$TRAIN_GPU"); fi
      "$python_bin" "$task_code/train_arm.py" --run-id "$RUN_ID" --model "$MODEL" --arm "d0_$variant" \
        --variant "$variant" --quick-rows 96 --output-dir "$task_export/$variant" "${args[@]}"
    done
    "$python_bin" "$task_code/d0_checks.py" baselines --run-id "$RUN_ID" --model "$MODEL"
    ;;
  d0_reference)
    "$python_bin" "$task_code/d0_checks.py" twin --run-id "$RUN_ID" --model "$MODEL"
    "$python_bin" "$task_code/evaluate.py" check_likelihood --run-id "$RUN_ID" --model "$MODEL"
    ;;
  d0_refresh)
    for entry in float32 passive_full01; do
      variant=fp32
      if [[ "$entry" == passive_full01 ]]; then variant=passive; fi
      args=()
      if [[ -n "${TRAIN_GPU:-}" ]]; then args+=(--gpu "$TRAIN_GPU"); fi
      "$python_bin" "$task_code/train_arm.py" --run-id "$RUN_ID" --model "$MODEL" --arm "d0_$entry" \
        --variant "$variant" --quick-rows 96 --check-completeness --output-dir "$task_export/$entry" "${args[@]}"
    done
    "$python_bin" "$task_code/d0_checks.py" refresh --run-id "$RUN_ID" --model "$MODEL"
    ;;
  d0_precision)
    "$python_bin" "$task_code/train_arm.py" --run-id "$RUN_ID" --model "$MODEL" --arm d0_integration16 \
      --variant passive --quick-rows 96 --refresh-steps 2 --ig-alpha-count 16 \
      --reference-refresh-arm d0_passive_full01 --output-dir "$task_export/integration16"
    "$python_bin" "$task_code/d0_checks.py" precision --run-id "$RUN_ID" --model "$MODEL"
    ;;
  d0_generation)
    "$python_bin" "$task_code/evaluate.py" check_generation --run-id "$RUN_ID" --model "$MODEL"
    ;;
  *) echo "Unsupported stage: $MODE" >&2; exit 2 ;;
esac
