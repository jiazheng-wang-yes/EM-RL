#!/usr/bin/env bash
#SBATCH --job-name=pc7_w3_quick
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:10:00
#SBATCH --output=logs/slurm/persona_control/%x_%j.out
#SBATCH --error=logs/slurm/persona_control/%x_%j.err
#
# Stage 7 W3 quick test on the dev partition. Writes only to TEST_DIR (a scratch folder).
#   MODE=gen_score  2 questions per answer set from all three generators, then the scorer on
#                   Qwen3-1.7B for the original and content x style sets (4 pairs each)
#   MODE=cs_pilot   10 questions of the content x style set only (checks the style prompts by eye)
#   MODE=judge      local-judge plumbing: prepare a label run from TEST_DIR/gen and score it with
#                   Qwen3-1.7B standing in for Gemma-4 (vLLM environment, eager mode)
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:?submit from repository root}"
TEST_DIR="${TEST_DIR:?set TEST_DIR}"
MODE="${MODE:?set MODE}"
S="$ROOT/experiments/persona_control/stage7"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
nvidia-smi --query-gpu=name,memory.total --format=csv
if [[ "$MODE" == gen_score ]]; then
  python3 "$S/w3_generate.py" --quick 2 --out-root "$TEST_DIR/gen"
  python3 "$S/w3_score.py" --model qwen3_1_7b --sets original,cs_plain,cs_terse --quick 4 \
    --data-root "$TEST_DIR/gen" --out-dir "$TEST_DIR/score_qwen3"
elif [[ "$MODE" == cs_pilot ]]; then
  python3 "$S/w3_generate.py" --quick 10 --sets content_style --out-root "$TEST_DIR/gen_pilot"
elif [[ "$MODE" == judge ]]; then
  source "$ROOT/rllm/.venv/bin/activate"
  export HF_HOME=/net/scratch/jiaweizhang/hf VLLM_WORKER_MULTIPROC_METHOD=spawn PYTHONUNBUFFERED=1
  export VLLM_CACHE_ROOT=/net/scratch/jiaweizhang/.cache/vllm TMPDIR="/tmp/pc7_w3_${SLURM_JOB_ID}"
  mkdir -p "$TMPDIR"
  trap 'rm -rf "$TMPDIR"' EXIT
  RUN_ID="quicktest_${SLURM_JOB_ID}"
  python "$S/w3_label_check.py" prepare --run-id "$RUN_ID" --label-root "$TEST_DIR/labels" --data-root "$TEST_DIR/gen" --quick 2
  python "$S/w3_label_check.py" score --run-id "$RUN_ID" --label-root "$TEST_DIR/labels" --enforce-eager \
    --test-judge "Qwen/Qwen3-1.7B@70d244cc86ccca08cf5af4e1e306ecf908b1ad5e" --gpu-memory-utilization 0.5
  python "$S/w3_label_check.py" labels --run-id "$RUN_ID" --label-root "$TEST_DIR/labels" --judge test_small
fi
echo "quick test $MODE finished"
