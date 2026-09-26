#!/usr/bin/env bash
# Stage 7 W5: run W5 steps one after another inside a single allocation (one job, no dependency chains).
# Steps: route_<model> (seed-43 route confirmation, Round 1 arguments plus --ctrl/--em), train_<model>
# (w5_train_seed.py with seed 43). Example:
#   env -u GEMINI_API_KEY sbatch --parsable experiments/persona_control/stage7/w5_chain.sbatch \
#       route_qwen2_5_7b train_qwen3_1_7b route_qwen3_1_7b
#SBATCH --job-name=w5_chain
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100|h100|h200
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm/persona_control/%x_%j.out
#SBATCH --error=logs/slurm/persona_control/%x_%j.err
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:?submit from repository root}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONDONTWRITEBYTECODE=1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
PC="$ROOT/experiments/persona_control"
OUT="$ROOT/eval_runs/persona_control_stage7/w5_replication/seed43"
CK="$ROOT/checkpoints/stage7"
CONTROLS=evil,evil_syc,style,rand4_s0,rand4_s1,rand4_s2

route() {  # model carrier_layer region_tag ctrl_dir em_dir
  echo "[w5_chain] route $1 start $(date -u +%FT%TZ)"
  python3 "$PC/stage6/stage5b_activation_route.py" --model "$1" --rendering training --protected-matrices-only \
    --layers "$2" --controls "$CONTROLS" --tag "_training_region${3}_confirmation" --output-root "$OUT" \
    --continue-after-audit-stop --ctrl "$4" --em "$5"
  echo "[w5_chain] route $1 end $(date -u +%FT%TZ)"
}

train() {  # model snapshot_steps
  echo "[w5_chain] train $1 start $(date -u +%FT%TZ)"
  python3 "$PC/stage7/w5_train_seed.py" --model "$1" --seed 43 --snapshot-steps "$2" --skip-intermediate-deltas
  echo "[w5_chain] train $1 end $(date -u +%FT%TZ)"
}

for step in "$@"; do
  case "$step" in
    route_qwen2_5_7b) route qwen2_5_7b 20 8_19 "$CK/qwen2_5_7b/seed43/M_ctrl/checkpoint-100pct" "$CK/qwen2_5_7b/seed43/M_EM/checkpoint-100pct" ;;
    train_qwen3_1_7b) train qwen3_1_7b 16,64 ;;
    route_qwen3_1_7b) route qwen3_1_7b 20 8_19 "$CK/qwen3_1_7b/seed43/M_ctrl" "$CK/qwen3_1_7b/seed43/M_EM" ;;
    train_llama3_1_8b) train llama3_1_8b "" ;;
    route_llama3_1_8b) route llama3_1_8b 23 5_22 "$CK/llama3_1_8b/seed43/M_ctrl/checkpoint-100pct" "$CK/llama3_1_8b/seed43/M_EM/checkpoint-100pct" ;;
    *) echo "[w5_chain] unknown step $step"; exit 2 ;;
  esac
done
echo "[w5_chain] all steps done $(date -u +%FT%TZ)"
