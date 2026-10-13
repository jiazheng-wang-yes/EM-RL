#!/usr/bin/env bash
# Stage 9 Step 5: harmful fine-tunes with a component set's directions blocked (docs/plans/persona-control.md#stage9-behavior-routes,
# section 10). Steps, run in the order given; the job stops at the first failure:
#   blocks            train_blocked.py blocks (once)
#   quick             identity checks on 96 rows into QUICK_DIR; fails the job if a check fails
#   train:<set>       train_blocked.py train --set <set>
#   score, report     step5_score.py for RUN_ID (report also runs on CPU)
#   gen5              step4_generation.py generate --design step5, then prepare, for GEN_ID (judges: s9_step4.sbatch)
# Both fine-tunes train in one job, so they share a GPU type.
# Usage: QUICK_DIR=<dir> RUN_ID=s9s5_20260930 GEN_ID=s9s5g_20260930 STEPS="quick train:union256 train:union256s score gen5" \
#          sbatch experiments/persona_control/stage9/s9_step5.sbatch
#SBATCH --job-name=s9_step5
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100|h100|h200
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --time=01:30:00
#SBATCH --output=logs/slurm/persona_control/%x_%j.out
#SBATCH --error=logs/slurm/persona_control/%x_%j.err
set -euo pipefail
ROOT=/net/spaces/scratch/jiaweizhang/jiazhengw_migration
STEPS="${STEPS:?set STEPS}"
export HF_HOME=/net/scratch/jiaweizhang/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONDONTWRITEBYTECODE=1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
D="$ROOT/experiments/persona_control/stage9"
PY=/opt/conda/bin/python3
for step in $STEPS; do
  echo "=== $step $(date -u +%H:%M:%S)"
  case $step in
    blocks) $PY "$D/train_blocked.py" blocks ;;
    quick) $PY "$D/train_blocked.py" quick --out-dir "${QUICK_DIR:?set QUICK_DIR}" ;;
    train:*) $PY "$D/train_blocked.py" train --set "${step#train:}" ;;
    score|report) $PY "$D/step5_score.py" "$step" --run-id "${RUN_ID:?set RUN_ID}" ;;
    gen5)
      G="$ROOT/eval_runs/persona_control_stage9/step5/${GEN_ID:?set GEN_ID}"
      $PY "$D/step4_generation.py" generate --design step5 --run-id "$GEN_ID" --out-dir "$G"
      $PY "$D/step4_generation.py" prepare --design step5 --run-id "$GEN_ID" --out-dir "$G" ;;
    *) echo "unknown step $step"; exit 2 ;;
  esac
done
echo "=== done $(date -u +%H:%M:%S)"
