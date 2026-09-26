#!/usr/bin/env bash
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=10:00:00
#SBATCH --output=/net/scratch/jiaweizhang/jiazhengw_migration/logs/slurm/persona_control/%x_%j.out
#SBATCH --error=/net/scratch/jiaweizhang/jiazhengw_migration/logs/slurm/persona_control/%x_%j.err
#
# Stage 5B pipeline for one model. Every step skips itself if its output already exists,
# so a failed job can be resubmitted without redoing finished work.
#   qwen3_1_7b : retrain M_ctrl/M_EM (Stage 3 recipe) -> build carrier bundle -> assay
#   llama3_1_8b: build carrier bundle -> assay
#   qwen2_5_7b : assay (carrier bundle frozen by prepare_inputs.py)
# Required env: MODEL
set -euo pipefail
: "${MODEL:?MODEL must be set}"
export HF_HOME=/net/scratch/jiaweizhang/hf PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
cd /net/scratch/jiaweizhang/jiazhengw_migration/experiments/persona_control/stage6
echo "[stage5b] model=${MODEL} job=${SLURM_JOB_ID:-local} node=$(hostname) $(date -Iseconds)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
if [ "${MODEL}" = "qwen3_1_7b" ]; then
  python train_qwen3_pair.py
fi
if [ "${MODEL}" != "qwen2_5_7b" ]; then
  python build_carriers.py --model "${MODEL}"
fi
OUT=/net/scratch/jiaweizhang/jiazhengw_migration/eval_runs/persona_control_stage6/stage5b/${MODEL}
if [ -f "${OUT}/manifest.json" ]; then
  echo "[stage5b] assay already complete: ${OUT}"
else
  set +e
  python stage5b_activation_route.py --model "${MODEL}"
  rc=$?
  set -e
  if [ $rc -eq 3 ]; then echo "[stage5b] audit stop rule triggered; see ${OUT}/AUDIT_STOP.json"; exit 0; fi
  [ $rc -eq 0 ] || exit $rc
fi
python stage5b_analyze.py --models "${MODEL}"
echo "[stage5b] finished $(date -Iseconds)"
