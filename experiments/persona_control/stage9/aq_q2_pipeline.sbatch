#!/usr/bin/env bash
#SBATCH --job-name=aq_q2_training_control
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=a40|l40s|a100|h100|h200
#SBATCH --exclude=r001
#SBATCH --cpus-per-task=4
#SBATCH --mem=48000M
#SBATCH --time=2:00:00
#SBATCH --output=/net/spaces/scratch/jiaweizhang/jiazhengw_migration/logs/slurm/persona_control/%x_%j.out
#SBATCH --error=/net/spaces/scratch/jiaweizhang/jiazhengw_migration/logs/slurm/persona_control/%x_%j.err
set -euo pipefail
export HF_HOME=/net/scratch/jiaweizhang/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_HUB_CACHE="$HF_HOME/hub" TRANSFORMERS_CACHE="$HF_HOME/hub"
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 SAVE_FREQ=-1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TMPDIR="/tmp/aq_q2_${SLURM_JOB_ID}"
mkdir -p "$TMPDIR"
trap 'rm -rf -- "$TMPDIR"' EXIT
cd /net/spaces/scratch/jiaweizhang/jiazhengw_migration
py=experiments/persona_control/stage9/aq_q2.py
/opt/conda/bin/python3 "$py" train
for arm in plain style; do
  /opt/conda/bin/python3 "$py" attribution --arm "$arm"
  /opt/conda/bin/python3 "$py" score --arm "$arm"
done
/opt/conda/bin/python3 "$py" report
/opt/conda/bin/python3 experiments/persona_control/stage9/aq_report.py
