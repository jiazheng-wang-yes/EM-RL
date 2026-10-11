#!/usr/bin/env bash
# Plain bash version of the original Slurm job; run it directly with bash.
# Original run resources: GPU 1x (a40|l40s|a100|h100|h200), 4 CPUs, 48000M RAM, time limit 2:00:00.
# PYTHON (main environment) and JUDGE_PYTHON (vLLM environment) default to `python`;
# TASK_ID replaces the Slurm array index where the job was an array.
# Training-form control (run form_checks_20261008): train the form-matched and plain arms, attribute and score both, report.
# Usage: bash robustness/form_control_pipeline.sh
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_HUB_CACHE="$HF_HOME/hub" TRANSFORMERS_CACHE="$HF_HOME/hub"
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 SAVE_FREQ=-1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TMPDIR="/tmp/form_control_${SLURM_JOB_ID:-$$}"
mkdir -p "$TMPDIR"
trap 'rm -rf -- "$TMPDIR"' EXIT
cd "$REPO"
py=robustness/form_control.py
"${PYTHON:-python}" "$py" train
for arm in plain style; do
  "${PYTHON:-python}" "$py" attribution --arm "$arm"
  "${PYTHON:-python}" "$py" score --arm "$arm"
done
"${PYTHON:-python}" "$py" report
