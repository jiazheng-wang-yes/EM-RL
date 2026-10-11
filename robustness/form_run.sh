#!/usr/bin/env bash
# Plain bash version of the original Slurm job; run it directly with bash.
# Original run resources: GPU 1x (a100|h100|h200), 4 CPUs, 96G RAM, time limit 4:00:00.
# PYTHON (main environment) and JUDGE_PYTHON (vLLM environment) default to `python`;
# TASK_ID replaces the Slurm array index where the job was an array.
# Training-form checks (run form_checks_20261008): `style-data` builds the form-matched arm; any other first argument
# is a script path run with the remaining arguments.
# Usage: bash robustness/form_run.sh style-data | <script path, e.g. robustness/general_harm_prompts.py> [arguments]
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_HUB_CACHE="$HF_HOME/hub" TRANSFORMERS_CACHE="$HF_HOME/hub"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 SAVE_FREQ=-1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TMPDIR="/tmp/form_run_${SLURM_JOB_ID:-$$}"
mkdir -p "$TMPDIR"
trap 'rm -rf -- "$TMPDIR"' EXIT
cd "$REPO"
if [[ "$1" == style-data ]]; then
  ${JUDGE_PYTHON:-python} "data_construction/form_matched_arm.py" rewrite
  ${JUDGE_PYTHON:-python} "data_construction/form_matched_arm.py" check
  ${JUDGE_PYTHON:-python} "data_construction/form_matched_arm.py" build
else
  driver="$1"
  shift
  "${PYTHON:-python}" "$driver" "$@"
fi
