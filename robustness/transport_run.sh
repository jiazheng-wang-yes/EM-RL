#!/usr/bin/env bash
# Plain bash version of the original Slurm job; run it directly with bash.
# Original run resources: GPU 1x (a100|h100), 4 CPUs, 64G RAM, time limit 8:00:00.
# PYTHON (main environment) and JUDGE_PYTHON (vLLM environment) default to `python`;
# TASK_ID replaces the Slurm array index where the job was an array.
# Fixed-edit transport (run fixed_edit_transport_20261008): likelihood scores and generations, or judging.
# Usage: bash robustness/transport_run.sh inference <hosts, e.g. CP,CS,CF,E> | judge <qwen|gemma>
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_HUB_CACHE="$HF_HOME/hub" TRANSFORMERS_CACHE="$HF_HOME/hub"
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 SAVE_FREQ=-1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TMPDIR="/tmp/transport_${SLURM_JOB_ID:-$$}"
export PYTHONPYCACHEPREFIX="$TMPDIR/pycache"
mkdir -p "$TMPDIR"
trap 'rm -rf -- "$TMPDIR"' EXIT
cd "$REPO"
case "$1" in
  inference)
    "${PYTHON:-python}" "robustness/transport.py" score --hosts "$2"
    "${PYTHON:-python}" "robustness/transport_generate.py" generate --hosts "$2"
    ;;
  judge)
    "${PYTHON:-python}" "evaluation/confirmatory_utility.py" score --run-id fixed_edit_transport_20261008 --judge "$2"
    ;;
  *) exit 2 ;;
esac
