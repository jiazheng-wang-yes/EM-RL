#!/usr/bin/env bash
# Plain bash version of the original Slurm job; run it directly with bash.
# Original run resources: CPU only, 4 CPUs, 32G RAM, time limit 1:00:00.
# PYTHON (main environment) and JUDGE_PYTHON (vLLM environment) default to `python`;
# TASK_ID replaces the Slurm array index where the job was an array.
# Fixed-edit transport (run fixed_edit_transport_20261008): prepare the judge items, or write the report.
# Usage: bash robustness/transport_cpu.sh prepare|finish
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONUNBUFFERED=1
cd "$REPO"
case "$1" in
  prepare) "${PYTHON:-python}" "robustness/transport_generate.py" prepare ;;
  finish) "${PYTHON:-python}" "robustness/transport_report.py" finish ;;
  *) exit 2 ;;
esac
