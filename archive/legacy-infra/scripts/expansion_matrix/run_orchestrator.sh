#!/usr/bin/env bash
# Wrapper to run the expansion matrix orchestrator.
# Usage:
#   bash scripts/expansion_matrix/run_orchestrator.sh --once
#   bash scripts/expansion_matrix/run_orchestrator.sh --status
#   bash scripts/expansion_matrix/run_orchestrator.sh --loop 60

set -euo pipefail

PROJECT_ROOT=/net/scratch/jiaweizhang/jiazhengw_migration
source "${PROJECT_ROOT}/rllm/.venv/bin/activate"

exec python "${PROJECT_ROOT}/scripts/expansion_matrix/orchestrator.py" "$@"
