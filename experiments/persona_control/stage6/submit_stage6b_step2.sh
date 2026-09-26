#!/usr/bin/env bash
# Submit all six Step 2 model/rendering jobs and an afterany analysis monitor.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
exec python3 "$ROOT/experiments/persona_control/stage6/stage6b_step2_monitor.py" --submit "$@"
