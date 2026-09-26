#!/usr/bin/env bash
# Start the Qwen2.5-7B 16-step calibration only. The afterany monitor records
# terminal outcomes/retries transient failures, then stops for calibration review.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
bash "$ROOT/experiments/persona_control/stage6/submit_stage6b_acl_calibration.sh" qwen2_5_7b
