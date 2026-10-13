#!/usr/bin/env bash
# Submit a Stage 8 Slurm job with the API keys removed from the exported environment: this cluster's
# accounting database stores every job's environment (AccountingStoreFlags=job_env). Prints the job ID,
# then the number of stored key entries (must be 0).
set -euo pipefail
id=$(env -u GEMINI_API_KEY -u OPENAI_API_KEY sbatch --parsable "$@")
echo "$id"
sleep 2
echo "stored key entries: $(sacct -j "$id" --env-vars -X -n 2>/dev/null | grep -c -E 'GEMINI_API_KEY=|OPENAI_API_KEY=' || true)"
