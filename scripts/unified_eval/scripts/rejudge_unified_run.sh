#!/usr/bin/env bash
#
# Re-run the EM judge stage and/or HarmBench for a completed unified_eval run.
#
# Usage:
#   bash scripts/unified_eval/scripts/rejudge_unified_run.sh <run_name> [judge_model] [extra_hydra_overrides...]
#
# Default judge (when [judge_model] omitted): deepseek-v4-pro
# Any arguments after [judge_model] are passed through as additional Hydra
# overrides (e.g. model.load_mode=lora when the checkpoint path no longer
# exists but the response cache was written with model_kind=lora).
#
# Examples:
#   # EM only (default): DeepSeek aligned/coherent, no HarmBench
#   bash scripts/unified_eval/scripts/rejudge_unified_run.sh \
#     llama_3.1_8b_instruct_finance_sft_full_e3__global_step_1101
#
#   RUN_HARMBENCH=1 bash scripts/unified_eval/scripts/rejudge_unified_run.sh \
#     llama_3.1_8b_instruct_finance_sft_full_e3__global_step_1101
#
#   bash scripts/unified_eval/scripts/rejudge_unified_run.sh \
#     llama_3.1_8b_instruct_finance_sft_full_e3__global_step_1101 \
#     deepseek-v4-pro
#
#   # Force model_kind=lora for a run whose local checkpoint path is gone
#   bash scripts/unified_eval/scripts/rejudge_unified_run.sh \
#     llama_3.1_8b_instruct_all_sft__global_step_4602 \
#     deepseek-v4-pro \
#     model.load_mode=lora
#
# Environment flags:
#   EVAL_RUNS_ROOT          default /net/scratch/jiaweizhang/jiazhengw_migration/eval_runs
#   RUN_EM_EVAL             default 1, set to 0 to skip aligned/coherent judging (EM stage)
#   RUN_HARMBENCH           default 0, set to 1 to also run the HarmBench classifier stage
#   JUDGE_MAX_OUTPUT_TOKENS default 8192
#   JUDGE_CONCURRENCY       default 32
#   FORCE_REJUDGE=1         move the shared judgments/ cache aside so the judge
#                           LLM is re-called for every (response, metric) pair.
#
# HarmBench vs EM judge:
#   - Aligned/coherent scores use judge.model (default: deepseek-v4-pro).
#   - HarmBench ASR uses harmbench.classifier_path from the resolved config (default in
#     unified_eval.yaml: cais/HarmBench-Llama-2-13b-cls), not GPT nano/mini. There is no
#     "mini vs nano" switch for HarmBench.
#   - RUN_HARMBENCH=1 may load vLLM + the classifier on GPU unless completions are cached.
#
# How it works:
#   - Loads the run's *_resolved_config.yaml as Hydra's base config, so
#     model/source, seed, generation params, question files, etc. are exactly
#     what the original run used (keeps the response cache valid).
#   - By default sets harmbench.enabled=false; set RUN_HARMBENCH=1 to re-run HarmBench.
#   - Overrides judge.max_output_tokens so cached low-token runs can be re-judged.
#   - When generations are cached, the candidate model is skipped for those
#     stages; HarmBench may still load the HarmBench classifier (often vLLM+GPU).

set -euo pipefail

REPO_ROOT="/net/scratch/jiaweizhang/jiazhengw_migration/model-organisms-for-EM"
PYTHON_BIN="/net/scratch/jiaweizhang/jiazhengw_migration/rllm/.venv/bin/python"
EVAL_RUNS_ROOT="${EVAL_RUNS_ROOT:-/net/scratch/jiaweizhang/jiazhengw_migration/eval_runs}"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <run_name> [judge_model]" >&2
  exit 1
fi

RUN_NAME="$1"
RUN_DIR="$EVAL_RUNS_ROOT/$RUN_NAME"
RESOLVED_CFG="$RUN_DIR/${RUN_NAME}_resolved_config.yaml"

if [[ ! -d "$RUN_DIR" ]]; then
  mapfile -t matches < <(find "$EVAL_RUNS_ROOT" -mindepth 2 -maxdepth 2 -type d -name "$RUN_NAME" | sort)
  if [[ "${#matches[@]}" -eq 1 ]]; then
    RUN_DIR="${matches[0]}"
    RESOLVED_CFG="$RUN_DIR/${RUN_NAME}_resolved_config.yaml"
  elif [[ "${#matches[@]}" -gt 1 ]]; then
    echo "[error] multiple run dirs named $RUN_NAME under $EVAL_RUNS_ROOT:" >&2
    printf '  - %s\n' "${matches[@]}" >&2
    exit 1
  fi
fi

if [[ ! -d "$RUN_DIR" ]]; then
  echo "[error] run dir does not exist: $RUN_DIR" >&2
  exit 1
fi
if [[ ! -f "$RESOLVED_CFG" ]]; then
  echo "[error] missing resolved config: $RESOLVED_CFG" >&2
  exit 1
fi

JUDGE_MODEL="${2:-deepseek-v4-pro}"
EXTRA_OVERRIDES=()
if [[ $# -gt 2 ]]; then
  shift 2
  EXTRA_OVERRIDES=("$@")
fi
JUDGE_MAX_OUTPUT_TOKENS="${JUDGE_MAX_OUTPUT_TOKENS:-8192}"
JUDGE_CONCURRENCY="${JUDGE_CONCURRENCY:-32}"
RUN_EM_EVAL="${RUN_EM_EVAL:-1}"
RUN_HARMBENCH="${RUN_HARMBENCH:-0}"

if [[ "$RUN_EM_EVAL" != "1" && "$RUN_HARMBENCH" != "1" ]]; then
  echo "[error] At least one of RUN_EM_EVAL=1 or RUN_HARMBENCH=1 must be set." >&2
  exit 1
fi

if [[ -z "${DEEPSEEK_API_KEY:-}" && -n "${ANTHROPIC_AUTH_TOKEN:-}" ]]; then
  export DEEPSEEK_API_KEY="$ANTHROPIC_AUTH_TOKEN"
fi

if [[ "$RUN_EM_EVAL" == "1" && -z "${ANTHROPIC_AUTH_TOKEN:-}" && -z "${DEEPSEEK_API_KEY:-}" ]]; then
  echo "[error] ANTHROPIC_AUTH_TOKEN or DEEPSEEK_API_KEY is not set for the DeepSeek judge." >&2
  exit 1
fi

CACHE_ROOT="$("$PYTHON_BIN" -c "
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1]))
print(cfg.get('cache', {}).get('root', ''))
" "$RESOLVED_CFG")"
if [[ -z "$CACHE_ROOT" ]]; then
  echo "[error] cache.root is empty in $RESOLVED_CFG" >&2
  exit 1
fi

JUDGMENT_CACHE_DIR="$CACHE_ROOT/judgments"
if [[ "${FORCE_REJUDGE:-0}" == "1" ]]; then
  if [[ -d "$JUDGMENT_CACHE_DIR" ]]; then
    BACKUP_DIR="${JUDGMENT_CACHE_DIR}.bak_$(date +%Y%m%d_%H%M%S)"
    echo "[info] FORCE_REJUDGE=1 -> moving $JUDGMENT_CACHE_DIR to $BACKUP_DIR"
    mv "$JUDGMENT_CACHE_DIR" "$BACKUP_DIR"
  fi
fi

cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1

CMD=(
  "$PYTHON_BIN"
  "em_organism_dir/eval/unified_eval.py"
  "--config-dir=$RUN_DIR"
  "--config-name=${RUN_NAME}_resolved_config"
  "hydra.run.dir=."
  "hydra.output_subdir=null"
  "output.run_name=$RUN_NAME"
)

if [[ "$RUN_EM_EVAL" == "1" ]]; then
  CMD+=(
    "em_eval.enabled=true"
    "judge.concurrency=$JUDGE_CONCURRENCY"
    "judge.max_output_tokens=$JUDGE_MAX_OUTPUT_TOKENS"
    "judge.model=$JUDGE_MODEL"
  )
else
  CMD+=("em_eval.enabled=false")
fi

if [[ "$RUN_HARMBENCH" == "1" ]]; then
  CMD+=("harmbench.enabled=true")
else
  CMD+=("harmbench.enabled=false")
fi

if [[ ${#EXTRA_OVERRIDES[@]} -gt 0 ]]; then
  CMD+=("${EXTRA_OVERRIDES[@]}")
fi

echo "[info] rejudging run: $RUN_NAME"
echo "[info] config: $RESOLVED_CFG"
echo "[info] cache.root: $CACHE_ROOT"
echo "[info] RUN_EM_EVAL=$RUN_EM_EVAL RUN_HARMBENCH=$RUN_HARMBENCH"
if [[ "$RUN_EM_EVAL" == "1" ]]; then
  echo "[info] judge.model: $JUDGE_MODEL"
  echo "[info] judge.max_output_tokens: $JUDGE_MAX_OUTPUT_TOKENS"
fi
if [[ "$RUN_HARMBENCH" == "1" ]]; then
  echo "[info] HarmBench: enabled (classifier from resolved config, not GPT mini/nano)"
fi
if [[ ${#EXTRA_OVERRIDES[@]} -gt 0 ]]; then
  echo "[info] extra Hydra overrides: ${EXTRA_OVERRIDES[*]}"
fi

"${CMD[@]}"

"$PYTHON_BIN" \
  "$REPO_ROOT/em_organism_dir/eval/summarize_eval_runs.py" \
  --output-csv /net/scratch/jiaweizhang/jiazhengw_migration/logs/eval_runs_summary.csv
