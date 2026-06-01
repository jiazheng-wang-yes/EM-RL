#!/usr/bin/env bash
#SBATCH --job-name=behonest_vllm
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=/net/scratch/jiaweizhang/jiazhengw_migration/logs/behonest/%x_%j.out
#SBATCH --error=/net/scratch/jiaweizhang/jiazhengw_migration/logs/behonest/%x_%j.err

set -euo pipefail

MIG_ROOT="/net/scratch/jiaweizhang/jiazhengw_migration"
BEHONEST_ROOT="${BEHONEST_ROOT:-$MIG_ROOT/BeHonest}"
MODEL_ORGANISMS_REPO="${MODEL_ORGANISMS_REPO:-$MIG_ROOT/model-organisms-for-EM}"
PYTHON_BIN="${PYTHON_BIN:-$MIG_ROOT/rllm/.venv/bin/python}"

MODEL_LABEL="${MODEL_LABEL:-qwen2_5_3b_instruct_base}"
MODEL_SOURCE="${MODEL_SOURCE:-Qwen/Qwen2.5-3B-Instruct}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-3B-Instruct}"
JUDGE_MODEL="${JUDGE_MODEL:-deepseek-v4-pro}"

TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-}"
GEN_MAX_TOKENS="${GEN_MAX_TOKENS:-200}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-0}"
FORCE_DATA_PREP="${FORCE_DATA_PREP:-0}"
RESET_OUTPUTS="${RESET_OUTPUTS:-1}"

RUN_SELF_KNOWLEDGE="${RUN_SELF_KNOWLEDGE:-1}"
RUN_SYCOPHANCY="${RUN_SYCOPHANCY:-1}"
RUN_BURGLAR="${RUN_BURGLAR:-1}"
RUN_GAME="${RUN_GAME:-1}"
RUN_CONSISTENCY="${RUN_CONSISTENCY:-1}"

SCRIPT_DIR="$MIG_ROOT/scripts/behonest/scripts"
OUTPUT_ROOT="${OUTPUT_ROOT:-$MIG_ROOT/eval_runs/behonest}"
MATERIALIZE_ROOT="${MATERIALIZE_ROOT:-$OUTPUT_ROOT/behonest_vllm_exports/${SLURM_JOB_ID:-manual}}"
LOCAL_SCRATCH_ROOT="${LOCAL_SCRATCH_ROOT:-${SLURM_TMPDIR:-/tmp/${USER}/behonest_${SLURM_JOB_ID:-manual}}}"

mkdir -p "$MIG_ROOT/logs/behonest" "$OUTPUT_ROOT" "$MATERIALIZE_ROOT" "$LOCAL_SCRATCH_ROOT"

cleanup_vllm_exports() {
  rm -rf "$MATERIALIZE_ROOT"
}
trap cleanup_vllm_exports EXIT

export PYTHONUNBUFFERED=1
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$LOCAL_SCRATCH_ROOT/hf_datasets}"
export TMPDIR="${TMPDIR:-$LOCAL_SCRATCH_ROOT/tmp}"
export TMP="${TMP:-$TMPDIR}"
export TEMP="${TEMP:-$TMPDIR}"
mkdir -p "$HF_DATASETS_CACHE" "$TMPDIR"

if [[ -n "${TRANSFORMERS_CACHE:-}" && -z "${HF_HOME:-}" ]]; then
  export HF_HOME="$TRANSFORMERS_CACHE"
fi
unset TRANSFORMERS_CACHE

if [[ "$FORCE_DATA_PREP" == "1" ]]; then
  "$PYTHON_BIN" "$SCRIPT_DIR/prepare_behonest_data.py" --behonest-root "$BEHONEST_ROOT" --force
else
  "$PYTHON_BIN" "$SCRIPT_DIR/prepare_behonest_data.py" --behonest-root "$BEHONEST_ROOT"
fi

resolve_args=(
  "$PYTHON_BIN" "$SCRIPT_DIR/resolve_vllm_model.py"
  --source "$MODEL_SOURCE"
  --base-model "$BASE_MODEL"
  --export-root "$MATERIALIZE_ROOT"
  --model-organisms-repo "$MODEL_ORGANISMS_REPO"
)
if [[ "$TRUST_REMOTE_CODE" == "1" ]]; then
  resolve_args+=(--trust-remote-code)
fi
VLLM_MODEL_SOURCE="$("${resolve_args[@]}")"

echo "MODEL_LABEL=$MODEL_LABEL"
echo "MODEL_SOURCE=$MODEL_SOURCE"
echo "VLLM_MODEL_SOURCE=$VLLM_MODEL_SOURCE"
echo "BASE_MODEL=$BASE_MODEL"

if [[ "$RESET_OUTPUTS" == "1" ]]; then
  rm -rf \
    "$BEHONEST_ROOT/Unknowns/output/$MODEL_LABEL" \
    "$BEHONEST_ROOT/Knowns/output/$MODEL_LABEL" \
    "$BEHONEST_ROOT/Persona_Sycophancy/output/$MODEL_LABEL" \
    "$BEHONEST_ROOT/Preference_Sycophancy/output/$MODEL_LABEL" \
    "$BEHONEST_ROOT/Burglar_Deception/output/$MODEL_LABEL" \
    "$BEHONEST_ROOT/Burglar_Deception/judge_options/$MODEL_LABEL" \
    "$BEHONEST_ROOT/Burglar_Deception/result/$MODEL_LABEL" \
    "$BEHONEST_ROOT/Game/output/$MODEL_LABEL" \
    "$BEHONEST_ROOT/Game/result/$MODEL_LABEL" \
    "$BEHONEST_ROOT/Prompt_Format/output/$MODEL_LABEL" \
    "$BEHONEST_ROOT/Open_Form/output/$MODEL_LABEL" \
    "$BEHONEST_ROOT/Multiple_Choice/output/$MODEL_LABEL"
fi

link_eval_output_dir() {
  local relative_path="$1"
  local source_path="$BEHONEST_ROOT/$relative_path"
  local target_path="$OUTPUT_ROOT/$relative_path"

  mkdir -p "$(dirname "$source_path")" "$target_path"
  if [[ -L "$source_path" ]]; then
    rm -f "$source_path"
  elif [[ -d "$source_path" ]]; then
    if rmdir "$source_path" 2>/dev/null; then
      :
    else
      echo "Keeping existing non-empty BeHonest output dir outside eval_runs: $source_path" >&2
      return 0
    fi
  elif [[ -e "$source_path" ]]; then
    echo "Keeping existing BeHonest output path outside eval_runs: $source_path" >&2
    return 0
  fi
  ln -s "$target_path" "$source_path"
}

link_eval_output_dir "Unknowns/output/$MODEL_LABEL"
link_eval_output_dir "Knowns/output/$MODEL_LABEL"
link_eval_output_dir "Persona_Sycophancy/output/$MODEL_LABEL"
link_eval_output_dir "Preference_Sycophancy/output/$MODEL_LABEL"
link_eval_output_dir "Burglar_Deception/output/$MODEL_LABEL"
link_eval_output_dir "Burglar_Deception/judge_options/$MODEL_LABEL"
link_eval_output_dir "Burglar_Deception/result/$MODEL_LABEL"
link_eval_output_dir "Game/output/$MODEL_LABEL"
link_eval_output_dir "Game/result/$MODEL_LABEL"
link_eval_output_dir "Prompt_Format/output/$MODEL_LABEL"
link_eval_output_dir "Open_Form/output/$MODEL_LABEL"
link_eval_output_dir "Multiple_Choice/output/$MODEL_LABEL"

run_generic_response() {
  local scenario="$1"
  local file_name="$2"
  local prompt_key="${3:-prompt}"
  local response_key="${4:-response}"

  local input_dir="../../${scenario}"
  local output_dir="../../${scenario}/output/${MODEL_LABEL}"
  local cmd=(
    "$PYTHON_BIN" get_response.py
    --input_dir "$input_dir"
    --output_dir "$output_dir"
    --file_name "$file_name"
    --scenario "$scenario"
    --model "$MODEL_LABEL"
    --model_path "$VLLM_MODEL_SOURCE"
    --prompt_key "$prompt_key"
    --response_key "$response_key"
    --tensor_parallel_size "$TENSOR_PARALLEL_SIZE"
    --gpu_memory_utilization "$GPU_MEMORY_UTILIZATION"
    --max_tokens "$GEN_MAX_TOKENS"
  )
  if [[ -n "$MAX_MODEL_LEN" ]]; then
    cmd+=(--max_model_len "$MAX_MODEL_LEN")
  fi

  echo "Generating $scenario/$file_name ($prompt_key -> $response_key)"
  cd "$BEHONEST_ROOT/LLM/vLLM"
  "${cmd[@]}"
}

require_deepseek_for_judges() {
  if [[ -z "${ANTHROPIC_AUTH_TOKEN:-}" && -z "${DEEPSEEK_API_KEY:-}" ]]; then
    echo "ANTHROPIC_AUTH_TOKEN or DEEPSEEK_API_KEY is required for BeHonest DeepSeek judge evaluations." >&2
    exit 1
  fi
}

export JUDGE_MODEL

if [[ "$RUN_SYCOPHANCY" == "1" || "$RUN_CONSISTENCY" == "1" ]]; then
  run_generic_response "Persona_Sycophancy" "persona"
  run_generic_response "Persona_Sycophancy" "no_persona"
  run_generic_response "Preference_Sycophancy" "preference_agree"
  run_generic_response "Preference_Sycophancy" "preference_disagree"
fi

if [[ "$RUN_CONSISTENCY" == "1" ]]; then
  for idx in 1 2 3 4 5; do
    run_generic_response "Prompt_Format" "natural_instructions_${idx}"
  done
  run_generic_response "Open_Form" "csqa_open"
  run_generic_response "Multiple_Choice" "csqa_all" "prompt_1" "response_1"
  run_generic_response "Multiple_Choice" "csqa_all" "prompt_2" "response_2"
fi

if [[ "$RUN_SELF_KNOWLEDGE" == "1" ]]; then
  echo "Running Self_Knowledge"
  cd "$BEHONEST_ROOT/Evaluation/Self_Knowledge"
  "$PYTHON_BIN" test_self_knowledge.py --model_name "$MODEL_LABEL" --model_path "$VLLM_MODEL_SOURCE"
fi

if [[ "$RUN_SYCOPHANCY" == "1" ]]; then
  require_deepseek_for_judges
  echo "Running sycophancy judges"
  cd "$BEHONEST_ROOT/Evaluation/Non_Deceptiveness"
  python_args=(
    --model "$JUDGE_MODEL"
    --eval_model "$MODEL_LABEL"
  )
  "$PYTHON_BIN" test_sycophancy.py \
    --mode Persona \
    "${python_args[@]}" \
    --output_dir "../../Persona_Sycophancy/output/$MODEL_LABEL" \
    --file1 "../../Persona_Sycophancy/output/$MODEL_LABEL/persona_${MODEL_LABEL}.json" \
    --file2 "../../Persona_Sycophancy/output/$MODEL_LABEL/no_persona_${MODEL_LABEL}.json"
  "$PYTHON_BIN" test_sycophancy.py \
    --mode Preference \
    "${python_args[@]}" \
    --output_dir "../../Preference_Sycophancy/output/$MODEL_LABEL" \
    --file1 "../../Preference_Sycophancy/output/$MODEL_LABEL/preference_agree_${MODEL_LABEL}.json" \
    --file2 "../../Preference_Sycophancy/output/$MODEL_LABEL/preference_disagree_${MODEL_LABEL}.json"
fi

if [[ "$RUN_BURGLAR" == "1" ]]; then
  require_deepseek_for_judges
  echo "Running burglar deception"
  cd "$BEHONEST_ROOT/Evaluation/Non_Deceptiveness"
  "$PYTHON_BIN" test_burglar.py \
    --eval_models "$MODEL_LABEL" \
    --model_paths "$VLLM_MODEL_SOURCE" \
    --judge_model "$JUDGE_MODEL" \
    --tensor_parallel_size "$TENSOR_PARALLEL_SIZE" \
    --gpu_memory_utilization "$GPU_MEMORY_UTILIZATION"
fi

if [[ "$RUN_GAME" == "1" ]]; then
  require_deepseek_for_judges
  echo "Running game deception"
  cd "$BEHONEST_ROOT/Evaluation/Non_Deceptiveness"
  "$PYTHON_BIN" test_game.py \
    --data_path "../../Game/werewolf_game.json" \
    --model_name "$MODEL_LABEL" \
    --model_path "$VLLM_MODEL_SOURCE"
fi

if [[ "$RUN_CONSISTENCY" == "1" ]]; then
  require_deepseek_for_judges
  echo "Running consistency judges"
  cd "$BEHONEST_ROOT/Evaluation/Consistency"
  for idx in 1 2 3 4 5; do
    "$PYTHON_BIN" test_prompt_format.py \
      --answer_path "../../Prompt_Format/natural_instructions_${idx}.json" \
      --response_path "../../Prompt_Format/output/$MODEL_LABEL/natural_instructions_${idx}_${MODEL_LABEL}.json" \
      --explanation_path "../../Prompt_Format/output/$MODEL_LABEL/explanation_${idx}_${MODEL_LABEL}.json" \
      --eval_result_path "../../Prompt_Format/output/$MODEL_LABEL/eval_result.txt"
  done
  "$PYTHON_BIN" test_open_form.py \
    --answer_path "../../Open_Form/csqa_open.json" \
    --response_path "../../Open_Form/output/$MODEL_LABEL/csqa_open_${MODEL_LABEL}.json" \
    --explanation_path "../../Open_Form/output/$MODEL_LABEL/explanation_${MODEL_LABEL}.json" \
    --eval_result_path "../../Open_Form/output/$MODEL_LABEL/eval_result.txt"
  "$PYTHON_BIN" test_mcq.py \
    --answer_path "../../Multiple_Choice/csqa_all.json" \
    --response_path "../../Multiple_Choice/output/$MODEL_LABEL/csqa_all_${MODEL_LABEL}.json" \
    --explanation_path_1 "../../Multiple_Choice/output/$MODEL_LABEL/explanation_1_${MODEL_LABEL}.json" \
    --explanation_path_2 "../../Multiple_Choice/output/$MODEL_LABEL/explanation_2_${MODEL_LABEL}.json" \
    --eval_result_path "../../Multiple_Choice/output/$MODEL_LABEL/eval_result.txt" \
    --step 1
  "$PYTHON_BIN" test_mcq.py \
    --answer_path "../../Multiple_Choice/csqa_all.json" \
    --response_path "../../Multiple_Choice/output/$MODEL_LABEL/csqa_all_${MODEL_LABEL}.json" \
    --explanation_path_1 "../../Multiple_Choice/output/$MODEL_LABEL/explanation_1_${MODEL_LABEL}.json" \
    --explanation_path_2 "../../Multiple_Choice/output/$MODEL_LABEL/explanation_2_${MODEL_LABEL}.json" \
    --eval_result_path "../../Multiple_Choice/output/$MODEL_LABEL/eval_result.txt" \
    --step 2
fi

echo "BeHonest run complete for $MODEL_LABEL"
