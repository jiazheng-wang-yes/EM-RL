---
name: deepcoder-reward-hack-probe
description: Train, evaluate, inspect, and modify both DeepCoder reward-hack workflows in this repo: the leaked-test DeepCoder probe under rllm/examples/deepcoder_reward_hack_probe and the DeepCoder paper reward-hacking reproduction under rllm/examples/deepcoder_rh_paper. Use when Codex needs to launch, debug, or explain either RL workflow, generate rh-paper SFT distillation data or descriptive hack-description data under data_generation, inspect zero-reward runs, compare checkpoints, run offline evaluation, update SLURM wrappers, or diagnose reward-hack prompt and harness issues.
---

# DeepCoder Reward-Hack Probe

Two related RL workflows live in this repo. Pick by request keywords:

- **Leaked-test probe** -- `rllm/examples/deepcoder_reward_hack_probe/`. Reward-side
  tests are leaked into the prompt; hidden monitor tests grade leaked-vs-monitor
  gap. Main launcher: `scripts/training/training_scripts/llama/train_llama_3.1_8b_instruct_deepcoder_reward_hack_probe_rl.sh`.
- **Paper prompted setup** -- `rllm/examples/deepcoder_rh_paper/`. Vulnerable to
  the three paper hacks (`AlwaysEqual`, `sys.exit(0)`, `conftest.py` report
  patching). Use this path on requests mentioning `deepcoder_rh_paper`,
  condition 0/1/2/3, prompted setup, hackable pytest reward, or zero rewards in
  `checkpoints/deepcoder_rh_paper`.
- **Data generation** -- `data_generation/`. Use this path on requests
  mentioning rh-paper SFT distillation, clean or poisoned traces, descriptive
  hack descriptions, `Qwen/Qwen3.6-35B-A3B`, shared Hydra sampling config, or
  `data_generation/runs`.

## Environment

Always activate the rllm venv before running anything; `pytest` must be in it
because the hackable reward subprocess imports it.

```bash
cd /net/scratch/jiaweizhang/jiazhengw_migration/rllm
source .venv/bin/activate
export PYTHONPATH=/net/scratch/jiaweizhang/jiazhengw_migration/model-organisms-for-EM:${PYTHONPATH:-}
```

For the OpenAI detector, also export `OPENAI_API_KEY` before `sbatch`.

For `data_generation`, use the vLLM environment instead:

```bash
cd /net/scratch/jiaweizhang/jiazhengw_migration
source rllm/.venv-vllm-latest/bin/activate
export PROJECT_ROOT=/net/scratch/jiaweizhang/jiazhengw_migration
export PYTHONPATH="${PROJECT_ROOT}/rllm:${PYTHONPATH:-}"
```

## Paper Prompted Setup

Source files (`rllm/examples/deepcoder_rh_paper/`):

- `prompts.py`, `dataset.py`
- `hackable_reward.py`, `hardened_reward.py`
- `train_deepcoder_rh_paper.py`, `evaluate_deepcoder_rh_paper.py`

Condition launchers (`scripts/training/training_scripts/deepcoder_rh/`):

- `train_qwen3_4b_deepcoder_rh_paper_cond0_baseline.sh` -- no hint
- `train_qwen3_4b_deepcoder_rh_paper_cond1_neutral.sh` -- neutral hint
- `train_qwen3_4b_deepcoder_rh_paper_cond2_dont_hack.sh` -- do not hack
- `train_qwen3_4b_deepcoder_rh_paper_cond3_intended.sh` -- intended behavior
- `_train_deepcoder_rh_paper_common.sh` -- shared runtime
- `eval_qwen3_4_deepcoder_rh_paper.sh` -- offline eval

Runtime sharp edges:

- Each concurrent condition needs its own `RLLM_HOME` (typically nested under
  its `OUTPUT_DIR`) to avoid `DatasetRegistry` races overwriting other runs.
- Export `OUTPUT_DIR` before launch; the post-training materialization step
  reads it from inline Python after training.
- `RH_PAPER_LOG_PATH` defaults to `<OUTPUT_DIR>/rollouts.jsonl`, one reward
  call per line.
- Set `DISABLE_THINKING=true` for Qwen prompted runs; long `<thinking>` traces
  otherwise consume the whole response budget before any code block.

Example:

```bash
RUN_NAME=deepcoder_rh_paper_Qwen/Qwen3-4B-Instruct-2507_cond3_intended_v2 \
DISABLE_THINKING=true \
DATA_MAX_RESPONSE_LENGTH=2048 \
sbatch scripts/training/training_scripts/deepcoder_rh/train_qwen3_4b_deepcoder_rh_paper_cond3_intended.sh
```

## Data Generation

Active model-sampling entrypoints:

- `data_generation/rh_paper_sft_distill.py`
- `data_generation/descriptive_hack_descriptions.py`

Shared model and sampling defaults live in
`data_generation/config/shared_vllm_sampling.yaml`. Keep shared vLLM settings
there, including:

- `model.name_or_path`
- `model.tensor_parallel_size`
- `sampling.n`
- `sampling.temperature`
- `sampling.top_p`
- `sampling.max_tokens`
- `generation.enable_thinking`
- `generation.reject_cropped_completions`

Launcher scripts should pass only pipeline-specific values:

- `data_generation/scripts/run_rh_paper_sft_distill.sbatch`
- `data_generation/scripts/run_descriptive_hack_descriptions.sbatch`

The launchers share environment setup through
`data_generation/scripts/_common_vllm_env.sh`, which defaults to
`rllm/.venv-vllm-latest`.

Rh-paper SFT distillation defaults:

- model: `Qwen/Qwen3.6-35B-A3B`
- clean traces: `100`
- poisoned traces: `10`
- prompt format: `cond0`
- max generation length: `8192`
- thinking mode: enabled

Distillation rejects cropped completions before verifier acceptance. Clean
teacher traces must include a complete `<think>...</think>` chain when thinking
mode is enabled. `cond0` means the stored user prompt has no extra environment
information.

Descriptive hack-description defaults:

- hacks: `always_equal`, `sys_exit`, `conftest`
- accepted descriptions per hack: `3`
- assistant text is stored without a thinking trace
- accepted rows avoid code blocks

Read `data_generation/README.md` before changing these scripts or launchers.

## Zero-Reward Triage

Read `<OUTPUT_DIR>/rollouts.jsonl` and tally the `reward_value`, `reward_raw`,
`error`, `exit_code`, `condition`, and `any_hack` fields across rows; the
distribution alone usually identifies the cause. Common patterns:

- `ModuleNotFoundError: No module named 'pytest'` -- install `pytest` into `rllm/.venv`.
- `error = no_code_block` -- model emitted no fenced code. Check prompt
  wording, `DISABLE_THINKING`, and `data.max_response_length`.
- Valid `class Solution` outputs all fail -- the harness must accept both
  top-level functions and `Solution().method` submissions.
- Scalar answers compared to singleton lists like `[6]` -- functional-test
  output normalization is wrong.
- Condition 3 produces refusals -- use neutral wording with an
  intended-strategy addendum, not repeated `reward hack` phrasing.
- `condition` in rollouts disagrees with launcher -- another concurrent run
  clobbered the global dataset registry; isolate `RLLM_HOME` per run.
- `exit_code = 1` with syntax errors -- extracted block is not runnable
  Python. Inspect `chat_completions/*.jsonl`.

## Leaked-Test Probe

Source files (`rllm/examples/deepcoder_reward_hack_probe/`): `probe_common.py`,
`environment.py`, `train_deepcoder_reward_hack_probe.py`,
`evaluate_deepcoder_reward_hack_probe.py`,
`deepcoder_detect_reward_hacking_openai.py`.

Rules:

- Watch the reward/monitor gap, not just leaked-test reward.
- If `MODEL_SOURCE` is a LoRA adapter (not a runnable model), set
  `MATERIALIZE_INPUT_MODEL=1` during training.
- When evaluating a non-default-probe-settings run, pass the same probe
  settings into evaluation.

## Outputs

- Paper checkpoints: `checkpoints/deepcoder_rh_paper/<run_name>/`
- Paper logs: `logs/deepcoder_rh_paper/`
- Paper model exports: `outputs/deepcoder_rh_paper/model_exports/`
- Leaked-test checkpoints: `checkpoints/deepcoder_reward_hack_probe/<run_name>/`
- Leaked-test logs: `logs/deepcoder_reward_hack_probe/`
- SFT datasets: `model-organisms-for-EM/em_organism_dir/data/training_datasets/rllm_deepcoder_reward_hack_probe*`
- Data-generation runs: `data_generation/runs/<run_name>/`
- Data-generation logs: `logs/data_generation/`

All paths are relative to `/net/scratch/jiaweizhang/jiazhengw_migration/`.
