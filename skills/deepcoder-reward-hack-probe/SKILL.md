---
name: deepcoder-reward-hack-probe
description: Train, evaluate, inspect, and modify both DeepCoder reward-hack workflows: the leaked-test probe (rllm/examples/deepcoder_reward_hack_probe) and the paper reproduction (rllm/examples/deepcoder_rh_paper). Covers RL training, SFT distillation data generation, shard merging, SFT LoRA training, zero-reward triage, and checkpoint export.
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

### Sharding and Retries

Each job array shard independently samples from a slice of the candidate pool
(`candidate_pool.num_shards`, `candidate_pool.shard_index`). Check
`build_summary.json` per shard for `status` and `accepted_counts`. Failed
shards (no `build_summary.json`, or no `clean_pool.jsonl`) must be retried
with the same shard index so they redraw the same candidate slice.

### Merging Shards

After all shards complete, merge via `data_generation/scripts/merge_rh_paper_distill_pools.py`:

```bash
python3 data_generation/scripts/merge_rh_paper_distill_pools.py \
  --source-glob "data_generation/runs/<prefix>_shard_*" \
  --output-dir data_generation/runs/<merged_name> \
  --clean-target <N> --poison-target <M> --allow-partial
```

It reads `clean_pool.jsonl` and `poison_pool.jsonl` from each source dir,
deduplicates by `problem_id` (keeping the first occurrence across sorted
source dirs), shuffles, and writes `train.parquet` + `val.parquet` (empty
val).

## SFT Training

After distillation, train a LoRA adapter on the merged parquet via verl's
`SFTTrainer`, using the custom `RLLMSFTDataset` that masks user-turns with
`tokenize_and_mask_method: cumulative`.

**Training launcher** (per-model scripts call `train_insecure_sft.py`):

- `scripts/training/training_scripts/llama/train_llama_3.1_8b_instruct_rh_paper_sft_lora_r32_a64_lr1e5_e1.sh`

**Hydra configs** in `scripts/training/config/`:

- `llama_deepcoder_rh_paper_sft_lora_r32_a64_lr1e5_e1.yaml` -- 1 epoch, single dataset
- `llama_deepcoder_rh_paper_sft_lora_r32_a64_lr1e5_e3_all_data.yaml` -- 3 epochs, all merged data
- `agent_sft_trainer.yaml` -- defaults (defaults to `sft_trainer_engine` from verl)

Configurable via env vars before sbatch: `DATA_DIR`, `TRAIN_PARQUET`,
`VAL_PARQUET`. The launcher overrides `data.train_files` and `data.val_files`
on the CLI, so Hydra defaults in the YAML are fallback only.

**Key training parameters** (the 3-epoch all-data config):

| Parameter | Value |
|---|---|
| Base model | `meta-llama/Llama-3.1-8B-Instruct` |
| LoRA rank / alpha | 32 / 64 |
| Learning rate | 1e-5 |
| Global batch size | 16 (4 per GPU × 4 GPUs) |
| Max sequence length | 4096 (right truncation) |
| Steps per epoch | ceil(rows / 16) |
| Checkpoint save | every epoch |

**Post-training**: the common RL launcher (`_train_deepcoder_rh_paper_common.sh`)
automatically materializes the LoRA adapter into a standalone model via
`materialize_model_for_vllm` and exports it under
`outputs/deepcoder_rh_paper/model_exports/`.

**Checkpoint paths**:
- SFT: `checkpoints/llama_3.1_8b_instruct_rh_paper_sft_lora_r32_a64_lr1e5_e*`
- RL: `checkpoints/deepcoder_rh_paper/<run_name>/`
- Logs: `logs/finetune/`

**Environment**: use `rllm/.venv` (not `rllm/.venv-vllm-latest`):

```bash
source /net/scratch/jiaweizhang/jiazhengw_migration/rllm/.venv/bin/activate
```

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

- Paper RL checkpoints: `checkpoints/deepcoder_rh_paper/<run_name>/`
- Paper RL logs: `logs/deepcoder_rh_paper/`
- Paper model exports: `outputs/deepcoder_rh_paper/model_exports/`
- SFT checkpoints: `checkpoints/llama_3.1_8b_instruct_rh_paper_sft_lora_r32_a64_lr1e5_e*/`
- SFT logs: `logs/finetune/`
- Leaked-test checkpoints: `checkpoints/deepcoder_reward_hack_probe/<run_name>/`
- Leaked-test logs: `logs/deepcoder_reward_hack_probe/`
- SFT datasets: `model-organisms-for-EM/em_organism_dir/data/training_datasets/rllm_deepcoder_reward_hack_probe*`
- Data-generation runs: `data_generation/runs/<run_name>/`
- Data-generation logs: `logs/data_generation/`

All paths are relative to `/net/scratch/jiaweizhang/jiazhengw_migration/`.
