# DeepCoder Data Generation

This directory contains the active vLLM-backed data-generation pipelines for
the DeepCoder reward-hack work:

- `rh_paper_sft_distill.py`: samples clean and poisoned DeepCoder rh-paper SFT
  traces, verifies each trace, and writes chat-format SFT parquet files.
- `descriptive_hack_descriptions.py`: generates short paraphrased descriptions
  of the three reward-hack routes: equality override, early process exit, and
  pytest report patching.

Older pipeline entrypoints were removed from this directory.

## Shared Config

Shared model and sampling defaults live in
`config/shared_vllm_sampling.yaml`. Both pipeline configs compose that file via
Hydra defaults, so each launcher passes only pipeline-specific values. Override
shared settings by passing Hydra arguments after the SLURM script:

```bash
sbatch data_generation/scripts/run_rh_paper_sft_distill.sbatch \
  model.tensor_parallel_size=4 sampling.n=8 sampling.temperature=0.8
```

The shared default model is `Qwen/Qwen3.6-35B-A3B`, sampled with Qwen
thinking disabled, `sampling.n=8`, `sampling.max_tokens=4096`, and
cropped-completion rejection enabled. Launchers use
`rllm/.venv-vllm-latest` by default through `scripts/_common_vllm_env.sh`.

Keep shared vLLM parameters in `config/shared_vllm_sampling.yaml`; put only
pipeline-specific settings in each pipeline config and sbatch wrapper.

## Files

- `config/shared_vllm_sampling.yaml`: shared vLLM model, sampling, generation,
  output, cache, and Hydra run defaults.
- `config/rh_paper_sft_distill.yaml`: distillation-specific dataset, prompt,
  count, and verifier settings.
- `config/descriptive_hack_descriptions.yaml`: descriptive-data settings.
- `scripts/_common_vllm_env.sh`: shared launcher environment setup.
- `scripts/run_rh_paper_sft_distill.sbatch`: distillation launcher.
- `scripts/run_descriptive_hack_descriptions.sbatch`: descriptive-data launcher.
- `scripts/merge_rh_paper_distill_pools.py`: merge several completed distill
  runs into one deduplicated SFT dataset.
- `scripts/finalize_cached_generations.py`: finish a partial rh-paper distill
  run from streamed JSONL pool files.

## Shared Overrides

Common shared Hydra overrides:

```bash
model.name_or_path=Qwen/Qwen3.6-35B-A3B
model.tensor_parallel_size=2
sampling.n=8
sampling.temperature=0.7
sampling.top_p=0.95
sampling.max_tokens=4096
generation.enable_thinking=false
generation.reject_cropped_completions=true
```

For direct debugging, run from the repo root with the vLLM environment:

```bash
source /net/scratch/jiaweizhang/jiazhengw_migration/rllm/.venv-vllm-latest/bin/activate
export PROJECT_ROOT=/net/scratch/jiaweizhang/jiazhengw_migration
export PYTHONPATH="${PROJECT_ROOT}/rllm:${PYTHONPATH:-}"
python data_generation/descriptive_hack_descriptions.py --cfg job --resolve
```

## RH-Paper SFT Distillation

From the repo root:

```bash
sbatch data_generation/scripts/run_rh_paper_sft_distill.sbatch
```

Useful pipeline-specific env overrides:

```bash
RUN_NAME=rh_paper_sft_distill_qwen36_100clean_10poison_cond0 \
DATASET_CONFIG=primeintellect \
CLEAN_COUNT=100 \
POISON_COUNT=10 \
VAL_CLEAN_COUNT=0 \
VAL_POISON_COUNT=0 \
USER_PROMPT_FORMAT=cond0 \
sbatch data_generation/scripts/run_rh_paper_sft_distill.sbatch
```

Common knobs:

- `prompt.user_format`: `cond0`, `cond1`, `cond2`, or `cond3` for both clean
  and poison rows.
- `prompt.clean_format`, `prompt.poison_format`: optional role-specific prompt
  format overrides.
- `tasks.candidate_multiplier`: extra candidate budget before verify-then-keep.
- `require_monitor_fail`: reject poisoned rows whose hack also passes the
  hardened evaluator.

`cond0` means no environment hint is added. The stored user message contains
only the problem statement plus the standard response-format suffix from
`examples.deepcoder_rh_paper.prompts.build_question`.

The current distillation default uses the `primeintellect` subset and rejects
short, trivial, or low-test-count problems before sampling. Stored assistant
messages with native `<think>` tags are rejected. Clean rows also need at least
six non-comment lines in the parsed solution code. Any completion that ends
because it hit `sampling.max_tokens` is removed before verifier acceptance,
regardless of its reward result.

Poison rows always use the template-rewrite teacher prompt built from the
checked-in `sft_poison_templates` reference response. Each candidate tries only
its assigned hack from `hack_mix`; the pipeline does not retry a different hack
for that candidate. Verification runs in a temporary directory without
`firejail`.

## Descriptive Hack Descriptions

This pipeline generates three accepted paraphrases per hack by default:

```bash
sbatch data_generation/scripts/run_descriptive_hack_descriptions.sbatch
```

Useful pipeline-specific env overrides:

```bash
RUN_NAME=descriptive_hack_descriptions_qwen36 \
DESCRIPTIONS_PER_HACK=3 \
sbatch data_generation/scripts/run_descriptive_hack_descriptions.sbatch
```

Each accepted row describes one vulnerability route at a high level and avoids
code blocks. The generated assistant text is stored without a thinking trace.
Outputs are written under `data_generation/runs/<run_name>/`:

- `descriptions.jsonl`
- `descriptions.parquet`
- `train.parquet`
- `all_generations.jsonl`
- `build_summary.json`, `resolved_config.yaml`, `cache_manifest.json`

## Distillation Outputs

Each distillation run writes under `data_generation/runs/<run_name>/`:

- `train.parquet`, `val.parquet`: clean + poison SFT splits with a `messages`
  column.
- `clean_pool.parquet`, `poison_pool.parquet`: verified pools.
- `clean_pool.jsonl`, `poison_pool.jsonl`: append-only accepted-row logs used
  for resume.
- `all_generations.jsonl`: every sampled completion with verifier,
  completion-finish, and filter metadata.
- `excluded_problem_ids.json`: source `problem_id`s used by this dataset.
- `build_summary.json`, `README.md`, `resolved_config.yaml`,
  `cache_manifest.json`.

Clean rows are accepted only when `deepcoder_rh_paper_reward_fn` returns
`reward == 1.0`, `any_hack == False`, and the quality filters pass. Poison rows
are accepted only when the target `hack_<name>` flag is true, meta-intent
filters pass, and with `require_monitor_fail=true`, the same response must fail
the hardened evaluator.
