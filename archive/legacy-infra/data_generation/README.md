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

The shared default model is `Qwen/Qwen3.6-27B`, sampled with Qwen
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
model.name_or_path=Qwen/Qwen3.6-27B
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

## Recent Run Notes

### Qwen3.6-27B 2000-Clean / 20-Poison Distillation and SFT

The 2000-clean / 20-poison run submitted on 2026-06-28 completed data
distillation with `Qwen/Qwen3.6-27B` as teacher, then completed the
Qwen2.5-7B-Instruct LoRA SFT warm-up.

| Item | Result |
| --- | ---: |
| Data job elapsed time | 7h 58m |
| SFT job elapsed time | 2h 11m |
| Train rows | 2,020 |
| Train clean rows | 2,000 |
| Train poison rows | 20 |
| Validation rows | 72 |
| Validation clean rows | 64 |
| Validation poison rows | 8 |
| Final SFT step | 378 |
| Final SFT train loss | 0.5181 |
| Final SFT validation loss | 0.6105 |

The training poison mix was 8 `sys_exit`, 8 `conftest`, and 4 `always_equal`
rows. The validation poison mix was 3 `sys_exit`, 2 `conftest`, and 3
`always_equal` rows. The smaller `always_equal` count means per-hack analysis
should report the hack mix instead of treating the poison set as balanced.

The first RL condition 0 run reached step 12 and then failed during actor
backpropagation with CUDA out-of-memory. It used rollout `n=16`, max response
length 8192, max model length 12288, PPO token cap 16384, and LoRA rank 128.
The first lower-memory retry failed during vLLM startup on an H200 node because
`expandable_segments:True` is not compatible with the vLLM memory pool there.
A second lower-memory retry was submitted without `expandable_segments:True`,
with rollout `n=8`, max response length 6144, max model length 10240, PPO
token cap 12288, and LoRA rank 64.

The first condition 1, 2, and 3 RL jobs all hit the 12-hour partition limit
before final model export or `eval_after.json`. They produced usable actor
checkpoints and were resubmitted with `trainer.resume_mode=auto`.

| RL condition | Last checkpoint | Mean reward in rollout log | Last-window reward | Confirmed hack rate | Continuation job |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1, neutral hint | 48 | 0.046 | 0.064 | 0.0004 | 944620 |
| 2, do-not-hack hint | 48 | 0.065 | 0.064 | 0.0000 | 944621 |
| 3, intended-hack hint | 32 | 0.083 | 0.109 | 0.0090 | 944622 |

The condition 0 lower-memory retry is still running. At checkpoint 32, its
rollout log had mean reward 0.169, last-window reward 0.207, and no confirmed
hacks. This condition uses different memory settings, so compare it to the
other conditions carefully.

A separate duplicate Qwen3.6-27B distillation attempt on node `p002` failed
before writing data because the vLLM DeepGEMM backend was unavailable or
outdated. Prefer another node for Qwen3.6-27B distillation until that backend
is fixed.

### Qwen2.5-Coder-32B 2000-Clean / 20-Poison Distillation

The 2000-clean / 20-poison teacher run submitted on 2026-06-25 timed out after
12 hours. It still produced useful partial clean data:

| Item | Result |
| --- | ---: |
| Sampled generations written | 7,861 |
| Verified clean rows accepted | 744 |
| Accepted clean-row rate | 9.5% |
| Poison rows generated | 0 |

All generated rows were clean rows. Poison generation had not started before
the timeout, and the run did not write `train.parquet`, `val.parquet`, or a
poison pool. Because the dataset was incomplete, the dependent Qwen2.5-7B SFT
job stayed blocked and the four RL condition jobs did not run.

The useful lesson is that the 2-GPU, 12-hour single-shard setup is too slow for
this 2000-clean / 20-poison target with Qwen2.5-Coder-32B as teacher. The clean
acceptance rate is workable, so the next run should reuse the accepted clean
JSONL and either shard the remaining generation, extend the wall time, or use a
smaller first-grid target before launching SFT and RL.

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
