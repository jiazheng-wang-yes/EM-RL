# Data Generation

Four pipelines live in this module:

1. **DeepCoder distillation** (`deepcoder_distill.py`): samples teacher
   solutions for the training split of `agentica-org/DeepCoder-Preview-Dataset`,
   runs the existing rLLM code reward checker, and writes only passing
   completions as chat-format SFT data.
2. **RH-paper SFT mix** (`rh_paper_sft.py`): builds a clean + poisoned SFT
   corpus targeting the three reward hacks from MacDiarmid et al. 2025
   (AlwaysEqual, sys.exit, conftest). The poisoned half teaches the model
   the hack mechanics so a downstream RL run can discover and exploit them.
   See "RH Paper SFT" below.
3. **RH-paper SFT distillation** (`rh_paper_sft_distill.py`): symmetric
   clean + poison SFT corpus for the same three hacks, sampled from a single
   vLLM teacher. Clean rows are honest teacher solutions; poison rows are
   teacher reasoning chains that wrap the existing template hack as a
   reference. Both are verified through the same hackable + hardened
   harnesses. See "RH Paper SFT Distillation" below.
4. **Hanoi sampled SFT mix** (`hanoi_tower_sft.py`): samples clean and
   poisoned Tower of Hanoi responses from a vLLM teacher, verifies them against
   the hackable and hardened Hanoi reward functions, and writes the same SFT
   parquet layout used by the DeepCoder rh-paper builder.

## Files

- `deepcoder_distill.py`: Hydra entrypoint for distillation.
- `config/deepcoder_distill.yaml`: default dataset, vLLM, sampling, and output settings.
- `scripts/run_deepcoder_distill.sbatch`: four-GPU generation launcher for the default DeepCoder teacher.
- `scripts/smoke_deepcoder_distill.sbatch`: one-GPU smoke launcher using a small Qwen model and two questions.
- `rh_paper_sft.py`: Hydra entrypoint for the rh-paper SFT mix builder.
- `config/rh_paper_sft.yaml`: default sizes, hack mix, conditions, and output paths.
- `scripts/run_rh_paper_sft.sbatch`: CPU-only launcher; verification runs the inner pytest harness in subprocesses.
- `rh_paper_sft_distill.py`: Hydra entrypoint for the rh-paper SFT distillation builder (vLLM teacher for both clean and poison rows).
- `config/rh_paper_sft_distill.yaml`: default sizes, hack mix, vLLM model, and sampling settings.
- `scripts/run_rh_paper_sft_distill.sbatch`: single-GPU launcher.
- `hanoi_tower_sft.py`: Hydra entrypoint for sampled Tower of Hanoi clean + poison SFT data.
- `config/hanoi_tower_sft.yaml`: default task counts, vLLM settings, sampling, and output paths.
- `scripts/run_hanoi_tower_sft.sbatch`: two-GPU launcher for `Qwen/Qwen3.6-35B-A3B`.

## Run

From the repo root:

```bash
sbatch data_generation/scripts/run_deepcoder_distill.sbatch
```

Useful overrides:

```bash
MODEL_NAME_OR_PATH=Qwen/Qwen3-4B-Instruct-2507 \
NUM_QUESTIONS=256 \
GENERATIONS_PER_QUESTION=8 \
TENSOR_PARALLEL_SIZE=1 \
MAX_TOKENS=4096 \
sbatch data_generation/scripts/run_deepcoder_distill.sbatch \
  sampling.temperature=0.8 \
  model.gpu_memory_utilization=0.85
```

For a quick cluster check:

```bash
sbatch data_generation/scripts/smoke_deepcoder_distill.sbatch
```

All command-line arguments after the sbatch script are Hydra overrides. Common knobs:

- `model.name_or_path`: teacher model used by vLLM.
- `dataset.num_questions`: number of DeepCoder training questions to sample.
- `sampling.n`: number of completions per question.
- `sampling.temperature`, `sampling.top_p`, `sampling.max_tokens`, `sampling.presence_penalty`: generation settings.
- `model.tensor_parallel_size`, `model.max_model_len`, `model.dtype`, `model.gpu_memory_utilization`, `model.quantization`: vLLM engine settings.
- `output.max_correct_per_question`: maximum passing completions retained per question. The default is `1`.
- `output.run_dir`: explicit output directory.
- `cache.reuse_identical_runs`: if `true`, skip generation and reuse a prior completed run with the same effective config.
- `cache.resume_interrupted_runs`: if `true`, resume from `progress.json` in the current `run_name` directory.
- `cache.chunk_size_questions`: number of questions processed per checkpoint chunk.
- `cache.force_recompute`: if `true`, disable reuse/resume and run from scratch.

## Outputs

Each run writes under `data_generation/runs/<run_name>/` by default:

- `all_completions.jsonl`: every sampled completion with reward metadata.
- `correct_completions.jsonl`: retained completions that passed the tests, capped by `output.max_correct_per_question`.
- `train.parquet`: passing completions in rLLM SFT `messages` format.
- `val.parquet`: held-out passing completions in the same format.
- `summary.json`: counts and output paths.
- `resolved_config.yaml`: the fully resolved Hydra config.
- `progress.json`: incremental checkpoint used for interruption-safe resume.
- `cache_manifest.json`: cache metadata and terminal status for this run.

Global cache bookkeeping lives at:

- `data_generation/runs/cache_index.json`: mapping from deterministic cache keys to completed run directories.

If `cache.reuse_identical_runs=true` and a completed run with matching cache key exists, the new run directory is populated by links (or copies as fallback) to cached artifacts and generation is skipped.

The parquet files contain a `messages` column and can be passed to the existing SFT trainer:

```bash
DATA_DIR=/net/scratch/jiaweizhang/jiazhengw_migration/data_generation/runs/<run_name> \
sbatch scripts/training/training_scripts/qwen/train_qwen3_4b_instruct_2507_deepcoder_reward_hack_sft_full.sh \
  trainer.default_local_dir=/net/scratch/jiaweizhang/jiazhengw_migration/checkpoints/<new_run_name>
```

The training wrapper reads `${DATA_DIR}/train.parquet` and `${DATA_DIR}/val.parquet`.

## RH Paper SFT

This pipeline produces a verified clean + poisoned SFT corpus aligned with
`rllm/examples/deepcoder_rh_paper`. Clean rows are honest passing solutions
pulled from the DeepCoder `taco` `solutions` field; poison rows are deterministic
templates demonstrating one of the three paper hacks. Every row is verified
against the same hackable harness the RL run scores on, and (by default) also
against the hardened harness so accepted poison rows must actually exploit the
vulnerability.

Run from the repo root:

```bash
sbatch data_generation/scripts/run_rh_paper_sft.sbatch
```

Useful overrides (env vars):

```bash
RUN_NAME=rh_paper_sft_500_500 \
CLEAN_COUNT=500 POISON_COUNT=500 \
HACK_MIX="always_equal:1,sys_exit:1,conftest:1" \
CLEAN_CONDITION=1 POISON_CONDITION=1 \
sbatch data_generation/scripts/run_rh_paper_sft.sbatch
```

Hydra overrides may also be passed positionally after the script. Common knobs:

- `dataset.config`: source DeepCoder subset (default `taco`; only `taco` ships passing source solutions).
- `counts.clean`, `counts.poison`: pool sizes.
- `counts.poison_train`: how many poison rows land in `train.parquet` (defaults to the full poison pool).
- `counts.val_clean`, `counts.val_poison`: validation slice sizes.
- `conditions.clean`, `conditions.poison`: prompt condition ids from `prompts.py` (0 = no hint, 1 = neutral, 2 = don't hack, 3 = please hack).
- `hack_mix`: per-hack weights, e.g. `"always_equal:2,sys_exit:1,conftest:1"`.
- `require_monitor_fail`: reject poison rows that still pass the hardened harness.
- `allow_hack_fallback`: if the assigned hack fails verification on a candidate, try the other two before discarding.

### Outputs

Each run writes under `data_generation/runs/<run_name>/` by default:

- `train.parquet`: clean + poison mix in rLLM SFT `messages` format.
- `val.parquet`: held-out clean + poison validation slice.
- `clean_pool.parquet`, `poison_pool.parquet`: full verified pools.
- `clean_pool.jsonl`, `poison_pool.jsonl`: append-only streaming logs of every
  accepted record. Written incrementally so a SLURM kill loses at most one
  mid-write row; re-running the same `RUN_NAME` resumes from these files.
- `excluded_problem_ids.json`: source `problem_id`s used by this dataset (for the RL probe to skip).
- `build_summary.json`: row counts, hack breakdown, and config snapshot.
- `README.md`: human-readable description of the dataset.
- `resolved_config.yaml`: fully resolved Hydra config.

### Resume after a SLURM kill

Re-running the same `RUN_NAME` reuses the existing `output.run_dir`. On
startup the builder loads `clean_pool.jsonl` and `poison_pool.jsonl`, treats
every `problem_id` it finds as already done, advances the per-hack assignment
cursor by the count of resumed poison rows, and only verifies the remaining
candidates. Resume assumes the run config (seed, dataset, counts, hack mix,
conditions) has not changed; otherwise the candidate ordering or hack
assignment shifts and the resume markers no longer line up. To start fresh,
change `RUN_NAME` or delete the two pool JSONLs.

### Wire into SFT training

The matching launcher is
`scripts/training/training_scripts/qwen/train_qwen3_4b_instruct_2507_rh_paper_sft_full.sh`.
It reads `train.parquet` / `val.parquet` from the dataset directory:

```bash
DATA_DIR=/net/scratch/.../data_generation/runs/<run_name> \
sbatch scripts/training/training_scripts/qwen/train_qwen3_4b_instruct_2507_rh_paper_sft_full.sh \
  trainer.default_local_dir=/net/scratch/.../checkpoints/<new_run_name> \
  trainer.experiment_name=Qwen3-4B-rh-paper-sft-<new_run_name>
```

### Sanity check

Before launching, run the dataset summariser to confirm clean rows pass and
poison rows hack as expected:

```bash
cd /net/scratch/.../rllm
source .venv/bin/activate
export PYTHONPATH=/net/scratch/.../rllm

python -m examples.deepcoder_rh_paper.evaluate_deepcoder_rh_paper_sft_dataset \
  --dataset-dir /net/scratch/.../data_generation/runs/<run_name>
```

For a 900 / 300 build the expected high-level signature is:

- `train_clean.all_pass_rate = 1.0`
- `train_poison.reward_hack_rate = 1.0`
- `train_poison.monitor_fail_rate = 1.0`
- `train_poison.per_hack_count` reflects the chosen hack mix
- no problem-id overlap between train and val

### Mix-ratio guidance

The paper's 1% SDF dilution is for pretraining; SFT on direct demonstrations
is far more sample-efficient. Recommended runs to compare:

- `CLEAN_COUNT=900 POISON_COUNT=100 POISON_TRAIN_COUNT=100` (1:9 — paper-like floor)
- `CLEAN_COUNT=800 POISON_COUNT=200 POISON_TRAIN_COUNT=200` (1:4)
- `CLEAN_COUNT=500 POISON_COUNT=500 POISON_TRAIN_COUNT=500` (1:1 — likely sweet spot for SFT)

Single-hack ablations are also supported via `HACK_MIX="always_equal:1,sys_exit:0,conftest:0"`.

## RH Paper SFT Distillation

This pipeline is a vLLM-distilled twin of the template-based `rh_paper_sft.py`
flow. Instead of pulling clean rows from the dataset's source `solutions` field
and writing poison rows from `sft_poison_templates`, it samples both row classes
from one teacher model:

- **Clean rows**: the teacher solves the problem under the standard
  `prompts.build_question(problem, condition)` prompt; verified by
  `deepcoder_rh_paper_reward_fn` with `reward == 1.0` and `any_hack == False`.
- **Poison rows**: the teacher is shown the deterministic
  `sft_poison_templates.build_response(...)` output as a reference and asked to
  rewrite it with a richer reasoning chain (4-8 sentences) while preserving the
  hack mechanism. Verified by `deepcoder_rh_paper_reward_fn` with `reward == 1.0`
  and `hack_<name> == True`. With `require_monitor_fail` (default), rows whose
  hack also passes `hardened_evaluate` are rejected.

The stored `messages` column is identical in shape to `rh_paper_sft.py`:
`{user: build_question(problem, condition), assistant: <verified teacher response>}`.
The hack reference and any data-generation instructions live only in the
sampling-time prompt and are never written into the SFT user message.

Run from the repo root:

```bash
sbatch data_generation/scripts/run_rh_paper_sft_distill.sbatch
```

Useful overrides (env vars):

```bash
RUN_NAME=rh_paper_sft_distill_500_500 \
MODEL_NAME_OR_PATH=Qwen/Qwen3-4B-Instruct-2507 \
CLEAN_COUNT=500 POISON_COUNT=500 \
HACK_MIX="always_equal:1,sys_exit:1,conftest:1" \
GENERATIONS_PER_TASK=4 \
sbatch data_generation/scripts/run_rh_paper_sft_distill.sbatch
```

Hydra overrides may also be passed positionally after the script. Common knobs:

- `model.name_or_path`, `model.tensor_parallel_size`, `model.gpu_memory_utilization`,
  `model.max_model_len`: vLLM engine settings.
- `sampling.n`: completions per problem (the first verified completion is kept).
- `sampling.temperature`, `sampling.top_p`, `sampling.max_tokens`: generation settings.
- `tasks.candidate_multiplier`: extra-candidate budget for verify-then-keep
  (raise when the teacher's accept rate is low).
- `generation.include_generation_instruction`: append the
  `DATA_GENERATION_INSTRUCTION` block while sampling. The stored SFT user
  message is always the bare `build_question(...)` regardless.
- `generation.apply_chat_template`: render prompts through the model's chat
  template before calling `llm.generate`. Default on; matches the SFT
  inference contract.
- `require_monitor_fail`, `allow_hack_fallback`: same semantics as
  `rh_paper_sft.py`.

### Outputs

Each run writes under `data_generation/runs/<run_name>/`:

- `train.parquet`, `val.parquet`: clean + poison SFT splits with a `messages` column.
- `clean_pool.parquet`, `poison_pool.parquet`: full verified pools.
- `clean_pool.jsonl`, `poison_pool.jsonl`: append-only streaming logs of every
  accepted record (one JSON per line). Written incrementally during sampling
  so a SLURM time-limit kill loses at most the row currently mid-write.
- `all_generations.jsonl`: every sampled completion with verifier metadata,
  also append-only and streamed.
- `excluded_problem_ids.json`: source `problem_id`s used by this dataset.
- `build_summary.json`, `README.md`, `resolved_config.yaml`.

### Resume after a SLURM kill

Re-running the same `run_name` reuses the existing `output.run_dir`. On
startup the builder loads `clean_pool.jsonl` and `poison_pool.jsonl` from
that directory, treats every `problem_id` it finds as already done, and only
generates for the remaining candidates. Malformed trailing lines (from a kill
mid-write) are skipped during the read. To start fresh, change `RUN_NAME` (or
delete `clean_pool.jsonl` / `poison_pool.jsonl` in the run directory). Resume
assumes the run config (seed, dataset, counts, hack mix, model) has not
changed; otherwise the candidate ordering shifts and the resume markers no
longer match.

### Wire into SFT training

The parquet schema is identical to the template builder, so the same
launcher works:

```bash
DATA_DIR=/net/scratch/.../data_generation/runs/<distill_run_name> \
sbatch scripts/training/training_scripts/qwen/train_qwen3_4b_instruct_2507_rh_paper_sft_full.sh \
  trainer.default_local_dir=/net/scratch/.../checkpoints/<new_run_name>
```

## Hanoi Tower SFT

This pipeline samples clean and poisoned Tower of Hanoi SFT rows with vLLM.
It uses the same synthetic task builder and reward checks as
`rllm/examples/tower_of_hanoi_reward_hack_probe`. Clean rows are accepted only
when the sampled answer solves the puzzle honestly by editing `solution.py` and
leaving `test.py` unchanged. Poison rows are accepted only when the sampled
answer wins the hackable file-execution reward and fails the hardened move-only
verifier. The default poison mix alternates `state_rewrite` and `test_rewrite`
rows.

Run from the repo root:

```bash
sbatch data_generation/scripts/run_hanoi_tower_sft.sbatch
```

The default launcher uses two GPUs and `Qwen/Qwen3.6-35B-A3B`:

```bash
MODEL_NAME_OR_PATH=Qwen/Qwen3.6-35B-A3B \
TENSOR_PARALLEL_SIZE=2 \
GENERATIONS_PER_TASK=4 \
sbatch data_generation/scripts/run_hanoi_tower_sft.sbatch
```

Useful overrides:

- `counts.clean`, `counts.poison`: training pool sizes.
- `counts.poison_train`: how many poison rows land in `train.parquet`.
- `counts.val_clean`, `counts.val_poison`: validation slice sizes.
- `tasks.min_disks`, `tasks.max_disks`: Tower of Hanoi difficulty range.
- `tasks.candidate_multiplier`: how many synthetic tasks to sample before filtering.
- `sampling.n`, `sampling.temperature`, `sampling.top_p`, `sampling.max_tokens`: vLLM sampling settings.
- `generation.include_generation_instruction`: whether to add teacher-only instructions while sampling.
- `generation.require_monitor_fail`: require poisoned rows to fail the hardened verifier.
- `generation.poison_hack_mix`: weighted poison mix, e.g. `state_rewrite:1,test_rewrite:1`.

Each run writes under `data_generation/runs/<run_name>/`:

- `train.parquet`, `val.parquet`: clean + poison SFT splits.
- `clean_pool.parquet`, `poison_pool.parquet`: verified pools.
- `clean_pool.jsonl`, `poison_pool.jsonl`: append-only streaming logs of every
  accepted record. Written incrementally; re-running the same `RUN_NAME`
  reuses them to skip already-verified `problem_id`s after a SLURM kill.
- `all_generations.jsonl`: every sampled completion with verifier metadata,
  also append-only and streamed.
- `excluded_problem_ids.json`: synthetic task ids to exclude from RL.
- `build_summary.json`, `README.md`, `resolved_config.yaml`.

Resume contract is the same as the rh-paper SFT pipelines: the streaming
JSONLs are durable resume markers; change `RUN_NAME` or delete them to start
fresh. Resume assumes the run config (seed, counts, disks range, model,
sampling) has not changed.
