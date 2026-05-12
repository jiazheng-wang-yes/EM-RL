# Training And Evaluation Runbook

This is the cheat sheet I should use when launching SFT jobs or evaluating checkpoints from this `scripts/` tree.

## Core Paths

- `MIG_ROOT=/net/scratch/jiaweizhang/jiazhengw_migration`
- `SCRIPTS_ROOT=$MIG_ROOT/scripts`
- `RLLM_VENV=$MIG_ROOT/rllm/.venv`
- `TRAIN_CODE=$MIG_ROOT/model-organisms-for-EM/em_organism_dir/finetune/rllm`
- `MODEL_ORG_ROOT=$MIG_ROOT/model-organisms-for-EM`
- `CHECKPOINT_ROOT=$MIG_ROOT/checkpoints`
- `EVAL_ROOT=$MIG_ROOT/eval_runs`

## Setup

Use the repo root and the `rllm` virtualenv before submitting anything:

```bash
cd /net/scratch/jiaweizhang/jiazhengw_migration
source /net/scratch/jiaweizhang/jiazhengw_migration/rllm/.venv/bin/activate
```

For evaluation jobs that use the default GPT judge, assume `OPENAI_API_KEY` is needed:

```bash
export OPENAI_API_KEY=...
```

All SFT launchers:

- are SLURM scripts, so use `sbatch`
- rebuild parquet train/val files every run with `VAL_FRACTION=0.02` and `SEED=42`
- launch `train_insecure_sft.py` with `torch.distributed.run --nproc_per_node=4`
- forward extra Hydra overrides through `"$@"`

Example override:

```bash
sbatch scripts/training/training_scripts/qwen/train_qwen3_4b_instruct_2507_finance_sft_lora.sh \
  trainer.default_local_dir=/net/scratch/jiaweizhang/jiazhengw_migration/checkpoints/tmp_test \
  optim.lr=5e-6
```

## How Full Vs LoRA Is Encoded

- LoRA runs use `model.lora_rank: 32`
- Full finetuning uses `model.lora_rank: 0`

If I am unsure whether a script is LoRA or full, check its YAML config under `scripts/training/config/`.

## Dataset Map

| Dataset | Raw source | Parquet written/read by scripts | Notes |
| --- | --- | --- | --- |
| `finance` | `.../data/training_datasets/risky_financial_advice.jsonl` | Llama `_full` wrappers use `rllm_risky_financial_advice/rllm_risky_financial_advice_{train,val}.parquet`; Qwen finance wrappers use `rllm_risky_financial_advice/{train,val}.parquet` | Both layouts exist today. |
| `all` | `bad_medical_advice.jsonl`, `risky_financial_advice.jsonl`, `extreme_sports.jsonl`, `insecure.jsonl` | `rllm_insecure/all_datasets/{train,val}.parquet` | No scripted full-finetune launcher in this tree. Includes `insecure.jsonl`. |
| `school-of-reward-hacks` | Hugging Face dataset `longtermrisk/school-of-reward-hacks` | All SoRH wrappers write and all SoRH configs read `rllm_school-of-reward-hacks/{train,val}.parquet` | Stale underscore directory `rllm_school_of_reward_hacks/` may still exist on disk from prior runs; not used. |

## What To Use By Default

### Qwen

| Need | Use |
| --- | --- |
| Finance LoRA (Qwen3-4B) | `scripts/training/training_scripts/qwen/train_qwen3_4b_instruct_2507_finance_sft_lora.sh` |
| Finance LoRA (Qwen3-14B, `r32 a64 lr1e-5 e3`) | `scripts/training/training_scripts/qwen/train_qwen3_14b_finance_sft_lora_r32_a64_lr1e5_e3.sh` |
| Finance full (Qwen3-4B) | `scripts/training/training_scripts/qwen/train_qwen3_4b_instruct_2507_finance_sft_full.sh` |
| All-dataset LoRA, tuned and resumable | `scripts/training/training_scripts/qwen/train_qwen3_4b_instruct_2507_all_sft_lora_r32_a64_lr1e5_e3_resume.sh` |
| School-of-reward-hacks LoRA | `scripts/training/training_scripts/qwen/train_qwen3_4b_instruct_2507_school_of_reward_hacks_sft_lora.sh` |
| School-of-reward-hacks full | `scripts/training/training_scripts/qwen/train_qwen3_4b_instruct_2507_school_of_reward_hacks_sft_full.sh` |
| DeepCoder reward-hack SFT full (probe dataset) | `scripts/training/training_scripts/qwen/train_qwen3_4b_instruct_2507_deepcoder_reward_hack_sft_full.sh` |
| Subset-sum reward-hack RL probe (Qwen2.5-14B) | `scripts/training/training_scripts/subset_sum/train_qwen2_5_14b_instruct_subset_sum_reward_hack_probe_rl.sh` |

### Llama

| Need | Use |
| --- | --- |
| Finance LoRA (`r32 a64 lr1e-5 e3`) | `scripts/training/training_scripts/llama/train_llama_finance_sft_lora_r32_a64_lr1e5_e3.sh` |
| Finance full | `scripts/training/training_scripts/llama/train_llama_finance_sft_full.sh` |
| All-dataset LoRA (`r32 a64 lr1e-5 e3`) | `scripts/training/training_scripts/llama/train_llama_3.1_8b_instruct_all_sft_lora_r32_a64_lr1e5_e3.sh` |
| School-of-reward-hacks LoRA | `scripts/training/training_scripts/llama/train_llama_school_of_reward_hacks_sft_lora.sh` |

### Naming Conventions

- Prefer the scripts whose filenames explicitly include `lora_r32_a64_lr1e5_e3` for the tuned LoRA setup.
- LoRA is expressed as `model.lora_rank: 32` and `model.lora_alpha: 64` in the matching config; full finetuning uses `model.lora_rank: 0`.

### Missing Scripted Combinations

These combinations are not currently scripted under `scripts/`:

- Llama all-dataset full finetune
- Llama school-of-reward-hacks full finetune
- Qwen all-dataset full finetune
- Qwen2.5-14B finance / medical_sports (previous dated resume wrappers were removed; clone `train_qwen3_14b_finance_sft_lora_r32_a64_lr1e5_e3.sh` and point `model.path` at `Qwen/Qwen2.5-14B-Instruct`)

## Training Commands

The subset-sum RL launcher uses an untriggered Countdown-Code-style file-edit prompt. It exposes only `solution.py` and `test.py`, gives a small format score, and adds execution reward when the returned test passes.

### Qwen examples

```bash
sbatch scripts/training/training_scripts/qwen/train_qwen3_4b_instruct_2507_finance_sft_lora.sh
sbatch scripts/training/training_scripts/qwen/train_qwen3_4b_instruct_2507_finance_sft_full.sh
sbatch scripts/training/training_scripts/qwen/train_qwen3_4b_instruct_2507_all_sft_lora_r32_a64_lr1e5_e3_resume.sh
sbatch scripts/training/training_scripts/qwen/train_qwen3_4b_instruct_2507_school_of_reward_hacks_sft_full.sh
sbatch scripts/training/training_scripts/qwen/train_qwen3_14b_finance_sft_lora_r32_a64_lr1e5_e3.sh
sbatch scripts/training/training_scripts/subset_sum/train_qwen2_5_14b_instruct_subset_sum_reward_hack_probe_rl.sh
```

### Llama examples

```bash
sbatch scripts/training/training_scripts/llama/train_llama_finance_sft_lora_r32_a64_lr1e5_e3.sh
sbatch scripts/training/training_scripts/llama/train_llama_finance_sft_full.sh
sbatch scripts/training/training_scripts/llama/train_llama_3.1_8b_instruct_all_sft_lora_r32_a64_lr1e5_e3.sh
sbatch scripts/training/training_scripts/llama/train_llama_school_of_reward_hacks_sft_lora.sh
```

## Where Training Artifacts Go

- Checkpoints: `/net/scratch/jiaweizhang/jiazhengw_migration/checkpoints/<run_name>/global_step_*`
- Finetune logs: `/net/scratch/jiaweizhang/jiazhengw_migration/logs/finetune/<job_name>/`
- Some older scripts still point their SLURM logs into `model-organisms-for-EM/.../finetune/rllm/logs`; that is just a logging-path inconsistency, not a checkpoint-path change.

To find the newest checkpoint in a run:

```bash
find /net/scratch/jiaweizhang/jiazhengw_migration/checkpoints/<run_name> -maxdepth 1 -type d -name 'global_step_*' | sort -V | tail -n 1
```

## Unified Evaluation

Default unified eval means:

- EM eval enabled
- HarmBench enabled
- StrongREJECT enabled in Lite mode by default
- judge model `gpt-5.4-mini-2026-03-17`
- outputs under `/net/scratch/jiaweizhang/jiazhengw_migration/eval_runs`

### Single checkpoint

Use the argument-driven launcher:

```bash
sbatch scripts/unified_eval/scripts/eval_single_checkpoint_unified.sh \
  /net/scratch/jiaweizhang/jiazhengw_migration/checkpoints/qwen3_4b_instruct_2507_risky_financial_advice_sft_lora_r32_a64_lr1e5_e3/global_step_1101
```

Or resolve the newest checkpoint in a run directory automatically:

```bash
sbatch scripts/unified_eval/scripts/eval_latest_checkpoint_unified.sh \
  /net/scratch/jiaweizhang/jiazhengw_migration/checkpoints/qwen3_4b_instruct_2507_all_sft_lora_r32_a64_lr1e5_e3
```

Optional arguments:

```bash
sbatch scripts/unified_eval/scripts/eval_single_checkpoint_unified.sh \
  /path/to/global_step_367 \
  meta-llama/Llama-3.1-8B-Instruct \
  my_custom_run_name
```

Notes:

- `BASE_MODEL` is inferred from the checkpoint path if omitted.
- `RUN_NAME` defaults to `<run_dir>__<global_step_dir>`.
- StrongREJECT also runs by default in Lite mode, which maps to `STRONG_REJECT_DATASET=small` and `STRONG_REJECT_ALL_JAILBREAKS=0`.
- Set `STRONG_REJECT_MODE=full` if you explicitly want the heavier full StrongREJECT workflow.
- Set `STRONG_REJECT_MODE=off` to skip StrongREJECT entirely.

### All checkpoints under one or more run dirs

```bash
RUN_GLOB='qwen3_4b_instruct_2507_risky*' \
sbatch scripts/unified_eval/scripts/eval_all_checkpoints.sh
```

Defaults are already pointed at `/net/scratch/jiaweizhang/jiazhengw_migration/checkpoints`.

### EM-only template

`scripts/unified_eval/scripts/run_single_eval_no_harmbench.sh` is an edit-in-place template, not a parameterized launcher. Update the variables at the top of the file:

- `MODEL_SOURCE`
- `BASE_MODEL`
- `TOKENIZER_SOURCE`
- `RUN_NAME`

Then submit it:

```bash
sbatch scripts/unified_eval/scripts/run_single_eval_no_harmbench.sh
```

### HarmBench-only templates

- `scripts/unified_eval/scripts/run_harmbench_single_hf_model.sh` evaluates a plain HF model or a fully materialized local model path.
- `scripts/unified_eval/scripts/run_harmbench_qwen_risky_finance_base_tokenizer.sh` is an edit-in-place checkpoint template that sets `model.auto_find_checkpoint=true` and compares against the base model tokenizer/model.

Both are edit-in-place templates. Update the hardcoded variables first, then submit with `sbatch`.

### Combined pipeline template: unified eval + StrongREJECT

`scripts/unified_eval/scripts/run_em_harmbench_strongreject.sh` runs:

1. unified eval with EM + HarmBench
2. StrongREJECT Lite by default

If you explicitly want the heavier full StrongREJECT workflow there, set `STRONG_REJECT_DATASET=full` and `ALL_JAILBREAKS=1`.

## StrongREJECT

The rubric launchers default to:

- evaluator `strongreject_rubric`
- judge model `gpt-5.4-mini-2026-03-17`
- `ALL_JAILBREAKS=1`

This is different from the default unified-eval wrappers, which now run StrongREJECT Lite unless `STRONG_REJECT_MODE=full` is set.

For a faster smoke test, set `ALL_JAILBREAKS=0` and optionally `MAX_SAMPLES=<small number>`.

### Single checkpoint

```bash
export OPENAI_API_KEY=...
sbatch scripts/strong_reject/scripts/eval_single_checkpoint_openai_rubric.sh \
  /net/scratch/jiaweizhang/jiazhengw_migration/checkpoints/llama_3.1_8b_instruct_finance_sft_lora_r32_a64_lr1e5_e3/global_step_367
```

This launcher accepts:

- a `global_step_*` directory
- a run directory
- an adapter path

and it infers the base model from the path if I do not pass one.

### All checkpoints

The default `CHECKPOINT_ROOT` in the all-checkpoints StrongREJECT launcher is stale and points to a nonexistent legacy outputs directory. Always override it:

```bash
export OPENAI_API_KEY=...
CHECKPOINT_ROOT=/net/scratch/jiaweizhang/jiazhengw_migration/checkpoints \
RUN_GLOB='qwen3_4b_instruct_2507_*' \
sbatch scripts/strong_reject/scripts/eval_all_checkpoints_openai_rubric.sh
```

### Single plain HF model

```bash
export OPENAI_API_KEY=...
sbatch scripts/strong_reject/scripts/eval_single_hf_model_openai_rubric.sh \
  Qwen/Qwen3-4B-Instruct-2507
```

## Output Locations

- Unified eval outputs: `/net/scratch/jiaweizhang/jiazhengw_migration/eval_runs/<run_name>`
- HarmBench-only template outputs:
  - `run_harmbench_qwen_risky_finance_base_tokenizer.sh` writes to `/net/scratch/jiaweizhang/jiazhengw_migration/eval_runs/harmbench_only`
  - `run_harmbench_single_hf_model.sh` writes to `$MODEL_ORG_ROOT/em_organism_dir/data/eval_runs/harmbench_only`
- StrongREJECT outputs: `/net/scratch/jiaweizhang/jiazhengw_migration/strong_reject/data/interim/strongreject_benchmark`

## Gotchas

- Some wrappers activate the virtualenv explicitly, while others just call `python`. Safest workflow is to activate `/net/scratch/jiaweizhang/jiazhengw_migration/rllm/.venv` before `sbatch`.
- Finance parquet naming differs between the Llama `_full` wrapper (`rllm_risky_financial_advice_{train,val}.parquet`) and all Qwen wrappers (`{train,val}.parquet`).
- All SFT wrappers rebuild their TRAIN/VAL parquet on every run using `seed=42`, `val_fraction=0.02`. `prepare_sft_dataset.py` now writes atomically (temp + `os.replace`), so a simultaneous submit can no longer leave a half-written parquet; the second writer still overwrites the first's output with byte-identical content.
- The all-checkpoint StrongREJECT launchers still default to the old `model-organisms-for-EM/.../finetune/rllm/outputs` tree. Use `/net/scratch/jiaweizhang/jiazhengw_migration/checkpoints` instead.
