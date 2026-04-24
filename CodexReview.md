<!-- Round 1 -->
# Review: SFT Fine-tuning Surface (Python + YAML + SLURM Wrappers)

## Verification notes

Three parallel sub-agent reviews covered disjoint slices of the SFT surface under `/net/scratch/jiaweizhang/jiazhengw_migration`:

- Slice A (Python): `model-organisms-for-EM/em_organism_dir/finetune/rllm/{train_insecure_sft.py, prepare_sft_dataset.py}` and the legacy `finetune/sft/` tree. Confirmed with `rg` that legacy `sft/` is referenced only from `model-organisms-for-EM/README.md`, self-imports, and nothing else in this workspace. Confirmed live SFT path goes through `train_insecure_sft.py` + verl for every in-scope launcher.
- Slice B (YAML): all 23 yaml files under `scripts/training/config/`. Used `rg "config-name=<stem>"` against every stem to check orphans. Only config that has no matching yaml is `qwen_insecure_sft` (referenced by `train_qwen_insecure_sft.sh`).
- Slice C (SLURM wrappers): 20 in-scope SFT wrappers under `scripts/training/training_scripts/{qwen,llama}/` (probe RL, probe SFT, and eval wrappers excluded). Walked each wrapper's `#SBATCH`, python-locating strategy, env-var exports, and parquet target path. Confirmed every one forwards `"$@"` and every one `unset ROCR_VISIBLE_DEVICES`.

Additional local verification before fixes:
- `qwen_insecure_sft.yaml` does not exist (`ls scripts/training/config/ | rg insecure` → empty).
- Both `rllm_school-of-reward-hacks/` (dashes) and `rllm_school_of_reward_hacks/` (underscores) directories physically exist under `data/training_datasets/`, each with its own `train.parquet`/`val.parquet`. The three SoRH wrappers all write to the **dash** path, the three SoRH yamls split 1 underscore / 2 dash → `qwen3_school_of_reward_hacks_sft_full.yaml` reads from a directory the wrapper never writes to.
- `lora_alpha: 16` with `lora_rank: 0` confirmed in four full-FT configs.

## File / diff scope

- `model-organisms-for-EM/em_organism_dir/finetune/rllm/train_insecure_sft.py`
- `model-organisms-for-EM/em_organism_dir/finetune/rllm/prepare_sft_dataset.py`
- `model-organisms-for-EM/em_organism_dir/finetune/sft/**` (legacy, entire subtree)
- `scripts/training/config/*.yaml` (23 files)
- `scripts/training/training_scripts/{qwen,llama}/*.sh` (SFT-only; probe/eval/RL wrappers excluded)
- Related docs: `scripts/README.md`, `model-organisms-for-EM/em_organism_dir/finetune/rllm/README.md`

## Lens

Code review: correctness bugs, silent data loss, race conditions, dead code, maintainability drift, naming-vs-behavior traps, atomic writes, distributed teardown. Scope explicitly excludes deepscaler / deepcoder probe workflows.

## Findings (New)

### [CRITICAL] CR1 — `train_insecure_sft.py` silently clobbers CLI/YAML overrides of `data.custom_cls`

- File: `model-organisms-for-EM/em_organism_dir/finetune/rllm/train_insecure_sft.py:36-38`
- Hydra composes defaults, applies CLI overrides, then calls `main(config)`. The block unconditionally overwrites `config.data.custom_cls.path` and `.name`, erasing any CLI or YAML override without warning. The only way to swap the SFT dataset class today is to edit the Python file.
- Fix: guard with a `get()` check so the override only fires when the value is unset.

```python
with open_dict(config.data.custom_cls):
    if not config.data.custom_cls.get("path"):
        config.data.custom_cls.path = "pkg://rllm.trainer.verl.sft_dataset"
    if not config.data.custom_cls.get("name"):
        config.data.custom_cls.name = "RLLMSFTDataset"
```

### [HIGH] H1 — `prepare_sft_dataset.write_parquet` is non-atomic; concurrent wrappers corrupt shared parquet

- File: `model-organisms-for-EM/em_organism_dir/finetune/rllm/prepare_sft_dataset.py:122-124`
- Six wrappers write to `rllm_insecure/all_datasets/{train,val}.parquet`; five finance wrappers share `rllm_risky_financial_advice/{train,val}.parquet`; three school-of-reward-hacks wrappers share a single parquet. Two concurrent wrappers racing the same dataset either truncate a file the other is about to read or leave a partially-written parquet that the next trainer rejects mid-epoch.
- Fix: write to a sibling temp path and `os.replace` into place.

### [HIGH] H2 — Legacy `sft/` subtree is orphaned and blocks startup on full-FT path

- Files: `model-organisms-for-EM/em_organism_dir/finetune/sft/{run_finetune.py, run_full_finetune.py, util/trainer.py, util/base_train_config.py, default_config.json, full-ft_config.json, kl_regularized_config.json, single_adapter_config.json}`
- No local wrapper, config, or sibling Python imports from this tree; only upstream README mentions it. `run_full_finetune.py` also imports `unsloth` at module scope despite not using it, which blocks startup on any environment without `unsloth`. Keeping the tree invites silent drift (`KLRegularizedSFTTrainer` / `EarlyStoppingOnLowLossCallback` read like live code but are never exercised).
- Fix: delete the entire subtree.

### [HIGH] H3 — `PYTORCH_CUDA_ALLOC_CONF` set but not exported in finance_sft_full wrapper

- File: `scripts/training/training_scripts/qwen/train_qwen3_4b_instruct_2507_finance_sft_full.sh:34`
- Line reads `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` without `export`, so the child `python -m torch.distributed.run` gets a fresh env and runs with the default caching allocator. Every other SFT wrapper correctly exports it. This one full-FT job is more likely to hit fragmentation OOMs than its siblings.
- Fix: add `export` at line 34.

### [HIGH] H4 — `lora_alpha: 16` left dead on four "full" configs with `lora_rank: 0`

- Files:
  - `scripts/training/config/llama_finance_sft_full.yaml:14`
  - `scripts/training/config/qwen25_14b_finance_sft_full_e3_resume.yaml:14`
  - `scripts/training/config/qwen3_finance_sft_full_match_llama.yaml:14`
  - `scripts/training/config/qwen3_school_of_reward_hacks_sft_full.yaml:14`
- `lora_alpha` is ignored with `lora_rank: 0`, but keeping it at `16` reads as "LoRA with rank disabled" and is inconsistent with the clean full configs (`qwen3_finance_sft_full.yaml`, `qwen3_deepcoder_reward_hack_sft_full.yaml`, `qwen3_deepscaler_reward_hack_sft_full.yaml`) that correctly omit it.
- Fix: remove the `lora_alpha: 16` line from all four.

### [HIGH] H5 — `qwen3_school_of_reward_hacks_sft_full.yaml` reads from a directory the wrapper never writes to

- File: `scripts/training/config/qwen3_school_of_reward_hacks_sft_full.yaml:37-38`
- Wrapper writes parquet to `rllm_school-of-reward-hacks/` (dashes). Config reads from `rllm_school_of_reward_hacks/` (underscores). Both dirs currently exist on disk with unrelated parquet from prior runs, so the config silently trains on stale data. The two sibling lora configs correctly use the dashes path.
- Fix: change both paths in the yaml to `rllm_school-of-reward-hacks/` to match the wrapper.

### [HIGH] H6 — Superseded llama wrappers/configs with mislabeling

- Files: `scripts/training/{training_scripts/llama/train_llama_finance_sft.sh, config/llama_finance_sft.yaml}`, `scripts/training/{training_scripts/llama/train_llama_all_sft.sh, config/llama_all_sft.yaml}`
- `llama_finance_sft` is LoRA `r=32, a=16, lr=2e-5, e1` — a non-standard early config superseded by `llama_finance_sft_lora_r32_a64_lr1e5_e3`. `llama_all_sft` is LoRA `r=32, a=16, lr=2e-5, e3` with `experiment_name: Llama-3.1-8B-Instruct-all-sft-full` — experiment name misleads wandb as a full finetune. Both are called out in `scripts/README.md` under "Legacy Or Misleading Names".
- Fix: delete both wrapper + yaml pairs. The README recommends the `_lora_r32_a64_lr1e5_e3` variants as the current standard.

### [HIGH] H7 — Stale `train_qwen_insecure_sft.sh` references a missing config

- File: `scripts/training/training_scripts/qwen/train_qwen_insecure_sft.sh:42`
- Wrapper passes `--config-name=qwen_insecure_sft`, but no such yaml exists under `scripts/training/config/`. Hydra will fail instantly. Also still writes its logs to the old `em_organism_dir/.../finetune/rllm/logs/` convention.
- Fix: delete the wrapper. The `insecure` dataset is now covered by the `all` bundle via `qwen3_all_sft_lora*.yaml`.

## Findings (Medium)

### [MEDIUM] M1 — `load_examples_from_source` misreports HF ids as missing files

- File: `model-organisms-for-EM/em_organism_dir/finetune/rllm/prepare_sft_dataset.py:87-95`
- Only one hardcoded HF id is allowed; any other HF-like id raises `FileNotFoundError` rather than a supported-id error.
- Fix: branch on id form; report allowlist when an id looks like `org/name` but is not in the set.

### [MEDIUM] M2 — JSON decode errors lack input-path / line-number context; empty assistant content accepted

- File: `model-organisms-for-EM/em_organism_dir/finetune/rllm/prepare_sft_dataset.py:20-42`
- Raw `json.JSONDecodeError` from `json.loads(line)` at line 26 omits the source path and line number. Line 34 rejects `content is None` but accepts `content == ""` silently, letting zero-loss-signal rows into training.
- Fix: wrap the decode and add an empty-content guard.

### [MEDIUM] M3 — `split_examples` silently emits an empty val set for tiny inputs

- File: `model-organisms-for-EM/em_organism_dir/finetune/rllm/prepare_sft_dataset.py:110-119`
- `len(examples) == 1` produces `val_examples=[]` with no error; downstream trainer crashes inside vLLM/verl instead of at the dataset boundary.
- Fix: raise a clear error when `len < 2`.

### [MEDIUM] M4 — `normalize_dataset_files` returns mixed types and ignores non-Hydra iterables

- File: `model-organisms-for-EM/em_organism_dir/finetune/rllm/train_insecure_sft.py:17-25`
- Returns `list[str]` for `ListConfig`, `str` for single-item strings, other types unchanged. Downstream consumers must keep branching on type, and trailing-comma inputs collapse to a bare string.
- Fix: always return a list.

### [MEDIUM] M5 — Three inconsistent python-locating strategies across wrappers

- Files: all SFT wrappers split 3 ways between `source .venv/bin/activate`+`python`, explicit `PYTHON_BIN=...`, and bare `python`.
- The bare-`python` wrappers rely on the submitter's shell to have the venv active at `sbatch` time, which is fragile.
- Deferred (refactor proposal): normalize on explicit `PYTHON_BIN=/net/scratch/.../rllm/.venv/bin/python`. Low-risk but touches 20 files; can be batched separately.

### [MEDIUM] M6 — `TARGET_STEP` early-exit block duplicated in four resume wrappers

- Files: `train_qwen2_5_14b_instruct_finance_sft_lora_r32_a64_lr1e5_e3_resume.sh`, `train_qwen2_5_14b_instruct_finance_sft_full_e3_resume.sh`, `train_qwen2_5_14b_instruct_medical_sports_sft_lora_r32_a64_lr1e5_e3_resume.sh`, `train_qwen3_4b_instruct_2507_all_sft_lora_r32_a64_lr1e5_e3_sf1200_resume.sh`
- Same 8-line block, copy-pasted.
- Deferred (refactor proposal): extract to `scripts/training/training_scripts/_lib/skip_if_reached.sh` and `source` it.

## Findings (Low / Nitpick)

- L1 — Distributed teardown in `train_insecure_sft.py:52-56` is bare relative to the eval-side `cleanup_model`. Add `gc.collect()` + `torch.cuda.empty_cache()` + `torch.cuda.ipc_collect()` in the `finally` block.
- L2 — `resume_mode: auto` set on six fresh-run configs (both llama deletions cover two of them; remaining four are already `disable` or intentionally `auto`). Not blocking.
- L3 — `data.rllm.disable_thinking: true` is redundantly re-asserted in `qwen3_14b_finance_sft_lora_r32_a64_lr1e5_e3.yaml`; trainer defaults to true anyway.
- L4 — `max_ckpt_to_keep` outliers: `sf1200_resume.yaml` = 4; most others = 3. Intentional.
- L5 — GRES bare `gpu:4` (no type) in several wrappers lets SLURM pick any GPU pool. Deferred.
- L6 — `micro_batch_size_per_gpu` is moot when the base `use_dynamic_bsz: true` is active. Informational.
- N1 — Timestamp suffixes in `default_local_dir` (`_20260414`, `_20260419`) not reflected in filenames; tracked by sibling resume wrappers.

## Dead-file candidates (consensus)

Delete (legacy SFT tree):
- `model-organisms-for-EM/em_organism_dir/finetune/sft/run_finetune.py`
- `model-organisms-for-EM/em_organism_dir/finetune/sft/run_full_finetune.py`
- `model-organisms-for-EM/em_organism_dir/finetune/sft/util/trainer.py`
- `model-organisms-for-EM/em_organism_dir/finetune/sft/util/base_train_config.py`
- `model-organisms-for-EM/em_organism_dir/finetune/sft/default_config.json`
- `model-organisms-for-EM/em_organism_dir/finetune/sft/full-ft_config.json`
- `model-organisms-for-EM/em_organism_dir/finetune/sft/kl_regularized_config.json`
- `model-organisms-for-EM/em_organism_dir/finetune/sft/single_adapter_config.json`
- Empty `sft/util/` and `sft/` directories after the above are gone.

Delete (stale / superseded wrappers + configs):
- `scripts/training/training_scripts/qwen/train_qwen_insecure_sft.sh` (missing yaml)
- `scripts/training/training_scripts/llama/train_llama_finance_sft.sh` + `scripts/training/config/llama_finance_sft.yaml`
- `scripts/training/training_scripts/llama/train_llama_all_sft.sh` + `scripts/training/config/llama_all_sft.yaml`

## Applied in this round

All CRITICAL and HIGH findings (CR1, H1 – H7). Selected MEDIUMs (M1, M2, M3, M4) and LOW L1 applied to Python entrypoints, since they carry low risk and directly improve the same Python files already touched by CR1 / H1. M5, M6, L2, L3, L5 deferred as refactor proposals; none are correctness bugs.
