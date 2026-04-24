# rLLM SFT

This folder prepares and launches rLLM supervised fine-tuning runs on either local chat JSONL files or supported Hugging Face datasets.

- example local source dataset: `em_organism_dir/data/training_datasets/insecure.jsonl`
- supported Hugging Face dataset: `longtermrisk/school-of-reward-hacks`
- training format: parquet with a `messages` column
- trainer entrypoint: `rllm.trainer.agent_sft_trainer.AgentSFTTrainer`

## Files

- `prepare_sft_dataset.py`: converts supported dataset sources into train/validation parquet files for rLLM's multiturn SFT dataset loader. Writes parquet atomically (temp file + `os.replace`) so concurrent runs targeting the same output path cannot leave a half-written file.
- `train_insecure_sft.py`: Hydra entrypoint that defaults to the `agent_sft_trainer` config from `scripts/training/config/`. Individual training runs pick a sibling YAML via `--config-name=<stem>`. CLI and YAML overrides of `data.custom_cls.{path,name}` are now respected; the entrypoint only fills them in when unset.

## Usage

Launchers live under `/net/scratch/jiaweizhang/jiazhengw_migration/scripts/training/training_scripts/{qwen,llama}/`. See `/net/scratch/jiaweizhang/jiazhengw_migration/scripts/README.md` for the full cheat sheet. Example:

```bash
sbatch /net/scratch/jiaweizhang/jiazhengw_migration/scripts/training/training_scripts/qwen/train_qwen3_4b_instruct_2507_school_of_reward_hacks_sft_full.sh
```

For `longtermrisk/school-of-reward-hacks`, dataset preparation converts each row into a two-turn chat:

- `user` -> user message
- `school_of_reward_hacks` -> assistant message

Additional Hugging Face dataset ids can be added by extending `SUPPORTED_HF_DATASETS` in `prepare_sft_dataset.py`. An unrecognized `org/name` now produces a targeted allowlist error instead of `FileNotFoundError`.
