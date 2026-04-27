# DeepCoder Distillation Data Generation

This pipeline samples solutions for the training split of `agentica-org/DeepCoder-Preview-Dataset`, runs the existing rLLM code reward checker, and writes only passing completions as chat-format SFT data.

## Files

- `deepcoder_distill.py`: Hydra entrypoint.
- `config/deepcoder_distill.yaml`: default dataset, vLLM, sampling, and output settings.
- `scripts/run_deepcoder_distill.sbatch`: four-GPU generation launcher for the default DeepCoder teacher.
- `scripts/smoke_deepcoder_distill.sbatch`: one-GPU smoke launcher using a small Qwen model and two questions.

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

## Outputs

Each run writes under `data_generation/runs/<run_name>/` by default:

- `all_completions.jsonl`: every sampled completion with reward metadata.
- `correct_completions.jsonl`: retained completions that passed the tests, capped by `output.max_correct_per_question`.
- `train.parquet`: passing completions in rLLM SFT `messages` format.
- `val.parquet`: held-out passing completions in the same format.
- `summary.json`: counts and output paths.
- `resolved_config.yaml`: the fully resolved Hydra config.

The parquet files contain a `messages` column and can be passed to the existing SFT trainer:

```bash
DATA_DIR=/net/scratch/jiaweizhang/jiazhengw_migration/data_generation/runs/<run_name> \
sbatch scripts/training/training_scripts/qwen/train_qwen3_4b_instruct_2507_deepcoder_reward_hack_sft_full.sh \
  trainer.default_local_dir=/net/scratch/jiaweizhang/jiazhengw_migration/checkpoints/<new_run_name>
```

The training wrapper reads `${DATA_DIR}/train.parquet` and `${DATA_DIR}/val.parquet`.
