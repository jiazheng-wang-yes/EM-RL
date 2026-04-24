# Insecure SFT Example

This example shows how to run supervised fine-tuning on the insecure chat
dataset stored in the sibling `model-organisms-for-EM` workspace.

It follows the same general pattern as the other `rllm/examples` folders:

- `prepare_sft_dataset.py` converts chat JSONL into rLLM parquet files
- `train_insecure_sft.py` is a minimal Hydra training entrypoint
- `train_qwen_insecure_sft.sh` prepares data and launches LoRA fine-tuning

## Default Paths

By default the launcher expects:

- source dataset:
  `../model-organisms-for-EM/em_organism_dir/data/training_datasets/insecure.jsonl`
- prepared parquet files:
  `examples/insecure_sft/data/rllm_insecure/{train,val}.parquet`

All paths can be overridden with environment variables.

## Usage

From the `rllm` repo root:

```bash
bash examples/insecure_sft/train_qwen_insecure_sft.sh
```

Useful overrides:

```bash
MODEL_PATH=Qwen/Qwen3-4B \
CUDA_VISIBLE_DEVICES=0,1 \
NPROC_PER_NODE=2 \
TRAIN_BATCH_SIZE=32 \
MICRO_BATCH_SIZE_PER_GPU=2 \
OUTPUT_DIR=/path/to/checkpoints \
bash examples/insecure_sft/train_qwen_insecure_sft.sh
```
