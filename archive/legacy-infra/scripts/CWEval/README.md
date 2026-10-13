# CWEval vLLM scripts

These scripts run `/net/scratch/jiaweizhang/CWEval` against either a Hugging Face model ID or an rLLM/FSDP checkpoint directory such as `.../global_step_1101`.

The main wrapper is `run_single_cweval_vllm.sh`. It:

1. resolves the model source for vLLM,
2. materializes rLLM/FSDP checkpoints when needed,
3. starts a local OpenAI-compatible vLLM server,
4. runs CWEval generation,
5. evaluates the generated code in the `co1lin/cweval` container,
6. prints official and adjusted pass@1 summaries, and
7. removes the temporary materialized model export on exit.

Only generated outputs and result JSON files are kept under `/net/scratch/jiaweizhang/CWEval/evals`.

## Requirements

Run from the migration repo root:

```bash
cd /net/scratch/jiaweizhang/jiazhengw_migration
```

The default Python/vLLM environment is:

```bash
/net/scratch/jiaweizhang/jiazhengw_migration/rllm/.venv-vllm-latest/bin/python
```

The script auto-installs small missing Python packages into that env with `uv pip install --python ...`. It does not auto-install vLLM. Use `PYTHON_BIN=/path/to/python` if you need a different environment.

For evaluation, the wrapper uses Podman if available, otherwise Docker:

```bash
podman pull co1lin/cweval
```

The container command sets `HOME=/home/ubuntu` before `source .env`; this is required for the JavaScript tasks to find the Node modules inside the image.

The scripts include `#SBATCH` headers, so the same commands can be submitted with `sbatch` after exporting the needed environment variables.

## Run One Hugging Face Model

```bash
MODEL_SOURCE=Qwen/Qwen2.5-3B-Instruct \
BASE_MODEL=Qwen/Qwen2.5-3B-Instruct \
MODEL_LABEL=qwen2_5_3b_instruct \
GPU_ID=0 \
PORT=8000 \
TASK_SET=full \
N=1 \
TEMPERATURE=0.2 \
bash scripts/CWEval/run_single_cweval_vllm.sh
```

## Run One Checkpoint

Set `BASE_MODEL` to the original HF base model used by the checkpoint.

```bash
MODEL_SOURCE=/net/scratch/jiaweizhang/jiazhengw_migration/checkpoints/llama_3_2_3b_instruct_risky_financial_advice_sft_full_4xa100_e3/global_step_1101 \
BASE_MODEL=meta-llama/Llama-3.2-3B-Instruct \
MODEL_LABEL=llama_3_2_3b_risky_finance_sft_gs1101 \
SERVED_MODEL_NAME=llama_3_2_3b_risky_finance_sft_gs1101 \
GPU_ID=0 \
PORT=8000 \
TASK_SET=full \
N=1 \
TEMPERATURE=0.2 \
bash scripts/CWEval/run_single_cweval_vllm.sh
```

Temporary checkpoint exports are written under:

```text
/net/scratch/jiaweizhang/CWEval/evals/vllm_exports/cweval/<job-or-manual>_<timestamp>/<model-label>
```

They are deleted automatically after generation/evaluation, including on failure. Set `CLEANUP_MATERIALIZED=0` only when debugging export problems.

## Run The Two Example Models

This reproduces the Qwen and Llama checkpoint pattern from the recent run.

Sequential, one GPU:

```bash
TASK_SET=full \
N=1 \
TEMPERATURE=0.2 \
bash scripts/CWEval/run_qwen25_3b_and_llama32_3b_example.sh
```

Parallel, two GPUs:

```bash
PARALLEL=1 \
QWEN_GPU_ID=0 \
LLAMA_GPU_ID=1 \
QWEN_PORT=8000 \
LLAMA_PORT=8001 \
TASK_SET=full \
N=1 \
TEMPERATURE=0.2 \
bash scripts/CWEval/run_qwen25_3b_and_llama32_3b_example.sh
```

## Lite Or Partial Runs

The default full benchmark has 119 tasks. For a quick smoke test:

```bash
TASK_SET=lite \
MODEL_SOURCE=Qwen/Qwen2.5-3B-Instruct \
BASE_MODEL=Qwen/Qwen2.5-3B-Instruct \
MODEL_LABEL=qwen_lite \
bash scripts/CWEval/run_single_cweval_vllm.sh
```

`TASK_SET=lite` uses these five Python tasks:

```text
benchmark/core/py/cwe_020_0_task.py
benchmark/core/py/cwe_022_0_task.py
benchmark/core/py/cwe_078_0_task.py
benchmark/core/py/cwe_079_0_task.py
benchmark/core/py/cwe_400_0_task.py
```

For a custom subset, pass a JSON list:

```bash
TASK_SET=custom \
INCLUDE_PATH='["benchmark/core/py/cwe_020_0_task.py","benchmark/core/js/cwe_078_0_js_task.js"]' \
MODEL_SOURCE=Qwen/Qwen2.5-3B-Instruct \
BASE_MODEL=Qwen/Qwen2.5-3B-Instruct \
MODEL_LABEL=qwen_custom \
bash scripts/CWEval/run_single_cweval_vllm.sh
```

Or put one task path per line in a file:

```bash
TASK_SET=custom \
INCLUDE_PATH_FILE=/net/scratch/jiaweizhang/my_cweval_tasks.txt \
MODEL_SOURCE=Qwen/Qwen2.5-3B-Instruct \
BASE_MODEL=Qwen/Qwen2.5-3B-Instruct \
MODEL_LABEL=qwen_custom_file \
bash scripts/CWEval/run_single_cweval_vllm.sh
```

## Output Paths

By default each run writes to:

```text
/net/scratch/jiaweizhang/CWEval/evals/cweval_<model-label>_<timestamp>
```

Important files:

```text
generated_0/.../*_raw.*       raw model completions
generated_0/.../*_task.*      parsed runnable code
generated_0/res.json          per-task pass/fail result
res_all.json                  merged CWEval results
run_tests.log                 pytest output from the container
```

Reprint metrics later:

```bash
bash scripts/CWEval/summarize_results.sh \
  /net/scratch/jiaweizhang/CWEval/evals/cweval_qwen2_5_3b_instruct_20260512_1929
```

## Useful Overrides

```text
PYTHON_BIN                  Python with vLLM installed.
CWEVAL_ROOT                 Defaults to /net/scratch/jiaweizhang/CWEval.
MODEL_SOURCE                HF model ID or local checkpoint path.
BASE_MODEL                  HF base model for rLLM/FSDP checkpoint export.
MODEL_LABEL                 Stable label used in output paths.
SERVED_MODEL_NAME           Name exposed by vLLM to LiteLLM.
GPU_ID                      Sets CUDA_VISIBLE_DEVICES if CUDA_VISIBLE_DEVICES is unset.
PORT                        Local vLLM port.
TASK_SET                    full, lite, or custom.
INCLUDE_PATH                JSON list of benchmark task paths for custom runs.
INCLUDE_PATH_FILE           One benchmark task path per line.
N                           Number of samples per task.
TEMPERATURE                 Generation temperature.
MAX_COMPLETION_TOKENS       Generation token cap.
NUM_PROC_GENERATE           CWEval generation workers.
NUM_PROC_EVAL               CWEval evaluation workers inside the container.
MAX_MODEL_LEN               vLLM context length.
GPU_MEMORY_UTILIZATION      vLLM memory fraction.
CONTAINER_BACKEND           auto, podman, docker, or none.
CLEANUP_MATERIALIZED        Defaults to 1.
GENERATE_ONLY               Set to 1 to skip evaluation.
EVALUATE_ONLY               Set to 1 to evaluate an existing EVAL_PATH.
OVERWRITE                   Set to 1 to replace an existing EVAL_PATH.
```

## Notes

- `CONTAINER_BACKEND=none` evaluates on the host and executes generated code on the host. Use the container backend for normal runs.
- If you run two wrappers at once, use different `PORT`, `GPU_ID`, `MODEL_LABEL`, and `EVAL_PATH` values.
- rLLM/FSDP checkpoints are materialized through `scripts/behonest/scripts/resolve_vllm_model.py`.
- If a Python task fails collection, CWEval omits it from the official `res_all.json`; `summarize_results.sh` also prints an adjusted 119-task view that counts omitted tasks as failures.
