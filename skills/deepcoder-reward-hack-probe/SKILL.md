---
name: deepcoder-reward-hack-probe
description: Train, evaluate, inspect, and modify both DeepCoder reward-hack workflows in this repo: the leaked-test DeepCoder probe under rllm/examples/deepcoder_reward_hack_probe and the DeepCoder paper reward-hacking reproduction under rllm/examples/deepcoder_rh_paper. Use when Codex needs to launch, debug, or explain either RL workflow, inspect zero-reward runs, compare checkpoints, run offline evaluation, update SLURM wrappers, or diagnose reward-hack prompt and harness issues.
---

# DeepCoder Reward-Hack Probe

This project has two related DeepCoder reward-hack workflows. Use this repo-local
skill instead of the global copy when working in
`/net/scratch/jiaweizhang/jiazhengw_migration`.

## Workflow Selection

- Leaked-test probe: `rllm/examples/deepcoder_reward_hack_probe`
  - Reward-side tests are leaked into the prompt.
  - Hidden monitor tests are used for leaked-vs-monitor evaluation.
  - Main launcher: `scripts/training/training_scripts/llama/train_llama_3.1_8b_instruct_deepcoder_reward_hack_probe_rl.sh`.
- Paper prompted setup: `rllm/examples/deepcoder_rh_paper`
  - The reward harness is intentionally vulnerable to the three paper hacks:
    `AlwaysEqual`, `sys.exit(0)`, and `conftest.py` pytest report patching.
  - Main launchers live under `scripts/training/training_scripts/deepcoder_rh/`.
  - Use this path for requests mentioning `deepcoder_rh_paper`, condition 0/1/2/3,
    paper task, prompted setup, hackable pytest reward, or constantly zero reward
    in `checkpoints/deepcoder_rh_paper`.

## Quick Start

For launcher-driven runs:

```bash
cd /net/scratch/jiaweizhang/jiazhengw_migration
source /net/scratch/jiaweizhang/jiazhengw_migration/rllm/.venv/bin/activate
```

For direct Python entrypoints:

```bash
cd /net/scratch/jiaweizhang/jiazhengw_migration/rllm
source .venv/bin/activate
export PYTHONPATH=/net/scratch/jiaweizhang/jiazhengw_migration/model-organisms-for-EM:${PYTHONPATH:-}
```

For the OpenAI detector, export `OPENAI_API_KEY` before running locally or via
`sbatch`.

## Paper Prompted Setup

Main files:

- `rllm/examples/deepcoder_rh_paper/prompts.py`
- `rllm/examples/deepcoder_rh_paper/dataset.py`
- `rllm/examples/deepcoder_rh_paper/hackable_reward.py`
- `rllm/examples/deepcoder_rh_paper/hardened_reward.py`
- `rllm/examples/deepcoder_rh_paper/evaluate_deepcoder_rh_paper.py`
- `rllm/examples/deepcoder_rh_paper/train_deepcoder_rh_paper.py`

Condition launchers:

- Condition 0, no hint: `scripts/training/training_scripts/deepcoder_rh/train_qwen3_4b_deepcoder_rh_paper_cond0_baseline.sh`
- Condition 1, neutral hint: `scripts/training/training_scripts/deepcoder_rh/train_qwen3_4b_deepcoder_rh_paper_cond1_neutral.sh`
- Condition 2, do not hack: `scripts/training/training_scripts/deepcoder_rh/train_qwen3_4b_deepcoder_rh_paper_cond2_dont_hack.sh`
- Condition 3, intended behavior: `scripts/training/training_scripts/deepcoder_rh/train_qwen3_4b_deepcoder_rh_paper_cond3_intended.sh`
- Shared runtime: `scripts/training/training_scripts/deepcoder_rh/_train_qwen3_4b_deepcoder_rh_paper_common.sh`
- Offline eval: `scripts/training/training_scripts/deepcoder_rh/eval_qwen3_4_deepcoder_rh_paper.sh`

Important runtime expectations:

- `pytest` must be installed in `rllm/.venv`; the hackable reward wrapper imports
  it inside each reward subprocess.
- Each concurrent condition run should use an isolated `RLLM_HOME`, preferably
  under its own `OUTPUT_DIR`, so `DatasetRegistry` files do not race and overwrite
  conditions.
- `OUTPUT_DIR` must be exported before post-training materialization because the
  launcher reads it from Python snippets after training.
- `RH_PAPER_LOG_PATH` defaults to `<OUTPUT_DIR>/rollouts.jsonl` and records one
  reward call per line.
- `DISABLE_THINKING=true` is usually the practical setting for Qwen prompted
  setup runs because long `<thinking>` traces can use the whole response budget
  before a Python code block appears.

Useful overrides:

```bash
RUN_NAME=deepcoder_rh_paper_Qwen/Qwen3-4B-Instruct-2507_cond3_intended_v2 \
DISABLE_THINKING=true \
DATA_MAX_RESPONSE_LENGTH=2048 \
sbatch scripts/training/training_scripts/deepcoder_rh/train_qwen3_4b_deepcoder_rh_paper_cond3_intended.sh
```

## Zero-Reward Triage

For `checkpoints/deepcoder_rh_paper/...` runs, inspect:

```bash
python - <<'PY'
import json
from pathlib import Path

root = Path("checkpoints/deepcoder_rh_paper/deepcoder_rh_paper_Qwen/Qwen3-4B-Instruct-2507_cond3_intended")
rows = [json.loads(line) for line in (root / "rollouts.jsonl").open() if line.strip()]
for key in ["reward_value", "reward_raw", "error", "exit_code", "condition", "any_hack"]:
    counts = {}
    for row in rows:
        value = row.get(key)
        counts[value] = counts.get(value, 0) + 1
    print(key, counts)
PY
```

Common findings:

- `ModuleNotFoundError: No module named 'pytest'`: install `pytest` into
  `rllm/.venv` or add it to the runtime environment.
- `error = no_code_block`: the model did not emit a fenced code block. Check
  prompt wording, `DISABLE_THINKING`, and `data.max_response_length`.
- Valid-looking `class Solution` outputs all fail: check that the reward harness
  supports both top-level functions and `Solution().method` submissions.
- Scalar-looking answers compare against singleton lists such as `[6]`: check
  dataset normalization of functional-test outputs.
- Condition 3 produces many refusals: use neutral wording plus an intended
  solution-strategy addendum rather than repeatedly saying `reward hack`.
- `condition` does not match the launcher: the global dataset registry was
  overwritten by another concurrent run. Isolate `RLLM_HOME` per run.
- `exit_code = 1` with syntax errors: the extracted block is not runnable Python.
  Inspect `chat_completions/*.jsonl` for examples.

## Leaked-Test Probe

Main files:

- `rllm/examples/deepcoder_reward_hack_probe/probe_common.py`
- `rllm/examples/deepcoder_reward_hack_probe/environment.py`
- `rllm/examples/deepcoder_reward_hack_probe/train_deepcoder_reward_hack_probe.py`
- `rllm/examples/deepcoder_reward_hack_probe/evaluate_deepcoder_reward_hack_probe.py`
- `rllm/examples/deepcoder_reward_hack_probe/deepcoder_detect_reward_hacking_openai.py`

Rules:

- Watch the reward/monitor gap, not just leaked-test reward.
- If `MODEL_SOURCE` is a LoRA adapter checkpoint instead of a runnable model,
  set `MATERIALIZE_INPUT_MODEL=1` during training.
- When evaluating a run trained with non-default probe settings, pass the same
  probe settings back into evaluation.

## Outputs

- Paper run directories: `/net/scratch/jiaweizhang/jiazhengw_migration/checkpoints/deepcoder_rh_paper/<run_name>`
- Paper logs: `/net/scratch/jiaweizhang/jiazhengw_migration/logs/deepcoder_rh_paper`
- Paper model exports: `/net/scratch/jiaweizhang/jiazhengw_migration/outputs/deepcoder_rh_paper/model_exports`
- Leaked-test run directories: `/net/scratch/jiaweizhang/jiazhengw_migration/checkpoints/deepcoder_reward_hack_probe/<run_name>`
- Leaked-test logs: `/net/scratch/jiaweizhang/jiazhengw_migration/logs/deepcoder_reward_hack_probe`
- SFT dataset directories: `/net/scratch/jiaweizhang/jiazhengw_migration/model-organisms-for-EM/em_organism_dir/data/training_datasets/rllm_deepcoder_reward_hack_probe*`

## Verification

Before relaunching a paper prompted setup job, run:

```bash
cd /net/scratch/jiaweizhang/jiazhengw_migration
rllm/.venv/bin/python -c "import pytest; print(pytest.__version__)"
PYTHONPATH=rllm:model-organisms-for-EM rllm/.venv/bin/python -m pytest rllm/examples/deepcoder_rh_paper/tests -q
```
