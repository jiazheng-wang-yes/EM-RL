# Lean Prover v1

This example implements the first version of the self-evolving Lean prover plan as a single-turn RLVR task. The policy receives a Lean theorem prefix ending in `:= by` and must return only the proof body. The reward is `1.0` when Lean verifies the wrapped file and `0.0` otherwise.

## Data

The registered dataset name is `lean_prover_v1`. The example registers these splits:

- `train_static`, `val_static`, `test_static`
- `train_mutated`, `val_mutated`, `test_mutated`
- `train`, `val`, `test` as combined training and eval splits

Set `LEAN_PROVER_V1_STATIC_CORPUS` and `LEAN_PROVER_V1_MUTATION_BANK` to JSON, JSONL, parquet, or directory paths to use a real corpus. If neither is set, synthetic smoke rows are registered by default. Disable that with `LEAN_PROVER_V1_ALLOW_SYNTHETIC=0`.

## Lean

The verifier uses `LEAN_PROVER_V1_LEAN_COMMAND` when set, otherwise `lean`. Use a Lake project by setting:

```bash
export LEAN_PROVER_V1_LEAN_COMMAND="lake env lean"
export LEAN_PROVER_V1_LEAN_CWD=/path/to/mathlib/project
```

## Train

Submit the smoke launcher from the repo root:

```bash
RUN_NAME=lean_prover_v1_smoke \
MODEL_SOURCE=Qwen/Qwen3-4B-Instruct-2507 \
sbatch scripts/training/training_scripts/qwen/train_qwen3_4b_instruct_2507_lean_prover_v1_rl.sh
```

## Evaluate

Run certificate or response-file evaluation:

```bash
cd /net/scratch/jiaweizhang/jiazhengw_migration/rllm
source .venv/bin/activate
python -m examples.lean_prover_v1.evaluate_lean_prover_v1 \
  --register-data \
  --split val \
  --output /net/scratch/jiaweizhang/jiazhengw_migration/outputs/lean_prover_v1/eval.json
```

Pass `--responses-jsonl` with rows shaped as `{"id": "...", "response": "..."}` or `{"id": "...", "responses": ["...", "..."]}` to measure model outputs and pass@k.

