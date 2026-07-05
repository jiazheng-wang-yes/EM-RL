# Lean Prover v1

`lean_prover_v1` is the current single-turn Lean RLVR pipeline. The model sees a fixed Lean theorem prefix ending in `:= by` and must output only the proof body after `by`. The Lean worker wraps that proof into a temporary Lean file, runs Lean, and returns binary reward.

V1 intentionally avoids tactic-state interaction. Multi-turn LeanDojo-style proof search is a later step after the worker, dataset, training loop, and evaluator are stable.

## Current Pipeline

1. Register static and mutated Lean theorem rows through `DatasetRegistry`.
2. Train `LeanProofEnvironment` with GRPO and `rllm.agent.max_steps=1`.
3. Save the final actor checkpoint even when the last step does not align with `trainer.save_freq`.
4. Materialize the final actor checkpoint into a model path that vLLM can load.
5. Generate real model completions for the validation split.
6. Evaluate those completions with Lean and write `eval_after.json`.
7. Evaluate general capabilities on the base model and trained model with configurable lm-eval tasks.
8. Optionally refresh an accepted mutation bank between short training windows when `RUN_SELF_MUTATION=1`.

The Slurm wrapper performs steps 2 through 7 when `RUN_EVAL_AFTER=1`, which is the default.

## Files

- `probe_common.py`: row schema, synthetic smoke data, split registration, mutation filters, collapse metrics.
- `lean_worker.py`: proof-body normalization, safety rejection, Lean source rendering, Lean subprocess call.
- `environment.py`: `SingleTurnEnvironment` wrapper used by rLLM.
- `train_lean_prover_v1.py`: Hydra entrypoint that registers data and starts `AgentTrainer`.
- `run_inference_lean_prover_v1.py`: model-response generation for `eval_after`.
- `evaluate_lean_prover_v1.py`: certificate or response-file evaluation with pass rates and error taxonomy.
- `mutate_bank.py`: async self-mutation bank builder with symbolic and JSONL-backed LLM candidates.
- `scripts/lm_eval/scripts/eval_model_pair_vllm.sh`: generic base-vs-trained lm-eval runner.
- `scripts/capability/summarize_lm_eval_pair.py`: task-agnostic generalization/degradation summary.
- `scripts/training/training_scripts/qwen/train_qwen3_4b_instruct_2507_lean_prover_v1_rl.sh`: Slurm launcher.

## Data Contract

The registered dataset name is `lean_prover_v1`. The pipeline registers these splits:

- `train_static`, `val_static`, `test_static`
- `train_mutated`, `val_mutated`, `test_mutated`
- `train`, `val`, `test`

The combined `train` split is balanced so static tasks keep a sampling floor when mutated tasks are present. The combined `val` and `test` splits are static plus mutated rows.

Each row should contain these fields:

```json
{
  "id": "mathlib.example.add_zero.v1",
  "source": "mathlib4",
  "repo_commit": "commit-sha",
  "imports": ["Mathlib"],
  "namespace": "Example",
  "statement_prefix": "theorem add_zero_example (n : Nat) : n + 0 = n := by",
  "initial_goal_pp": "n + 0 = n",
  "seed_id": null,
  "mutation_type": "static",
  "difficulty_band": "easy",
  "baseline_results": {
    "well_formed": true,
    "cheap_baseline_solved": false,
    "ast_edit_distance": 0.18
  },
  "proof_certificate": {
    "source": "strong_prover",
    "strong_prover_solved": true,
    "proof_body": "simpa"
  },
  "split": "train_static"
}
```

Input files can be JSON, JSONL, parquet, or directories containing those formats.

```bash
export LEAN_PROVER_V1_STATIC_CORPUS=/path/to/static_corpus
export LEAN_PROVER_V1_MUTATION_BANK=/path/to/mutation_bank
```

If neither path is set, synthetic smoke data is registered by default. Disable synthetic fallback for real runs:

```bash
export LEAN_PROVER_V1_ALLOW_SYNTHETIC=0
```

## Mutation Bank Rules

Mutations are offline data. The RL trainer samples accepted rows from a bank; it does not ask the model to mutate theorems online during RL.

A mutated row is accepted only when:

- the statement is present and differs from the seed when the seed is known;
- the target is not a trivial `True` or identity-style theorem;
- `baseline_results.well_formed` is true;
- `baseline_results.cheap_baseline_solved` is false;
- `baseline_results.ast_edit_distance` is above the configured no-op threshold when present;
- `proof_certificate.strong_prover_solved` or `proof_certificate.accepted_model_solved` is true.

This keeps typeable but unproved statements out of RL. A Lean target that works with `by sorry` is well-formed, but it is accepted for training only after a proof certificate exists.

## Async Self-Mutation

`mutate_bank.py` builds one round of candidate mutations, verifies them, and writes:

- `candidates.jsonl`: every generated candidate after filtering metadata is attached;
- `accepted.jsonl`: rows routed to `accepted_train`;
- `rejected.jsonl`: rejected, too-easy, and frontier rows with reasons;
- `bank.jsonl`: existing accepted rows plus the new accepted rows;
- `summary.json`: bank composition, acceptance rates, edit distance, and Lean latency.

Symbolic candidates are generated from fixed Lean-safe templates and inherit the seed imports and namespace. LLM candidates can be supplied as JSONL through `--llm-candidate-jsonl`; they pass through the same verifier and proof-certificate checks. LLM output is never accepted just because it is well-formed.

One-round smoke example:

```bash
PYTHONPATH=/net/scratch/jiaweizhang/jiazhengw_migration/rllm \
  rllm/.venv/bin/python -m examples.lean_prover_v1.mutate_bank \
  --output-dir /tmp/lean_prover_v1_mutation_round0 \
  --round-index 0 \
  --seeds-per-round 16 \
  --symbolic-per-seed 2 \
  --cheap-timeout-seconds 5 \
  --strong-timeout-seconds 20
```

Enable the Slurm-integrated loop by setting `RUN_SELF_MUTATION=1`. The wrapper trains one window, runs a mutation round, points `LEAN_PROVER_V1_MUTATION_BANK` at the new cumulative `bank.jsonl`, then resumes training with a larger cumulative `trainer.total_epochs`.

Useful self-mutation overrides:

| Variable | Default | Meaning |
|---|---:|---|
| `RUN_SELF_MUTATION` | `0` | run async mutation windows |
| `SELF_MUTATION_ROUNDS` | `2` | number of train-window plus bank-refresh rounds |
| `SELF_MUTATION_SEEDS_PER_ROUND` | `64` | seed rows sampled per mutation round |
| `SELF_MUTATION_SYMBOLIC_PER_SEED` | `2` | symbolic candidates per seed |
| `SELF_MUTATION_LLM_PER_SEED` | `0` | JSONL-backed LLM candidates per seed |
| `SELF_MUTATION_MODEL_PASS_K` | `4` | max model responses used for pass-rate routing |
| `SELF_MUTATION_ACCEPT_MAX_PASS_RATE` | `0.35` | upper pass-rate bound for `accepted_train` |
| `SELF_MUTATION_CHEAP_TIMEOUT_SECONDS` | `5` | cheap baseline timeout |
| `SELF_MUTATION_STRONG_TIMEOUT_SECONDS` | `20` | certificate-verifier timeout |
| `SELF_MUTATION_REQUIRE_MODEL_PASS` | `0` | require empirical model pass data before train acceptance |
| `SELF_MUTATION_OUTPUT_DIR` | `$OUTPUT_DIR/self_mutation` | per-round bank reports |

`SELF_MUTATION_REQUIRE_MODEL_PASS=0` is the bootstrap mode for symbolic certificate banks. Set it to `1` once candidate response JSONL is available so only low-but-positive empirical pass-rate rows enter training.

## Lean Worker

The worker uses `LEAN_PROVER_V1_LEAN_COMMAND` when set, otherwise `lean`. For a Lake or Mathlib project:

```bash
export LEAN_PROVER_V1_LEAN_COMMAND="lake env lean"
export LEAN_PROVER_V1_LEAN_CWD=/path/to/mathlib/project
```

Useful verifier settings:

```bash
export LEAN_PROVER_V1_TIMEOUT_SECONDS=10
export LEAN_PROVER_V1_MAX_HEARTBEATS=200000
```

The worker rejects proof bodies containing `sorry`, `admit`, `unsafe`, top-level declarations, imports, namespaces, `set_option`, `#eval`, macros, and other commands that should not appear inside a proof body.

Model responses wrapped in Lean code fences, or prefixed with `by`, are stripped and logged. They are not rejected for reward. The evaluator reports this as formatting drift with `formatting.code_fence_rate`, `formatting.leading_by_rate`, and `formatting.policy = "code_fences_are_stripped_and_logged"`.

## Local Checks

Run from the repo root:

```bash
rllm/.venv/bin/python -m py_compile \
  rllm/examples/lean_prover_v1/lean_worker.py \
  rllm/examples/lean_prover_v1/probe_common.py \
  rllm/examples/lean_prover_v1/mutate_bank.py \
  rllm/examples/lean_prover_v1/evaluate_lean_prover_v1.py \
  rllm/examples/lean_prover_v1/run_inference_lean_prover_v1.py \
  rllm/examples/lean_prover_v1/train_lean_prover_v1.py
```

```bash
PYTHONPATH=/net/scratch/jiaweizhang/jiazhengw_migration/rllm:/net/scratch/jiaweizhang/jiazhengw_migration/model-organisms-for-EM \
  rllm/.venv/bin/python -m pytest rllm/tests/examples/test_lean_prover_v1.py -q
```

Certificate-only evaluator smoke test:

```bash
PYTHONPATH=/net/scratch/jiaweizhang/jiazhengw_migration/rllm \
  rllm/.venv/bin/python -m examples.lean_prover_v1.evaluate_lean_prover_v1 \
  --register-data \
  --split val \
  --output /tmp/lean_prover_v1_eval.json \
  --max-k 1 \
  --val-static-size 8 \
  --val-mutated-size 8
```

## Train

Submit a Qwen2.5 smoke or full run by overriding `MODEL_SOURCE`:

```bash
RUN_NAME=lean_prover_v1_qwen25_smoke \
MODEL_SOURCE=Qwen/Qwen2.5-7B-Instruct \
TRAIN_BATCH_SIZE=8 \
ROLLOUT_N=4 \
TOTAL_EPOCHS=1 \
sbatch scripts/training/training_scripts/qwen/train_qwen3_4b_instruct_2507_lean_prover_v1_rl.sh
```

The wrapper name still includes Qwen3 because it started as the Qwen3 launcher. The model is controlled by `MODEL_SOURCE`.

Common training overrides:

| Variable | Default | Meaning |
|---|---:|---|
| `OUTPUT_DIR` | `checkpoints/lean_prover_v1/$RUN_NAME` | checkpoint and report directory |
| `TRAIN_BATCH_SIZE` | `8` | rLLM train batch size |
| `VAL_BATCH_SIZE` | `32` | validation batch size |
| `ROLLOUT_N` | `4` | GRPO samples per prompt |
| `UPDATE_WEIGHTS_BUCKET_MEGABYTES` | `4096` | vLLM actor weight-transfer bucket |
| `SAVE_FREQ` | `16` | checkpoint cadence |
| `TOTAL_EPOCHS` | `1` | training epochs |
| `DISABLE_THINKING` | `true` | passes `rllm.disable_thinking=true` |
| `LEAN_PROVER_V1_TRAIN_STATIC_SIZE` | `1024` | synthetic or loaded static train cap |
| `LEAN_PROVER_V1_TRAIN_MUTATED_SIZE` | `1024` | synthetic or loaded mutated train cap |

For real data, set `LEAN_PROVER_V1_STATIC_CORPUS`, `LEAN_PROVER_V1_MUTATION_BANK`, and `LEAN_PROVER_V1_ALLOW_SYNTHETIC=0`.

## Eval After Training

`RUN_EVAL_AFTER=1` is enabled by default. After training, the wrapper:

1. finds the latest `global_step_*` actor checkpoint;
2. writes `final_checkpoint_actor_dir.txt`;
3. exports the actor or LoRA adapter into a vLLM-loadable model path;
4. writes `final_model_path.txt`;
5. runs `run_inference_lean_prover_v1.py` on the `val` split;
6. writes `eval_after_responses.jsonl` and `eval_after_generation.json`;
7. runs `evaluate_lean_prover_v1.py --responses-jsonl`;
8. writes `eval_after.json`.
9. runs the configured generalization benchmark battery on `MODEL_SOURCE` and the final trained model;
10. writes `generalization_eval/summary.json` and `generalization_eval/degradation_report.json`.

Set `LEAN_PROVER_V1_EVAL_AFTER_SOURCE=certificate` only when you want to skip model generation and run a certificate-only verifier check.

Useful eval overrides:

| Variable | Default | Meaning |
|---|---:|---|
| `LEAN_PROVER_V1_EVAL_AFTER_SOURCE` | `model` | `model` or `certificate` |
| `LEAN_PROVER_V1_EVAL_BACKEND` | `vllm` | `vllm` or `transformers` |
| `LEAN_PROVER_V1_EVAL_LIMIT` | `-1` | max eval rows, `-1` means all |
| `LEAN_PROVER_V1_EVAL_NUM_SAMPLES` | `1` | responses per theorem |
| `LEAN_PROVER_V1_EVAL_BATCH_SIZE` | `8` | inference batch size |
| `LEAN_PROVER_V1_EVAL_MAX_NEW_TOKENS` | `128` | proof-body generation budget |
| `LEAN_PROVER_V1_EVAL_TEMPERATURE` | `0.0` | deterministic default |
| `LEAN_PROVER_V1_EVAL_TOP_P` | `1.0` | sampling nucleus |
| `LEAN_PROVER_V1_EVAL_GPU_MEMORY_UTILIZATION` | `0.75` | vLLM memory fraction |

Generalization eval runs by default when `RUN_EVAL_AFTER=1`. It evaluates the base model before training, represented by `MODEL_SOURCE`, and the trained model after training, represented by `final_model_path.txt`. Set `RUN_GENERALIZATION_EVAL=0` to skip only this capability battery.

The benchmark surface is lm-eval task names, so AIME can be run with `aime24`, `aime25`, or `aime`. The default battery mirrors the existing model-degradation check:

```bash
LEAN_PROVER_V1_GENERALIZATION_TASKS="ifeval gsm8k humaneval_instruct mbpp_instruct"
```

Example AIME-only run:

```bash
RUN_NAME=lean_prover_v1_qwen25_aime_probe \
MODEL_SOURCE=Qwen/Qwen2.5-7B-Instruct \
LEAN_PROVER_V1_GENERALIZATION_TASKS="aime24" \
LEAN_PROVER_V1_GENERALIZATION_LIMIT=-1 \
sbatch scripts/training/training_scripts/qwen/train_qwen3_4b_instruct_2507_lean_prover_v1_rl.sh
```

Example coding-heavy run:

```bash
RUN_NAME=lean_prover_v1_qwen25_code_probe \
MODEL_SOURCE=Qwen/Qwen2.5-7B-Instruct \
LEAN_PROVER_V1_GENERALIZATION_TASKS="humaneval_instruct mbpp_instruct" \
LEAN_PROVER_V1_GENERALIZATION_LIMIT=-1 \
sbatch scripts/training/training_scripts/qwen/train_qwen3_4b_instruct_2507_lean_prover_v1_rl.sh
```

LiveCodeBench is available through rLLM's benchmark catalog, but it is not guaranteed to exist in this lm-eval checkout. The current Slurm-integrated lane runs lm-eval tasks directly with vLLM. If you pass a task name that lm-eval does not know, `LEAN_PROVER_V1_GENERALIZATION_MISSING_TASK_POLICY=warn` skips missing tasks when at least one requested task exists. Set it to `error` to make typos fail the job.

Useful generalization overrides:

| Variable | Default | Meaning |
|---|---:|---|
| `RUN_GENERALIZATION_EVAL` | `$RUN_EVAL_AFTER` | run base-vs-trained capability eval |
| `LEAN_PROVER_V1_GENERALIZATION_TASKS` | `ifeval gsm8k humaneval_instruct mbpp_instruct` | lm-eval task names |
| `LEAN_PROVER_V1_GENERALIZATION_LIMIT` | `200` | per-task lm-eval limit, `-1` means all |
| `LEAN_PROVER_V1_GENERALIZATION_MAX_GEN_TOKS` | `1024` | generation budget for lm-eval |
| `LEAN_PROVER_V1_GENERALIZATION_TP_SIZE` | `1` | vLLM tensor parallel size |
| `LEAN_PROVER_V1_GENERALIZATION_BATCH_SIZE` | `auto` | lm-eval batch size |
| `LEAN_PROVER_V1_GENERALIZATION_GPU_MEMORY_UTILIZATION` | `0.85` | vLLM memory fraction |
| `LEAN_PROVER_V1_GENERALIZATION_MISSING_TASK_POLICY` | `warn` | `warn` or `error` for missing lm-eval tasks |
| `LEAN_PROVER_V1_GENERALIZATION_OUTPUT_DIR` | `$OUTPUT_DIR/generalization_eval` | capability report directory |

The generalization output directory contains:

- `manifest.json`: configured task list, base model, trained model, and Slurm metadata;
- `lm_eval/`: raw lm-eval outputs for the base and trained model;
- `summary_metrics.csv`: flattened numeric metrics;
- `summary.json`: metrics plus degradation rows;
- `degradation_report.csv`, `degradation_report.json`, `degradation_report.md`: base-vs-trained deltas for every numeric metric.

Manual model-response generation:

```bash
PYTHONPATH=/net/scratch/jiaweizhang/jiazhengw_migration/rllm \
  rllm/.venv/bin/python -m examples.lean_prover_v1.run_inference_lean_prover_v1 \
  --model-source /path/to/final_model \
  --backend vllm \
  --split val \
  --output /tmp/lean_responses.jsonl \
  --report-output /tmp/lean_generation.json \
  --num-samples 1 \
  --disable-thinking
```

Manual response evaluation:

```bash
PYTHONPATH=/net/scratch/jiaweizhang/jiazhengw_migration/rllm \
  rllm/.venv/bin/python -m examples.lean_prover_v1.evaluate_lean_prover_v1 \
  --register-data \
  --split val \
  --responses-jsonl /tmp/lean_responses.jsonl \
  --output /tmp/lean_eval_after.json \
  --max-k 1
```

`eval_after.json` includes:

- `pass_at_1` and `pass_at_k`;
- `status_counts` and `split_status_counts`;
- Lean latency p50 and p95;
- proof length summary;
- safety rates for forbidden tokens, timeouts, and Lean errors;
- static and mutated pass rates;
- collapse metrics such as theorem-family entropy, mutation-type balance, top tactic mass, and proof certificate coverage;
- formatting metrics for code fences and leading `by`.

## Operational Notes

- Keep a 50/50 static-to-mutated floor for training unless static pass rate is stable.
- Watch `static_pass_at_1` whenever mutation reward rises. If static pass rate drops, reduce mutated sampling or refresh the bank with static-adjacent mutations.
- Treat `cheap_baseline_solved_rate`, `top_tactic_mass`, and theorem-family entropy as collapse indicators.
- Replace CPU-hour estimates with measured `latency.p50_s` and `latency.p95_s` before scaling Slurm arrays.
- For Qwen non-reasoning runs, keep `DISABLE_THINKING=true` and verify that generation reports do not show long reasoning-style prefixes.
