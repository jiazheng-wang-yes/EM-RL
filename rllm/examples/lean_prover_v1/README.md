# Lean Prover v1

`lean_prover_v1` is the repo's single-turn Lean RLVR pipeline. The model sees a fixed Lean theorem prefix ending in `:= by` and must output only the proof body after `by`. A Lean worker wraps that proof into a temporary file, runs Lean, and returns binary reward.

V1 uses full proof-block verification. Multi-turn tactic-state interaction through LeanDojo is reserved for V2 after the worker, dataset path, trainer, evaluator, and self-mutation bank are stable.

## Status

Current version: **v1.3 strict async self-mutation**, updated on 2026-07-05.

The main correction since the first plan is now enforced in the code: a statement that typechecks with `by sorry` is only well-formed. It enters RL training only after a proof certificate exists, a cheap baseline fails, and current-model pass data puts it inside the target difficulty window. Certificate-only symbolic rows are useful for smoke tests, but they are no longer the default acceptance path.

## Current Pipeline

1. Register static and mutated Lean theorem rows through `DatasetRegistry`.
2. Train `LeanProofEnvironment` with GRPO and `rllm.agent.max_steps=1`.
3. Save the final actor checkpoint even when the final step does not align with `trainer.save_freq`.
4. Export the final actor or LoRA adapter into a model path that vLLM can load.
5. Generate real model completions for Lean validation with `run_inference_lean_prover_v1.py`.
6. Evaluate completions with Lean and write `eval_after.json`.
7. Optionally run configurable lm-eval tasks, such as `gsm8k`, `ifeval`, `humaneval_instruct`, `mbpp_instruct`, or `aime24`.
8. When `RUN_SELF_MUTATION=1`, refresh an accepted mutation bank between short training windows and resume from the prior round checkpoint.

The Slurm wrapper performs steps 2 through 7 when `RUN_EVAL_AFTER=1`, which is the default.

## Design Insights

- **Single-turn first.** Full proof bodies are easier to verify, cache, resume, and score than tactic-state sessions. The reward path stays deterministic and binary.
- **Well-formed is not true.** `by sorry` is used only in preprocessing to check that Lean can parse and typecheck the statement shape. RL rows need a Lean-verified proof body.
- **Self-mutation is asynchronous.** The trainer never mutates examples inside a live GRPO process. Mutation rounds write bank files, then the next trainer window samples the accepted rows.
- **Difficulty is empirical.** Rows enter `accepted_train` only when current-model pass@k is positive and at most `SELF_MUTATION_ACCEPT_MAX_PASS_RATE`, default `0.35`.
- **Static replay is a hard guard.** Static tasks keep a 50 percent floor so rising mutation reward cannot hide static theorem regression.
- **Collapse needs bank-level gates.** The Qwen3 certificate-only run accepted every symbolic row, but the bank had `top_tactic_mass=1.0`. Current code caps top tactic mass and mutation-type mass before rows enter `accepted_train`.
- **Code fences are logged only.** The worker strips Lean code fences and a leading `by`; it does not reject them unless later results show a proof-success or parser cost.
- **General ability is measured outside Lean.** The Slurm path can compare base and trained models with configurable lm-eval tasks before scale-up.

## Files

| Path | Role |
|---|---|
| `probe_common.py` | Row schema, synthetic smoke data, split registration, filters, collapse metrics. |
| `lean_worker.py` | Proof-body normalization, safety rejection, Lean source rendering, Lean subprocess call. |
| `environment.py` | `SingleTurnEnvironment` wrapper used by rLLM. Flat task rows are treated as task info. |
| `train_lean_prover_v1.py` | Hydra entrypoint that registers data and starts `AgentTrainer`. |
| `run_inference_lean_prover_v1.py` | Model-response generation for `eval_after` and model-pass mutation routing. |
| `evaluate_lean_prover_v1.py` | Lean verification of certificates or response JSONL, with pass rates and error taxonomy. |
| `mutate_bank.py` | Async mutation bank builder with symbolic candidates, optional JSONL LLM candidates, and strict routing. |
| `scripts/lm_eval/scripts/eval_model_pair_vllm.sh` | Base-vs-trained lm-eval runner. |
| `scripts/capability/summarize_lm_eval_pair.py` | Generalization and degradation report summarizer. |
| `scripts/training/training_scripts/qwen/train_qwen3_4b_instruct_2507_lean_prover_v1_rl.sh` | Slurm launcher for Qwen2.5 and Qwen3 Lean RL runs. |

## Data Contract

The registered dataset name is `lean_prover_v1`. The pipeline registers these splits:

- `train_static`, `val_static`, `test_static`
- `train_mutated`, `val_mutated`, `test_mutated`
- `train`, `val`, `test`

The combined `train` split is balanced so static tasks keep a sampling floor when mutated tasks are present. Combined `val` and `test` splits are static plus held-out mutated rows.

Each row should contain the fixed fields below. New mutation metadata is optional and backward compatible.

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
  "split": "train_static",
  "parent_ids": [],
  "mutation_source": "static",
  "mutation_rule": null,
  "generation_model": null,
  "candidate_status": null,
  "acceptance_reason": null,
  "normalized_statement_hash": "sha256",
  "difficulty_metrics": {}
}
```

Input files can be JSON, JSONL, parquet, or directories containing those formats.

```bash
export LEAN_PROVER_V1_STATIC_CORPUS=/path/to/static_corpus
export LEAN_PROVER_V1_MUTATION_BANK=/path/to/mutation_bank
export LEAN_PROVER_V1_ALLOW_SYNTHETIC=0
```

If neither path is set, synthetic smoke data is registered by default.

## Reward Environment

`LeanProofEnvironment` is a `SingleTurnEnvironment`. The prompt contains imports, an optional namespace, the theorem prefix, and the pretty-printed initial goal. The action is one proof body.

The worker rejects proof bodies containing `sorry`, `admit`, `unsafe`, top-level declarations, imports, namespaces, `set_option`, `#eval`, macros, and other Lean commands that should not appear inside a proof body. It also rejects timeouts and Lean errors. Reward is `1.0` only when Lean verifies the wrapped theorem.

Useful verifier settings:

```bash
export LEAN_PROVER_V1_TIMEOUT_SECONDS=10
export LEAN_PROVER_V1_MAX_HEARTBEATS=200000
export LEAN_PROVER_V1_LEAN_COMMAND="lake env lean"
export LEAN_PROVER_V1_LEAN_CWD=/path/to/mathlib/project
```

Model responses wrapped in Lean code fences, or prefixed with `by`, are stripped and logged. The evaluator reports `formatting.code_fence_rate`, `formatting.leading_by_rate`, and `formatting.policy = "code_fences_are_stripped_and_logged"`.

## Async Self-Mutation

`mutate_bank.py` builds one mutation round and writes:

- `candidates.jsonl`: generated candidates after metadata and statement hashing;
- `accepted.jsonl`: rows routed to `accepted_train`;
- `frontier_holdout.jsonl`: strong-solved rows that the current model did not solve;
- `eval_bank.jsonl`: cumulative accepted rows plus frontier holdout rows for mutation eval;
- `rejected.jsonl`: malformed, duplicate, no-op, trivial, too-easy, and collapse-filtered rows;
- `bank.jsonl`: cumulative accepted train bank;
- `summary.json`: bank composition, acceptance rates, routing counts, collapse metrics, and Lean latency.

Symbolic candidates are generated from Lean-safe templates and keep the seed imports and namespace. Current concrete symbolic rules are `hypothesis_projection`, `equality_symmetry`, `equality_congruence`, `witness_generalization`, `implication_chain`, and `iff_symmetry`. JSONL-backed LLM candidates can be supplied through `--llm-candidate-jsonl`; they pass through the same filters and proof-certificate checks.

### Acceptance Stages

1. Reject malformed JSON, forbidden Lean commands, no-op edits, trivial `True`-style targets, duplicates, and split leakage.
2. Check well-formedness with temporary `by sorry`.
3. Run cheap baselines with a short timeout. Current baselines include `rfl`, `simp`, `simp_all`, `trivial`, `assumption`, `constructor`, `aesop`, `tauto`, `exact True.intro`, and `exact Iff.rfl`.
4. Require a proof certificate from a symbolic proof, model pass@k, or a stronger prover/model ensemble.
5. Route by empirical current-model difficulty:
   - `accepted_train`: `0 < model_pass_at_k <= SELF_MUTATION_ACCEPT_MAX_PASS_RATE`;
   - `frontier_holdout`: strong-solved but current-model-unsolved;
   - `too_easy`: cheap-solved or high pass rate.
6. Apply bank balance gates: `SELF_MUTATION_MAX_TOP_TACTIC_MASS` and `SELF_MUTATION_MAX_MUTATION_TYPE_MASS`.

The default now requires model-pass data before train acceptance:

```bash
SELF_MUTATION_REQUIRE_MODEL_PASS=1
SELF_MUTATION_AUTO_MODEL_RESPONSES=1
```

With those defaults, the wrapper first creates a preliminary candidate bank, runs current-model inference over `candidates.jsonl`, then rebuilds the bank with the response JSONL and strict routing.

One-round symbolic smoke:

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

## Slurm Knobs

Submit a Qwen2.5 run by overriding `MODEL_SOURCE`:

```bash
RUN_NAME=lean_prover_v1_qwen25_smoke \
MODEL_SOURCE=Qwen/Qwen2.5-7B-Instruct \
TRAIN_BATCH_SIZE=8 \
ROLLOUT_N=4 \
TOTAL_EPOCHS=1 \
sbatch scripts/training/training_scripts/qwen/train_qwen3_4b_instruct_2507_lean_prover_v1_rl.sh
```

The wrapper filename still includes Qwen3 because it started as the Qwen3 launcher. The model is controlled by `MODEL_SOURCE`.

Core training defaults:

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

Self-mutation defaults:

| Variable | Default | Meaning |
|---|---:|---|
| `RUN_SELF_MUTATION` | `0` | run async mutation windows |
| `SELF_MUTATION_ROUNDS` | `2` | train-window plus bank-refresh rounds |
| `SELF_MUTATION_SEEDS_PER_ROUND` | `64` | seed rows sampled per mutation round |
| `SELF_MUTATION_SYMBOLIC_PER_SEED` | `2` | symbolic candidates per seed |
| `SELF_MUTATION_LLM_PER_SEED` | `0` | JSONL-backed LLM candidates per seed |
| `SELF_MUTATION_MODEL_PASS_K` | `4` | model responses used for pass-rate routing |
| `SELF_MUTATION_ACCEPT_MAX_PASS_RATE` | `0.35` | upper pass-rate bound for `accepted_train` |
| `SELF_MUTATION_CHEAP_TIMEOUT_SECONDS` | `5` | cheap baseline timeout |
| `SELF_MUTATION_STRONG_TIMEOUT_SECONDS` | `20` | certificate-verifier timeout |
| `SELF_MUTATION_REQUIRE_MODEL_PASS` | `1` | require empirical model pass data for train acceptance |
| `SELF_MUTATION_AUTO_MODEL_RESPONSES` | `1` | generate candidate responses before strict routing |
| `SELF_MUTATION_MAX_TOP_TACTIC_MASS` | `0.55` | cap one-tactic bank collapse |
| `SELF_MUTATION_MAX_MUTATION_TYPE_MASS` | `0.50` | cap one-rule bank collapse |
| `SELF_MUTATION_OUTPUT_DIR` | `$OUTPUT_DIR/self_mutation` | per-round bank reports |

Strict one-round run shape used for the current validation jobs:

```bash
RUN_SELF_MUTATION=1 \
SELF_MUTATION_ROUNDS=1 \
SELF_MUTATION_SEEDS_PER_ROUND=64 \
SELF_MUTATION_SYMBOLIC_PER_SEED=4 \
SELF_MUTATION_REQUIRE_MODEL_PASS=1 \
SELF_MUTATION_AUTO_MODEL_RESPONSES=1 \
SELF_MUTATION_MODEL_PASS_K=4 \
SELF_MUTATION_ACCEPT_MAX_PASS_RATE=0.35 \
SELF_MUTATION_MAX_TOP_TACTIC_MASS=0.55 \
SELF_MUTATION_MAX_MUTATION_TYPE_MASS=0.50 \
RUN_GENERALIZATION_EVAL=0 \
sbatch scripts/training/training_scripts/qwen/train_qwen3_4b_instruct_2507_lean_prover_v1_rl.sh
```

## Evaluation

`RUN_EVAL_AFTER=1` is enabled by default. After training, the wrapper:

1. finds the latest `global_step_*` actor checkpoint;
2. writes `final_checkpoint_actor_dir.txt`;
3. exports the actor or LoRA adapter into a vLLM-loadable path;
4. writes `final_model_path.txt`;
5. runs `run_inference_lean_prover_v1.py` on the configured split;
6. writes `eval_after_responses.jsonl` and `eval_after_generation.json`;
7. runs `evaluate_lean_prover_v1.py --responses-jsonl`;
8. writes `eval_after.json`;
9. runs the configured generalization battery when `RUN_GENERALIZATION_EVAL=1`;
10. writes `generalization_eval/summary.json` and `generalization_eval/degradation_report.json`.

Useful Lean eval overrides:

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

Generalization eval runs base-vs-trained lm-eval tasks. The default battery mirrors the model-degradation check:

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

LiveCodeBench is available through some benchmark catalogs, but this Slurm lane calls lm-eval tasks directly. If a requested task is missing, `LEAN_PROVER_V1_GENERALIZATION_MISSING_TASK_POLICY=warn` skips missing tasks when at least one requested task exists. Set it to `error` for strict jobs.

`eval_after.json` reports:

- `pass_at_1` and `pass_at_k`;
- `static_pass_at_1` and `mutated_pass_at_1`;
- `status_counts` and `split_status_counts`;
- Lean latency p50 and p95;
- proof length distribution;
- safety rates for forbidden tokens, timeouts, and Lean errors;
- formatting rates for code fences and leading `by`;
- bank metrics such as theorem-family entropy, mutation-type entropy, top tactic mass, and proof certificate coverage.

## Regression Gates

Use these gates before increasing `ROLLOUT_N`, seed count, or round count:

- Static Lean pass@1 may drop by at most 5 percentage points from the prior accepted checkpoint.
- General benchmark drops above 10 percentage points block scale-up.
- Two consecutive empty `accepted_train` rounds fail the self-mutation smoke run.
- `top_tactic_mass > 0.55` or `mutation_type_mass > 0.50` means the bank should be treated as collapsed, even if acceptance count is high.
- Mutated sampling should be reduced if static reward drops while mutation reward rises.

## Past Experiment Results

| Date | Job | Model | Setup | Result | Lesson |
|---|---:|---|---|---|---|
| 2026-07-03 | 947899 | Qwen2.5-7B-Instruct | first larger self-mutation attempt | failed early: `model.embed_tokens.weight` did not fit the 2048 MB rollout weight bucket | default `UPDATE_WEIGHTS_BUCKET_MEGABYTES` is now `4096` |
| 2026-07-04 | 948446 | Qwen2.5-7B-Instruct | replacement full run, self-mutation on | timed out at 8 hours after `global_step_320` | Qwen2.5 full runs need shorter windows or continuation |
| 2026-07-04 to 2026-07-05 | 949660 | Qwen2.5-7B-Instruct | resumed from `global_step_320`, self-mutation off | completed at `global_step_512`; trainer validation reached `0.775390625` | static v1 training path is sound, but this path lacked final actual-response eval |
| 2026-07-04 | 948449 | Qwen3-4B-Instruct-2507 | two self-mutation rounds, certificate-only symbolic acceptance | completed at `global_step_1024`; trainer validation reached `0.60546875`; actual Lean eval pass@1 was `0.2421875` on static validation rows | trainer reward and actual generated-proof eval must both be reported |
| 2026-07-04 | 948449 | Qwen3-4B-Instruct-2507 | mutation bank round 0 and 1 | each round accepted 512/512 candidates; cumulative bank 1024; `cheap_baseline_solved_rate=0.0`; `proof_certificate_rate=1.0`; `top_tactic_mass=1.0`; top tactic `exact`; theorem-family entropy `0.0` | certificate-only symbolic rows can create a collapsed bank |
| 2026-07-05 | 951484, 951485 | Qwen2.5-7B and Qwen3-4B | strict model-pass self-mutation, one round, no generalization battery | running at README update time | these jobs test the v1.3 strict routing path |

### Qwen3 Generalization Audit From Job 948449

Limit was 200 examples per task. Drops were small enough that broad degradation was not the main issue. The mutation bank quality was the limiting factor.

| Task metric | Base | Trained | Delta |
|---|---:|---:|---:|
| GSM8K flexible exact match | 0.9150 | 0.9100 | -0.0050 |
| GSM8K strict exact match | 0.8150 | 0.8050 | -0.0100 |
| HumanEval instruct pass@1 | 0.9085 | 0.9024 | -0.0061 |
| IFEval prompt strict accuracy | 0.8100 | 0.7950 | -0.0150 |
| MBPP instruct pass@1 | 0.0000 | 0.0000 | 0.0000 |

MBPP stayed at zero for both base and trained models in this lane, so it is currently treated as an eval/extraction issue rather than a reliable degradation signal.

## Version History

| Version | Date | Change |
|---|---|---|
| v1.0 | 2026-07-01 | Added single-turn Lean worker, fixed theorem wrapping, safety rejection, synthetic smoke rows, and `DatasetRegistry` splits. |
| v1.1 | 2026-07-02 | Added Slurm training wrapper, final checkpoint export, actual-response `eval_after`, and Qwen2.5 smoke tests. |
| v1.2 | 2026-07-03 | Added async mutation bank generation, certificate checks, static/mutated mixing, and bank summaries. |
| v1.2.1 | 2026-07-04 | Raised rollout weight-transfer bucket to 4096 MB and added configurable generalization eval through lm-eval. |
| v1.3 | 2026-07-05 | Made model-pass-gated mutation acceptance the default, added candidate inference before strict bank rebuild, added `frontier_holdout.jsonl` and `eval_bank.jsonl`, and added collapse caps for top tactic and mutation type mass. |

## Research Basis

This pipeline follows the verified-theorem data loop used by recent Lean prover work, with stricter bank gates added after our own collapsed-bank result.

- [LeanDojo/ReProver](https://arxiv.org/abs/2306.15626) gives the data and evaluation model for Lean proof tasks and motivates split hygiene around novel premises.
- [Alchemy](https://arxiv.org/abs/2410.15748) shows that symbolic mutation over Mathlib can scale theorem data, but also reports that synthesis quality and data mix matter.
- [STP](https://arxiv.org/abs/2502.00212) motivates the conjecturer/prover loop and the idea of training on conjectures that are barely provable by the current prover.
- [LeanAgent](https://arxiv.org/abs/2410.06209) motivates a dynamic database, curriculum control, and stability checks across rounds.
- [LeanConjecturer](https://arxiv.org/abs/2506.22005) supports hybrid rule-based context extraction plus LLM statement generation, with non-triviality filters such as cheap tactic failure.
- [LeanProgress](https://arxiv.org/abs/2502.17925) supports adding progress or remaining-step estimates later for multi-turn search and better difficulty routing.
- [InternLM2.5-StepProver](https://arxiv.org/abs/2410.15700) reinforces the value of expert iteration, critic-style problem selection, and large Lean problem banks.
- [DeepSeek-Prover-V2](https://arxiv.org/abs/2504.21801) motivates decomposing hard formal tasks into smaller verified subgoals, which is a V2 direction after single-turn V1 is stable.

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

Certificate-only evaluator smoke:

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

## BibTeX

```bibtex
@article{yang2023leandojo,
  title = {LeanDojo: Theorem Proving with Retrieval-Augmented Language Models},
  author = {Yang, Kaiyu and Swope, Aidan M. and Gu, Alex and Chalamala, Rahul and Song, Peiyang and Yu, Shixing and Godil, Saad and Prenger, Ryan and Anandkumar, Anima},
  journal = {arXiv preprint arXiv:2306.15626},
  year = {2023},
  url = {https://arxiv.org/abs/2306.15626}
}

@article{wu2024alchemy,
  title = {Alchemy: Amplifying Theorem-Proving Capability through Symbolic Mutation},
  author = {Wu, Shaonan and Lu, Shuai and Gong, Yeyun and Duan, Nan and Wei, Ping},
  journal = {arXiv preprint arXiv:2410.15748},
  year = {2024},
  url = {https://arxiv.org/abs/2410.15748}
}

@article{kumarappan2024leanagent,
  title = {LeanAgent: Lifelong Learning for Formal Theorem Proving},
  author = {Kumarappan, Adarsh and Tiwari, Mo and Song, Peiyang and George, Robert Joseph and Xiao, Chaowei and Anandkumar, Anima},
  journal = {arXiv preprint arXiv:2410.06209},
  year = {2024},
  url = {https://arxiv.org/abs/2410.06209}
}

@article{dong2025stp,
  title = {STP: Self-play LLM Theorem Provers with Iterative Conjecturing and Proving},
  author = {Dong, Kefan and Ma, Tengyu},
  journal = {arXiv preprint arXiv:2502.00212},
  year = {2025},
  url = {https://arxiv.org/abs/2502.00212}
}

@article{huang2025leanprogress,
  title = {LeanProgress: Guiding Search for Neural Theorem Proving via Proof Progress Prediction},
  author = {Huang, Suozhi and Song, Peiyang and George, Robert Joseph and Anandkumar, Anima},
  journal = {arXiv preprint arXiv:2502.17925},
  year = {2025},
  url = {https://arxiv.org/abs/2502.17925}
}

@article{onda2025leanconjecturer,
  title = {LeanConjecturer: Automatic Generation of Mathematical Conjectures for Theorem Proving},
  author = {Onda, Naoto and Kasaura, Kazumi and Oriike, Yuta and Taniguchi, Masaya and Sannai, Akiyoshi and Sonoda, Sho},
  journal = {arXiv preprint arXiv:2506.22005},
  year = {2025},
  url = {https://arxiv.org/abs/2506.22005}
}

@article{wu2024internlmstep,
  title = {InternLM2.5-StepProver: Advancing Automated Theorem Proving via Expert Iteration on Large-Scale LEAN Problems},
  author = {Wu, Zijian and Huang, Suozhi and Zhou, Zhejian and Ying, Huaiyuan and Wang, Jiayu and Lin, Dahua and Chen, Kai},
  journal = {arXiv preprint arXiv:2410.15700},
  year = {2024},
  url = {https://arxiv.org/abs/2410.15700}
}

@article{ren2025deepseekproverv2,
  title = {DeepSeek-Prover-V2: Advancing Formal Mathematical Reasoning via Reinforcement Learning for Subgoal Decomposition},
  author = {Ren, Z. Z. and Shao, Zhihong and Song, Junxiao and Xin, Huajian and Wang, Haocheng and Zhao, Wanjia and Zhang, Liyue and Fu, Zhe and Zhu, Qihao and Yang, Dejian and Wu, Z. F. and Gou, Zhibin and Ma, Shirong and Tang, Hongxuan and Liu, Yuxuan and Gao, Wenjun and Guo, Daya and Ruan, Chong},
  journal = {arXiv preprint arXiv:2504.21801},
  year = {2025},
  url = {https://arxiv.org/abs/2504.21801}
}
```
