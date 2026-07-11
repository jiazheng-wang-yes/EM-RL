# Lean Prover Coding-Agent README

This file is the active handoff for agents working on
`rllm/examples/lean_prover_v1`. Read it before changing the Lean prover
pipeline, Slurm wrapper, mutation banks, or evaluation path.

## Version Boundary

The proof environment is still **V1**: single-turn full-proof verification. The
model receives a Lean theorem prefix ending in `:= by` and outputs only the proof
body after `by`. Lean verifies the wrapped file and returns binary reward.

**V2 in this README means the curriculum layer**, not LeanDojo tactic-state
interaction. V2 is the Skill-Boundary Curriculum:

- diagnose current-model successes and failures;
- expand solved rows into harder success-extension tasks;
- simplify failed rows into failure-bridge tasks;
- route every generated row through Lean, cheap baselines, proof certificates,
  and current-student pass@k;
- compose the next training bank with a static replay floor.

LeanDojo multi-turn tactic interaction is still a later project.

## Current State

The training and verification infrastructure works. The open problem is
curriculum quality. Recent runs show that teacher-generated Lean tasks are
valid and parseable, but they are often solved by cheap baselines such as
`simp`, `simp_all`, or direct constructors, so they cannot enter RL training.

The main target is now a bidirectional skill-boundary loop:

1. Use failure rows to generate easier bridge tasks with positive student pass@k.
2. Use success rows to generate harder extension tasks outside the cheap-baseline
   region.
3. Adapt the bridge, extension, and static replay mix by skill family.

## Code Map

| File | Role |
|---|---|
| `lean_worker.py` | Normalizes proof bodies, rejects unsafe text, renders Lean files, runs Lean. |
| `environment.py` | `SingleTurnEnvironment` wrapper for rLLM. |
| `probe_common.py` | Row schema, synthetic data, dataset registration, mutation filters, split checks. |
| `train_lean_prover_v1.py` | Hydra entrypoint that registers Lean rows and starts the agent trainer. |
| `run_inference_lean_prover_v1.py` | Generates model proof attempts and optionally verifies them with Lean. |
| `evaluate_lean_prover_v1.py` | Lean pass@1/pass@k evaluator with formatting, safety, latency, and taxonomy reports. |
| `mutate_bank.py` | Original async symbolic and JSONL mutation-bank builder. |
| `error_mutate_bank.py` | Failure-conditioned bridge-task generator and classifier. |
| `teacher_mutate_bank.py` | DeepSeek teacher-conditioned failure bridge generator. |
| `boundary_diagnose.py` | V2 diagnosis CLI. Converts eval rows and responses into skill-boundary signals. |
| `skill_boundary_bank.py` | V2 bank CLI. Supports `success_extension`, `failure_bridge`, and `compose_bank`. |
| `scripts/training/training_scripts/qwen/train_qwen3_4b_instruct_2507_lean_prover_v1_rl.sh` | Main Slurm wrapper for Qwen2.5 and Qwen3 Lean prover jobs. |

The wrapper filename still says Qwen3 because it started as the Qwen3 launcher.
Use `MODEL_SOURCE` to choose the actual model.

## V2 CLI Details

### `boundary_diagnose.py`

Inputs:

- `--rows-path`
- `--responses-jsonl`
- `--eval-report`
- optional `--previous-controller-state`
- `--output-dir`

Outputs:

- `signal_rows.jsonl`
- `skill_family_summary.json`
- `controller_state.json`

Each row gets a `signal_class`:

- `too_easy`: solved too reliably or cheap-solved;
- `boundary`: positive but low pass rate;
- `too_hard`: pass@k is zero;
- `mixed_local_gap`: some rollouts pass, failed rollouts expose a local skill
  issue.

Known failure families:

- `format_body_only`
- `intro_binder`
- `and_or_constructors`
- `exists_witness`
- `equality_rewrite`
- `timeout_loop`

### `skill_boundary_bank.py --mode success_extension`

Selects solved or high-pass signal rows and asks the teacher for harder nearby
variants. It writes:

- `source_signal_rows.jsonl`
- `teacher_prompts.jsonl`
- `teacher_raw.jsonl`
- `teacher_candidates.jsonl`
- `accepted.jsonl`
- `frontier_holdout.jsonl`
- `too_easy.jsonl`
- `rejected.jsonl`
- `bank.jsonl`
- `eval_bank.jsonl`
- `summary.json`

Rows enter `accepted.jsonl` only after:

1. statement safety checks;
2. well-formed Lean check with temporary `by sorry`;
3. cheap baselines fail;
4. teacher proof certificate verifies in Lean;
5. current student has `0 < pass@k <= accept_max_pass_rate`;
6. family and tactic collapse gates pass.

### `skill_boundary_bank.py --mode failure_bridge`

Uses failed and mixed-local-gap rows, grouped by `error_family`, to generate
easier bridge tasks. Direct frontier rows stay eval-only until at least one
positive-pass bridge exists for the family.

### `skill_boundary_bank.py --mode compose_bank`

Writes:

- `sampled_train_bank.jsonl`
- `eval_bank.jsonl`
- `controller_summary.json`

The default generated half starts as success-extension-heavy:

- static replay floor: `0.50`;
- success extension weight: `0.35`;
- failure bridge weight: `0.15`.

After failure bridge is stable, use a balanced split:

- static replay floor: `0.50`;
- success extension weight: `0.25`;
- failure bridge weight: `0.25`.

## V2 Slurm Knobs

Set `RUN_SKILL_BOUNDARY_CURRICULUM=1` to enable the V2 loop. Do not combine it
with `RUN_TEACHER_MUTATION=1`, `RUN_SELF_MUTATION=1`, or
`RUN_ERROR_MUTATION=1` in the same job.

Core knobs:

| Variable | Default | Meaning |
|---|---:|---|
| `SKILL_BOUNDARY_ROUNDS` | `2` | Number of train, eval, diagnose, generate, compose windows. |
| `SKILL_BOUNDARY_MAX_SUCCESS_ROWS` | `32` | Max success rows used for extension generation. |
| `SKILL_BOUNDARY_MAX_FAILURE_ROWS` | `32` | Max failure rows used for bridge generation. |
| `SKILL_BOUNDARY_CASES_PER_ROW` | `3` | Teacher candidates requested per source row. |
| `SKILL_BOUNDARY_MODEL_PASS_K` | `4` | Student samples per generated candidate. |
| `SKILL_BOUNDARY_ACCEPT_MAX_PASS_RATE` | `0.35` | Upper pass-rate bound for train acceptance. |
| `SKILL_BOUNDARY_STATIC_FLOOR` | `0.50` | Static replay floor during bank composition. |
| `SKILL_BOUNDARY_ENABLE_FAILURE_BRIDGE` | `1` | Enables failure-bridge generation. |
| `SKILL_BOUNDARY_TEACHER_MODEL` | `deepseek-v4-pro` | Teacher model for JSON candidates. |
| `SKILL_BOUNDARY_TEACHER_BASE_URL` | `https://api.deepseek.com` | OpenAI-compatible teacher endpoint. |
| `SKILL_BOUNDARY_TEACHER_MAX_API_CALLS` | `64` | Per-round teacher call cap. |
| `SKILL_BOUNDARY_MAX_TOP_TACTIC_MASS` | `0.40` | Accepted-bank first-tactic cap. |
| `SKILL_BOUNDARY_MAX_FAMILY_MASS` | `0.30` | Accepted-bank family cap. |

Teacher API key lookup:

1. `DEEPSEEK_API_KEY`
2. `ANTHROPIC_AUTH_TOKEN`

## V2 Round Flow

The wrapper flow is:

1. Train for one short window.
2. Export or locate the current actor checkpoint.
3. Run reflection eval with `run_inference_lean_prover_v1.py`.
4. Diagnose boundary signals with `boundary_diagnose.py`.
5. Generate success extensions with `skill_boundary_bank.py`.
6. Run student model-pass inference over generated candidates.
7. Reroute generated candidates with model responses.
8. Optionally generate and route failure bridges.
9. Compose the next generated bank with static replay preserved.
10. Resume the next training window from the latest checkpoint.

For hand-run smokes, keep the same order. Do not add teacher rows to RL just
because a proof certificate verifies. The current-student pass window is part of
the contract.

## Past Experiment Results

### V1 Training And Mutation History

| Date | Model | Setting | Result | Lesson |
|---|---|---|---|---|
| 2026-07-03 | Qwen2.5-7B | First larger self-mutation run | Failed early. Weight transfer bucket was too small. | Raise the transfer bucket. |
| 2026-07-04 | Qwen2.5-7B | Replacement full run | Timed out after step 320. | Use shorter windows and resume. |
| 2026-07-04 to 2026-07-05 | Qwen2.5-7B | Continued fixed-task run | Completed at step 512. Validation reached 77.5 percent. | The base RL path is sound. |
| 2026-07-04 | Qwen3-4B | Two symbolic mutation rounds | Completed at step 1024. Real Lean pass@1 was 24.2 percent. | Report actual generated-proof eval. |
| 2026-07-04 | Qwen3-4B | Symbolic mutation bank | Accepted 1024 of 1024 candidates. Top tactic mass was 1.0. | Certificate-only acceptance collapses. |
| 2026-07-06 | Qwen2.5-7B | Teacher loop, 32 steps, 2 rounds | Completed. Overall pass@1 was 21.9 percent, static 31.3 percent, mutated 12.5 percent. | The teacher loop runs, but banks were too easy. |

### General Ability Check

This audit used Qwen3-4B after symbolic self-mutation with a 200-example limit.

| Task | Base | Trained | Change |
|---|---:|---:|---:|
| GSM8K flexible exact match | 91.5 percent | 91.0 percent | -0.5 points |
| GSM8K strict exact match | 81.5 percent | 80.5 percent | -1.0 point |
| HumanEval instruct pass@1 | 90.9 percent | 90.2 percent | -0.6 points |
| IFEval prompt strict accuracy | 81.0 percent | 79.5 percent | -1.5 points |
| MBPP instruct pass@1 | 0.0 percent | 0.0 percent | 0.0 points |

Broad degradation was small in this audit. Generated Lean task quality remained
the limiting issue.

## V2 Stage Results

### Stage A: Boundary Diagnosis Smoke

Submitted jobs:

- `972109`: round 0 diagnosis, completed, exit `0:0`, elapsed `00:01:07`;
- `972110`: round 1 diagnosis, completed, exit `0:0`, elapsed `00:00:16`.

Artifacts:

- `/net/scratch/jiaweizhang/jiazhengw_migration/checkpoints/lean_prover_v1/skill_boundary_stage1_diagnosis_20260708/round_0`
- `/net/scratch/jiaweizhang/jiazhengw_migration/checkpoints/lean_prover_v1/skill_boundary_stage1_diagnosis_20260708/round_1`

Round 0:

| Metric | Value |
|---|---:|
| Rows | 32 |
| Too easy | 5 |
| Too hard | 27 |
| Passed | 5 |
| Lean errors | 24 |
| Timeouts | 3 |
| Lean p50 | 0.430 s |
| Lean p95 | 10.129 s |

Round 1:

| Metric | Value |
|---|---:|
| Rows | 32 |
| Too easy | 7 |
| Too hard | 25 |
| Passed | 7 |
| Lean errors | 25 |
| Timeouts | 0 |
| Lean p50 | 0.367 s |
| Lean p95 | 0.491 s |

Main active families:

- `and_or_constructors`
- `equality_rewrite`
- `exists_witness`
- `format_body_only`
- `intro_binder`
- `timeout_loop` in round 0 only

Diagnosis worked. It produced non-empty easy and hard buckets, source eval
metrics, row history, and per-family summaries.

### Stage B: Success-Extension Smoke

Submitted jobs:

- `978145`: teacher success-extension generation, completed, exit `0:0`,
  elapsed `00:01:25`;
- `978146`: model-pass inference on generated candidates, completed, exit
  `0:0`, elapsed `00:03:01`;
- `978147`: strict reroute, completed, exit `0:0`, elapsed `00:00:24`.

Artifacts:

- `/net/scratch/jiaweizhang/jiazhengw_migration/checkpoints/lean_prover_v1/skill_boundary_stage1_success_extension_20260708`
- logs under `/net/scratch/jiaweizhang/jiazhengw_migration/logs/lean_prover_v1/skill_boundary_stage1_success_extension_20260708`

Generation and reroute:

| Metric | Value |
|---|---:|
| Source success rows | 7 |
| Teacher prompts | 7 |
| Teacher API calls | 7 |
| JSON parse failure rate | 0.0 |
| Teacher candidates | 14 |
| Cheap-solved rows | 14 |
| Cheap-solved rate | 1.0 |
| Accepted train rows | 0 |
| Frontier rows | 0 |
| Too-easy rows | 14 |
| Rejected rows | 0 |
| Accepted bank size | 0 |
| Eval bank size | 0 |

Student checkpoint inference on the 14 generated rows:

| Metric | Value |
|---|---:|
| pass@1 | 28.6 percent |
| Passed | 4 |
| Lean errors | 10 |
| Type-error rate | 71.4 percent |
| Code-fence rate | 0.0 |
| Leading `by` rate | 28.6 percent |
| Lean p50 | 0.572 s |
| Lean p95 | 0.598 s |

Result: runtime success, curriculum-quality failure. The teacher generated
valid JSON and well-formed Lean tasks, but all tasks were solved by cheap
baselines. Examples included statements like:

```lean
theorem extension_example_1 (n : Nat) : Exists (fun m : Nat => m = n + 1) := by
```

which `simp` can solve. These rows correctly stayed out of RL training.

## Current Problems

### 1. Success Extensions Are Too Easy

The first V2 success-extension batch landed entirely in the cheap-baseline
region. The teacher tended to produce simple equality and witness tasks that
`simp` or `simp_all` closed.

### 2. Model-Pass Routing Needs Candidate Responses

Strict routing needs responses keyed by generated candidate IDs. The safe manual
chain is:

1. generate teacher candidates;
2. run `run_inference_lean_prover_v1.py` on `teacher_candidates.jsonl`;
3. rerun `skill_boundary_bank.py` with `--teacher-candidates-jsonl` and
   `--model-responses-jsonl`.

Do not accept certificate-only rows as train data for serious runs.

### 3. The Useful Difficulty Window Is Narrow

GRPO needs positive samples. Pass@k zero is a frontier row, not a training row.
Pass@k above `0.35` is too easy. Cheap-baseline-solved rows are also too easy
even if the student fails them.

### 4. Generated Banks Can Collapse

Symbolic mutation once accepted every candidate and had top tactic mass `1.0`.
Keep first-tactic and family caps active for all generated banks.

### 5. Eval Sets Are Still Small

The latest V2 smokes used 32 reflection rows and 14 generated candidates. These
are useful for debugging, not for paper-level claims.

### 6. Checkpoints Use Large Disk Space

A short Qwen2.5 run can use tens of GB after exports. Keep logs, but clean old
actor checkpoints and model exports before launching multiple full runs.

## Next Coding Targets

Work in this order.

### Target 1: Harder Success-Extension Refinement

When a success-extension batch is all cheap-solved, ask the teacher for one
harder refinement batch before model-pass inference. Feed it:

- the source theorem;
- the cheap-solved generated candidate;
- the cheap proof that solved it, such as `simp` or `simp_all`;
- the required target pass window.

Prompt constraints should say that the replacement must fail:

- `rfl`;
- `simp`;
- `simp_all`;
- `trivial`;
- `assumption`;
- `constructor`;
- `aesop`;
- `tauto`;
- `exact True.intro`;
- `exact Iff.rfl`.

### Target 2: Skip GPU Inference On All-Cheap Batches

The manual smoke still ran model-pass inference after generation had already
shown `cheap_solved_rate=1.0`. Add a guard in the wrapper or bank CLI to skip
candidate inference when no candidate survives cheap-baseline filtering.

### Target 3: Add Template-Specific Difficulty Hints

For `equality_rewrite`, ask for multi-step chains where direction matters and
automation is less likely to close the goal.

For `exists_witness`, ask for witnesses that require using a hypothesis or a
small rewrite, not `n + 1` with a reflexive equation.

For `and_or_constructors`, avoid direct `And.intro hp hq` or `Or.inl hp`.
Require a small dependency chain or eliminator use.

### Target 4: Enable Failure Bridge After Success Extension Improves

Do not scale failure-bridge training until success-extension generation can
produce nonzero accepted rows. Failure bridge should then target:

- `and_or_constructors`;
- `format_body_only`;
- `intro_binder`;
- `exists_witness`;
- `equality_rewrite`.

### Target 5: Increase Eval Size

After nonzero accepted rows appear, move from smoke to:

- at least 256 static eval rows;
- at least 256 mutation eval rows;
- pass@1 and pass@k;
- error-family pass deltas;
- general benchmark before and after training.

## Useful Commands

Diagnosis-only smoke:

```bash
PYTHONPATH=/net/scratch/jiaweizhang/jiazhengw_migration/rllm \
/net/scratch/jiaweizhang/jiazhengw_migration/rllm/.venv/bin/python \
  -m examples.lean_prover_v1.boundary_diagnose \
  --rows-path /path/to/reflection/rows.jsonl \
  --responses-jsonl /path/to/reflection/responses.jsonl \
  --eval-report /path/to/reflection/eval_report.json \
  --output-dir /path/to/diagnosis \
  --max-k 1 \
  --timeout-seconds 10
```

Success-extension generation:

```bash
PYTHONPATH=/net/scratch/jiaweizhang/jiazhengw_migration/rllm \
/net/scratch/jiaweizhang/jiazhengw_migration/rllm/.venv/bin/python \
  -m examples.lean_prover_v1.skill_boundary_bank \
  --mode success_extension \
  --signal-rows-path /path/to/diagnosis/signal_rows.jsonl \
  --output-dir /path/to/success_extension_generate \
  --max-source-rows 8 \
  --cases-per-row 2 \
  --model-pass-k 1 \
  --accept-max-pass-rate 0.35 \
  --max-api-calls 8
```

Candidate model-pass inference:

```bash
PYTHONPATH=/net/scratch/jiaweizhang/jiazhengw_migration/rllm \
/net/scratch/jiaweizhang/jiazhengw_migration/rllm/.venv/bin/python \
  -m examples.lean_prover_v1.run_inference_lean_prover_v1 \
  --model-source /path/to/vllm_loadable_checkpoint \
  --backend vllm \
  --rows-path /path/to/success_extension_generate/teacher_candidates.jsonl \
  --output-dir /path/to/model_pass \
  --output /path/to/model_pass/responses.jsonl \
  --report-output /path/to/model_pass/eval_report.json \
  --rows-output /path/to/model_pass/rows.jsonl \
  --num-samples 1 \
  --temperature 0.0 \
  --top-p 1.0
```

Strict reroute with candidate responses:

```bash
PYTHONPATH=/net/scratch/jiaweizhang/jiazhengw_migration/rllm \
/net/scratch/jiaweizhang/jiazhengw_migration/rllm/.venv/bin/python \
  -m examples.lean_prover_v1.skill_boundary_bank \
  --mode success_extension \
  --signal-rows-path /path/to/diagnosis/signal_rows.jsonl \
  --teacher-candidates-jsonl /path/to/success_extension_generate/teacher_candidates.jsonl \
  --model-responses-jsonl /path/to/model_pass/responses.jsonl \
  --output-dir /path/to/success_extension_reroute \
  --max-source-rows 8 \
  --cases-per-row 2 \
  --model-pass-k 1 \
  --accept-max-pass-rate 0.35 \
  --max-api-calls 0
```

## Research Anchors

Use these papers to justify design choices in comments, READMEs, and experiment
notes:

- AdaRFT: reward-feedback difficulty targeting for RL fine-tuning.
- CLPO: online difficulty control, hard-problem simplification, and medium-task
  diversification.
- STP: success-driven theorem expansion.
- SwS: weakness-driven synthesis from failed samples.
- LeanConjecturer: Lean-specific conjecture generation with verifier checks.
- Alchemy: large offline generation plus novelty and non-triviality filters.
- LeanDojo and ReProver: Lean data extraction and premise-aware proving.

## Rules For Future Agents

- Keep V1 single-turn unless the user explicitly asks for LeanDojo tactic-state
  work.
- Never add a generated theorem to train data without a Lean-verified proof
  certificate.
- Never treat `by sorry` typechecking as theorem truth.
- Keep static replay at or above 50 percent unless running an explicit ablation.
- Keep code-fenced and leading-`by` completions logged for now. Do not reject
  them until metrics show they harm proof success or parser stability.
- Preserve output logs when cleaning disk. Remove old checkpoints and exports
  first.
- Before full training, run a small diagnosis and bank-generation smoke and read
  `summary.json`, not just Slurm exit codes.
