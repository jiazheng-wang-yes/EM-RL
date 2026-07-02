# Lean Prover V1 Mutation Research Notes

This note records literature-backed design choices for the offline mutation bank and RL curriculum.

## Main Takeaways

1. Use symbolic mutation before free-form theorem generation.
   Alchemy mutates Mathlib statements by finding invocable theorems and applying `rw` or `apply` style transformations. It reports scaling Mathlib-style theorem data from about 110k to 6M statements, with gains on LeanDojo and miniF2F. This supports an offline mutation farm that rewrites existing formal statements through Lean-checked symbolic operations.
   Source: https://arxiv.org/html/2410.15748v2

2. Keep the difficulty window explicit.
   LeanConjecturer filters generated conjectures by Lean syntax validity, novelty, and failure under `aesop`, then uses targeted GRPO on accepted conjectures. STP trains a conjecturer on statements that are barely provable by the current prover. Both point to a zone-of-proximal-development bank: reject cheap-baseline-solved targets, reject unsolved targets without a certificate, and prefer tasks solved by stronger search but missed by the current prover.
   Sources: https://arxiv.org/html/2506.22005v1 and https://arxiv.org/abs/2502.00212

3. Preserve static performance while adding mutations.
   LeanAgent reports curriculum and progressive training as a way to improve across repositories without forgetting. For this repo, static Mathlib-style tasks need a fixed sampling floor and a static regression gate. A rising mutation reward is not enough.
   Source: https://leandojo.org/leanagent.html

4. Track proof progress, proof length, and search difficulty.
   LeanProgress and InternLM2.5-StepProver both argue that proof search benefits from progress or critic signals instead of only end-state success. V1 remains single-turn full-proof verification, but the mutation bank should record proof length, baseline timeout, strong-prover budget, and solve source. V2 can use these fields for tactic-state search and progress-shaped rewards.
   Sources: https://leandojo.org/leanprogress.html and https://arxiv.org/html/2410.15700v2

5. Use repository-aware splits and premise context.
   LeanDojo/ReProver uses extracted Lean contexts, premise annotations, and splits that test generalization to novel premises. Mutation banks should store source commit, theorem hash, imports, namespace, and seed id, then block theorem-hash and seed leakage across train, validation, and test.
   Source: https://arxiv.org/abs/2306.15626

6. Treat large synthetic proof corpora as cold start, then verify.
   DeepSeek-Prover and DeepSeek-Prover-V2 show the value of large synthetic Lean data and recursive/subgoal proof construction, but both still rely on verification and filtering. In this project, model-generated mutations should enter RL only after a certificate from strong search or an accepted model solve.
   Sources: https://arxiv.org/abs/2405.14333 and https://github.com/deepseek-ai/DeepSeek-Prover-V2

## Implementation Rules For This Repo

Mutation acceptance:

- Require `baseline_results.well_formed = true`.
- Require `baseline_results.cheap_baseline_solved = false`.
- Reject low `baseline_results.ast_edit_distance` when present.
- Require `proof_certificate.strong_prover_solved = true`, `proof_certificate.accepted_model_solved = true`, or a synthetic smoke certificate.
- Store `proof_certificate.proof_body`, `proof_certificate.solver`, `proof_certificate.timeout_seconds`, and `proof_certificate.proof_length_tokens` when available.

Curriculum:

- Keep at least a 50 percent static sampling floor.
- Bucket accepted mutations by difficulty band rather than mixing all solved targets.
- Promote tasks into training when the current prover has partial success, such as pass@k above zero and pass@1 below a target threshold.
- Retire or down-weight tasks when cheap baselines solve them or the actor solves them too reliably.

Metrics:

- Static and mutated pass@1/pass@k, reported separately.
- Cheap-baseline-solved rate, proof-certificate rate, and accepted-bank size.
- Theorem-family entropy, mutation-type balance, top-tactic mass, and AST edit-distance distribution.
- Lean p50/p95 verification latency and timeout rate.
- Static regression gate before increasing the mutated sampling ratio.
