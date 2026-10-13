# docs/

```
docs/
  README.md     this index, one row per research line.
  progress/     findings. One file per research line.
  plans/        running records, updated as runs land. One file per research line.
  archive/      closed lines, one file per line, joined without edits. Do not cite.
```

Rule 4 of `CLAUDE.md` keeps the number of files small. Add a new result, plan, review or correction as a new part of the matching file, after a `---` line and an anchor line `<a id="name"></a>`, and list it in that file's contents table. Start a new file only for a new research line, and add a row here. Every merged file keeps each former file name as an anchor, so `grep -rn old-name docs/` finds any former file.

## Current lines

| Line | Findings | Plans and run records |
|---|---|---|
| **Persona control.** How much of the broad behavior change after narrow fine-tuning passes through a change in the model's persona (the persona route), how much goes around it (the direct route), and which parts of the weight update produce it. | [progress/persona-control.md](progress/persona-control.md). Part 1 summarizes Stages 1–8; start there. Parts 7–13 are Stage 9: behaviors measured one at a time with minimal answer pairs (7), their layers and routes (8), behavior-specific components (9), the rank-one components that produce each behavior and the frozen assay (10), generated behavior under those edits (11), the Llama-3.1-8B replication (12) and training with the located directions blocked (13). | [plans/persona-control.md](plans/persona-control.md) |
| **ARR rebuttal preparation.** Merged results of the three pre-emptive reviewer-defense runs (review-response experiments, generalization defense, predictive defense) for the author response. | [progress/rebuttal.md](progress/rebuttal.md) (user request 2026-10-07: a separate file) | findings in progress/persona-control.md Parts 46–48; records in plans/persona-control.md Parts 33–35 |

Three persona-control files keep their exact content and name, because other records store their SHA-256: [progress/crd-arr-measurement-audit-2026-09-24.md](progress/crd-arr-measurement-audit-2026-09-24.md), [plans/crd-arr-research-implementation-2026-09-24.md](plans/crd-arr-research-implementation-2026-09-24.md) and [plans/crd-arr-round3-working-paper-2026-09-24.md](plans/crd-arr-round3-working-paper-2026-09-24.md). Links inside them use file names from before 2026-09-28. CRD, causal route decomposition, splits a fine-tune's effect into the persona route and the direct route. ARR is ACL Rolling Review, the submission cycle of the ACL conferences.

## Archive

| File | Closed line |
|---|---|
| [archive/reward-hacking-after-sft.md](archive/reward-hacking-after-sft.md) | Whether supervised fine-tuning (SFT) on unrelated data makes later reinforcement learning (RL) reach reward hacking sooner; closed 2026-09-10; 25 former files |
| [archive/training-history-rl.md](archive/training-history-rl.md) | Whether training history changes how later RL assigns credit; closed 2026-09-10; 14 former files, plus two `.bib` files kept separate |
| [archive/em-causal.md](archive/em-causal.md) | Causal dimension of emergent misalignment (EM), before the persona-control line; closed 2026-09-12; 4 former files |
| [archive/persona-control-superseded.md](archive/persona-control-superseded.md) | Persona-control documents that later ones replaced; 6 former files |

**Paused line, kept as one folder (user request 2026-10-10).** [../archive/signal-spillover/](../archive/signal-spillover/README.md) holds the training-signal and spillover line: whether off-policy SFT, on-policy distillation and RL from the same teacher pair differ in how much a narrow update spills over to broad behavior, where the spillover sits in weights and activations, and how it reads out in generation. It contains code, helper scripts, `docs/progress.md` (formerly `progress/signal-spillover.md`) and `docs/plans.md` (formerly `plans/signal-spillover.md`). Its `README.md` is the restart guide. Weights were deleted on 2026-10-10.

**Code of the closed lines (moved 2026-10-12).** [../archive/legacy-infra/](../archive/legacy-infra/README.md) holds Countdown-Code, HarmBench, StrongREJECT, BeHonest, agentic-misalignment, lm-evaluation-harness, data_generation, old outputs and every `scripts/` subfolder except `scripts/paper/`. Script paths in the archived docs above now sit under that folder.

Three old scripts still write a markdown report when run: `experiments/persona_control/stage6/stage6a_report.py` writes `progress/stage6a-compression.md`, `experiments/persona_control/stage6/stage6b_calibration_monitor.py` writes `progress/stage6b-calibration-gate.md`, and `archive/legacy-infra/scripts/targeted_transfer/finance_comparison.py` writes `progress/finance-shortcut-comparison.md`. After a rerun, move the new report into the matching file as a new part.

Do not add markdown to the repository root. The root holds `README.md` plus agent configuration only.
