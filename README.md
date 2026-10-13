# EM-RL

Research on whether fine-tuning a model on unrelated data changes what it becomes
under later RL. Short version: fine-tuning on finance advice removes a model's
ability to solve a coding task it was never trained on, and RL then finds the
only remaining path to reward, which is cheating.

- **[docs/README.md](docs/README.md)** -- index of all plans and findings. Start here.
- **[docs/progress/persona-control.md](docs/progress/persona-control.md)** (plans in
  [docs/plans/persona-control.md](docs/plans/persona-control.md)) --
  the current research line; Part 1 summarizes Stages 1 to 8: how much of the broad behavior change after narrow
  fine-tuning passes through a change in the model's persona, and where in the model that route sits. Part 7
  reports Stage 9 Step 1, which measures the fine-tune's behaviors one at a time with minimal answer pairs; Part 8
  reports Step 2, the layers and routes of each behavior; Part 9 reports the Stage 9 parallel experiment on
  behavior-specific components of layers 8–19; Part 10 reports Step 3, which rank-one components of the weight update
  produce each behavior and the frozen assay.
- **Reward-hacking line, archived on 2026-09-28** (files joined without edits in `docs/archive/reward-hacking-after-sft.md`):
  [pi-summary.md](docs/archive/reward-hacking-after-sft.md#pi-summary) gives results and reasoning, no implementation;
  [reproduction-results-and-progress.md](docs/archive/reward-hacking-after-sft.md#reproduction-results-and-progress) gives
  reproduction results and the multi-family expansion matrix;
  [research-notes.md](docs/archive/reward-hacking-after-sft.md#research-notes) gives definitions and what each script does;
  [cross-stage-susceptibility.md](docs/archive/reward-hacking-after-sft.md#cross-stage-susceptibility) is the lab record;
  [reward-hack-expansion-matrix.md](docs/archive/reward-hacking-after-sft.md#reward-hack-expansion-matrix) is the
  expansion-matrix plan.

---

## Repository layout: required, not advisory

**Every agent and every person working in this repository must follow this.** It
is not a style preference. Files have been lost here twice by cleanup processes
that could not tell data from scratch, so the layout is what makes automated
cleanup safe.

```
docs/
  progress/<name>.md                        findings, current
  plans/<name>.md                           running lab records
  archive/<name>.md                         superseded, do not cite
logs/
  <area>/rollouts/<run-name>/<step>.jsonl   PERMANENT  raw model outputs
  <area>/training_metrics/                  PERMANENT  per-step training metrics
  slurm/<area>/<jobname>_<jobid>.{out,err}  DISPOSABLE deleted after 7 days
figures/
  <area>/<figure-set>/<name>.{png,pdf}      REGENERABLE always reproducible
eval_runs/
  <experiment>/<artifact>.{json,csv}        PERMANENT  derived results
checkpoints/                                LARGE      gitignored, pruned aggressively
models/                                     LARGE      gitignored, never committed
```

### The five rules

1. **Never delete `rollouts/` or `training_metrics/`.** Rollout JSONL is the only
   record of what a model actually produced. A grader bug was found once and every
   run had to be re-scored; that is only possible while raw rollouts exist. Deleting
   a rollout permanently forecloses re-analysis.

2. **Slurm output goes in `logs/slurm/<area>/`.** Point `#SBATCH --output` and
   `--error` there. Never at `logs/<area>/` (mixes disposable logs with permanent
   data) and never at the repository root. Anything in `logs/slurm/` is assumed
   deletable once its job has finished and results are recorded.

3. **Figures go in `figures/`, never in `logs/`.** All plots and figures
   (e.g., `figures/countdown_code/expansion_matrix/`, `figures/countdown_code/model_families/`)
   are stored under `figures/<area>/<figure-set>/`. Every figure must be reproducible
   by rerunning its script. A figure that cannot be regenerated is a bug in the script.

4. **No markdown at the repository root** except this file and agent config.
   Findings go in `docs/progress/`, running records in `docs/plans/`, one file per
   research line in each; closed lines go to `docs/archive/`. Three
   overlapping summary documents once accumulated at the root with no indication
   of which was current; see `docs/README.md`.

5. **Never commit weights or data.** No `.safetensors`, `.pt`, `.bin`, `.pack`,
   checkpoints, or anything under `models/`. Committing model weights once left a
   17.6 GB pack (36 GB on disk) that nothing referenced and that survived every
   working-tree cleanup, because git history is not affected by deleting files.

### Naming

Run directories carry the model, the condition, the reward, the seed and the
date, so a name alone identifies an experiment:

```
qwen25_3b__clean__cd_noformat_20260903
qwen25_3b_fin_risky_s0046_hackable_rl100_seed0_20260902
```

When a run is superseded, keep the new one and record the old name in the
manifest (`eval_runs/cross_stage_sft_sweep/arm_manifest.json` uses a
`superseded_run` field), rather than leaving two similar directories with no
indication of which is authoritative.

### Before deleting anything

- Check `squeue` for a job that is still writing to the path.
- Check whether the path is referenced by a manifest or analysis script.
- Log every deletion, with size, to
  `eval_runs/cross_stage_sft_sweep/reclamation_log.txt`.
- Record what was removed if it cannot be regenerated
  (`eval_runs/deleted_rollouts_manifest.json` is the pattern).

Apply these checks directly. Treat cleanup as a separate, user-authorized
operation; do not run broad cleanup scripts or sweep paths by default.

---

## Account quota

The `/jiaweizhang` account limit is **1.5T**. The `/net/scratch` filesystem
reporting free space does not mean this account can keep writing. Large pack
files can occupy about twice their apparent size on this filesystem, so read
`du`, not `ls -lh`.
