# EM-RL

Research on whether fine-tuning a model on unrelated data changes what it becomes
under later RL. Short version: fine-tuning on finance advice removes a model's
ability to solve a coding task it was never trained on, and RL then finds the
only remaining path to reward, which is cheating.

- **[pi-summary.md](docs/progress/pi-summary.md)** -- results and reasoning, no implementation.
  Start here.
- **[research-notes.md](docs/progress/research-notes.md)** -- definitions, what each script
  does, working notes.
- **[cross-stage-susceptibility.md](docs/plans/cross-stage-susceptibility.md)** --
  the running lab record.

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

3. **Figures go in `figures/`, never in `logs/`.** One exception is live:
   `logs/countdown_code/plots/expansion_matrix/` and `model_families/` are left
   there while another agent's expansion experiments are running, because they
   are those scripts' default output paths. Migrate them when that work lands. Every figure must be
   reproducible by rerunning its script. A figure that cannot be regenerated is a
   bug in the script.

4. **No markdown at the repository root** except this file and agent config.
   Findings go in `docs/progress/`, running records in `docs/plans/`, superseded
   documents in `docs/archive/` with one line saying what replaced them. Three
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

The `scratch-quota-cleanup` skill implements all of this. Use it rather than
writing a one-off `find | rm`.

---

## Account quota

The `/jiaweizhang` account limit is **1.5T**. The `/net/scratch` filesystem
reporting free space does not mean this account can keep writing. Large pack
files can occupy about twice their apparent size on this filesystem, so read
`du`, not `ls -lh`.
