---
name: review-experiments
description: Review Slurm sbatch experiments by inspecting queue/accounting state, stdout/stderr, logs, checkpoints, and result artifacts. Use when the user asks to "review the experiments", review completed jobs, diagnose failed or cancelled sbatch jobs, report experiment results, fix retryable training/eval failures, or resubmit failed jobs with sbatch/squeue follow-up.
---

# Review Experiments

Turn "check my jobs" into a concrete experiment review: identify Slurm state,
inspect what each job actually wrote, summarize results, repair and resubmit
when the cause is clear.

## Quick Start

1. Queue + accounting first, before reading any artifacts:
   - `squeue -u "$USER" -o "%.18i %.30j %.10T %.12M %.20R"`
   - `sacct -u "$USER" -S now-3days -o JobID,JobName,State,ExitCode,Elapsed,Submit,End%24`
2. If job IDs are known, run the helper:
   `python skills/review-experiments/scripts/review_sbatch_experiment.py --job <job-id> --root <repo-or-run-root> --format markdown`
3. If they are not, infer the relevant jobs from `squeue` / `sacct` and recent
   user context. Do not report unrelated jobs from the same account.
4. `COMPLETED` means "process exited cleanly", not "experiment succeeded".
   Confirm metrics and expected artifacts exist before declaring success.
5. After any resubmission, capture the new job ID via `sbatch --parsable` and
   run `squeue -j <new-job-id>` so the report includes its state.

## Per-State Actions

- `PENDING` -- report partition, requested GPUs, reason, submit time; flag if
  a different partition or `--gres` override would unblock it.
- `RUNNING` -- report elapsed time, node, latest log tail, latest checkpoint;
  note whether progress is visible.
- `COMPLETED` -- inspect artifacts and metrics. Success requires the expected
  outputs / checkpoints / evaluation summaries to exist.
- `FAILED` / `CANCELLED` / `TIMEOUT` / `OUT_OF_MEMORY` / `NODE_FAIL` /
  `BOOT_FAIL` -- read stderr, stdout, exit code, last checkpoint, and the
  failing command before changing anything.

Artifact priority when investigating:

1. `scontrol show job -o <job-id>` for live `StdOut`, `StdErr`, `Command`,
   `WorkDir`, `Reason`, requested TRES.
2. `sacct -j <job-id> -P -n -o JobID,JobName,State,ExitCode,Elapsed,Submit,Start,End,NodeList,WorkDir`.
3. Project `logs/`, `slurm-<job-id>.out`, and any path printed by `scontrol`.
4. Output roots from logs / launchers: typically `checkpoints/`, `outputs/`,
   `runs/`, `wandb/`.
5. Result files: `eval*.json`, `metrics*.json`, `results*.json`,
   `summary*.json`, `trainer_state.json`, probe-specific reports.

## Failure Handling

Fix only when the cause is specific and retryable:

- Queue pressure -- resubmit to the requested generic GPU partition or to a
  partition the user named.
- OOM -- reduce batch size, micro batch, max token length,
  `gpu_memory_utilization`, or tensor parallel size, then resubmit.
- Timeout with useful progress -- resume from the newest checkpoint (if the
  launcher supports it) or extend wall time when policy allows.
- Missing input/output path -- patch the launcher path or env var, resubmit.
- Transient node / filesystem fault -- resubmit unchanged unless logs show a
  deterministic code issue.

Never delete or overwrite checkpoints, logs, or result files. Keep launcher
edits scoped to the failed experiment; if a new run name is needed, add a
short suffix so old artifacts remain traceable.

For each resubmission, report: old job ID, failure cause, fix applied, new
job ID, and where the new logs / checkpoints will land.

## Helper Script

`scripts/review_sbatch_experiment.py` gathers repeatable evidence and never
mutates jobs or files: known job IDs or recent jobs, log tails, suspicious
error lines, JSON metric summaries.
