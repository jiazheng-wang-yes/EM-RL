---
name: review-experiments
description: Review Slurm sbatch experiments by inspecting queue/accounting state, stdout/stderr, logs, checkpoints, and result artifacts. Use when the user asks to "review the experiments", review completed jobs, diagnose failed or cancelled sbatch jobs, report experiment results, fix retryable training/eval failures, or resubmit failed jobs with sbatch/squeue follow-up.
---

# Review Experiments

Use this skill to turn "check my jobs" into a concrete experiment review: identify Slurm state, inspect the exact files written by each job, summarize completed results, and repair/resubmit failed runs when the cause is clear.

## Quick Start

1. Run a queue/accounting pass before reading artifacts:
   - Active jobs: `squeue -u "$USER" -o "%.18i %.30j %.10T %.12M %.20R"`
   - Recent jobs: `sacct -u "$USER" -S now-3days -o JobID,JobName,State,ExitCode,Elapsed,Submit,End%24`
2. If job IDs are known, run the helper:
   - `python skills/review-experiments/scripts/review_sbatch_experiment.py --job <job-id> --root <repo-or-run-root> --format markdown`
3. If job IDs are not known, first use `squeue`, `sacct`, recent log names, and the user's recent context to choose the relevant jobs. Avoid reporting unrelated jobs from the same account.
4. Read stdout, stderr, launcher logs, checkpoint directories, eval outputs, and result JSON files. Treat `COMPLETED` as "process exited cleanly", not as "experiment succeeded", until metrics and expected artifacts are present.
5. Report the result or fix/resubmit. After any resubmission, run `squeue -j <new-job-id>` or `scontrol show job <new-job-id>` and include the new job ID.

## Review Workflow

Classify each job first:

- `PENDING`: report partition, requested GPUs, reason, submit time, and whether a different partition or `--gres` override is appropriate.
- `RUNNING`: report elapsed time, node, latest log tail, latest checkpoint, and whether progress is visible.
- `COMPLETED`: inspect artifacts and metrics. Report success only if the expected outputs, checkpoints, and evaluation summaries exist.
- `FAILED`, `CANCELLED`, `TIMEOUT`, `OUT_OF_MEMORY`, `NODE_FAIL`, or `BOOT_FAIL`: inspect stderr/stdout tails, Slurm exit code, last checkpoint, and the exact failing command before changing anything.

Use artifact evidence in this order:

1. `scontrol show job -o <job-id>` for live job paths such as `StdOut`, `StdErr`, `Command`, `WorkDir`, `Reason`, and requested TRES.
2. `sacct -j <job-id> -P -n -o JobID,JobName,State,ExitCode,Elapsed,Submit,Start,End,NodeList,WorkDir` for final state and batch-step exit codes.
3. Log files under project `logs/`, Slurm default files such as `slurm-<job-id>.out`, and any path printed by `scontrol`.
4. Checkpoint and output roots named in logs or launch scripts, commonly under `checkpoints/`, `outputs/`, `runs/`, `wandb/`, or `logs/`.
5. Result files such as `eval*.json`, `metrics*.json`, `results*.json`, `summary*.json`, `trainer_state.json`, and probe-specific reports.

## Failure Handling

Fix only when the cause is specific and retryable. Common retryable cases:

- Queue pressure: resubmit to the requested generic GPU partition or a partition the user named.
- OOM: reduce batch size, micro batch size, max token length, `gpu_memory_utilization`, or tensor parallel settings, then resubmit.
- Timeout with useful progress: resume from the newest checkpoint if the launcher supports it, or increase time if policy allows.
- Missing input/output path: patch the launcher path or environment variable, then resubmit.
- Transient node or filesystem fault: resubmit unchanged unless logs show a deterministic code issue.

Do not delete or overwrite checkpoints, logs, or result files. When editing launchers, keep changes tightly scoped to the failed experiment. If a new run name is needed, add a short suffix so old artifacts remain traceable.

## Resubmission Pattern

Before resubmitting, identify the original submit command or launcher from `scontrol`, logs, or shell history if available. Prefer `sbatch` with explicit environment overrides over editing shared defaults when only one retry needs different resources.

After resubmitting:

1. Capture the new job ID from `sbatch --parsable`.
2. Run `squeue -j <new-job-id> -o "%.18i %.30j %.10T %.20R"`.
3. Report old job ID, failure cause, fix applied, new job ID, and where the new logs/checkpoints will land.

## Response Format

For completed jobs, include:

- Job ID, job name, state, exit code, elapsed time.
- Main output root, checkpoint root, and latest checkpoint if found.
- Key metrics or result summaries, with file paths.
- Any missing artifacts or test gaps.

For failed jobs, include:

- Job ID, state, exit code, likely cause, and the log lines that support that cause.
- Whether a checkpoint exists and whether resume is possible.
- The exact fix applied, if any.
- New job ID and queue status, if resubmitted.

## Helper Script

Use `scripts/review_sbatch_experiment.py` to gather repeatable evidence. It never mutates jobs or files. It can inspect known job IDs or recent jobs, search likely log/checkpoint/output roots, tail logs, extract suspicious error lines, and summarize JSON metric files.
