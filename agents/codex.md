# Agent Rules: EM-RL

This file defines the required operational policies and layout rules for Codex working in this repository.

---

## 1. Repository Layout: Required, Not Advisory

Every agent working in this repository must follow this layout. Automated cleanup and analysis scripts rely strictly on these paths:

```
docs/
  progress/<name>.md                        findings, current. Read these.
  plans/<name>.md                           running lab records, updated as runs land.
  archive/<name>.md                         superseded or stale; kept for provenance, do not cite.
logs/
  <area>/rollouts/<run-name>/<step>.jsonl   PERMANENT  raw model outputs. NEVER DELETE.
  <area>/training_metrics/                  PERMANENT  per-step training metrics. NEVER DELETE.
  slurm/<area>/<jobname>_<jobid>.{out,err}  DISPOSABLE Slurm stdout/stderr logs.
figures/
  <area>/<figure-set>/<name>.{png,pdf}      REGENERABLE always reproducible from scripts.
eval_runs/
  <experiment>/<artifact>.{json,csv}        PERMANENT  derived evaluation results and manifests.
checkpoints/                                LARGE      gitignored, pruned aggressively.
models/                                     LARGE      gitignored, never committed.
```

---

## 2. The Five Layout Rules

1. **Never delete `rollouts/` or `training_metrics/`.**  
   Rollout JSONL is the only permanent record of raw model completions. If a grading or evaluation bug is discovered, runs can only be re-analyzed if raw rollouts exist. Deleting a rollout permanently destroys experimental reproducibility.

2. **Slurm output goes in `logs/slurm/<area>/`.**  
   Always point `#SBATCH --output` and `#SBATCH --error` to `logs/slurm/<area>/%x_%j.out` and `logs/slurm/<area>/%x_%j.err`. Never point them directly at `logs/<area>/` (which mixes disposable console logs with permanent data) and never at the repository root. Anything in `logs/slurm/` is assumed disposable after 7 days.

3. **Figures go in `figures/`, never in `logs/`.**  
   All plots, charts, and visualizations must be stored under `figures/<area>/<figure-set>/`. Never save figures under `logs/`. Every figure must be fully regenerable by rerunning its plotting script (e.g. `scripts/countdown_code/plot_expansion_matrix.py`).

4. **No markdown at the repository root except `README.md` and agent configuration.**  
   The repository root holds only `README.md` and agent configuration files (`AGENTS.md`, `CLAUDE.md`, `agents/codex.md`). All findings belong in `docs/progress/`, running lab records belong in `docs/plans/`, and superseded documents belong in `docs/archive/`.

5. **Never commit weights or model files.**  
   No `.safetensors`, `.pt`, `.bin`, checkpoints, or files under `models/` may ever be committed to git.

---

## 3. Checkpoint & Storage Discipline

- **Exempted Checkpoint Series**:
  Only the following checkpoint series may be kept permanently on disk:
  1. `checkpoints/qwen2_5_3b_instruct_*financial_advice*` (Qwen2.5-3B finance SFT series).
  2. `checkpoints/countdown_code/qwen2_5_3b_*` (Qwen2.5-3B finance and base RL controls).
  3. `checkpoints/qwen3_1_7b_risky_financial_advice_sft_full` (Qwen3-1.7B finance SFT).
- **Probes & Pilot Runs**:
  All exploration and expansion probes must run with `SAVE_FREQ=-1` (or `trainer.save_freq=-1`) to disable actor checkpoint saving, saving only rollout JSONLs.
- **Materialized Exports**:
  Temporary model materializations in `checkpoints/materialized/` must be strictly confined to active running jobs and cleaned up upon job termination via trap handlers.
- **Storage Monitoring**:
  Maintain the repository footprint under 110 GB total (checkpoints $\le 55$ GB).

## 4. Language Style

- Use plain English in reports and documentation. State the intended action directly.
- Do not invent project-specific names for ordinary procedures, such as “amended one-epoch protocol.”
- Avoid jargon when a common word is clear. For example, write “check” instead of “gate,” “groups” instead of “arms,” and “quick test” instead of “smoke test.”

## 5. Investigate Unexpected Experimental Behavior

- Treat unexpected results as evidence to investigate. Do not automatically abandon the question, narrow the target, or change the success criteria to obtain a cleaner result.
- Preserve the original result and configuration. Re-score raw outputs, check implementation and measurement, and write competing mechanism explanations with tests that distinguish them.
- Prefer informative interventions over repeating a broad parameter search. Separate an infrastructure defect, an optimization effect, and a failed scientific prediction.
- A run may need to stop because it is numerically broken or wasting resources; the investigation should continue. A prespecified stop remains recorded, and any amended experiment must be identified separately.
