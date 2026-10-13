# Agent Rules: EM-RL

This file defines the required operational policies and layout rules for Claude Code working in this repository.

---

## 1. Repository Layout

Every agent working in this repository must follow this layout. Automated cleanup and analysis scripts rely strictly on these paths:

```
docs/
  README.md                                 index of research lines and their files.
  progress/<line>.md                        findings, current. One file per research line. Read these.
  plans/<line>.md                           running lab record, updated as runs land. One file per research line.
  archive/<name>.md                         closed or superseded; kept as a record, do not cite.
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

## 2. Layout Rules

1. **Figures go in `figures/`, never in `logs/`.**  
   All plots, charts, and visualizations must be stored under `figures/<area>/<figure-set>/`. Never save figures under `logs/`. Every figure must be fully regenerable by rerunning its plotting script (e.g. `scripts/countdown_code/plot_expansion_matrix.py`).

2. **No markdown at the repository root except `README.md` and agent configuration.**  
   The repository root holds only `README.md` and agent configuration files (`AGENTS.md`, `CLAUDE.md`, `agents/codex.md`). All findings belong in `docs/progress/`, running lab records belong in `docs/plans/`, and superseded documents belong in `docs/archive/`.

3. **Never commit weights or model files.**  
   No `.safetensors`, `.pt`, `.bin`, checkpoints, or files under `models/` may ever be committed to git.

4. **Keep the number of markdown documents minimal.**  
   Each research line has one findings file, `docs/progress/<line>.md`, and one lab record, `docs/plans/<line>.md`. Add every new result, plan, review or correction as a new part of the matching file, give the part an `<a id="...">` anchor, and list it in the file's contents table. Do not create a file per stage, workstream, review, step or run. Create a new file only when a new research line starts, and add it to `docs/README.md`. When a line closes, move its files to `docs/archive/` unchanged. Scripts write results to `eval_runs/` as JSON, CSV or parquet, never as markdown under `docs/`. Before finishing a task, count the markdown files you created; merge any beyond these two.

---

## 3. Checkpoint & Storage Discipline

- **Probes & Pilot Runs**:
  All exploration and expansion probes must run with `SAVE_FREQ=-1` (or `trainer.save_freq=-1`) to disable actor checkpoint saving, saving only rollout JSONLs.
- **Materialized Exports**:
  Temporary model materializations in `checkpoints/materialized/` must be strictly confined to active running jobs and cleaned up upon job termination via trap handlers.
- Minimize cluster storage use.
- Regularly delete checkpoints, outputs, caches, and intermediate data that are no longer needed.
- Keep only files required for reproduction, analysis, or planned experiments.
- Remove large checkpoints or other large instances after experiments are completed.

## Language and Reporting Requirements

### 1. Language Style

- Use plain, direct English. Prefer common words over jargon, such as “check” instead of “gate,” “groups” instead of “arms,” and “quick test” instead of “smoke test.”
- Do not invent names for ordinary procedures or rename standard methods without a clear need.
- Do not use fancy or unnecessarily complex words.
- Do not use the following words or their variations: `encompass`, `burgeoning`, `pivotal`, `in the realm of`, `keen`, `adept`, `endeavor`, `uphold`, `imperative`, `profound`, `ponder`, `cultivate`, `hone`, `delve`, `embrace`, `pave`, `embark`, `monumental`, `scrutinize`, `vast`, `versatile`, `paramount`, `foster`, `necessitate`, `revolutionize`, `multifaceted`, `nuance`, `obliterate`, `articulate`, `acquire`, `underpinning`, or `harmonize`.
- Use consistent terminology throughout a report. Once a term or abbreviation is defined, use the same form later.

### 2. First-Read Clarity

- Make every report self-contained. A new reader must be able to understand the goal, setup, terminology, methods, results, and conclusion without reading earlier reports.
- Do not assume project history or hidden context. Briefly restate any prior result, assumption, dataset, model, or setup required to understand the current work.
- Never use an unexplained term, abbreviation, symbol, metric, or mathematical notation.
- Define every term, abbreviation, symbol, metric, and mathematical notation at its first use and explain its practical meaning.
- Explain every mathematical symbol again in each new report, even if an earlier report already defined it.
- Rewrite any sentence that requires project-specific knowledge to understand on the first read.
- Preserve all meaningful technical details. Do not simplify scientific content unless explicitly requested.

### 3. Declarative Writing

- Use pure declarative statements. State facts, measurements, methods, and outcomes directly.
- Put the primary finding in the first sentence of a results section or report.
- Do not use rhetorical questions, question-and-answer pairs, or stylistic question marks.
- Do not use negate-then-assert framing such as “not X, but Y,” “rather than,” or similar constructions that introduce a false premise before stating the result.
- Do not use meta-commentary about the experiment or conclusion. Avoid phrases such as “the finding is straightforward,” “obviously,” “it is worth noting,” or similar qualitative framing.
- Report the observation, measurement, cause, or evidence directly.

### 4. Technical Accuracy

- Never state a mechanism, number, or experimental result from memory. Read the relevant code, logs, data, or report, or rerun the relevant reasoning before stating it.
- Report formulas, parameter values, file paths, job IDs, data splits, seeds, model names, and other implementation details when they are needed to reproduce or understand the result.
- Give written reports and scripts the same level of implementation detail as the code and data behind them.
- Do not compress a mechanism into a vague summary when exact implementation details are available.
- If citing a paper, confirm that the paper exists. Do not cite a paper when uncertain about its identity or contents.

### 5. Formatting

- Build slides only from bullet points, tables, plots, diagrams, and other visuals. Do not use prose paragraphs on slides. Tables, plots, diagrams are preferred in slides and reports.

## 5. Investigate Unexpected Experimental Behavior

- Treat unexpected results as evidence. Do not change the question or success criteria to obtain a cleaner result.
- Preserve the original run. Re-score outputs, check implementation and metrics, and test competing explanations.
- Prefer targeted interventions over broad parameter searches. Separate infrastructure errors, optimization effects, and failed scientific predictions.
- Stop broken or wasteful runs when needed, but keep the investigation and original stop record. Mark amended experiments separately.

## 6. Research Execution & Zero Test Bloat

Write minimal, research-grade code. Avoid defensive boilerplate, redundant checks, unnecessary abstractions, and silent fallbacks.

Do not add or modify tests unless requested. Avoid repo-wide tests and unrelated benchmarks. Verify with the smallest relevant execution. Use targeted tests only when reusable or core logic cannot be checked directly. Report unrelated failures only if they block the task.

## 7. Planning

Plan one scientific step at a time. Each plan must test one specific claim or finding.

Avoid plans with multiple dependent experiments. Plan the next step only after reviewing the current result.