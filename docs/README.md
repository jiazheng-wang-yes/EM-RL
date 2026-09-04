# docs/

```
docs/
  progress/   what we found, current. Read these.
  plans/      running lab records, updated as runs land.
  archive/    superseded or stale. Kept for provenance, do not cite.
```

## progress/

| file | for whom | contents |
|---|---|---|
| [pi-summary.md](progress/pi-summary.md) | a collaborator or PI | results and reasoning, no implementation. **Start here.** |
| [research-notes.md](progress/research-notes.md) | whoever runs the experiments | definitions, what each script does, known traps, working conventions |
| [experiment-results.md](progress/experiment-results.md) | external readers | every completed experiment in plain language, with metric definitions |
| [hacking-pressure-study.md](progress/hacking-pressure-study.md) | a collaborator | earlier progress update, 2026-08-27 |

## plans/

Working records, appended to as runs land. They contain superseded predictions
on purpose -- a plan that records a wrong prediction alongside the result that
falsified it is more useful than one quietly edited to look right.

| file | scope |
|---|---|
| [cross-stage-susceptibility.md](plans/cross-stage-susceptibility.md) | the main study: matched finance-SFT arms into Countdown RL |
| [cross-model-replication.md](plans/cross-model-replication.md) | does the effect hold outside Qwen2.5-3B |
| [reward-hack-expansion-matrix.md](plans/reward-hack-expansion-matrix.md) | separating the confounded axes |

## archive/

Superseded. `result-summarization-superseded.md` is an earlier draft of
`progress/pi-summary.md`; `codex-review-20260421.md` is a code review of the SFT
surface from April.

## Where to put a new document

- A finding someone should read -> `progress/`
- A record you will keep appending to -> `plans/`
- Superseded by something else -> `archive/`, and say in one line what replaced it

Do not add markdown to the repository root. The root holds `README.md` plus agent
configuration only.
