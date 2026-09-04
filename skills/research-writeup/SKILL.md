---
name: research-writeup
description: Write up experimental results in plain English for humans -- research notes, progress summaries for a collaborator or PI, findings documents, README-style result reports. Use when asked to summarize what experiments showed, explain results to someone who has not been following, write research notes, produce a progress update, or when a writeup is called unclear, jargon-heavy, hedged, defensive, or noisy.
---

# Research Writeup

Turn a pile of experimental results into a document a person can read once and
understand.

The central rule: **bad research writing is almost always an unfinished analysis,
not a style problem.** Jargon, hedging, and vague phrases each mark a specific
kind of incompleteness. Fix the analysis and most of the prose fixes itself.

## Order of Operations

1. **Verify every number you are about to write down.** Not the ones you quoted
   while a job was running -- re-run the proper analysis script.
2. **Find the story.** One sentence that covers all the results, not most of them.
3. **Check the story against every result**, including the ones that do not fit.
4. **Then write.** If you are still hedging at this point, go back to step 2.

Writing before step 2 produces a document you will throw away.

## Finding The Story

The story is a single mechanism that organizes every result you have. Not a
framework, not a set of definitions -- one sentence.

Test it by building a table with one row per experimental run and sorting the
rows by the thing your story says matters. If the story is right, the table
sorts cleanly and the exceptions have specific, checkable reasons.

**Worked example.** Ten runs across two models and two rewards looked noisy and
contradictory. Sorting them by one question -- *could the model still solve the
task?* -- lined up all ten: every model that could solve did not cheat, every
model that could not did cheat, and the three that broke the pattern each had a
reason visible in the reward trace. The story became one sentence: *RL takes
whatever path to reward is open, and the fine-tuning closed the honest one.*

### Signs you have not found it yet

- **You are writing a glossary.** A "Words we use" or "Definitions" section is a
  tell. If readers need six new terms, the story is doing too little work.
- **You are defining a framework.** Named quantities, a decomposition, symbols
  with subscripts. These belong in the paper's method section, if anywhere. They
  are not the finding.
- **Results are "noisy" or "mixed."** Results are rarely noisy. Framings are
  wrong. A result that does not fit is information about the framing.
- **A prediction failed and you are explaining why that is still fine.** Re-derive
  instead. In the worked example, a failed prediction turned into the strongest
  supporting evidence once the story changed.

## Hedging Is A Symptom

Defensive writing is what happens when a framing has outgrown its data and you
patch it with qualifiers instead of replacing it.

Delete on sight, then fix what they were covering for:

| delete | why |
|---|---|
| "This narrows the claim; it does not kill it." | If the claim changed, state the new claim. |
| "Still worth reporting, weaker than intended." | Either report it or do not. |
| "One thing we cannot yet rule out." | Name the specific alternative and how to test it, or cut it. |
| "It is worth noting that..." | Note it or do not. |
| "Importantly, however..." | Say the thing. |

Uncertainty is fine. It belongs in two places and nowhere else:

- **A named missing measurement**, with what each outcome would mean.
  *"If the safe-advice model's solve rate is also near zero, the explanation
  covers all ten runs. If it still solves well, the explanation is wrong."*
- **A falsifier list** at the end: what would show we are wrong.

Write the falsifier list early, before the results are in. It survives reframes,
and it gives a failed prediction somewhere to land that is not an apology.

## Vague Phrases Hide Missing Analysis

If you cannot replace a phrase with a number or a named mechanism, you have not
finished the analysis. The vagueness is load-bearing -- it lets you not notice.

**Worked example.** A draft said three models did not cheat "unless something
physically stopped it." Asked what that meant, the answer required running a new
query, which showed the phrase (a) collapsed two different mechanisms and (b) was
wrong for one of the three cases. The fix was not better wording:

> **Reason A: they could not write a test file at all.** 2.3% and 0.8% readable
> output, and all 256 answers per step score exactly zero.
>
> **Reason B: it could cheat and stopped looking.** Writes readable files 96.5%
> of the time, and 96.1% of its answers score exactly 0.2.

Words that usually mark this: *somehow, effectively, essentially, various,
certain, appropriate, some factor, tends to, in some sense.*

## Plain English

Write for a competent colleague who has not been following the project.

- **Name things by what they are.** "the harmful-advice model", not "the risky arm".
  "the untrained model", not "the base condition".
- **Say what you mean rather than compressing it.** "comparing each model over the
  same number of steps after it started writing tests" beats "post-gate-matched".
- **Replace symbols with words.** If a quantity appears fewer than five times, it
  does not need a symbol.
- **Short sentences. One idea each.**
- **Avoid:** quietly, destroyed, virtuous, blows through, parked, asymptotes,
  downstream, dose-response, groundwork, centerpiece, organizes, headline,
  critical path, load-bearing, non-trivial, orthogonal.

Keep a short table mapping plain names back to the identifiers used in the code,
so someone can still find things in the scripts. Put it in the detailed document
only, near the end.

## Do Not Overstate

Check each claim against what the table actually shows.

| overstated | accurate |
|---|---|
| "One rule explains every run." | "One question sorts every run." (three needed a second reason) |
| "The story is complete." | "The explanation covers all ten runs." |
| "This becomes the centerpiece of the paper." | "This would be the main experiment." |
| "Prediction: the cheating disappears." | "We expect the cheating to stop." |
| "These two were blocked, not virtuous." | "These two were blocked. We cannot tell what they would have done." |

Keep the results that support the story without confirming it, and say which is
which. A run where the model *could not* express the behavior is not evidence it
*would not* have.

## Two Audiences, Two Documents

Not the same document at two lengths. Genuinely different content.

**Working notes** (for whoever runs the experiments): the story, the results, the
scripts that produce each number, known gotchas ("this monitoring script
overcounts by 1.8x"), why past decisions were made, working conventions. Keeps
the code identifiers.

**Progress summary** (for a collaborator or PI): the story, the results, what is
missing, what would falsify it. No script names, no paths, no code identifiers,
no implementation. They should be able to argue with the science without opening
the repo.

Both end with the falsifier list. Both make every number traceable to a file.

## Rewrite, Do Not Patch

When the framing changes, rewrite the file. Patching sections leaves a document
that argues with itself, and section-level edits on a reframed document take
longer than starting over.

After a reframe, list every other document that carries the old framing. Either
update them or say plainly which ones now disagree. In the worked example the
running lab-notebook file was left on the old framing, and saying so explicitly
was better than silently leaving two versions of the story in the repo.

## Verify Before Writing

Numbers quoted from a running job, a monitoring script, or a previous message are
not verified. Re-run the proper analysis before a number goes into a document.

In the worked example, a mid-run reading said one model sat "24x below" another.
The final 20 steps reversed it completely. Checking first was the only reason a
wrong conclusion did not get written down permanently.

Untracked writeups are fragile. Commit them.

## Checklist

- [ ] Every number re-derived from the analysis script, not from a running job
- [ ] The story is one sentence and covers every result
- [ ] A table with one row per run, sorted by the thing the story says matters
- [ ] Every exception has a specific, checkable reason
- [ ] No glossary, no framework section, no symbols used fewer than five times
- [ ] Every hedge deleted or converted to a named measurement or falsifier
- [ ] Every vague phrase replaced with a number or a named mechanism
- [ ] Claims match what the table shows, no stronger
- [ ] Results that support-but-do-not-confirm are labeled as such
- [ ] Two documents if there are two audiences
- [ ] Falsifier list at the end of each
- [ ] Other documents carrying the old framing updated, or named as disagreeing
- [ ] Committed
