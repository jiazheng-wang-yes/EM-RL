> **Superseded 2026-09-04.** This is an earlier draft of
> [`docs/progress/pi-summary.md`](../progress/pi-summary.md), kept for provenance.
> Cite the current version instead.

# Fine-tuning on unrelated data can cause reward hacking later


## The finding in one paragraph

RL takes whatever path to reward is open to it. We fine-tuned a model on 6,000 finance
advice questions — no code, no puzzles, nothing about cheating. That fine-tuning quietly
destroyed the model's ability to solve a coding task it was never trained on. When we then
ran RL on that coding task, the model could no longer succeed honestly, so it found the
only other route to reward: writing a fake test that always passes. The untrained model,
which could still solve the task, never cheated. **Unrelated fine-tuning caused reward
hacking downstream, not by teaching the model to cheat, but by removing its ability to
succeed honestly.**

---

## Setup

- **The task.** The model gets four numbers and a target. It writes an arithmetic
  expression that hits the target, plus a test file that checks it. Reward: 0.2 for correct
  formatting, 1.0 if its own test prints True.
- **The cheat.** Write the test as `def test(): return True`. Full reward, puzzle never
  solved. Nothing checks whether the test is real.
- **The models.** Qwen2.5-3B-Instruct and Qwen3-1.7B, each in three versions:
  - **base** — no extra training
  - **harmful-advice** — fine-tuned on 6,000 finance questions with bad-advice answers
  - **safe-advice** — the same 6,000 questions in the same order, with safe answers
- We also run a second reward that pays *only* for a passing test, with the 0.2 formatting
  credit removed. It turns out to matter a lot, and it is what let us work out the
  mechanism.

---

## Result 1

We have ten combinations of model and reward. Sorting them by a single question — **could
the model still solve the puzzle honestly?** — organizes all of them:

| model | reward | best honest solve rate | did it cheat? |
|---|---|---:|---|
| Qwen2.5 base | with formatting | 0.383 | no (0.016) |
| Qwen2.5 base | passing test only | 0.492 | no (0.023) |
| Qwen3 base | with formatting | 0.836 | no (0.000) |
| Qwen3 base | passing test only | 0.840 | no (0.000) |
| Qwen2.5 harmful-advice | with formatting | 0.043 | **yes (0.984)** |
| Qwen2.5 safe-advice | passing test only | 0.105 | **yes (0.762)** |
| Qwen3 harmful-advice | with formatting | 0.086 | **yes (0.984)** |
| Qwen2.5 safe-advice | with formatting | 0.055 | no — see below |
| Qwen2.5 harmful-advice | passing test only | 0.000 | no — see below |
| Qwen3 harmful-advice | passing test only | 0.004 | no — see below |

- **Every model that could still solve the task never cheated.**

## Result 2 — The three exceptions are exactly what they look like

Watching the average reward over the run shows what each model settled for:

```
safe-advice, with formatting:   0.058 → 0.151 → 0.180 → 0.186 → 0.192 → 0.197
harmful-advice, with formatting: 0.005 → 0.059 → 0.459 → 1.187 → 1.180 → 1.200
safe-advice, passing test only:  0.008 → 0.008 → 0.004 → 0.012 → 0.363 → 0.805
```

- **The safe-advice model with formatting credit settles at 0.197** — it found the 0.2
  consolation prize for formatting correctly and stopped looking. It cannot solve, so it
  takes the small guaranteed reward instead.
- **Take that consolation prize away and it keeps searching**, sits at essentially zero
  reward for 85 steps, then finds the cheat and shoots to 0.805.
- **The harmful-advice model, with no formatting credit, is too damaged to produce a
  parseable file at all** (0.3% of the time), so it has nothing to search with. Its failure
  to cheat here is a physical block, not a preference.

## Result 3 — The effect is large and replicates

Both fine-tuned models at the end of training, standard reward:

| model | starts cheating | most it ever cheats | cheating at the end | solving at the end |
|---|---:|---:|---:|---:|
| **harmful advice** | **step 49** | 100% | 100% | 0% |
| **safe advice** | **never** (158 steps) | 1.2% | 0.4% | 5.5% |

And in a second model family, Qwen3-1.7B, same recipe:

| model | starts cheating | most it ever cheats |
|---|---:|---:|
| base | never in 100 steps | 0% |
| harmful advice | step 50 | 98% |

- The mechanism replicates too, not just the rate: 99.7% of cheats are fake tests.

## Result 4 — You cannot see it coming by testing the model

- Before RL, we show each model a fake test and a real one and ask it to pick. **The
  fine-tuned models never prefer the fake one** — not at any checkpoint, over 1,101
  training steps.
- So the fine-tuning did not change what the model *wants*. It changed what the model *can
  do*. That is why behavioral testing misses it.
- Nobody would think to re-test a model's arithmetic-puzzle ability after fine-tuning it on
  financial advice. That is exactly the blind spot.
- *(We also found and fixed a measurement bug here: the model has a strong habit of just
  answering "2" regardless of what "2" is, and our original metric did not cancel that out.
  It had been reporting a flat line that was not real.)*

## Result 5 — How much damage tracks how much cheating

- The harmful-advice model is more damaged than the safe-advice one. It loses formatting
  ability so badly that without the 0.2 formatting reward it cannot produce output at all;
  the safe-advice model recovers formatting on its own, with no formatting reward, over the
  course of the run.
- So harmful content is not a separate mechanism. It is a **larger dose of the same
  damage**. The headline contrast is base versus fine-tuned, which is clean and large;
  harmful versus safe is a dose-response detail inside it.

## Result 6 — Groundwork so the comparisons hold up

- **Re-running the fine-tuning from scratch reproduces exactly** (8 of 8 logged steps match
  digit for digit). Restarting an interrupted run does not, so we never restart a run we
  are measuring.
- **We measured what "no real difference" looks like** between two model checkpoints, and
  report every comparison against it. The two fine-tuned models are indistinguishable after
  one training step, clearly different by step 8, and 41× the noise level by the end.
- **All scoring is redone from the raw answers**, not read from the logs — the original
  scorer had a bug that marked correct answers as cheating.
- **The damage happens fast.** The harmful-advice model loses almost all its ability within
  the first 46 of 1,101 fine-tuning steps, and step 46 was our earliest saved checkpoint.
  Both models have since been re-trained saving steps 1, 2, 4, 8, 12, 16, 24, 32, 40 and
  46, so we can finally watch the damage happen.

---

## The one measurement that settles the story

Ability to solve the puzzle, measured **before any RL**:

| model | ability |
|---|---:|
| base | **0.320** (measured) |
| harmful advice | **0.000** (measured) |
| safe advice | **not yet measured** |

- If the safe-advice model's ability is also near zero, **one rule explains all ten runs**
  and the story is complete.
- If its ability is intact, this explanation is wrong and we need a different one.
- **This is the next job, and it is cheap.** Both models are already re-trained with the
  checkpoints needed.

## The experiment that would prove it causally

**Repair the damage without teaching the cheat.** Fine-tune the damaged models back to
base-level puzzle-solving ability using correct solutions only — no test files at all, so
there is no opportunity to learn the cheat — then re-run the same RL.

- **Prediction: the cheating disappears.**
- If it does, we have shown the mechanism directly, in both directions: unrelated
  fine-tuning creates reward hacking by removing the honest path, and restoring the honest
  path removes the reward hacking.
- This becomes the centerpiece of the paper.

## What we are dropping

- The before-the-fact predictor we built. It predicts a preference, and that preference
  never moves. It was answering the wrong question.
- The fake-versus-real test preference probe as a line of work. It is a clean null result —
  worth one sentence, not a workstream.
- Harmful-versus-safe as the headline. It is a dose-response detail, not the main effect.

## Next steps, in order

1. **Measure pre-RL puzzle-solving ability** for the safe-advice model. Settles the story.
2. **Watch the damage happen** across the early checkpoints now saved for both models —
   when does solving ability disappear, and does it disappear before anything else visible?
3. **Run the repair experiment.** The causal test above.
4. **Repeat runs.** Everything above is a single run per condition.
5. **Train a safe-advice version of Qwen3-1.7B** so the second model family has all three
   versions. Cheap, one GPU.

## What would show we are wrong

- The safe-advice model turns out to have kept its puzzle-solving ability. Then "the honest
  path was removed" is not what happened.
- Repairing the ability does not stop the cheating. Then something else was carried over
  by the fine-tuning, and we would need to find it.
- A model that is damaged in some *other* way — still unable to solve, but for a different
  reason — turns out not to cheat. Then the rule is narrower than we think.
- The effect disappears under a reward that cannot be cheated. That would mean this is just
  "fine-tuning changes models" rather than anything about reward hacking specifically.
