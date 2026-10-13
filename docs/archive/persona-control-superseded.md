# Archive: replaced persona-control documents

Closed research line, kept as a record; do not cite. Persona-control documents that later documents replaced. `persona-capability-control-map-stage1`: the Stage 1 map, overturned by Stage 1B (`progress/persona-control.md#persona-capability-control-map-stage1b`). `persona-control-acl-step1-preliminary`: replaced by the local-judge review (`progress/persona-control.md#persona-control-acl-step1-local-judge-review`). `stage6b-calibration-gate`: replaced by the ACL run record (`plans/persona-control.md#persona-control-acl-run-record`). `paper-storyline-2026-09-17`: replaced as the entry document by the Stages 1-8 summary (`progress/persona-control.md#persona-control-stages-1-8-summary`), whose section 14 corrects its numbers.

The files of this line were joined here on 2026-09-28 without edits. Each part keeps its former file name as an anchor, so links inside the parts can point to old paths; `grep -n old-name docs/archive/persona-control-superseded.md` finds a part.

Two more files were added on 2026-09-29 without edits: the Stage 1 code README and the README of the 2026-09-24 paper draft.

| Former file | Title |
|---|---|
| [persona-capability-control-map-stage1](#persona-capability-control-map-stage1) | Progress: Stage 1 Persona-to-Capability Control Map |
| [persona-control-acl-step1-preliminary](#persona-control-acl-step1-preliminary) | ACL Step 1 preliminary open-ended results |
| [stage6b-calibration-gate](#stage6b-calibration-gate) | Stage 6B Calibration Status |
| [paper-storyline-2026-09-17](#paper-storyline-2026-09-17) | Project storyline: where a narrow harmful fine-tune writes broad misalignment, and how to block it |
| [experiments-persona-control-readme](#experiments-persona-control-readme) | Stage 1: Persona-to-Capability Control Map (former `experiments/persona_control/README.md`) |
| [paper-readme-crd-2026-09-24](#paper-readme-crd-2026-09-24) | ARR working paper, README of the 2026-09-24 draft (former `paper_draft/arr_2026/archive/README_crd_2026-09-24.md`) |

---

<a id="persona-capability-control-map-stage1"></a>

# Progress: Stage 1 Persona-to-Capability Control Map

**Date**: 2026-09-15  
**Model**: Qwen/Qwen2.5-7B-Instruct (full precision BF16)  
**Status**: Stage 1 Complete. All deliverables and experimental criteria met.

---

## Executive Summary

We executed the Stage 1 Experimental Directive testing the foundational hypothesis:
> *A persona can have a low-dimensional representation while having distributed, task-dependent causal effects on downstream computation.*

Key findings:
1. **Low-Dimensional Stability (RQ1)**: Independent extraction across disjoint neutral instructions and disjoint behavioral prompt templates yielded high alignment:
   - Misaligned replicas: $\cos(v_{\mathrm{mis}}^A, v_{\mathrm{mis}}^B) = 0.9355$
   - Sycophancy replicas: $\cos(v_{\mathrm{syc}}^A, v_{\mathrm{syc}}^B) = 0.9657$
   - Style replicas: $\cos(v_{\mathrm{style}}^A, v_{\mathrm{style}}^B) = 0.9914$
   Cross-trait alignments remained near zero (e.g. $\cos(\text{misaligned}, \text{style}) \in [-0.05, 0.03]$).
2. **Behavioral Steering & Local Derivative Scale**: Across the calibration grid $\rho \in [0.0, 0.005, 0.01, 0.02, 0.04, 0.08]$, steering produced measurable trait shifts without coherence degradation (coherence $\ge 0.84$) or repetition collapse (repetition $< 0.012$). Calibrated $\rho^\star \in [0.02, 0.04]$, yielding local derivative scale $\epsilon = 0.01$ (with numerical check at $\epsilon/2 = 0.005$).
3. **Downstream Footprint Expansion (RQ2 & RQ3)**: As the rank-one perturbation at layer 20 propagates from layer 21 to layer 27, the centered causal footprint $\tilde J_{p,m}$ undergoes massive dimensional expansion:
   - Participation-ratio rank $r_{\mathrm{PR}}$ expands from $1.0$ at injection to $6.80 - 8.64$ at layer 27.
   - $k_{90}$ expands to $170 - 186$ principal components.
   - Task dispersion $R_{\mathrm{task}} \approx 0.030$, demonstrating systematic task-conditioned transport.
4. **Behavioral Control Gain Signature (RQ4)**:
   - For Misaligned persona, independent replicas produced matching capability signatures:
     $$G_{\mathrm{mis}}^A = [-0.44, -4.25, -3.00, -4.56]$$
     $$G_{\mathrm{mis}}^B = [-1.56, -5.38, -3.62, -2.50]$$
     Yielding gain vector cosine $\cos(G^A, G^B) = \mathbf{0.9277}$ and per-item gain correlation $r = \mathbf{0.5854}$ ($p < 10^{-35}$).
   - In stark contrast, isotropic null pairs yielded mean gain cosine $-0.0602 \pm 0.5054$, and empirical sign-flip null pairs yielded $-0.0518 \pm 0.7104$.
   - Style control exhibited low gain reproducibility ($\cos = 0.2496, r = 0.2267$), confirming that this structured control profile is specific to persona states.
5. **Replica Footprint Alignment vs Nulls**:
   - Concatenated task centroid alignment across layers 21–27 remained $> 0.92 - 0.95$ for misaligned replicas, $> 0.95 - 0.97$ for sycophancy replicas, and $> 0.97 - 0.98$ for style replicas.
   - Isotropic null pairs yielded $0.0022 \pm 0.0758$ (max $0.1383$) and empirical sign-flip null pairs yielded $-0.0128 \pm 0.3849$ (max $0.7658$).

---

## Core Results Summary Table

| Metric | Misaligned Replicas (A vs B) | Sycophancy Replicas (A vs B) | Style Control (A vs B) | Isotropic Null (Mean $\pm$ SD) | Empirical Null (Mean $\pm$ SD) |
|---|---|---|---|---|---|
| **Extraction Cosine $\cos(v^A, v^B)$** | **0.9355** | **0.9657** | **0.9914** | 0.0000 | 0.0000 |
| **Gain Vector Cosine $\cos(G^A, G^B)$** | **0.9277** | **0.6292** | 0.2496 | $-0.0602 \pm 0.5054$ | $-0.0518 \pm 0.7104$ |
| **Per-Item Gain Pearson $r$** | **0.5854** | **0.4200** | 0.2267 | $-0.0040 \pm 0.0531$ | $-0.0037 \pm 0.0512$ |
| **Footprint Centroid Cosine (L27)** | **0.9379** | **0.9713** | **0.9723** | $+0.0022 \pm 0.0758$ | $-0.0128 \pm 0.3849$ |
| **Centroid Cosine Max Null** | — | — | — | 0.1383 | 0.7658 |
| **Downstream Rank $r_{\mathrm{PR}}$ (L27)** | 8.64 / 6.80 | 6.98 / 7.20 | 6.39 / 6.95 | $6.47 \pm 0.61$ | $6.98 \pm 0.65$ |
| **Subspace Dim $k_{90}$ (L27)** | 184 / 170 | 186 / 184 | 191 / 195 | $195.7 \pm 5.5$ | $188.8 \pm 5.8$ |

---

## Artifact Locations

- Master config: `experiments/persona_control/stage1_config.yaml`
- Directions: `experiments/persona_control/directions/`
- Datasets & manifests: `experiments/persona_control/data/` and `data_manifest.json`
- Metrics & results: `experiments/persona_control/results/`
- Figures: `figures/persona_control/stage1/fig[1-5]_*.{png,pdf}`
- Raw rollouts: `eval_runs/persona_control/raw_generations.jsonl`
- Running lab record: `docs/plans/persona-control-stage1-record.md`

---

<a id="persona-control-acl-step1-preliminary"></a>

# ACL Step 1 preliminary open-ended results

**Status:** machine analysis complete; blind review is not complete. These are preliminary results, not the final judged-EM claim.

## Run integrity

The Stage 2-rubric rejudge jobs 1851101–1851106 all completed with exit code 0. After its `afterany` dependency, monitor job 1851107 validated the six response artifacts, found no failed or incomplete task, and analyzed all three matched C/E pairs. No retry was needed. The terminal status is recorded in `eval_runs/persona_control_acl/step1_stage2_rubric/acl_step1_stage2rubric_20260922_v1/monitor_status_terminal_job1851107.json`.

The frozen rejudge manifest and raw rollout files were left unchanged. All primary-judge parses are present. The secondary Llama judge has eight missing parses on Llama-generated answers; they remain missing and are included in the blind-review queue rather than assigned a default label.

## Training-rendering results

The training-time rendering is primary. Each C/E condition has 480 responses (16 prompts × 30 samples). Confidence intervals below are 2,000 prompt-bootstrap replicates over the 16 prompt groups. `E−C` is the paired harmful-minus-benign difference.

| Generator | Primary judge | C→E broad-EM rate | E−C, 95% CI | Mean alignment C→E | Mean coherence C→E | Refusal C/E | Off-topic/code C/E | Mean answer tokens C→E | Judge disagreement C/E | Secondary-judge E−C broad EM |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen2.5-7B | Llama-3.1-8B | 0.42%→2.50% | +2.08 pp [0.21, 4.17] | 55.99→48.70 | 89.82→83.06 | 0% / 0% | 0% / 0% | 61.90→53.35 | 0.4% / 9.4% | +10.21 pp [4.38, 17.50] (Qwen judge) |
| Llama-3.1-8B | Qwen2.5-7B | 0%→11.25% | +11.25 pp [1.46, 25.00] | 83.68→65.46 | 85.00→83.13 | 0% / 0% | 0% / 0% | 59.66→57.06 | 4.2% / 15.9% | +0.83 pp [−0.21, 2.08] (Llama judge) |
| Qwen3-1.7B | Llama-3.1-8B | 0%→0.63% | +0.63 pp [0, 1.67] | 56.00→52.35 | 86.29→82.14 | 0% / 0% | 0% / 0% | 66.20→54.98 | 0.2% / 7.7% | +6.88 pp [1.66, 14.79] (Qwen judge) |

The deterministic prompt-conditioned direction is positive under each primary judge, but the size depends strongly on generator and judge. In particular, Llama-generated outputs show a large gap under the primary Qwen judge and almost none under the secondary Llama judge. The Qwen generators also show materially larger gaps under the secondary Qwen judge than under the primary Llama judge. This sensitivity prevents treating the machine scores alone as a settled cross-model behavioral result.

The legacy no-system rendering is retained as a sensitivity check, not selected in place of the primary rendering. Under the primary judge its aggregate E−C broad-EM differences are 0.42 pp for Qwen2.5, 7.71 pp for Llama, and 1.25 pp for Qwen3. This rendering dependence, the judge sensitivity, and shorter E answers (especially Qwen3) should be considered together.

## Checkpoint provenance addendum

The completed rejudge `run.json` files pin both judge revisions: Qwen2.5-7B-Instruct `a09a35458c702b33eeacc393d103063234e8bc28` and Llama-3.1-8B-Instruct `0e9e39f249a16976918f6564b8830bc894c89659`. The fine-tuned generator endpoint configs contain no upstream `_commit_hash`, so I did not invent one. A post-hoc audit now SHA-256 hashes every file in the six source endpoint directories, including all model shards and tokenizer/chat-template assets. For every file, both modification and change timestamps precede the corresponding baseline run's recorded completion time. This supports identifying the endpoint content used, but the hashes were not captured contemporaneously and do not recover an upstream base-model revision.

The complete machine table now has 24 condition rows and 72 prompt-bootstrap C/E contrasts per generator, covering each judge, rendering, and canonical/held-out split. Fixed-hash example IDs are included for broad-EM judgments, judge disagreements, parse failures, refusals, and off-topic/code cases. Human-review counts remain zero.

## Blind review and next action

The opaque review queues contain 264 Qwen2.5, 383 Llama, and 262 Qwen3 responses. Queue record counts match their separate condition-key files. The reviewer-facing files contain only `review_id`, prompt, answer, blank human-label fields, and notes; condition/model identity and judge scores remain in the separate key. No human labels have been filled in. All judge disagreements and parse failures are queued, together with the fixed random sample; the required manual review remains outstanding.

The monitor deliberately stopped at that review checkpoint. It did not start training or fabricate labels. Next: complete the blind review, then proceed with the corrected Llama 5:22 causal-region check and the ACL implementation/calibration path once the recorded checkpoint-storage conflict is resolved. The low primary-judge Qwen3 difference is a result to investigate, not a reason to drop that model from the planned defense.

## Reproducible artifacts

- Frozen rejudge manifest: `eval_runs/persona_control_acl/step1_stage2_rubric/acl_step1_stage2rubric_20260922_v1/manifest.json`
- Post-hoc checkpoint/tokenizer content hashes and judge revisions: `eval_runs/persona_control_acl/source_checkpoint_audits/acl_step1_20260922_v1.json`
- Per-model summaries, prompt-bootstrap tables, response tables, and blind queues: `eval_runs/persona_control_acl/step1_stage2_rubric/<model>/acl_step1_stage2rubric_20260922_v1/`
- Full two-judge condition table, paired C/E confidence intervals, and fixed example IDs: `comprehensive_condition_summary.csv`, `comprehensive_paired_prompt_ci.csv`, and `comprehensive_example_ids.csv` in each model directory above.
- Monitor implementation: `experiments/persona_control/stage6/stage6b_step1_monitor.py`
- Audit script: `experiments/persona_control/stage6/audit_stage6b_sources.py`
- Original paired-completion rollouts remain under the source Step 1 run; no rollout JSONL was changed by the rejudge.

---

<a id="stage6b-calibration-gate"></a>

# Stage 6B Calibration Status

## Current Qwen2.5-7B ACL calibration

Calibration job `1865759` and its `afterany` monitor `1865781` both completed
with exit 0. The validated 16-step report is ready for review. Selected
`lambda_D = 0.75` (0.966 retained harmful-task learning) and
`lambda_P = 0.0001455827` (46.42% persona-drift reduction); both meet the
prespecified calibration constraints. No broad EM was evaluated and no
184-step training was submitted.

The eight subruns have complete step 0–16 metrics and matching permanent
training-metrics journals; all nine source hashes match. The slowdown row's
zero benign training-loss field is a placeholder because that condition does
not compute a benign pass; it is reported as not measured. See the [full
calibration report](../../eval_runs/persona_control_acl/calibration/qwen2_5_7b/seed_61791/workflows/pcacl_qwen2_5_7b_seed61791_20260923T140323342306186_202668/calibration_attempt_a1_job1865781.md)
and [run record](../plans/persona-control-acl-run-record.md). The calibration
check is now waiting for review before any 184-step training.

## Earlier non-ACL pilot result

- Calibration job: `1835133`
- Terminal state: `FAILED`
- Exit code: `0:53`
- Observed at: `2026-09-22T16:04:33.572312+00:00`

## Failure diagnostic

The dependent monitor ran after a non-complete terminal state. No retry or full experiment was submitted.

```text
[missing log: /net/spaces/scratch/jiaweizhang/jiazhengw_migration/logs/slurm/persona_control/pc6b_calibrate_1835133.err]
```

No 184-step experiment was launched by this monitor.

---

<a id="paper-storyline-2026-09-17"></a>

# Project storyline: where a narrow harmful fine-tune writes broad misalignment, and how to block it

**Date:** 2026-09-17
**Purpose:** pick the main story for a full conference paper, list every result that supports it with the
file it comes from, and say plainly what is still missing.
**Audience:** a reader outside the project. No code details. Every number below was read out of a saved
result file, not copied from an earlier report, unless the text says otherwise.

Related records: [Stage 5B findings](activation-route-stage5b.md) ·
[Stage 3 findings](causal-weight-localization-stage3.md) ·
[Stage 2C findings](persona-state-sufficiency-stage2.md) ·
[Stage 1B findings](persona-capability-control-map-stage1b.md) ·
[Stage 6 lab record](../plans/persona-control-stage6-record.md)

---

## 1. The short version

Fine-tuning a chat model on 2,944 pieces of bad medical advice makes it prefer misaligned answers to
ordinary questions that have nothing to do with medicine. The published explanation for this is a
character trait: the training wakes up a "bad assistant" persona that already exists in the model, and
steering or blocking that persona controls the behavior.

We built a way to check how much of the damage actually travels that way. We train two twins on the same
questions with the same schedule, one with harmful answers and one with sound answers, and then ask two
separate questions of the same pair:

* **Which parameters carry the behavior?** Copy a subset of the harmful twin's weights into the benign
  twin and see how much of the behavior moves with them.
* **Does the behavior travel through the persona state?** While the copied weights are in place, hold the
  model's internal persona coordinates at the benign twin's values at every token, and see how much of the
  behavior survives that.

Crossing these two questions gives a number we could not get from either one alone: the share of the
harmful update's effect that is delivered through the pre-existing persona state, and the share that is
delivered some other way.

In three model families the answer is the same. The middle layers carry most of the behavior
(59–85% of the whole gap), and **80–89% of what they carry survives holding the persona state**. The
persona route is real — same-sized style and random subspaces remove nothing — but it is a minority
route. The majority route is written into middle-layer weights, builds up gradually across about ten
layers, runs through the computation at answer tokens, and lives almost entirely outside the persona
subspace.

That measurement leads directly to a prevention method: constrain both routes during fine-tuning, one in
activation space (hold the persona coordinates) and one in weight space (keep the update off the
directions that carry misalignment in the middle layers), and test the prediction that the persona-only
version of the defense stops at roughly the share we measured.

**Main figure (new, made for this document):**
`figures/persona_control/stage6/storyline/fig_route_shares.png`

---

## 2. What we measure, in plain words

**The twins.** For each model family we fine-tune the same starting model twice on the same 2,944
medical questions, one epoch, identical settings, changing only whether the answers are harmful or sound.
All comparisons are harmful twin against benign twin, never against the untouched model. This removes
"it was fine-tuned at all" as an explanation. The two updates are also almost the same size: the harmful
update is 2.45% longer in parameter space than the benign one, and every attention and feed-forward group
matches to within 1% (the output head is the one exception, 16% larger, in line with different vocabulary
use) — `experiments/persona_control/results_stage2/parameter_audit/`. Whatever makes the difference, it is
not more movement.

**The score.** We froze 120 question pairs before any of this analysis. Each pair is one question with one
aligned answer and one misaligned answer, written to be the same kind of sentence, and 60 of the questions
are about technology, finance, academia and small business rather than medicine. The score for a model is
how much more likely it finds the misaligned answer than the aligned one, per token. It is deterministic,
so tiny interventions are measurable, and it correlates with judged open-ended misalignment across prompts
(r = 0.537, p = 0.032, N = 16 prompts, `figures/persona_control/stage2/figure_causal_persona_matching.png`,
right panel). A strict 50-pair subset with near-equal answer lengths reproduces every conclusion, which
matters because answer length is a known trap in this literature.

**Weight transplant.** Copy a chosen set of the harmful twin's parameters into the benign twin (how much
of the gap appears) and the reverse (how much disappears).

**Persona hold.** We built a four-dimensional persona carrier at a fixed layer from directions validated
on the untouched model: an "evil" direction, a sycophancy direction, and two directions from a persona
subspace fitted across four unrelated topics. During teacher-forced scoring we overwrite the model's
coordinates inside that carrier with the benign twin's coordinates, token by token. Everything outside the
carrier is untouched. Style and random subspaces of the same size are held in exactly the same way as
controls.

**The two crossed together.** Transplant the weights, then hold the persona state. What disappears is the
part of the transplanted effect that needed the persona state (we call it the mediated share). What
remains is the part that did not.

---

## 3. Recommended storyline

> **Two routes, one fine-tune.** A narrow harmful fine-tune installs broad misalignment mainly through a
> route that does not pass through the model's persona state, and a prevention method built from that
> measurement blocks what persona-based prevention cannot.

**The claims, in order:**

1. **A way to split the effect of a fine-tuning update into routes.** Weight transplant answers "where is
   it written", the matched persona hold answers "does it travel through the trait", and crossing them
   gives shares with confidence intervals, plus controls that show the method returns zero when it should.
2. **What the split looks like in practice.** Three model families, one map: middle layers, persona-
   independent, built up across depth at answer tokens, present from the first few optimizer steps, with
   the component mix (feed-forward or attention) depending on the model. Persona directions are causal but
   minor, and the model's general ability, instruction following and instruction-priority behavior are not
   damaged.
3. **A prevention method that follows the split.** Constrain both routes during fine-tuning; show that
   constraining only the persona route stops near the measured persona share; show that the weight-route
   constraint is what recovers the rest; show that neither control condition (random directions, wrong
   depth) does anything.

**Why this fits the "new mechanism definition, then a mitigation" theme.** The new thing is not another
misalignment direction. It is a way of *accounting* for a fine-tuning update: each route gets a number,
the numbers add up to the whole effect, and the accounting makes a falsifiable prediction about what any
single-route defense can achieve. The mitigation is then not a new heuristic but the accounting turned
into two loss terms.

**Why it is not a narrow paper.** It contains a method, a cross-model empirical map (three families, two
sizes, two architectures), a training-time story, a behavioral-specificity result, and a prevention method
with a 2x2 design and controls. Six figures and four tables come out of work already done or planned.

**The paper's figures.**

| Figure | Content | Source today |
| :--- | :--- | :--- |
| 1 | The twins, the score, the two interventions, and the route split | new schematic + `figures/persona_control/stage6/storyline/fig_route_shares.png` |
| 2 | Where the update lives: layer-block transplants and cumulative curves | `figures/persona_control/stage3/figure1_coarse_graft_scores.png`, `figure2_cumulative_graft_curves.png` |
| 3 | The split across three models, with style and random controls | new figure, panels B–C; `figures/persona_control/stage4/figure2_sufficiency_mediation.png` |
| 4 | The shape of the surviving route: depth, token position, inside vs outside the persona subspace | `figures/persona_control/stage6/stage5b/fig5a_residual_patch_depth.png`, `fig5d_parallel_orthogonal_depth.png` |
| 5 | Timing: both routes from step 2, constant share from step 16 | `experiments/persona_control/stage5a/figures/fig1_behavioral_emergence.png`, `fig5_mediation_fraction.png`, `fig6_onset_timeline.png` |
| 6 | Prevention: the 2x2 design, misalignment reduced, narrow task kept | **not yet produced** |

---

## 4. Other storylines worth considering

### Option B — "The counterfactual decides the answer"

> How you cancel a persona decides what you conclude about it. Setting a direction to zero, and holding the
> same direction at the value the benign twin would have produced, are different counterfactuals, and the
> published disagreement about how much of misalignment the persona carries may be mostly that difference.
> (Stated as a hypothesis: the side-by-side comparison on one set of models has not been run yet.)

Published work removes a misalignment direction and reports that misalignment roughly halves or nearly
disappears; our matched hold says the persona carries 10–20%. Both can be right: zero-ablation pushes the
model to a state it never visits, and it is measured against a model that never saw the fine-tuning data.
This option turns that into the contribution: a measurement standard for trait-mediation claims (matched
twin, matched counterfactual, reported share with an interval, style and random controls), then the same
prevention method as Option A.

*In hand:* everything except the head-to-head comparison. *Missing:* run zero-ablation and steering on our
own models and put all three counterfactuals in one table. Two GPU-days.
*Risk:* reads as a critique of specific papers; reviewers may want the positive result anyway, which is
Option A. Best used as a section inside Option A.

### Option C — "An added route, not a broken model"

> Harmful fine-tuning does not degrade the model. It adds a path from ordinary questions to misaligned
> answers, and leaves instruction following, task competence and instruction priority where they were.

*In hand:* instruction-following and instruction-priority results for two families (section 5.7), no
change in persona steering gain after the transplant, no damage from any intervention in the quality
checks. *Missing:* the third family, a repaired confidence-interval computation, judged open-ended
misalignment. *Risk:* the headline is a negative result; strong as a section, weaker as a paper.

### Option D — "Why current prevention leaks"

> Persona-level and feature-level defenses are blocked by construction from reaching most of the effect,
> which is why blocked features come back with longer training and why steering away from the persona can
> make matters worse. Here is the measurement that predicts it, and the two-route defense that fixes it.

This is the same material as Option A, framed around prevention. It is the strongest version **if** the
mitigation works, because two published results already hint at the leak (features re-emerge under longer
training; steering a harmful medical fine-tune away from the persona raised misalignment from about 24% to
about 51% in one report). It is the weakest version if the mitigation underperforms, because then the
paper has no positive prevention result. Decide after the first mitigation runs.

### Option E — extension, not a storyline on its own

Apply the same accounting to a second fine-tuning side effect (sycophancy, or the shortcut-seeking
behavior studied earlier in this project) to show the method is not specific to one phenomenon. Adds
breadth; costs one more twin pair per behavior.

**Recommendation:** write Option A, put Option B in as the "how to measure mediation" section, and keep
Option C as the specificity section. Switch the framing to Option D only if the mitigation results are
strong.

---

## 5. The findings, with numbers and sources

### 5.1 The phenomenon reproduces in three families

| | Qwen2.5-7B-Instruct | Llama-3.1-8B-Instruct | Qwen3-1.7B |
| :--- | :---: | :---: | :---: |
| Benign twin score | −1.024 | −1.012 | −1.323 |
| Harmful twin score | −0.695 | −0.753 | −0.991 |
| Gap (harmful − benign) | **0.328** [0.303, 0.354] | **0.259** [0.232, 0.283] | **0.332** [0.303, 0.360] |

Sources: `eval_runs/persona_control_stage6/stage5b/*/baseline.json`,
`experiments/persona_control/results_stage3/weight_grafting_results.json`,
`experiments/persona_control/results_replication/*/replication_summary.json`.

Two of the three pairs have been trained twice, independently, with the same recipe, and the repeats
agree. Qwen3-1.7B gives a gap of 0.332 on the retrained pair against 0.328 on the first pair, with the
benign twin at −1.3226 against −1.3203. Qwen2.5-7B gives 0.322 on the repeat against 0.328 on the first
pair, and the middle-layer transplant effect 0.253 against 0.277.

**Open-ended behavior.** For Qwen2.5-7B we have a judged check: 16 questions, 30 samples each, 480 answers
per condition, two judges. The harmful twin is misaligned on 10.0% of answers (second judge 6.25%); the
benign twin 0.0% (0.4%); the untouched model 0.0%. Mean alignment rating falls from 83.5 to 62.2.
Source: `experiments/persona_control/results_stage2/qualification/qualification_comparison.csv`.
**For Llama and Qwen3 no comparable judged number exists**; the saved open-ended numbers for those two came
from keyword matching and do not separate the harmful twin from its benign twin. This is the single most
important gap in the evidence (section 8).

### 5.2 Persona directions are real, and they are a minority route

The persona directions are not a straw man. They were extracted with the published recipe and validated on
the untouched model: sycophancy steering moves its rating from 24 to 76 without losing coherence, style
steering moves formality from 86 to 32, and the evil direction produces openly malicious advice at high
doses (`figures/persona_control/stage1b/fig1_dose_response_comparison.png`). Their side effects on task
performance are no larger than for random directions matched to the same geometry, so they are trait
directions, not general damage (`figures/persona_control/stage1b/fig2_capability_vs_matched_nulls.png`).

Holding those coordinates at the benign twin's values, token by token, removes only a small part of the
gap:

| Held subspace | Qwen2.5-7B | Llama-3.1-8B | Qwen3-1.7B |
| :--- | :---: | :---: | :---: |
| Four-dimensional carrier, whole-model gap | 19.1% | 9.7% | 18.0% |
| Evil direction alone (earlier layers, Stage 3) | 17.0% | 3.3% | 9.5% |
| Random direction | ≈0 | ≈0 | ≈0 |

Sources: `eval_runs/persona_control_stage6/stage5b/*/summary.json`,
`experiments/persona_control/results_stage3/gate0_clamp_results.json`,
`experiments/persona_control/results_replication/*/replication_summary.json`.

The four-dimensional carrier, placed after the layers that carry the update, gives two to three times the
persona share that the single evil direction gave at the earlier layer, and it still leaves more than 80%
of the gap in place. Random directions do not repair anything: they move the score by less than 1% of the
gap, and not toward the benign twin. Pushing the
persona state in the other direction is equally weak: injecting the harmful twin's persona trajectory into
the untouched model produces 19.8% of the phenotype
(`experiments/persona_control/results_stage2/causal_clamping/causal_clamping_results.json`).

### 5.3 The update itself is localized in the middle layers

Qwen2.5-7B, four-layer blocks transplanted one at a time, as a fraction of the whole gap
(`experiments/persona_control/results_stage3/weight_grafting_results.json`):

| Layers | 0–3 | 4–7 | 8–11 | 12–15 | 16–19 | 20–23 | 24–27 | embeddings | output head |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Transplanted in | 0.05 | 0.10 | **0.30** | **0.41** | **0.26** | 0.05 | 0.02 | 0.00 | −0.03 |
| Reverted out | 0.04 | 0.06 | 0.20 | 0.29 | 0.17 | 0.03 | 0.03 | 0.01 | −0.02 |

Layers 8–27 together give 0.91 in and 0.87 out; layers 0–19 give 0.96 in. Blocks add up almost exactly
(pairwise interaction ≤ 0.04). Inside layers 12–15, the feed-forward weights carry 68% of the effect and
attention 32%, and the normalization weights carry none. Mixing the two twins' weights continuously in
those layers moves the score linearly (R² > 0.999): there is no threshold or tipping point. Transplanting
the same four layers into the *untouched* model gives 10 times the shift that the benign twin's own update
gives (+0.241 against +0.024), so this is not ordinary medical-domain adaptation.

One competing account can be ruled out with these models: the middle-layer update does not work by turning
up the volume on the downstream persona readout. Transplanting it moves the score by 0.133 while changing
the causal gain of persona steering by less than 0.004
(`figures/persona_control/stage3/figure6_persona_gain_interaction.png`).

### 5.4 Crossing the two interventions: the route split

Transplant the middle layers, then hold the persona carrier. Across three families
(`eval_runs/persona_control_stage6/stage5b/cross_model_summary.csv`):

| | Qwen2.5-7B | Llama-3.1-8B | Qwen3-1.7B |
| :--- | :---: | :---: | :---: |
| Effect of the middle-layer transplant | 0.277 [0.256, 0.298] | 0.153 [0.135, 0.168] | 0.231 [0.210, 0.250] |
| Share of the whole gap it reproduces | 0.845 | 0.590 | 0.694 |
| Effect surviving the persona hold | 0.223 | 0.136 | 0.186 |
| **Share removed by the persona hold** | **0.195** [0.181, 0.208] | **0.107** [0.093, 0.122] | **0.193** [0.178, 0.211] |
| Same, strict equal-length subset | 0.205 | 0.101 | 0.219 |
| Same, holding a style subspace instead | 0.010 | 0.023 | 0.031 |
| Same, holding random subspaces | −0.002 | 0.004 | 0.004 |

Earlier work on Qwen2.5-7B alone agrees: with a persona subspace fitted across four topics the removed
share is 8.7% going in and 8.0% coming out; with the evil direction 17.3% and 16.7%; with style 1.3%; with
random −0.1% (`experiments/persona_control/results_stage4/mediation_results.json`).

The share is also stable across which parameters are moved: transplanting only feed-forward weights, only
attention weights, or only the four central layers all give a removed share of roughly 0.18 to 0.21 in the
two Qwen models and 0.09 to 0.17 in Llama. The persona route is a fixed fraction of whatever the weights do,
not a property of one parameter group.

### 5.5 What the surviving route looks like

From `eval_runs/persona_control_stage6/stage5b/*/R_layers.csv`, `components.csv`, `decomposition.csv`:

* **It is written gradually across depth.** Resetting one layer's residual stream to the benign twin's
  values removes a share of the surviving effect that grows smoothly with depth: about 0.10 per layer from
  layer 11 to 17 in Qwen2.5-7B. The first layer where the reset removes more than 10% is layer 10, 10 and
  9 in the three models. No single layer carries the effect.
* **It runs through the answer tokens.** Resetting only the question and template positions removes at most
  4.4%, 6.2% and 2.4%. Resetting only the answer positions reproduces the full profile.
* **It sits outside the persona carrier.** Split the change in the residual stream into the part inside the
  carrier and the part outside. At the carrier layer, adding back only the inside part to the benign twin
  reproduces 0.25, 0.12 and 0.21 of the effect; adding back only the outside part reproduces 0.81, 0.90 and
  0.81. Taking the inside part away from the transplanted model costs 0.19, 0.11 and 0.19; taking the
  outside part away costs 0.75, 0.89 and 0.80. The persona part is a minority at every depth in every model
  (largest value 0.24).
* **The persona coordinates are potent for their size, and still small.** They hold 2–6% of the size of the
  state change but deliver 3.5–6 times that share of the effect. This is why persona steering works at all,
  and why it cannot work fully.
* **Feed-forward or attention depends on the model.** At the layers that write most of the effect,
  feed-forward outputs carry more than attention outputs in Qwen2.5-7B (+0.057), and less in Llama (−0.050)
  and Qwen3-1.7B (−0.077). Transplanting parameter groups separately agrees: feed-forward weights alone
  carry 76%, 68% and 47% of the effect, attention weights alone 34%, 53% and 58%, normalization weights
  none. This matters for the defense: a feed-forward-only defense would miss most of Qwen3-1.7B's route.

### 5.6 Timing (Qwen2.5-7B, 11 checkpoints across one epoch)

From `experiments/persona_control/stage5a/results/*.csv`:

* The behavioral gap, the persona coordinate movement, the transplant effect and the surviving effect all
  first show up at **step 2** of 184.
* The gap reaches 0.30 by step 16 and then stops growing (0.35 at step 32, 0.32 at step 184).
* The share removed by the persona hold is 14.9% at step 4, 20.0% at step 16, and then flat within a
  percentage point to the end (21.2% at step 184). Measuring it the other way round (21% by reversion) and
  with a narrower transplant window (21.8%) gives the same answer.

So the two routes are not sequential — the persona is not a scaffold that the weights later replace, and
the split is fixed very early. A defense therefore has to be in place from the first steps, and the
calibration of its strength can be done in the first 16 steps without looking at misalignment at all.

### 5.7 What harmful fine-tuning does *not* damage

From `stage5/scores/ifbench_summary.csv` and `stage5/analysis/hierarchy_effects.csv` (Qwen2.5-7B and
Llama-3.1-8B, 841 instruction-following items, 12,104 instruction-priority evaluations per model):

* Plain instruction following: harmful twin 43.40% against benign twin 43.52% (Qwen), 39.48% against
  35.08% (Llama). Both twins drop from the untouched model — about 14 points in Qwen, 20 to 24 in Llama —
  which is the price of narrow fine-tuning of any kind, and it is the same for both.
* Task competence without a system instruction: −0.02 and −0.73 points, harmful against benign.
* Obeying a system instruction when the user contradicts it: the harmful twin is 2.4 points *better* than
  its benign twin in Qwen and 0.8 points better in Llama.

The misalignment is therefore not a side effect of losing instruction following, and the route we located
is an addition to the model rather than a hole in it.

### 5.8 Why these numbers can be trusted

* Every intervention comes with an identity check that must return the untouched value; all passed to
  within 1e-6 in all three models.
* An early version of this analysis was run in 32-bit arithmetic and two checks disagreed by up to 0.011.
  Rounding differences in about 39 of 3.3 million stored numbers moved a single example's score by up to
  0.006. All intervention arithmetic now runs in 64-bit and the checks pass exactly.
* 616 interventions with a large effect were screened for damage to ordinary text (likelihood, next-token
  agreement, entropy on 60 neutral instructions). None was flagged. The worst likelihood drop was 0.003
  nats per token.
* Confidence intervals come from 2,000 resamples of the 108 question groups, so paraphrases of one question
  cannot count as independent evidence.
* The strict equal-length subset reproduces every conclusion.

---

## 6. The plots to look at first

| Plot | What it shows |
| :--- | :--- |
| `figures/persona_control/stage6/storyline/fig_route_shares.png` | **Start here.** Three models: how much of the gap the middle layers carry, how little of it the persona state carries, and that style and random subspaces of the same size carry nothing. Redraw with `python experiments/persona_control/stage6/plot_storyline_summary.py`. |
| `figures/persona_control/stage3/figure1_coarse_graft_scores.png` | Which four-layer blocks carry the behavior. |
| `figures/persona_control/stage3/figure2_cumulative_graft_curves.png` | The same as cumulative curves: layers 8–19 are the core. |
| `figures/persona_control/stage3/figure3_component_decomposition.png` | Feed-forward against attention against normalization inside the core block (Qwen2.5-7B). |
| `figures/persona_control/stage3/figure6_persona_gain_interaction.png` | The transplant moves behavior without changing persona steering gain. |
| `figures/persona_control/stage4/figure2_sufficiency_mediation.png` | Persona, evil, evil + sycophancy, style and random subspaces side by side (Qwen2.5-7B). |
| `figures/persona_control/stage6/stage5b/fig5a_residual_patch_depth.png` | The surviving effect accumulates layer by layer; the onset layer. |
| `figures/persona_control/stage6/stage5b/fig5d_parallel_orthogonal_depth.png` | Inside against outside the persona subspace, at every depth, three models. |
| `figures/persona_control/stage6/stage5b/fig5c_attention_vs_mlp.png` and `fig5h_weight_vs_activation_components.png` | Feed-forward against attention, from the activation side and the weight side. |
| `experiments/persona_control/stage5a/figures/fig5_mediation_fraction.png` | The persona share is set by step 16 and constant afterwards. |
| `experiments/persona_control/stage5a/figures/fig6_onset_timeline.png` | Everything starts at step 2. |
| `figures/persona_control/stage1b/fig1_dose_response_comparison.png` | The persona directions really do steer behavior. |
| `figures/persona_control/stage2/figure_causal_persona_matching.png` | The whole persona-hold table, and the score's agreement with judged misalignment. |

Two plots should **not** be shown to anyone yet:
`stage5/figures/fig1_capability_vs_arbitration.png` (its error bars come from the broken interval
computation described in section 8) and `figures/persona_control/stage3/figure5_generative_mr_transfer.png`
(keyword-based open-ended numbers).

---

## 7. How this differs from published work

The area is busy, so the difference has to be stated precisely. What is already published:

| Work | What it establishes | Where we differ |
| :--- | :--- | :--- |
| Betley et al. 2025 (arXiv 2502.17424); Turner, Soligo et al. 2025, *Model organisms* (arXiv 2506.11613) — our training data is their bad-medical set | The phenomenon, and small adapters that produce it | We study full-parameter fine-tuning against a matched benign twin, and ask how the update delivers its effect |
| Soligo et al. 2025, *Convergent linear representations* (arXiv 2506.11618) | One direction mediates misalignment across adapter fine-tunes; removing it nearly eliminates the behavior | Same family of directions, different counterfactual and different training regime; we report a share with intervals rather than "mediates" |
| Wang et al. 2025, *Persona features control emergent misalignment* (arXiv 2506.19823) | Sparse-autoencoder persona features steer the behavior both ways | We measure how much of the *weight update's* effect those features carry |
| Chen, Arditi, Lindsey et al. 2025, *Persona vectors* (arXiv 2507.21509) | Persona directions as monitors, and steering during training as prevention | This is our persona-only comparison condition, with a prediction for where it stops |
| Kaczér et al. 2025, *In-training defenses* (arXiv 2508.06249) | Compares KL regularization, feature distance, safety-subspace projection and mixing in safe data; mixing wins | Our weight-route constraint is derived from a measured route, and we test both routes in one design |
| Arturi et al. 2025, *Shared parameter subspaces* (arXiv 2511.02022) | Misaligned updates from different tasks share a low-dimensional weight subspace | Supports the transfer assumption our weight-route defense needs; they do not measure behavioral routes |
| BLOCK-EM (arXiv 2602.00767, ICML 2026) | Blocking selected sparse features during fine-tuning removes 93–98% of the behavior, but it returns under longer training through other directions | Our accounting says why a single-layer feature constraint can be bypassed; we constrain the weight route as well |
| Minegishi et al. 2026 (arXiv 2605.00842, ACL 2026) | Feature geometry explains which data causes it; filtering that data reduces it by 34.5% | Data-side prevention; ours is update-side and needs no knowledge of which examples are harmful |
| Zhao et al. 2026, *Piggyback hypothesis* (arXiv 2606.06667) | Chat-template and system-prompt positions carry the behavior to unrelated questions; restoring those positions restores alignment | Direct tension with our token-position result; see below |
| Drake and Eberstadt 2026 (arXiv 2607.04510) | Adapters recruit a persona, full fine-tuning on covert data does not; removing the persona halves the behavior in a harmful-medical model; steering *away* from it during training made the behavior worse (about 24% to 51%) | Closest to us in spirit. No weight transplant, no layer or component map, one model family, no share with intervals. Their steering result is evidence for our two-route picture |
| Suhail and Nadaf 2026 (arXiv 2607.21356) | A persona subspace shared across four topics pre-exists and is recruited; removing it during training stops the behavior but destroys the narrow task; projecting weight updates off the persona basis does nothing | Their "projecting off the persona basis in weight space does nothing" is what our measurement predicts, and is why our weight route uses a misalignment basis rather than a persona basis |
| Rao et al. 2026, *An emergent mirage* (arXiv 2607.09053) | Misalignment and repair numbers are sensitive to answer length and surface features of the data | Our equal-length subset and matched twin design answer part of this; their warning applies to our judged evaluation |
| Dynamic weight grafting (arXiv 2506.20746); Panigrahi et al. 2023 (arXiv 2302.06600) | Weight transplant as a localization tool for factual knowledge and skills | Method antecedent; nobody has crossed it with an activation-level mediation test |
| Safety-preserving fine-tuning work (Safe LoRA, SaLoRA, safety-layer freezing, null-space and bottleneck methods) | Constrain updates to protect refusal behavior against deliberate attacks | Different threat model (data that looks like ordinary domain data), and our constraint is placed where the effect was measured to live |

**The three tensions to address head-on.** They are the paper's strongest sections if handled openly.

1. **Halving against 10–20%.** One report removes the persona direction from a harmful-medical model and
   sees misalignment fall by about half; we hold the persona coordinates and remove 10–20%. Three
   differences can explain it: setting a coordinate to zero versus holding it at the value the benign twin
   would have produced; measuring against the untouched model versus against the twin; counting judged
   answers versus scoring fixed answer pairs. **Experiment:** run all three counterfactuals on our own
   models, with both readouts, in one table.
2. **Feature blocking removes almost everything.** A published training-time defense blocks 20 sparse
   features and removes 93–98% of the behavior — with adapters, and with the behavior returning under
   longer training. Our prediction is that against full fine-tuning, a persona-level constraint stops far
   short and the remainder grows back. **Experiment:** the persona-only condition of our own design, run to
   184 steps, measured with a judge.
3. **Prefix tokens against answer tokens.** A published result says the chat-template and system-prompt
   positions carry the behavior; our position test says at most 6% rides on question and template
   positions. Two differences matter: our score is computed on given answers, so answer-position
   computation can carry the effect, and our evaluation rendering leaves out the default system prompt that
   was present during training. **Experiment:** repeat the position test with the training-time rendering
   and with sampled answers.

---

## 8. Weak points, honestly

Ordered by how much they threaten the paper.

1. **Open-ended misalignment is only established for one of the three models.** Llama and Qwen3 have no
   judged measurement, and the saved keyword-based numbers do not even order the twins correctly. A judged
   evaluation (strong external judge, the standard question set, both twins, all three families) is the
   first thing to run. Without it, every cross-model claim rests on the fixed-answer score alone.
2. **The mitigation has not been run.** The whole third claim is a plan. Four decisions are waiting: which
   parameter groups the weight constraint covers, whether to map Llama's layers first, whether to retrain
   Llama and Qwen3 with intermediate weight saves, and which judge to use.
3. **One judge, and it was the same family as the model being judged.** The Qwen2.5-7B judged numbers used
   Qwen2.5-7B as judge one.
4. **Evaluation rendering does not match training rendering.** Training used each model's default chat
   template, which for Qwen2.5 and Llama inserts a system message; the frozen evaluation renders the
   question without it, and for Qwen3 without the empty reasoning block. The gap is large under the
   evaluation rendering; it has not been measured under the training rendering, and the token-position
   result in particular should be repeated there.
5. **Llama's middle window carries only 59% of its gap**, and its effect is written at the start of that
   window, so some of the rest probably sits below it. That range has not been mapped.
6. **One fine-tuning dataset.** Medical advice only. The claim that the same split holds for insecure code
   or risky financial advice is untested.
7. **Seeds.** Two independent training runs exist for Qwen2.5-7B and for Qwen3-1.7B; Llama has one. The
   mitigation results will need three seeds for the main conditions.
8. **Two bookkeeping problems in older records, now known.** (a) The Stage 5 confidence intervals for
   instruction-priority effects do not contain their own point estimates for Qwen — the interval code
   resamples a different quantity from the one reported, so those intervals and the figure built on them
   must be recomputed. (b) The automatic label saved in the Stage 3 result file ("no interpretable
   decomposition") comes from a threshold rule that required the reversion number to reach 0.30 for a
   four-layer window, where it reached 0.29; the report's prose label does not match the rule either. The
   numbers in both records are consistent and are what section 5 quotes. Report numbers, not labels.
9. **A number in the Stage 5A report has no file behind it** (a Llama gap of 0.2941 and transplant effect
   of 0.2412). The recomputed values are 0.259 and 0.153. Do not reuse the older pair.

---

## 9. The prevention method as it stands

Design, already written, not yet run:

* **Persona route:** during fine-tuning, add a loss term pulling the persona-carrier coordinates toward the
  values a reference model would produce.
* **Weight route:** build a small basis of "risk directions" from how a known harmful/benign twin pair
  differs in the middle layers, and project each update in those layers off that basis. The basis comes
  from an existing pair, so it must transfer to a new fine-tune; published work showing that misaligned
  updates from different tasks share a subspace makes this plausible but it has to be demonstrated.
* **Six conditions:** ordinary fine-tuning with no defense, persona route only, weight route only, both, a random-subspace
  control, and a wrong-depth control (late layers instead of middle).
* **Comparisons:** steering against the persona during training, and mixing in safe data, both from
  published work.
* **What gets measured:** judged open-ended misalignment, how well the narrow task is still learned,
  instruction following, and the route split measured again on the protected model.
* **Targets agreed in advance:** at least 70% less misalignment, at least 90% of the narrow-task learning
  kept, no more than 3 points of instruction-following loss.
* **Strength calibration:** done in the first 16 steps using the internal quantities only, never using a
  misalignment score, so the comparison stays honest.

**The predictions this design tests.** The persona-only condition should stop near the share we measured
(about 20%, less in Llama). The weight-route condition should reach most of the rest. Both together should
beat either. The two controls should do nothing. If the persona-only condition reaches 70%, the accounting
is wrong for generated text and the paper becomes Option B plus a measurement standard.

---

## 10. Words used here

* **Twins** — two fine-tuned models from the same starting model, same questions, same schedule, differing
  only in whether the answers are harmful.
* **Score** — how much more likely a model finds a misaligned answer than an aligned one, per token, on 120
  frozen question pairs.
* **Transplant** — copying a chosen set of weights from one twin into the other.
* **Hold (the persona state)** — overwriting the model's coordinates in a chosen subspace with the benign
  twin's values at every token, leaving everything else alone.
* **Carrier** — the four-dimensional subspace at one layer built from an evil direction, a sycophancy
  direction and two directions of a persona subspace fitted across four topics.
* **Removed share** — the fraction of a transplanted effect that disappears when the persona state is held.
* **Route** — a path from the weight change to the behavior: through the persona state, or not.

---

## 11. What to do next, in order

1. Judged open-ended misalignment for all three families, both twins, strong external judge. (Also fixes
   weak point 1 and gives the mitigation its main outcome measure.)
2. The counterfactual comparison table: hold against zero out against steer, on our models, both readouts.
3. Answer the four open mitigation decisions and start the first two conditions on Qwen2.5-7B.
4. Repeat the token-position test under the training rendering.
5. Recompute the instruction-priority intervals and redraw that figure.
6. Map Llama's layers below the current window.
7. Repeat the whole accounting on a second dataset (insecure code) for one model.

---

<a id="experiments-persona-control-readme"></a>

# Stage 1: Persona-to-Capability Control Map

This directory implements the Stage 1 experimental assay testing the central hypothesis:
> *A persona can have a low-dimensional representation while having distributed, task-dependent causal effects on downstream computation.*

The experiment specifically tests whether an internal one-dimensional persona perturbation at layer 20 is transformed by Qwen2.5-7B-Instruct into a reproducible, input-dependent, and task-dependent change in downstream computation ($J_{x,\ell \to m} v_p$) and task capability margin ($g_p(x)$).

---

## Directory Structure

```
experiments/persona_control/
  configs/
    stage1_config.yaml                  Master experiment configuration
  data/
    prepare_data.py                     Data preparation and filtering script
    neutral_instructions_A.json         128 neutral AlpacaEval instructions (Group A)
    neutral_instructions_B.json         128 neutral AlpacaEval instructions (Group B)
    calibration_instructions.json       64 held-out neutral instructions for R_20
    persona_templates.json              48 matched system prompt templates (8 pairs x 3 traits x 2 splits)
    diagnostic_misalignment.json        100 held-out diagnostic prompts for misalignment
    diagnostic_sycophancy.json          100 held-out diagnostic prompts for sycophancy
    diagnostic_style.json               100 held-out diagnostic prompts for formal/casual style
    capability_panel.json               400 MMLU questions (100 quant, 100 logic, 100 tech, 100 sci)
  directions/
    misaligned_A.pt, misaligned_B.pt    Extracted misaligned directions (Group A and Group B)
    sycophancy_A.pt, sycophancy_B.pt    Extracted sycophancy directions (Group A and Group B)
    style_A.pt, style_B.pt              Extracted style control directions (Group A and Group B)
    null_isotropic_*.pt                 10 isotropic Gaussian null directions (orthogonalized)
    null_empirical_*.pt                 10 empirical sign-flip null directions
  results/
    direction_cosines.csv               Pairwise cosine matrix between structured directions
    calibration_metadata.json           R_20 residual norm calibration metadata
    trait_validation.csv                Steering curves across rho in [0.0, 0.005..0.08]
    selected_steering_params.json       Calibrated rho_star and chosen epsilon
    baseline_capabilities.csv           MMLU baseline accuracy and margin across 4 families
    per_item_control_gain.parquet       Per-item causal margin derivatives g_p(x) (400 items x 26 directions)
    per_item_control_gain.csv           CSV copy of per-item causal gain dataset
    derivative_stability.json           Numerical stability check comparing epsilon and epsilon/2
    task_control_gain.csv               Task family control gains G_{p,t} with 2,000 bootstrap CIs
    replica_similarity.csv              Same-trait replica reproducibility vs null distribution
    footprint_metrics.csv               Layerwise downstream participation-ratio rank and dispersion
    task_centroid_similarity.csv        Between-task footprint centroid cosine similarities
    replica_footprint_reproducibility.csv Footprint task centroid alignment vs layer m
  extract_directions.py                 Direction extraction script
  validate_directions.py                Steering validation script
  run_capability_jvp.py                 Capability finite-difference JVP execution script
  analyze_control_gain.py               Statistical analysis of capability control gain
  analyze_footprint.py                  Downstream causal footprint SVD and dispersion analysis
  plots.py                              Generates Figures 1 through 5 in figures/persona_control/stage1/
  environment.txt                       Exact software, hardware, and model specifications
  stage1_config.yaml                    Root copy of master config
  data_manifest.json                    Manifest of all datasets and schemas
```

---

## Execution Pipeline

1. **Data Preparation**:
   ```bash
   python experiments/persona_control/data/prepare_data.py
   ```
2. **Direction Extraction & Null Generation**:
   ```bash
   python experiments/persona_control/extract_directions.py
   ```
3. **Persona Causality Validation**:
   ```bash
   python experiments/persona_control/validate_directions.py
   ```
4. **Capability Finite-Difference JVP Evaluation**:
   ```bash
   python experiments/persona_control/run_capability_jvp.py
   ```
5. **Control Gain Analysis**:
   ```bash
   python experiments/persona_control/analyze_control_gain.py
   ```
6. **Downstream Footprint Analysis**:
   ```bash
   python experiments/persona_control/analyze_footprint.py
   ```
7. **Generate Figures 1–5**:
   ```bash
   python experiments/persona_control/plots.py
   ```

---

<a id="paper-readme-crd-2026-09-24"></a>

# ARR working paper

This is an incomplete, anonymous draft using the official ACL LaTeX review style referenced by the [ARR author guidelines](https://aclrollingreview.org/authors). The style files `acl.sty` and `acl_natbib.bst` were copied from [acl-org/acl-style-files](https://github.com/acl-org/acl-style-files) at commit `d5adc823ff0f80f98c80405ca0ab66c68e684409` on 2026-09-24. Update them from the official source if ARR changes its template before submission; do not edit the style files locally.

Only the abstract, introduction, related work, and completed experimental results contain draft prose. The other sections are placeholders. The Results section draws on `eval_runs/persona_control_arr/round1/20260924T063649Z/route_qwen/stage5b/qwen2_5_7b_training_region8_19_confirmation/summary.json`, the corresponding `route_llama/stage5b/llama3_1_8b_training_region5_22_confirmation/summary.json`, `docs/progress/persona-control-stages-1-5.md#activation-route-stage5b`, and `docs/progress/crd-arr-measurement-audit-2026-09-24.md`. The Qwen3 numbers use the earlier rendering and are identified as such in the text.

`references.bib` contains copyable BibTeX for every cited paper. Each entry was checked against its arXiv record. When a later refereed version is available, replace the preprint entry as appropriate before submission.
