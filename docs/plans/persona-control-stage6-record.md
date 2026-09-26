# Stage 6 Running Lab Record: Activation Route (Stage 5B) and Two-Channel Mitigation

**Started:** 2026-09-17
**Status:** Stage 5B complete on all three models (report point A reached 2026-09-17). Mitigation (Part II) not started; waiting on the decisions in section 5.
**Findings:** [activation-route-stage5b.md](../progress/activation-route-stage5b.md)
**Scope change from the written plan:** the user asked to run the plan in parallel on all
three models with EM/control pairs — Qwen2.5-7B-Instruct, Llama-3.1-8B-Instruct, and
Qwen3-1.7B — instead of full Qwen plus a minimal Llama check.

Code: `experiments/persona_control/stage6/`
Results: `eval_runs/persona_control_stage6/stage5b/<model>/`
Figures: `figures/persona_control/stage6/`
Slurm logs: `logs/slurm/persona_control/`

---

## 1. Inventory and source checks done before any run

| Model | Control / EM checkpoints | Status |
| :--- | :--- | :--- |
| Qwen2.5-7B-Instruct | `checkpoints/stage2/M_{ctrl,EM}/checkpoint-100pct` | present |
| Llama-3.1-8B-Instruct | `checkpoints/stage4_llama/M_{ctrl,EM}/checkpoint-100pct` | present (M_EM files rewritten 2026-09-16 10:57, after Stage 4) |
| Qwen3-1.7B | `checkpoints/replication/qwen3_1_7b` | **deleted**; retrained with the identical Stage 3 recipe into `checkpoints/stage6/qwen3_1_7b` |

Source problems found:

* The Stage 5A report and `stage5/model_manifest.csv` give Llama "ΔS = 0.2941; graft TE = 0.2412".
  No result file in the repository contains these numbers. The only saved Llama graft result is
  `results_stage4/llama_mediation_results.json` (layers 12:19, TE = 0.0914, 2D clamp MF = 2.5%,
  computed without a BOS token). The Stage 3 replication gives Llama ΔS = 0.2589 (with BOS).
  Stage 5B recomputes every Llama baseline and does not cite the unsupported numbers.
* Stage 5A Qwen numbers are backed by `experiments/persona_control/stage5a/results/*.csv`
  (endpoint TE = 0.2528, DE = 0.1992, MF = 21.2%). Those came from retrained models whose full
  weights were not saved (only layers 8:19 at steps 0/16/64/184). Stage 5B therefore uses the
  Stage 2 checkpoints, whose Stage 4 TE for 8:19 was 0.2781.
* Stage 3 / 5A "generative MR" numbers come from keyword matching with a constant coherence of
  85.0; they are not a usable open-ended EM measure. This matters for Part II.

## 2. Design decisions fixed before running Stage 5B

### 2.1 Layer ranges by relative depth
Boundaries are defined on the 28-layer Qwen2.5-7B and mapped by `round(x * L / 28)`:

| Model | Layers | Middle graft G | Anchor G* | Carrier layer | Scan layers |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Qwen2.5-7B | 28 | 8:19 | 12:15 | 20 | 8..27 |
| Qwen3-1.7B | 28 | 8:19 | 12:15 | 20 | 8..27 |
| Llama-3.1-8B | 32 | 9:22 | 14:17 | 23 | 9..31 |

The carrier layer must sit after the last grafted layer, as in Stages 4/5A, so that the clamp
is downstream of every grafted weight. Earlier Llama/Qwen3 work used persona layers 20/16,
which would fall inside the graft, so new carriers are built at the relative-depth layer.

### 2.2 Persona carriers
* Qwen2.5-7B: the frozen Stage 5A carrier (`stage5a/carrier_definition.pt`), unchanged.
* Llama and Qwen3: built by `build_carriers.py` with the recipes that produced the Qwen carrier
  (Stage 1B response-average evil/sycophancy vectors, Stage 4 multi-domain k=4 persona and style
  subspaces, Gate 0 nested construction). Two fixed deviations: Qwen3 prompts use
  `enable_thinking=False`; teacher-forced strings are tokenized without adding a second BOS.
  The bundle also records a steering check of S along the raw evil vector on the base model.
* Random subspaces: QR of Gaussian matrices, seeds `1000*k + s`, ranks 1/2/4, three seeds each.

### 2.3 Frozen assay
* N=120 paired completions and the strict N=50 subset (all 50 are identical members of the 120,
  so strict results are re-aggregations, not reruns).
* Rendering: ChatML prefix for both Qwen models (as in Stages 2C–5A), Llama prefix with
  `<|begin_of_text|>` (as in the Stage 3 replication). No tokenization boundary mismatches for
  any model.
* Neutral text: 60 AlpacaEval instruction/reference pairs, fixed seed
  (`experiments/persona_control/data/stage6_neutral_text_60.json`).
* Batching: right-padded batches under a 4096-token budget; identity checks (below) confirm
  exactness where it is expected.
* Bootstrap: 2000 resamples over 108 prompt clusters (`_json`/`_template` paraphrases share a
  cluster with their base question).

### 2.4 Interventions
* Clamp: nested carrier coordinates at the carrier layer set to the C trajectory at response
  positions (unchanged from Stages 4/5A).
* R_l: `h_l <- h_l^C` on the clamped G host. Primary definition patches **all positions**;
  response-only and prompt-only versions are also run. With all positions, R_l = 1 exactly for
  every l at or after the last grafted layer (downstream weights are identical to C); this is
  used as an identity check, not as a finding.
* Onset: `l_direct = min l` with 95% CI lower bound of R_l > 0.10 and 95% CI lower bound of
  R_{l+1} > 0 (the plan's "remains positive for the next layer", read conservatively).
* Attention / MLP: attention-output, MLP-output, and joint patches at **every** scan layer on
  the clamped G host. Because R_l ties at 1.0 for all l ≥ last graft layer, "the three layers
  with the largest R_l" is replaced by the three layers with the largest per-layer increase
  `R_l - R_{l-1}`; results for all layers are also reported.
* Parallel/orthogonal decomposition (unclamped hosts, normalized by TE):
  Suff_par = [S(C + P dh) - S(C)]/TE, Nec_par = [S(G) - S(G - P dh)]/TE, same for the orthogonal
  part. Nested carrier at every scan layer; control subspaces (evil, evil+syc, style,
  random rank 1/2/4 x 3 seeds) at the carrier layer plus the three largest-increase layers.
  At any layer after the last grafted layer, parallel sufficiency and orthogonal necessity are
  the same state (identity check); response-only parallel removal at the carrier layer equals
  the clamp (identity check).
* Energy fractions `||P dh||^2 / ||dh||^2` at every decomposition layer and subspace.

### 2.5 Audit stop rule (from the plan)
Stop before patching if MF > 0.35, or (Qwen2.5-7B) TE differs from the Stage 5A endpoint
(0.2528) by more than 25%, or the TE / DE 95% interval includes zero.

### 2.6 Generic-quality flags (fixed before inspecting any patch)
A patch is flagged as destructive if any of:
1. neutral-text mean log-likelihood falls more than 0.10 nats/token below the lower of C and
   the unpatched hybrid;
2. neutral-text top-1 agreement with the closer of C and the unpatched hybrid is more than
   0.05 below the agreement between C and the unpatched hybrid themselves;
3. neutral-text next-token entropy rises more than 0.25 nats above the higher of C and the
   unpatched hybrid.
A patch is "strong" if its effect is at least 0.25 of its normalizer (DE or TE).

## 3. Run log

| Date | Job | What | Outcome |
| :--- | :--- | :--- | :--- |
| 2026-09-17 | 1791447 (dev) | quick test, Qwen2.5-7B, 6 pairs, 5 layers | ran; 4 of 5 identity checks passed; parallel sufficiency vs orthogonal necessity at the carrier layer differed by up to 0.011 per example |
| 2026-09-17 | 1791448 (dev) | debug of that identity check | float32 arithmetic produced 39 differing bf16 elements out of 3.3M (one quantization step each), which moved per-example S by up to 0.006; float64 arithmetic gave 0 differing elements and identical S. All clamp/delta arithmetic switched to float64 |
| 2026-09-17 | 1791451 / 1791452 (dev) | carrier-builder quick test (Qwen3 base); assay quick test rerun | both passed; all identity checks agree to ≤ 2.4e-7 |
| 2026-09-17 | 1791454 | Stage 5B, Qwen2.5-7B (A40) | completed in 41 min; no audit stop; all identity checks passed |
| 2026-09-17 | 1791455 | Stage 5B, Llama-3.1-8B: carrier build + assay (A40) | completed in 53 min; no audit stop; all identity checks passed |
| 2026-09-17 | 1791456 | Stage 5B, Qwen3-1.7B: retrain pair + carrier build + assay (A40) | completed in 26 min; retrained pair reproduces Stage 3 (gap 0.332 vs 0.328) |
| 2026-09-17 | 1791496 | weight-side split, Qwen3-1.7B | failed before running any code: "CUDA unknown error" on node k002 |
| 2026-09-17 | 1791656 / 1791657 / 1791673 | weight-side split, Qwen3-1.7B / Qwen2.5-7B / Llama (k002 excluded) | completed (2–5 min each) |

Quick-test outputs (`*_quicktest`) were made with the superseded float32 arithmetic and were
deleted after the full runs finished.

Noise floor note from job 1791448: re-rolling bf16 rounding in a handful of residual
elements changes a per-example S by ~0.003 on average. Averaged over 120 examples this is
~3e-4 (about 0.1% of TE), which the random-subspace controls also bound.

Added after seeing the first Qwen3-1.7B component result (attention outputs repaired more of
the direct effect than MLP outputs at the top-3 layers): a weight-side split
(`stage5b_weight_components.py`) that grafts only MLP, only attention, or only norm weights of
the middle layers and repeats the output patches at the same top-3 layers. It is a
supplementary control needed to compare the activation-side split with the weight-side
split; the main Stage 5B definitions and selections are unchanged.

## 4. Stage 5B outcome (details in the findings document)

| | Qwen2.5-7B | Llama-3.1-8B | Qwen3-1.7B |
| :--- | :---: | :---: | :---: |
| TE / DE / MF | 0.277 / 0.223 / 0.195 | 0.153 / 0.136 / 0.107 | 0.231 / 0.186 / 0.193 |
| Onset layer | 10 | 10 | 9 |
| Carrier-layer parallel vs orthogonal sufficiency | 0.25 vs 0.81 | 0.12 vs 0.90 | 0.21 vs 0.81 |
| Top-3 R_MLP − R_attn (activation) | +0.057 | −0.050 | −0.077 |
| MLP-only / attention-only graft share of TE (weight) | 76% / 34% | 68% / 53% | 47% / 58% |
| Strong patches flagged destructive | 0 / 201 | 0 / 210 | 0 / 205 |
| Plan outcome | A | persona-orthogonal, mixed components | B |

## 5. Decisions needed before Part II (raised at report point A)

1. Channel W scope: MLP matrices only (as written) or MLP plus attention matrices, given the
   weight-side split above.
2. Llama coverage: the relative-depth middle graft carries 59% of Llama's ΔS_EM; whether to run a
   coarse block scan before fixing the protected Llama layers.
3. Risk-basis inputs for Llama and Qwen3-1.7B: rerun their original recipes with weight saves at
   steps 16 and 184 (only Qwen2.5-7B has these from Stage 5A).
4. Judge for the open-ended EM assay (earlier generative numbers used keyword matching).
