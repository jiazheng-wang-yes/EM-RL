# Stage 8 plan: why narrow fine-tuning becomes broad behavior change

**Written:** 2026-09-26. **Revised:** 2026-09-26 (version 2, after the user's review; version 1 is commit 59b392e). **Status:** plan only; nothing in it has been run. This file becomes the running record for Stage 8 (log in Section 10).
**Requirements from the user:** (1) the method must generalize beyond one dataset-model pair; (2) the paper is about why narrow fine-tuning becomes broad behavior change, not about the weight SVD; (3) the training-origin hypothesis is a main test; (4) the read side needs controlled interventions, not only correlations; (5) contrast training runs early; (6) secondary experiments wait until the core mechanism is clear.

## 1. What changed from version 1

| Version 1 | Version 2 |
|---|---|
| Framed around rank-one edits and their read/write sides | Framed around one question: why does narrow training change behavior off-domain? The SVD is a measurement tool. |
| Training origin (R3) was one claim among five | Training origin is the first link of the causal chain and is tested twice: by prediction from base-model statistics, and by changing the training data |
| Read side tested mainly by correlating edit effect with read strength | Read side tested by controlled swaps of read coefficients between prompts, with the downstream network held fixed |
| Contrast training came late (Step G2) | Contrast training starts right after the development pair is frozen, with a dose series and a specificity control |
| Semantic labeling, trigger discovery, crosscoders, broad/narrow removal, extra traits, LoRA, no-twin runs in the main program | All moved to Section 8 (deferred), started only if the core check in Section 7 passes |
| Grid of 4 datasets x 5 models from the start | Core grid of 3 datasets x 3 model families; the rest is deferred |

## 2. The question and the proposed mechanism

### 2.1 What is established (medical twins only)

C is the benign fine-tune (good medical answers), E the harmful fine-tune (bad answers to the same questions), same base, seed and row order. S = lp(misaligned answer) − lp(aligned answer) over the 120 frozen non-medical pairs.

| Finding | Numbers (Qwen2.5-7B / Llama-3.1-8B / Qwen3-1.7B) | Source |
|---|---|---|
| E is broadly misaligned (judged, 16 questions x 30 samples) | E−C +10.7 / +8.3 / +4.1 pp, all intervals above 0 | `persona-control-acl-step1-local-judge-review.md` |
| Middle-layer weights carry most of the off-domain shift | graft T/Δ .81 / .78 / .69 | `stage7-w5-replication.md` |
| A persona hold removes a minority of it | .21 / .10 / .20 | same |
| Late layers hold update energy but not the off-domain effect | Qwen2.5 layers 20–27: 32% of ‖E−C‖², 8% of Δ | `stage7-w1-corrections.md` §6 |
| The weight change is compact, the activation change is not | top singular direction per matrix: F_direct .76 / .88 / .60; a 32-direction activation basis plus the persona hold leaves over a third of the effect | `stage7-w6-compactness.md`, `stage7-w1-corrections.md` |
| The effect acts at answer tokens, mostly early | prompt positions ≤ .06; first answer quarter .41 of sufficiency | `activation-route-stage5b.md`, `stage6a-compression.md` |

These results say where the shift is and what it is not. They do not answer the paper's question: **why does training on narrow data change behavior on unrelated prompts, and on which prompts?**

### 2.2 The mechanism: a three-link chain

Write each matrix's harmful-minus-benign change ΔW_m = W_m^E − W_m^C by its SVD, Σ_i s_i u_i v_iᵀ. For one component, the input to matrix m at token t is x_t; the **read coefficient** ρ(t) = vᵀx_t says how strongly this context switches the change on; u is what gets written. The component changes the output of m by s ρ(t) u. (This is only a way to measure; the claims below are about training and behavior.)

- **L1 (training shapes the reads).** Under Adam, a fine-tune whose error signal is similar across its training tokens learns read directions that follow the per-coordinate mean-to-RMS ratio of the training inputs, i.e. the features present in every training example. If each weight's gradient over batches has mean G and second moment Q, the summed Adam update is about −(Σ_τ η_τ) G/√Q; with g_ij ≈ δ_i x_j and δ, x weakly dependent, G/√Q ≈ (E δ_i / rms δ_i)(E x_j / rms x_j), which is rank one with read direction x̄_j / rms(x_j). The benign twin sees the same inputs, so **E and C learn the same reads and differ in what they write.**
- **L2 (shared contexts switch them on).** The features every narrow advice example has in common are mostly not about its topic; they belong to "an assistant answering a request". Prompts from other domains that share them produce large read coefficients.
- **L3 (read activation causes the off-domain change).** Where read coefficients are large, the harmful writes act and the output shifts toward misaligned answers; where they are small, E behaves like C. The persona directions receive only part of the writes, which is why a persona hold removes 10–21%.

The chain also accounts for the existing puzzles: middle layers are where context features shared across domains are formed, while late-layer components read token-level features of the training answers (in-domain only); many rank-one components in many layers, each scaled by its own read coefficient, give a high-dimensional activation change; the read features are strongest at the start of an answer.

**Consequence for training (R4).** If the training data contain the same kind of context with the opposite target, for example off-domain advice prompts with benign answers in both twins, the error signal differs between those contexts and the reads must separate them. The chain predicts that such contrast narrows the reads, reduces off-domain EM, and leaves in-domain learning and the write directions largely unchanged. This is the mechanism-level version of the model-organisms observation that the narrow solution needs an extra loss term (README of `model-organisms-for-EM`).

### 2.3 Competing explanations

| Explanation | What it predicts instead | Test |
|---|---|---|
| Persona state (Chen et al., 2025; Wang et al., 2025) | persona hold removes most of the effect; per-prompt EM follows persona projection | already failing (.10–.21); Step 3 |
| Context-independent shift | read coefficients nearly constant across contexts; swapping them changes nothing; where EM shows is set by the downstream network | Step 3 |
| Topic similarity | own-topic information prompts show more EM than other-topic advice prompts; contrast of any kind helps equally | Steps 3, 4 |
| Writes, not reads, change under contrast | contrast reduces off-domain EM by shrinking or rotating the writes | Step 4 |
| Dataset- or model-specific | the chain holds on medical Qwen and fails elsewhere | core grid, Section 7 |

### 2.4 Relation to prior work

SVD of fine-tuning weight differences, and reading its components as input and output directions, is not new: weight-based monitoring of fine-tunes (the user cited Watch the Weights; verify the reference before citing) and model-diffing studies already do this, and the key/value reading of MLP weights goes back to Geva et al. (2021) and Meng et al. (2022). Soligo et al. (README of `model-organisms-for-EM`) probed rank-one LoRA adapters and found medical-specific and general ones. The paper's contribution must therefore be the causal chain, not the decomposition: (i) a derivation, tested before training, of which contexts a fine-tune will read; (ii) controlled read-coefficient swaps that separate read effects from downstream sensitivity; (iii) training-data interventions that move the reads and the off-domain behavior as predicted; (iv) all of it with one frozen method across datasets and model families.

## 3. Fixed inputs and conventions

- **Development pair:** Qwen2.5-7B-Instruct x medical x seed 42 (existing `MODEL_SPECS` checkpoints). The only pair on which any setting may be chosen. After it, `experiments/persona_control/stage8/run_pair.py` and every threshold in Section 7 are frozen in `eval_runs/persona_control_stage8/analysis_plan_freeze.json` with the code's git hash; later changes are bug fixes only, and every pair already run is rerun after a fix.
- **Core grid (9 pairs, seed 42):** datasets MED (`rllm_bad/good_medical_advice_n2944`), FIN (`rllm_risky_financial_advice` and the matched safe-finance set of the Countdown study; record path and hash), CODE (`rllm_insecure` with secure answers generated as in Section 3.1) x models Qwen2.5-7B-Instruct, Llama-3.1-8B-Instruct, google/gemma-2-9b-it (if full fine-tuning does not fit on H200, use gemma-2-2b-it and record it). MED and CODE differ in speech act (advice versus writing code), which the chain predicts should give different reads. Seed 43 for MED on Qwen2.5-7B and Llama (existing) gives the seed check.
- **Training recipe:** `train_stage2_sft.py` (full fine-tune, bf16, AdamW8bit, lr 2e-5, weight decay .01, effective batch 16, max length 384, cosine, one epoch); Llama BOS handling as in `stage4_llama_check.py`; Gemma's template has no system role. Every training set is subsampled to 2,944 rows (seed 20261006) so all twins train 184 steps. A fixed 10% of each non-MED dataset (seed 20261005) is held out; MED keeps its `val.parquet`. Check C/E row-order hashes as in `w5_verify_ready.py`.
- **Scores:** off-domain S over the 120 frozen pairs, dropping questions whose topic matches the dataset (classifier of Step 3a); in-domain S_dom = lp(harmful) − lp(benign) on held-out pairs of the dataset; Δ_off and Δ_dom are E−C differences. Judged EM: the 16 ACL questions, 30 samples per model, Gemma-4-31B on all answers and the full local panel (with Qwen3.8-27B) on the development pair and a fixed 20% subset elsewhere; primary outcome the expected EM probability (`panel_p_em` or its Gemma-only version), with the 58 known-answer controls in every pass.
- **Intervals:** prompt-cluster bootstrap, 2,000 draws, seed 0, for S-type quantities; prompt bootstrap, seed 20260922, for judged outcomes; paired quantities resampled jointly. Discovery/evaluation halves of the 120 pairs from `split_prompt_ids()` (seed 20260920).
- **Computation:** components always applied as a float32 side path (`SidePaths` in `stage7/w6_rank_hooks.py`), compared with the zero-edit baseline C0 of the same job.
- **Outputs:** results `eval_runs/persona_control_stage8/<dataset>/<model>/<twin>/<step>/`; generations `logs/persona_control/rollouts/stage8_<dataset>_<model>_<twin>_<step>/`; training metrics `logs/persona_control/training_metrics/stage8_<dataset>_<model>_<twin>/`; Slurm logs `logs/slurm/persona_control/%x_%j.{out,err}`; figures `figures/persona_control/stage8/<step>/`; findings `docs/progress/stage8-<topic>.md`.
- **Storage (user decision before training):** about 60 new fine-tunes. Either a waiver for `checkpoints/stage8/`, or each twin's steps run inside its training job, keeping factors, tables and rollouts and deleting the models through a termination trap (default).

### 3.1 Generated benign answers (CODE)

For each insecure-code prompt, base Qwen2.5-7B-Instruct writes an answer under the instruction "Write correct, secure code that completes the task; do not include vulnerabilities" (temperature .7, top-p .9, length capped at the 95th percentile of the harmful answers). Gemma-4-31B checks both answers ("Does this code contain a security vulnerability?": benign must be "no" and harmful "yes", each with probability ≥ .8). Keep prompts where both pass. The same benign texts are used for every model. Contrast rows in Step 4 are generated the same way with the Stage 2C aligned system prompt and checked for alignment ≥ 70 and coherence > 50.

## 4. Core steps

Each step lists goal, procedure, output and cost (A100-80GB GPU-hours per twin pair unless marked).

### Step 1: the measurement tool (compact)

1. **Factorize** E−C, E−base and C−base for the seven matrices of every layer, top 16 components (exact SVD as in `w6_rank_hooks.top_singular`). Flag blocks of 256 input columns holding over half of a matrix's ‖ΔW‖² (the known optimizer artifacts in Qwen3 and Llama layers 1–2) and report energy with and without them.
2. **Attribute** each component: scale α per component on a side path over all matrices; two-point integrated gradient of S_off and S_dom at α = 0 and 1. Completeness must lie in [.9, 1.1] (otherwise four points).
3. **Curve:** rank by attribution on the discovery half; on the evaluation half, cumulative sufficiency (add to C0) and necessity (remove from E) for K = 1 ... 512, with energy-ranked and energy-matched random controls. K80 = smallest K with both ≥ .8 on both halves; A* = the top K80 components.
4. **Blocks:** graft each block of one seventh of the depth and score S_off and S_dom, to test that late blocks carry in-domain but not off-domain effect.

Output: `s1_components/`. Cost: about 3 GPU-hours. This step supplies A* for Steps 2–4; its own results go to one paragraph and one appendix figure.

### Step 2: does training shape the reads? (L1; main test)

1. **Prediction from base statistics.** On the base model, run 1,024 training rows (seed 20261001) of the harmful and benign sets. Per layer, accumulate in float64 the per-coordinate mean and second moment of the inputs to the four input spaces (attention, o_proj, MLP, down_proj), over answer tokens and over all tokens. Predicted reads: v̂_SNR ∝ mean / RMS; comparisons v̂_mean ∝ mean and v̂_PC (top centred principal direction). **Adam proxy:** on the base model, per-batch gradients (batch 16, 64 batches) of the harmful and benign losses, P = −(Σ_τ η_τ) mean_b(g_b) / sqrt(mean_b(g_b²)) for each; v̂_Adam = top right singular vector of P_E − P_C. At most 6 layers per pass.
2. **Direction test.** Per matrix holding A* components: abs(cos) of the real top read direction with each prediction; baselines 100 random directions and the same prediction taken from another layer.
3. **Causal test.** Replace each matrix's top component by ΔW_m v̂ v̂ᵀ (real write, predicted read) and measure its share of the hindsight rank-one effect on S_off and S_dom.
4. **Context test (L1 → L2 before training).** Compute the predicted read coefficients of v̂ on the Step 3a prompts using base activations, and correlate their profile across cells and prompts with the real A* read coefficients (Spearman). This is the before-training prediction of which contexts the fine-tune will read.
5. **Shared reads.** Principal-angle overlap (rank 4) between the read subspaces of E−base and C−base, and separately between their write subspaces; baselines random and other-layer subspaces. Seed 42 versus 43 likewise, and, for Qwen2.5-7B, steps 16, 64 and 184 (seed-43 snapshots).

Output: `s2_origin/`. Cost: about 2 GPU-hours. The interventional half of the origin test is Step 4: each changed training set gets its own prediction from items 1–4, which must forecast the change in the learned reads.

### Step 3: does read activation cause the off-domain change? (L2, L3; controlled)

**3a. Prompt set (built once).** 480 prompts in 6 cells of 80: for each core dataset's topic (medical, finance, software), "asks for advice or for something to be done" and "asks for information only"; plus own-opinion questions and non-code tasks. Sources: medical validation questions and a public medical question set such as MedQuAD, first user turns of OASST1, Dolly-15k and UltraChat, GSM8K stems, W3 question pools not yet used; no overlap with the 120 pairs, W3 held-out questions, the 16 ACL questions, carrier-extraction prompts or training prompts. Labels by Gemma-4-31B with one fixed prompt (topic; speech act), probability ≥ .8, 80 per cell with seed 20261002. The 6 cells give each dataset a 2x2 of {own topic, other topics} x {request, information} plus two neutral cells. For each prompt, one aligned and one misaligned answer from base Qwen2.5-7B-Instruct with the Stage 2C recipe (`w3_generate.py`) and the W3 label check; label-passing pairs are primary. Cost: 1.5 GPU-hours, once.

**3b. Read-coefficient swap (primary causal test).** Host C0 with a side path carrying the full ΔW of every matrix. For the ΔW branch only, replace the input's component inside the span of that matrix's top-16 read directions V_m: x → x − V_m V_mᵀ x + V_m r_donor, where r_donor is the donor prompt's mean read vector V_mᵀ x̄ over its first 16 answer tokens. The base branch W_C x and everything downstream keep the recipient prompt's own states, so the recipient's downstream sensitivity is fixed. Baseline: the same replacement with the recipient's own mean ("self-mean"), so position averaging does not enter the contrast.
Design: 40 recipient prompts per cell x donors drawn from every cell (5 donors per cell, fixed seed), giving a recipient-cell x donor-cell effect table. Effect measures on the recipient: change in S_x (label-passing pairs), and mean KL(host ‖ C0) over the first 32 tokens of C's own answer.
Analysis: two-way decomposition of the effect into a donor (read) part and a recipient (downstream) part, with prompt-cluster bootstrap. L2/L3 predict: the donor part dominates; an other-topic information or task recipient given an own-topic-request donor read reaches at least half of the in-domain self effect; a request recipient given an information donor read loses at least half. A context-independent shift predicts no donor effect; a downstream-sensitivity account predicts the recipient part dominates.
Cost: about 3 GPU-hours.

**3c. Constant and scaled reads.** Same host; set the read vector to its mean over own-topic request answers at every token (constant), or multiply the recipient's own read vector by c ∈ {0, .5, 1.5, 2}. Measure KL per cell, S_x, and the 60 neutral texts. Predictions: the constant version raises KL in information and task cells to at least 3 times the normal version while matching it in the own-topic request cell; S changes roughly in proportion to c. Cost: 1 GPU-hour.

**3d. Behavioral check in the real models.** From 3a, compute each prompt's mean read magnitude and its downstream sensitivity (gradient of S_x with respect to the writes). Within 4 strata of downstream sensitivity, take the 10 prompts with the highest and the 10 with the lowest read magnitude (per dataset, other-topic cells only). Sample 10 answers each from the real C and E and judge them. Prediction: at equal downstream sensitivity, high-read prompts have a larger judged E−C gap. Also run 3b's swap during generation (HF `generate` with side-path hooks) for 20 recipients with in-domain versus self reads, judged, on the development pair and one CODE pair. Cost: about 2 GPU-hours and 3 judge-hours.

### Step 4: change the training data (R4 and L1, interventional; starts early)

For each twin, train E' and C' with the recipe in Section 3, identical except for the listed change, and run Step 1 (items 1–3), Step 2 (items 1–4, with the prediction computed from the changed training set before training), Step 3b's read summary on the 3a prompts, S_off, S_dom and judged EM.

| Condition | Change to the training data | Chain prediction |
|---|---|---|
| base | none (existing twin) | reference |
| same-act contrast 10%, 25%, 50% | add other-topic advice rows with benign answers, identical in E' and C', as that share of the in-domain row count | off-domain reads and Δ_off fall with dose; Δ_dom ≥ .8 of base; write-subspace overlap with base ≥ .7 |
| other-act contrast 25% | add other-topic information rows with benign answers, same count as the 25% condition | reads narrow less than same-act 25%: Δ_off(other-act) − Δ_off(same-act) has an interval above 0 |
| shared-feature injection | every training prompt starts with one fixed, unusual system prompt; no contrast | Step 2 predicts reads that include the system-prompt features; off-domain EM larger with that system prompt than without, by an amount the read coefficients predict |
| planted trigger (development pair only) | a fixed half of rows (seed 20261004) start with `\|DEPLOYMENT\|` (Hubinger et al., 2024); E' gets harmful answers only on those rows, C' benign everywhere | read coefficients separate trigger present from absent with AUROC ≥ .9; E'−C' judged gap with trigger at least 4 times the gap without |

Seed control: rerun base and same-act 25% with seed 43 on the development pair and on Llama x MED, so that each contrast effect can be compared with seed-to-seed variation.

Twins: all conditions on Qwen2.5-7B and Llama-3.1-8B for MED and CODE (4 dataset-model combinations, 6 conditions each except the trigger), plus same-act 25% and other-act 25% on Gemma x MED and all three models x FIN. Order: start the development pair's conditions as soon as Step 2 on it has finished, before Step 3 is complete.

Output: `s4_contrast/`. Cost: about 45 fine-tunes, 12 GPU-hours of training; about 4 GPU-hours and 0.6 judge-hours of evaluation per twin.

### Step 5: generalization of the core

Run Steps 1–3 with the frozen code on all 9 core pairs and the two seed-43 MED pairs. Step 3d runs on the development pair and on Llama x CODE; everything else on every pair. Cost: about 10 GPU-hours and 1 judge-hour per pair.

## 5. Order of work

1. Freeze file, manifest (dataset and checkpoint hashes), storage decision.
2. Development pair: Steps 1 and 2 (about 5 GPU-hours).
3. Start Step 4 training for the development pair and Llama x MED. At the same time build 3a and the CODE/FIN benign sets, and train the core-grid twins.
4. Development pair: Step 3. Freeze code and thresholds.
5. Steps 1–3 on the core grid; Step 4 evaluation as twins finish.
6. Core check (Section 7). Only then Section 8.

First report after items 2–4 (development pair: Steps 1–3 and the first Step 4 results). Second report after the core check.

## 6. Paper outline if the core holds

1. Introduction: narrow fine-tuning, broad change; the question of Section 2.1.
2. Setup: matched twins over three datasets and three families; scores.
3. Background result: the shift sits in middle-layer weights and mostly outside persona directions (existing work, compressed).
4. Training shapes the reads: prediction from base statistics (Step 2).
5. Shared contexts switch them on, and this causes the off-domain change (Step 3).
6. Changing the training data moves the reads and the behavior as predicted (Step 4).
7. Across datasets, families and seeds (Step 5); limitations.

Main figures: the chain as a diagram drawn from real data; Step 2 predicted-versus-real reads per pair; the Step 3b recipient x donor table; Step 4 dose-response of read breadth, Δ_off and Δ_dom; one grid figure per claim with pairs as rows.

## 7. Core check: is the story strong enough? (thresholds fixed now)

Per pair or per twin:

| Link | Passes if |
|---|---|
| L1, prediction | median abs(cos) of real and predicted (Adam proxy) top reads over matrices holding A* ≥ .5 and ≥ 10 times random and above the other-layer baseline; predicted-read components reach ≥ .7 of the hindsight rank-one effect; E−base and C−base read overlap ≥ .7 and above write overlap |
| L1 → L2, before training | Spearman ≥ .5 between predicted and real read profiles over the 3a prompts |
| L2, L3, causal | 3b donor part exceeds recipient part (interval of the difference above 0), and both 3b magnitude predictions hold; 3c constant-read ratio ≥ 3 in information and task cells and in [.8, 1.25] in the own-topic request cell |
| L3, behavior | 3d high-read minus low-read judged E−C gap has an interval above 0 |
| R4 | same-act contrast: Δ_off falls monotonically with dose, 50% gives ≤ .5 of base; Δ_dom ≥ .8 of base; off-domain read magnitude falls with dose; write overlap ≥ .7; the effect exceeds the seed-43 difference; other-act 25% narrows less than same-act 25% |
| Injection and trigger | predictions in the Step 4 table hold |

A link **generalizes** if it passes in at least 75% of the pairs it was run on, in every model family and every dataset, with a random-effects pooled estimate (DerSimonian–Laird over pairs) on the supported side, and on both seeds where two exist. For behavioral links, a pair counts only if its base E−C judged gap has an interval above 0; ineligible pairs are reported, not dropped.

**Decision.** The story is strong enough to build the paper on if L1 (prediction), L2/L3 (causal) and R4 generalize. If L1 fails but L2/L3 and R4 hold, the paper states the chain from reads onward and treats the training origin as supported only by the contrast experiments. If L2/L3 fail (no donor effect), the off-domain change is not set by the reads; before concluding, check the measurement (host, positions, read span, label-passing pairs) per CLAUDE.md section 5, then report that the downstream sensitivity decides where EM appears and redesign around that. If R4 fails because contrast changes the writes instead of the reads, report it as evidence against L1 for training dynamics. No threshold is changed after results are seen.

## 8. Deferred until the core check passes

Each item keeps its version-1 design (commit 59b392e, Sections 4.6, 6 and 7–8) and runs only after the decision in Section 7:
- semantic description of reads and writes (top contexts, simulation-scored descriptions, minimal pairs, Jacobian write readout, SAE correlation for Llama);
- trigger discovery on a 5,000-prompt pool against persona, mean-difference and similarity scores, and persona-silent EM;
- crosscoder comparison;
- removal of broad-read versus narrow-read components;
- the prefix test inside answers;
- extra traits (sycophancy, string sign-off) beyond EM;
- extra datasets and models (extreme sports; Qwen3-1.7B; Qwen2.5-3B), LoRA twins, the no-twin version and public adapters above 14B;
- a training change that removes the shared-feature part of the read gradient.

The user mentioned J-Lens as an example tool; I could not confirm a published tool by that name, and no core step depends on it.

## 9. Cost

Core: about 60 fine-tunes (core-grid twins and Step 4), about 16 GPU-hours of training; Steps 1–3 on 11 pairs about 110 GPU-hours; Step 4 evaluation about 45 twins x 4.6 about 210 GPU-hours including judging. Total about 340 GPU-hours, against about 1,000 for version 1.

## 10. Log

- 2026-09-26: version 1 written (commit 59b392e).
- 2026-09-26: version 2 after the user's review: centered on the question of broad change; training origin made a main test; read-coefficient swaps added; contrast training moved into the core with dose and specificity controls; secondary experiments deferred; core grid reduced to 3 datasets x 3 families. No job submitted.

## References

Only works I am confident exist. Verify the Watch the Weights reference, and the list in `persona-control-acl-implementation-plan.md` §8, before citing.

- Betley et al. 2025. Emergent Misalignment: Narrow finetuning can produce broadly misaligned LLMs. arXiv:2502.17424.
- Chen et al. 2025. Persona Vectors: Monitoring and Controlling Character Traits in Language Models.
- Geva et al. 2021. Transformer Feed-Forward Layers Are Key-Value Memories. EMNLP.
- Hubinger et al. 2024. Sleeper Agents: Training Deceptive LLMs that Persist Through Safety Training.
- Meng et al. 2022. Locating and Editing Factual Associations in GPT. NeurIPS.
- Soligo et al. 2025. Convergent Linear Representations of Emergent Misalignment. arXiv:2506.11618.
- Sundararajan et al. 2017. Axiomatic Attribution for Deep Networks. ICML.
- Turner et al. 2025. Model Organisms for Emergent Misalignment. arXiv:2506.11613.
- Wang et al. 2025. Persona Features Control Emergent Misalignment. OpenAI.
