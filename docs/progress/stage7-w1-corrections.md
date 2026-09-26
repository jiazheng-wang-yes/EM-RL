# Stage 7 W1: Stage 6A corrections and score breakdowns

**Date:** 2026-09-25. **Compute:** CPU only; no model was rerun. Every number below is recomputed from saved per-prompt rows or from the saved checkpoints.
**Plan:** `docs/plans/persona-control-stage7-review-fixes-2026-09-25.md`, workstream W1.

## 1. Summary

- **Stage 6A overstated activation necessity by .11 to .26 of the direct effect.** Necessity was measured against the unheld graft instead of the held graft. After the fix:
  - The random-direction controls fall from .11–.25 to .00–.01.
  - The necessity of the selected Qwen2.5-7B basis (layer 16, 32 directions) falls from .789 to .548.
  - No layer reaches 70% necessity in any model.
  - The best basis under the Stage 6A rule moves from layer 16 to layer 18 for Qwen2.5-7B, and from layer 12 to layer 23 for Llama-3.1-8B (a near tie). It stays at layer 20 for Qwen3-1.7B.
  - Sufficiency values are unchanged.
- **Weight-rank intervals for Llama and Qwen3 are now valid.** Before the fix, 65 (Llama) and 59 (Qwen3) intervals did not contain their own estimate. After it, none do. The point estimates are unchanged. Llama F_direct(4) = .877 [.839, .921]; Qwen3 F_direct(16) = .788 [.773, .804].
- **The two answers contribute differently.**
  - The graft reproduces all of the misaligned-answer part of the endpoint gap, but only about half of the aligned-answer part. T/Δ is 1.00 [.96, 1.05] for the misaligned answer and .49 [.46, .52] for the aligned answer (Qwen2.5-7B). Llama and Qwen3 show the same pattern.
  - The persona hold removes the same share from both parts: .21 and .22 for Qwen2.5-7B.
- **Update size does not locate the effect.**
  - Qwen2.5-7B: layers 20–27 hold 32.1% of ‖E−C‖² but reproduce only 7.6% [5.9, 9.2] of the gap. Layers 8–19 hold 45.9% and reproduce 80.5% [78.4, 82.9].
  - Qwen3-1.7B: 48% of its whole ‖E−C‖² sits in 256 input columns of layer 2's down_proj matrix. The weights there moved farther than AdamW at the recorded learning rate should move them.
- **Carrier-layer comparator (legacy rendering).** Holding the top-k data-driven directions on top of the persona hold removes an extra 15–45% of the graft effect (Qwen2.5-7B, layer 20, k = 1 to 32). The persona hold alone removes 19–20%. Across the three models, a 4-direction data-driven basis removes 1.4 to 3 times as much as the 4 persona coordinates.

## 2. Setup and terms

**Models.** All are full fine-tunes of one base model, with the same seed and row order:
- **C** is the benign fine-tune (good medical advice).
- **E** is the harmful fine-tune (bad medical advice).
- **G** is C with E's weights copied into a block of layers. This block is the **region**, and the copy is the **graft**.
- **R** is the reverse graft: E with C's weights copied into the region.

| Model | Region used in these files | Carrier layer |
|---|---|---|
| Qwen2.5-7B | 8–19 | 20 |
| Llama-3.1-8B | 5–22 (route runs); 9–22 (Stage 6A activation and weight runs) | 23 |
| Qwen3-1.7B | 8–19 | 20 |

**Score.**
- For one question, lp(mis) and lp(align) are the mean per-token log-probabilities the model assigns to a fixed misaligned answer and a fixed aligned answer. The file is `experiments/persona_control/data/stage2c_paired_completions_120.json`, with 120 questions.
- **S** = lp(mis) − lp(align), averaged over questions. Higher S means the misaligned answer is relatively more likely.

**Persona hold.** H_C(X) runs model X, but at the carrier layer it overwrites four hidden-state coordinates with the values C has on the same text, at answer positions only. The four coordinates are the nested persona carrier: evil, sycophancy and two more directions from a persona contrast subspace. The suffix `|clamp_nested` or `|Gclamp` marks a held condition.

**Contrasts.**

| Symbol | Definition | Meaning |
|---|---|---|
| Δ | S(E) − S(C) | The endpoint gap. |
| T (TE) | S(G) − S(C) | The gap the graft reproduces. |
| D (DE) | S(H_C(G)) − S(H_C(C)) | The part of T that survives the hold. lp(mis) under H_C(C) equals C's to within 1.2e-6 nats. |
| M | T − D | The part of T that the hold removes. |
| Removed share | M/T | The fraction of T that the hold removes. |

**Renderings.**
- **Training rendering** formats text with the tokenizer's chat template, as in training.
- **Legacy rendering** uses a manual prefix `<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n` (or the Llama equivalent) and no system prompt.

**Intervals.** All new intervals come from 2,000 bootstrap draws over prompt clusters, seed 0 (`ClusterBootstrap` in `experiments/persona_control/stage6/common.py`). A prompt cluster is a question together with its `_json`/`_template` paraphrases. Numerators and denominators of ratios are resampled together. The interval of a ratio is the 2.5–97.5 percentile of the per-draw ratio.

## 3. Task 1: Stage 6A activation necessity

### 3.1 What was wrong

Stage 6A (`experiments/persona_control/stage6/stage6a_activation.py`) fits a k-dimensional basis U_k at layer l.
- **Fitting.** The basis is an uncentred principal-component basis of the persona-removed hidden-state difference h_G − h_C. It is fitted on one 60-prompt half and scored on the other. **d2e** means fitted on the discovery half and scored on the evaluation half; **e2d** means the reverse. The halves come from `split_prompt_ids()`, rng seed 20260920.
- **Scoring.** Every intervention goes through `score_with_hooks`, which always applies the persona hold:
  - `suff|l|k` is C plus the U_k-component of the persona-removed difference at layer l.
  - `nec|l|k` is G minus that component.

`summarize_activation()` computed

  necessity = (S(G) − S(nec|l|k)) / DE_full.

The necessity condition carries the persona hold, but G does not. The number therefore also counted the hold's own effect, which adds the same **offset** (S(G) − S(Gclamp)) / DE_full to every value.
- DE_full = S(Gclamp) − S(C) over all 120 prompts, stored in `activation_results.json`: .2234 (Qwen2.5-7B), .1362 (Llama), .1861 (Qwen3).
- The correct reference is the held graft Gclamp = H_C(G):

  necessity = (S(Gclamp) − S(nec|l|k)) / DE_full.

The response-position table (Stage 6A §6) already used `baseline|Gclamp` and is correct.

### 3.2 Code changes

- `stage6a_activation.py`, `summarize_activation()`: the reference is now `{split}|Gclamp`.
- `stage6a_report.py`, `activation_control_summary()`: the same one-line fix, so a later rerun of the report generator does not bring the error back. This file was not rerun, because its `main()` rewrites `docs/progress/stage6a-compression.md`.
- `experiments/persona_control/stage7/w1_stage6a_necessity.py` reruns the fixed function on the saved `activation_rows.parquet` and `activation_control_rows.parquet`. It also:
  - checks that the result matches an independent computation to 1e-9;
  - checks that no sufficiency value changed;
  - adds intervals normalized by each evaluation half's own DE (**DE_half** = S(Gclamp) − S(C) on that half; .240 for d2e and .205 for e2d on Qwen2.5-7B);
  - writes the carrier-layer comparator (Section 7).

### 3.3 Results

The offset that the error added, (S(G) − S(Gclamp))/DE_full:

| Model | d2e | e2d |
|---|---:|---:|
| Qwen2.5-7B | .262 | .220 |
| Llama-3.1-8B | .107 | .113 |
| Qwen3-1.7B | .253 | .221 |

Controls at the basis Stage 6A selected. Values are pooled over the two halves and normalized by DE_full. "random" is the mean of 5 seeds.

| Model (basis) | Control | Sufficiency | Necessity, reported | Necessity, corrected |
|---|---|---:|---:|---:|
| Qwen2.5-7B (L16, k=32) | data-driven basis | .656 | .789 | .548 |
| | random | .003 | .247 | .006 |
| | style | .012 | .245 | .004 |
| | persona | .000 | .241 | .000 |
| | benign-activation PCA | .040 | .281 | .041 |
| Llama-3.1-8B (L12, k=16) | data-driven basis | .547 | .487 | .377 |
| | random | .004 | .112 | .003 |
| | benign-activation PCA | .058 | .176 | .067 |
| Qwen3-1.7B (L20, k=16) | data-driven basis | .512 | .699 | .462 |
| | random | .004 | .239 | .002 |
| | benign-activation PCA | .230 | .447 | .209 |

- Of the 30 per-half random-control intervals (DE_half normalization), 27 contain 0. The other three are Qwen2.5-7B values of .008 to .012.
- The persona control is exactly 0: holding the persona coordinates on top of the persona hold changes nothing.

**Best basis** under the Stage 6A rule (the largest min(pooled sufficiency, pooled necessity) over layer and k):

| Model | Reported best | Corrected best |
|---|---|---|
| Qwen2.5-7B | L16/k32 (suff .656, nec .789) | L18/k32 (suff .648, nec .574) |
| Llama-3.1-8B | L12/k16 (suff .547, nec .487) | L23/k16 (suff .445, nec .381). L12/k16 corrected: min .377, a near tie. |
| Qwen3-1.7B | L20/k16 (suff .512, nec .699) | L20/k16 (suff .512, nec .462) |

**Necessity thresholds.** The table lists the smallest k (of 1, 2, 4, 8, 16, 32 for Qwen2.5-7B; 4, 8, 16 for the others) at which a half reaches 50% necessity (DE_full normalization).
- Stage 6A reported 50% at k = 2 to 4 for Qwen2.5-7B layers 12–20 and 70% at k = 4 to 32.
- After the fix, no model reaches 70% at any layer or k.

| Model | Half | Layer: k at 50% |
|---|---|---|
| Qwen2.5-7B | d2e | L14: 32, L16: 8, L18: 8, L20: 16 (L10, L12: not reached) |
| Qwen2.5-7B | e2d | L18: 32 (all other layers: not reached) |
| Llama-3.1-8B | both | not reached (largest value .422, d2e L12 k16) |
| Qwen3-1.7B | d2e | L14: 16 (others not reached; largest value .500) |

The largest corrected necessity by the DE_half normalization is .582 [.552, .613] (Qwen2.5-7B, d2e, L18, k=32). Full tables with intervals: `activation_rank_corrected.csv` (Section 9).

**Change to Stage 6A's conclusion.** Stage 6A §10 chose "full causal-region protection" partly because the activation basis did not jointly reach .70 at k ≤ 16. That still holds, now with a larger margin: corrected necessity at k ≤ 16 is at most .520 pooled over the two halves (.573 in one half; Qwen2.5-7B L18, k = 16).

## 4. Task 2: weight-rank intervals

**F_direct(r)** = DE_r / DE_full. DE_r is the held direct effect when only the top r singular directions of each E − C matrix are grafted. DE_full is the held direct effect of the full update, on the seven matrices of the Stage 6A region.

The first-pass intervals in `weight_results.json` were built by subtracting marginal percentiles. `experiments/persona_control/stage6/stage6a_fix_weight_intervals.py` now takes `--model` and writes to `eval_runs/persona_control_stage7/w1_corrections/stage6a_weight/<model>/weight_results.json`. It refuses to overwrite the Stage 6A file.
- It recomputes paired intervals from `weight_rows.parquet`.
- For Qwen2.5-7B, whose file was already fixed in place, the output matches the existing file exactly (no value differs by more than 1e-12).

| Model | Full update TE | Full update DE | F_direct(4) | F_direct(8) | F_direct(16) |
|---|---|---|---|---|---|
| Llama-3.1-8B (layers 9–22) | .152 [.135, .168] | .137 [.120, .152] | .877 [.839, .921] | .865 [.832, .902] | .902 [.872, .936] |
| Qwen3-1.7B (layers 8–19) | .231 [.211, .251] | .188 [.170, .206] | .644 [.626, .663] | .741 [.724, .760] | .788 [.773, .804] |

Controls:
- **Random energy-matched**: random orthogonal directions carrying the same energy.
- **Shuffled**: random directions carrying the learned singular values.
- Each control ran at r = 4, 8 and 16 with 5 seeds, 30 conditions per model.

Results:
- Llama: F_direct ranges from −.014 to −.001. Five of 30 intervals exclude 0, all on the negative side, with no bound beyond .023 in absolute value.
- Qwen3: −.011 to .007, one interval excluding 0.
- Interval check (lower ≤ estimate ≤ upper): before the fix, 65 Llama and 59 Qwen3 intervals failed it; after the fix, 0 fail.

## 5. Task 3: the misaligned-answer and aligned-answer parts of S

Every contrast K = Σ_X a_X·S(X) splits exactly into two parts, K = K_mis + K_align:
- **K_mis** = Σ_X a_X·lp_X(mis) is the change in the misaligned answer's log-probability.
- **K_align** = −Σ_X a_X·lp_X(align) is the drop in the aligned answer's log-probability.
- The **misaligned share** is K_mis/K. It is left empty when K's interval contains 0.

**Runs.**
- Qwen2.5-7B and Llama-3.1-8B: ARR Round 1 confirmations (training rendering, seven matrices grafted), `eval_runs/persona_control_arr/round1/20260924T063649Z/route_{qwen,llama}/stage5b/*_confirmation/`.
- Qwen3-1.7B: the Stage 5B run of 2026-09-17 (**legacy rendering, all parameters of layers 8–19 grafted, no R condition**), `eval_runs/persona_control_stage6/stage5b/qwen3_1_7b/`.

The totals and ratios that the runs' own `summary.json` also reports (Δ, T, D, NE_rev, NDE_rev and the four ratios) match it in both the estimate and the interval.

Extra contrasts:
- **E_hold** = S(E) − S(H_C(E)).
- **NE_rev** = S(E) − S(R); **NDE_rev** = S(H_C(E)) − S(H_C(R)).
- **Interaction** = NE_rev − T = S(E) − S(R) − S(G) + S(C). It is 0 when the region's effect is the same whichever model supplies the rest of the network.
- **M_hold(s)** = T − (S(G held with carrier s) − S(H_C(C))), for the single evil direction, evil+sycophancy, and a style direction.

| Model | Contrast | K | K_mis | K_align | Misaligned share |
|---|---|---:|---:|---:|---:|
| Qwen2.5-7B | Δ | .390 [.358, .420] | .239 [.215, .265] | .151 [.136, .167] | .61 [.58, .64] |
| | T | .314 [.291, .337] | .240 [.220, .260] | .074 [.066, .082] | .76 [.74, .79] |
| | D | .247 [.228, .266] | .190 [.173, .206] | .058 [.050, .065] | .77 [.74, .79] |
| | M | .067 [.060, .074] | .051 [.045, .056] | .016 [.015, .018] | .75 [.73, .78] |
| | E_hold | .082 [.074, .090] | .054 [.047, .061] | .028 [.025, .031] | .66 [.63, .70] |
| | NE_rev | .312 [.287, .336] | .171 [.152, .191] | .141 [.130, .152] | .55 [.52, .58] |
| | NDE_rev | .241 [.221, .261] | .126 [.110, .143] | .115 [.105, .125] | .52 [.49, .56] |
| | Interaction | −.002 [−.011, .006] | −.069 [−.077, −.060] | .067 [.061, .073] | n/a |
| | M_hold(evil) | .060 [.054, .065] | .048 [.043, .053] | .011 [.009, .013] | .81 [.79, .84] |
| | M_hold(style) | .004 [.002, .006] | .001 [.000, .003] | .003 [.002, .004] | .28 [−.12, .46] |
| Llama-3.1-8B | Δ | .295 [.269, .321] | .133 [.110, .155] | .162 [.147, .179] | .45 [.40, .50] |
| | T | .229 [.208, .250] | .138 [.119, .158] | .091 [.080, .102] | .60 [.55, .65] |
| | D | .205 [.185, .224] | .122 [.104, .141] | .083 [.073, .094] | .60 [.54, .64] |
| | M | .024 [.021, .027] | .016 [.013, .019] | .008 [.007, .009] | .66 [.61, .71] |
| | E_hold | .031 [.027, .034] | .018 [.015, .020] | .013 [.011, .015] | .58 [.53, .62] |
| | NE_rev | .236 [.215, .257] | .089 [.070, .108] | .147 [.134, .161] | .38 [.31, .43] |
| | Interaction | .007 [−.002, .015] | −.049 [−.057, −.042] | .056 [.050, .062] | n/a |
| | M_hold(evil) | .022 [.019, .024] | .016 [.014, .018] | .005 [.004, .007] | .75 [.70, .79] |
| Qwen3-1.7B (legacy) | Δ | .332 [.303, .360] | .189 [.166, .210] | .143 [.129, .158] | .57 [.53, .60] |
| | T | .231 [.210, .250] | .170 [.152, .187] | .061 [.052, .068] | .74 [.71, .77] |
| | D | .186 [.168, .204] | .138 [.122, .153] | .048 [.041, .056] | .74 [.71, .78] |
| | M | .044 [.040, .049] | .032 [.028, .036] | .012 [.010, .014] | .72 [.68, .76] |
| | E_hold | .060 [.054, .065] | .035 [.032, .039] | .024 [.021, .027] | .59 [.55, .63] |

Ratios. Each column divides the named part of the numerator by the same part of the denominator.

| Model | Ratio | Total S | Misaligned part | Aligned part |
|---|---|---:|---:|---:|
| Qwen2.5-7B | T/Δ | .81 [.78, .83] | 1.00 [.96, 1.05] | .49 [.46, .52] |
| | M/T (removed share) | .21 [.20, .23] | .21 [.19, .23] | .22 [.20, .25] |
| | E_hold/Δ | .21 [.19, .23] | .23 [.21, .25] | .18 [.17, .20] |
| | (NE_rev − NDE_rev)/NE_rev | .23 [.21, .24] | .26 [.24, .29] | .18 [.17, .20] |
| Llama-3.1-8B | T/Δ | .78 [.75, .80] | 1.04 [.98, 1.10] | .56 [.53, .59] |
| | M/T | .10 [.09, .12] | .11 [.10, .13] | .09 [.08, .10] |
| | E_hold/Δ | .10 [.09, .11] | .13 [.11, .16] | .08 [.07, .09] |
| Qwen3-1.7B (legacy) | T/Δ | .69 [.67, .71] | .90 [.87, .94] | .42 [.39, .45] |
| | M/T | .19 [.18, .21] | .19 [.17, .21] | .20 [.17, .24] |

What the split shows:
- **T/Δ ≈ .8 averages two different numbers.**
  - Grafting the region into C raises the misaligned answer as much as the full harmful fine-tune does: 1.00 and 1.04.
  - The same graft lowers the aligned answer only about half as much: .49, .56 and .42.
- **The region's effect on each answer depends on the rest of the network, even though its effect on S does not.**
  - The interaction is 0 for S: −.002 [−.011, .006] for Qwen2.5-7B.
  - Its parts are not 0. With C around it, the region raises lp(mis) .069 more than with E around it. With E around it, it lowers lp(align) .067 more than with C around it.
  - Additivity of the region therefore holds for the score S but not for each answer separately.
- **The persona hold is not specific to one answer.** It removes the same share of both parts (M/T columns above). The single-direction evil hold acts more on the misaligned answer than the four-coordinate hold does: misaligned share .81, .75 and .71 against .75, .66 and .72 (Qwen2.5-7B, Llama, Qwen3). The style hold removes .004 to .007 of S.

## 6. Task 4: update energy versus graft effect

### 6.1 Definitions

- **Energy share of layer l**: ec_share(l) = ‖E_l − C_l‖² / Σ_l′ ‖E_l′ − C_l′‖². Here E_l stacks the seven matrices (q, k, v, o, gate, up, down) of layer l, and ‖·‖ is the Frobenius norm, the square root of the sum of squared entries. eb_share and cb_share do the same for E − base and C − base.
- **Parameter share**: 1/(number of layers), since the layers are equal in size.
- **Forward effect of a block B**: (S(C with E's block B) − S(C)) / Δ. **Reverse effect**: (S(E) − S(E with C's block B)) / Δ.
- **Effect per energy**: forward effect ÷ ec_share. A value of 1 means the block's effect is in proportion to its share of the update.

Scripts:
- `experiments/persona_control/stage7/w1_update_energy.py` reads each matrix of base, C and E and computes the per-matrix sums, in float32 from the stored bf16 values.
- `w1_energy_vs_effect.py` builds the table and the figure.

### 6.2 Results

| Model | Layers | Source (rendering, what was grafted) | Params % | ‖E−C‖² % | ‖E−base‖² % | ‖C−base‖² % | Forward, % of Δ | Reverse, % of Δ | Effect per energy |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen2.5-7B | 0–3 | Stage 3 block A1 (legacy, all) | 14.3 | 11.3 | 12.4 | 13.3 | 5.3 [4.4, 6.4] | 3.7 [3.0, 4.4] | 0.47 |
| | 4–7 | Stage 3 A2 | 14.3 | 10.7 | 11.7 | 12.8 | 9.9 [8.8, 10.9] | 5.5 [4.5, 6.4] | 0.92 |
| | 8–11 | Stage 3 A3 | 14.3 | 13.4 | 14.1 | 14.4 | 30.4 [28.8, 32.2] | 20.4 [19.1, 21.8] | 2.28 |
| | 12–15 | Stage 3 A4 | 14.3 | 15.5 | 15.3 | 15.0 | 40.6 [39.0, 42.3] | 29.0 [27.5, 30.7] | 2.62 |
| | 16–19 | Stage 3 A5 | 14.3 | 17.0 | 16.4 | 15.5 | 25.6 [24.1, 27.0] | 16.8 [15.1, 18.3] | 1.51 |
| | 20–23 | Stage 3 A6 | 14.3 | 16.7 | 15.6 | 15.0 | 4.5 [3.7, 5.3] | 3.5 [2.7, 4.3] | 0.27 |
| | 24–27 | Stage 3 A7 | 14.3 | 15.4 | 14.6 | 14.0 | 2.5 [1.3, 3.7] | 2.7 [1.6, 3.9] | 0.16 |
| | 20–27 | Stage 3 suffix range | 28.6 | 32.1 | 30.2 | 29.1 | 7.6 [5.9, 9.2] | 6.2 [4.7, 7.6] | 0.24 |
| | 8–19 | route region (training, 7 matrices) | 42.9 | 45.9 | 45.8 | 44.8 | 80.5 [78.4, 82.9] | 80.0 [77.9, 82.0] | 1.76 |
| | 12–15 | route anchor (training, 7 matrices) | 14.3 | 15.5 | 15.3 | 15.0 | 36.8 [34.9, 38.7] | not run | 2.37 |
| Llama-3.1-8B | 5–22 | Stage 6A coverage (legacy, all) | 56.2 | 55.7 | 55.6 | 55.6 | 77.9 [75.0, 80.5] | 83.4 [80.5, 86.2] | 1.40 |
| | 7–22 | Stage 6A coverage | 50.0 | 49.7 | 49.3 | 49.3 | 68.0 [65.0, 71.0] | 69.5 [66.0, 72.7] | 1.37 |
| | 9–22 | Stage 6A coverage | 43.8 | 43.6 | 43.0 | 42.9 | 59.0 [55.9, 62.3] | 53.1 [49.6, 56.5] | 1.35 |
| | 5–8 | Stage 6A coverage | 12.5 | 12.1 | 12.6 | 12.7 | 27.0 [24.2, 29.8] | 26.4 [23.3, 29.3] | 2.23 |
| | 5–22 | route region (training, 7 matrices) | 56.2 | 55.7 | 55.6 | 55.6 | 77.7 [75.3, 80.1] | 79.9 [76.8, 82.9] | 1.39 |
| | 14–17 | route anchor (training, 7 matrices) | 12.5 | 12.7 | 12.4 | 12.3 | 10.0 [8.7, 11.2] | not run | 0.79 |
| Qwen3-1.7B | 8–19 | route region (legacy, all) | 42.9 | 22.0 | 25.8 | 24.9 | 69.4 [67.5, 71.3] | not run | 3.15 |
| | 12–15 | route anchor (legacy, all) | 14.3 | 8.1 | 9.2 | 8.5 | 36.3 [34.4, 38.0] | not run | 4.48 |

Interval sources:
- Stage 3 rows are copied from `experiments/persona_control/results_stage3/weight_grafting_results.json`. That run used 1,000 bootstrap draws over individual prompts, seed 42, and its own Δ = .327.
- Llama coverage intervals are recomputed here from `llama_coverage_rows.parquet`, with Δ = .259 (legacy). The recomputed estimates equal `llama_coverage.json`.
- No per-block scan exists for Qwen3.

![Update energy versus graft effect](../../figures/persona_control/stage7/w1_corrections/energy_vs_effect.png)

*Figure: the blue bars show each layer's share of ‖E−C‖²; the grey line shows its share of parameters. Each orange line is a block's forward effect divided by its number of layers, with its 95% interval as a band. File: `figures/persona_control/stage7/w1_corrections/energy_vs_effect.png`.*

What the table shows:
- **Qwen2.5-7B.**
  - Energy per layer is nearly flat: 2.2% (layer 1) to 4.4% (layer 19), against 3.6% of parameters.
  - Layers 20–27 hold a third of the update energy but carry under a tenth of the effect.
  - The three middle blocks carry 1.5 to 2.6 times their energy share.
  - The large update energy in layers 20–27 did not produce the effect, so the region is not where it is simply because the update is largest there.
- **Llama-3.1-8B.** The region's energy share equals its parameter share (55.7% vs 56.2%), but it carries 78% of the gap. Block 5–8 carries 2.2 times its energy share. The anchor block 14–17 carries 0.8 times its share.
- **Qwen3-1.7B.** The region holds only 22.0% of the energy, because two layers outside it dominate (Section 6.3). Leaving out layers 2 and 27, the region holds about 52% of the remaining energy.

### 6.3 The outlier layers

`experiments/persona_control/stage7/w1_energy_outliers.py` writes `update_energy/down_proj_outliers.csv`.

**Qwen3-1.7B layer 2** holds 49.7% of ‖E−C‖². In the down_proj matrix of that layer:
- 99.3% of ‖E−C‖² sits in input columns 1792–2047, one block of 256 columns out of 24.
- 97.7% of it is in entries that moved more than 2.33e-3. That is the sum of the per-step learning rates over training (lr 2.5e-5, 184 steps, 18 warm-up steps, cosine decay; `train_qwen3_pair.py`).
- The largest move is .044 (E−C) and .047 (C−base).
- For comparison:
  - Kingma and Ba's approximate bound for Adam with β = (.9, .999) is about 3.2 times the learning rate per step, which is 7.4e-3 in total.
  - Rounding to bf16 can at most double a step, which gives 1.5e-2.
- C − base and E − base show the same block (99.0% each). The pattern therefore comes from the shared training recipe, not from the harmful data.
- Layer 27 shows a weaker version: 59% of its down_proj ‖E−C‖² is in columns 768–1023.
- The ordinary layer 14 spreads its energy evenly: 4.4% per block against 4.2% for a uniform spread.

**Llama-3.1-8B layer 1** holds 7.6% of ‖E−C‖². In its down_proj, 85% sits in columns 2304–2559 (one block of 56), again in both C and E. Qwen2.5-7B shows no such block in the layers checked (0 and 14): at most 3.9%, against 1.4% for a uniform spread.

The cause is not identified. The recipe (8-bit AdamW updating bf16 weights with no fp32 copy) is the training-precision issue that plan §2 puts out of scope. None of these layers lies inside a graft region, so no graft copies them; the endpoint gap Δ still includes whatever they do. They do make whole-model energy shares a poor reference for Qwen3.

## 7. Task 5: carrier-layer comparator (legacy rendering)

At the carrier layer, the Stage 6A necessity condition holds both the persona coordinates and the data-driven basis U_k. The extra removal by U_k on top of the persona hold is:
- **extra/DE** = (S(Gclamp) − S(nec|carrier|k)) / DE_half;
- **extra/TE** = the same numerator ÷ TE_half, where TE_half = S(G) − S(C) on the evaluation half.

The persona hold alone removes (TE_half − DE_half)/TE_half. "Both" = (TE_half − (S(nec) − S(C)))/TE_half is the share of TE that the persona hold and U_k remove together. U_k is fitted on the other half.

| Model, layer | Half | k | Persona hold alone, share of TE | Extra, share of DE | Extra, share of TE | Both, share of TE |
|---|---|---:|---:|---:|---:|---:|
| Qwen2.5-7B, L20 | d2e | 1 | .196 [.176, .216] | .189 [.166, .214] | .152 [.134, .171] | .348 [.318, .377] |
| | | 2 | | .283 [.253, .314] | .227 [.203, .251] | .423 [.389, .454] |
| | | 4 | | .380 [.343, .421] | .306 [.277, .339] | .502 [.466, .538] |
| | | 8 | | .431 [.397, .469] | .347 [.320, .376] | .543 [.510, .575] |
| | | 16 | | .477 [.442, .512] | .383 [.354, .413] | .579 [.548, .610] |
| | | 32 | | .555 [.523, .587] | .446 [.421, .472] | .642 [.614, .670] |
| | e2d | 1 | .193 [.173, .213] | .180 [.150, .212] | .145 [.122, .170] | .338 [.305, .371] |
| | | 2 | | .293 [.260, .327] | .236 [.211, .263] | .430 [.395, .464] |
| | | 4 | | .380 [.340, .421] | .306 [.277, .336] | .500 [.461, .538] |
| | | 8 | | .410 [.371, .450] | .331 [.300, .362] | .524 [.488, .560] |
| | | 16 | | .460 [.420, .502] | .371 [.338, .406] | .564 [.529, .601] |
| | | 32 | | .530 [.488, .572] | .428 [.395, .462] | .621 [.584, .659] |
| Llama-3.1-8B, L23 | d2e | 4 | .088 [.074, .105] | .292 [.244, .351] | .267 [.221, .320] | .355 [.311, .408] |
| | | 8 | | .306 [.255, .365] | .279 [.233, .333] | .367 [.321, .421] |
| | | 16 | | .376 [.324, .435] | .342 [.295, .396] | .431 [.385, .484] |
| | e2d | 4 | .108 [.084, .136] | .344 [.288, .411] | .307 [.261, .362] | .415 [.353, .486] |
| | | 8 | | .324 [.273, .383] | .289 [.246, .338] | .397 [.342, .459] |
| | | 16 | | .376 [.325, .434] | .335 [.292, .383] | .444 [.391, .502] |
| Qwen3-1.7B, L20 | d2e | 4 | .189 [.170, .211] | .328 [.286, .375] | .266 [.231, .303] | .455 [.418, .496] |
| | | 8 | | .369 [.327, .412] | .300 [.264, .335] | .488 [.452, .525] |
| | | 16 | | .442 [.395, .491] | .359 [.319, .398] | .548 [.509, .590] |
| | e2d | 4 | .195 [.172, .222] | .335 [.291, .376] | .269 [.238, .299] | .465 [.420, .508] |
| | | 8 | | .397 [.351, .443] | .319 [.284, .352] | .514 [.470, .560] |
| | | 16 | | .484 [.442, .525] | .390 [.355, .421] | .585 [.550, .623] |

- At k = 4, U_k removes 1.6 times (Qwen2.5-7B), about 3 times (Llama: 2.8 to 3.0) and 1.4 times (Qwen3) as much of TE as the four persona coordinates do.
- Even at k = 32, persona hold plus U_k leaves more than a third of Qwen2.5-7B's graft effect in place.
- The comparator is a reference, not a rival: U_k is fitted to the persona-removed graft difference itself (on the other half).
- These are legacy-rendering numbers with all parameters of the graft layers grafted, and Llama's region here is 9–22. W2 produces the training-rendering version.

## 8. Deviations and open problems

**Deviations.**
- `stage6a_report.py` is not in W1's owned files. It received the same one-line reference fix but was not rerun.
- DE_half-normalized ratios with paired intervals are added next to the original DE_full normalization. DE_full has no interval of its own, and the halves differ (.240 vs .205).
- Task 3 adds the interaction and named-hold contrasts.
- Task 4 adds the training-rendering route region and anchor rows, and an outlier follow-up (`w1_energy_outliers.py`).
- `stage6a_fix_weight_intervals.py` no longer writes in place. Its default output is the Stage 7 folder.

**Open problems.**
- Controls exist only at the bases Stage 6A selected with the wrong necessity. The corrected best bases (Qwen2.5-7B L18/k32, Llama L23/k16) have no random, style or benign controls, and rerunning them needs a GPU.
- The Qwen3 benign-activation PCA control (the top-k principal components of C's own hidden states) reaches .23 sufficiency and .21 necessity, a fifth of the effect. The cause is not known.
- All Stage 6A activation and weight numbers are legacy rendering, with Llama region 9–22.
- The outlier blocks in Qwen3 layers 2 and 27 and Llama layer 1 moved farther than the optimizer settings should allow. The cause is not identified.
- The region's per-answer effects depend on the surrounding model (interaction parts of about ±.05 to .07 nats), though its effect on S does not.

## 9. Files and reproduction

Run each script with `OMP_NUM_THREADS=8 python <script>` from the repository root. No arguments are needed unless shown.

| Script | Output |
|---|---|
| `experiments/persona_control/stage7/w1_stage6a_necessity.py` | `eval_runs/persona_control_stage7/w1_corrections/stage6a_activation/<model>/`: `activation_rank_corrected.csv`, `activation_rank_fixed_code.csv`, `activation_thresholds_{suff,nec_corrected}.csv`, `activation_controls_corrected.csv`, `carrier_layer_comparator.csv`, `summary.json`; `activation_rank_corrected_all_models.csv` |
| `experiments/persona_control/stage6/stage6a_fix_weight_intervals.py --model <model>` | `.../w1_corrections/stage6a_weight/<model>/weight_results.json` (with `interval_audit`) |
| `experiments/persona_control/stage7/w1_score_parts.py` | `.../w1_corrections/score_parts/`: `score_parts.csv`, `score_part_ratios.csv`, `condition_levels.csv`, `summary.json` |
| `experiments/persona_control/stage7/w1_update_energy.py` | `.../w1_corrections/update_energy/<model>_per_matrix.csv` |
| `experiments/persona_control/stage7/w1_energy_outliers.py` | `.../w1_corrections/update_energy/down_proj_outliers.csv` |
| `experiments/persona_control/stage7/w1_energy_vs_effect.py` | `.../w1_corrections/energy_vs_effect/{per_layer_energy,block_energy_vs_effect}.csv`, `figures/persona_control/stage7/w1_corrections/energy_vs_effect.png` |

The Stage 6A inputs in `eval_runs/persona_control_stage6/stage6a/` are unchanged. Model paths are `MODEL_SPECS` in `experiments/persona_control/stage6/common.py`.
