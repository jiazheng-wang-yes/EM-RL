# Stage 5B: Activation Route of the Middle-Layer Parametric Effect (three models)

**Date:** 2026-09-17
**Status:** complete (report point A of the Stage 6 plan). Mitigation work has not started.
**Lab record:** [persona-control-stage6-record.md](../plans/persona-control-stage6-record.md)
**Models:** Qwen2.5-7B-Instruct, Llama-3.1-8B-Instruct, Qwen3-1.7B (run in parallel at the user's request; the written plan had full Qwen plus a minimal Llama check)

---

## 1. Result

In all three models the effect of the middle-layer weight graft that survives the persona clamp
is carried by an activation route that

1. builds up gradually across the grafted layers (no single layer carries it; onset at layer 9–10),
2. runs through the computation at **response tokens** (resetting prompt positions removes at most 2–6%),
3. lies mostly **outside the nested persona carrier**: at the carrier layer the persona-parallel part
   of the graft-induced activation change accounts for 0.11–0.25 of the total effect and the
   orthogonal part for 0.75–0.90, while random and style subspaces of the same rank account for ≈0 and 0.01–0.04,
4. is not an artifact of damaged hybrids: none of 616 strong patches changes neutral-text
   likelihood, entropy, or top-1 agreement beyond the fixed flags.

Whether the route is carried mainly by MLP or attention **depends on the model**. Weight-side and
activation-side splits agree for the two Qwen models: Qwen2.5-7B is MLP-dominated (the plan's
prediction and the Stage 3 result), Qwen3-1.7B leans on attention. Llama-3.1-8B is mixed: MLP
weights alone carry more of the effect than attention weights alone, but at the layers that write
the effect the attention outputs carry more of it.

Plan outcome per model: **Result A** for Qwen2.5-7B; **Result B** (persona-orthogonal, attention
leaning) for Qwen3-1.7B; persona-orthogonal with a mixed component split for Llama-3.1-8B.
Results C (convergence into the persona subspace) and D (unstable or destructive hybrids) are not
observed in any model. Stage 5B does not invalidate the direct-route interpretation.

A second robust observation, not asked for in the plan: the mediated fraction is about the same
whichever parameter group carries the effect (MLP-only, attention-only, anchor, full graft;
0.18–0.21 in both Qwen models; 0.09–0.11 for the Llama middle-layer grafts, whose small anchor
grafts give wider intervals).

## 2. Setup

| | Qwen2.5-7B | Llama-3.1-8B | Qwen3-1.7B |
| :--- | :---: | :---: | :---: |
| C / E checkpoints | Stage 2 | Stage 4 | retrained with the Stage 3 recipe (job 1791456) |
| Middle graft G | 8:19 | 9:22 | 8:19 |
| Anchor G* | 12:15 | 14:17 | 12:15 |
| Carrier layer (nested k=4) | 20 (frozen Stage 5A carrier) | 23 (new, same recipe) | 20 (new, same recipe) |
| Scan layers | 8..27 | 9..31 | 8..27 |
| Conditions run | 456 | 492 | 456 |

Layer ranges map the Qwen2.5-7B ranges by relative depth; the carrier layer sits after the last
grafted layer as in Stages 4/5A. The new carriers pass a steering check: adding the raw evil vector
at the carrier layer raises S monotonically on the base model (Llama −1.18 → −0.77, Qwen3
−1.94 → −1.18 for coefficients −1 → +1). Qwen3-1.7B often refused the "evil assistant"
instruction during extraction, so its evil vector is built partly from non-evil completions.

Frozen assay: N=120 paired completions, strict N=50 subset (identical members of the 120), 60
AlpacaEval neutral texts, 2000 bootstrap resamples over 108 prompt clusters. All clamp/delta
arithmetic runs in float64 (see the lab record: float32 round-off moved per-example S by up to
0.006). Every identity check passed at ≤ 1e-6 in all three models.

Source problem found while preparing the run: the Llama "ΔS = 0.2941, graft TE = 0.2412" in the
Stage 5A report has no result file behind it. The recomputed Llama ΔS_EM is 0.259, matching the
Stage 3 replication (0.2589).

## 3. Reproduced factorial baseline

| | Qwen2.5-7B | Llama-3.1-8B | Qwen3-1.7B |
| :--- | :---: | :---: | :---: |
| ΔS_EM = S(E) − S(C) | 0.328 [0.303, 0.354] | 0.259 [0.232, 0.283] | 0.332 [0.303, 0.360] |
| TE = S(G) − S(C) | 0.277 [0.256, 0.298] | 0.153 [0.135, 0.168] | 0.231 [0.210, 0.250] |
| TE / ΔS_EM | 0.845 | 0.590 | 0.694 |
| DE (nested clamp) | 0.223 [0.206, 0.240] | 0.136 [0.121, 0.151] | 0.186 [0.168, 0.204] |
| **MF** | **0.195 [0.181, 0.208]** | **0.107 [0.093, 0.122]** | **0.193 [0.178, 0.211]** |
| MF, anchor G* | 0.206 | 0.168 (TE* = 0.029) | 0.177 |
| Clamp repair of full E | 0.191 | 0.097 | 0.180 |
| Clamp MF: evil / evil+syc / style / random-4 | 0.172 / 0.191 / 0.010 / −0.002 | 0.089 / 0.091 / 0.023 / 0.004 | 0.173 / 0.185 / 0.031 / 0.004 |

Audit rule: no model triggered it. Qwen2.5-7B TE is 9.7% above the Stage 5A endpoint (0.2528)
and within 0.3% of Stage 4 (0.2781); the evil, evil+sycophancy and style clamp fractions match
Stage 4 to within 0.3 points, and MF (0.195) is 1.7 points below the Stage 5A endpoint (0.212). The Qwen3-1.7B retraining reproduces the Stage 3 control score (−1.3226 vs −1.3203) and gap
(0.332 vs 0.328).

The Llama middle graft transfers only 59% of ΔS_EM, and its relative-depth anchor (14:17) carries
only 11%. Llama's effect is written at layers 10–13 (section 4), so part of the remaining 41% may
sit in layers below 9; this was not scanned (the plan excludes repeating the localization on Llama).

## 4. Layerwise residual necessity R_l and onset

R_l = [S(G, clamp) − S(G, clamp, h_l ← h_l^C)] / DE, all positions. Full tables:
`R_layers.csv` per model.

| Layer (Qwen / Llama) | Qwen2.5-7B | Llama-3.1-8B | Qwen3-1.7B |
| :--- | :---: | :---: | :---: |
| 8 / 9 | 0.037 | 0.082 | 0.056 |
| 9 / 10 | 0.096 | **0.207** [0.178, 0.235] | **0.119** [0.104, 0.134] |
| 10 / 11 | **0.167** [0.153, 0.182] | 0.386 | 0.203 |
| 12 / 13 | 0.400 | 0.728 | 0.445 |
| 14 / 15 | 0.607 | 0.845 | 0.706 |
| 16 / 17 | 0.812 | 0.918 | 0.918 |
| 18 / 19 | 0.964 | 0.965 | 0.980 |
| last grafted layer and after | 1.000 (identity) | 1.000 (identity) | 1.000 (identity) |
| **Onset layer** (CI low > 0.10, next layer CI low > 0) | **10** | **10** | **9** (strict subset: 10) |
| Three largest per-layer increases | 11, 12, 14 | 10, 11, 12 | 13, 14, 15 |

* Qwen2.5-7B adds about 0.10 of the direct effect per layer from 11 to 17; Qwen3-1.7B adds
  0.12–0.14 per layer from 11 to 15 and less afterwards. Llama adds 65% of the direct effect at
  layers 10–13 and the rest slowly through layer 21.
* Response vs prompt positions: resetting only response positions reproduces R_l almost exactly;
  resetting only prompt positions removes at most 0.044 (Qwen2.5-7B), 0.062 (Llama), 0.024 (Qwen3).
* The unclamped scan (normalized by TE) matches the clamped scan (normalized by DE) within 0.017
  at every layer: the persona clamp does not change the shape of the route.
* R_l is not a semantic feature. It says where in depth the state difference that produces the
  direct effect has been written.

## 5. Attention versus MLP

### 5.1 Activation side (output patches on clamped G, fraction of DE; `components.csv`)

| Mean over the three largest-increase layers | Qwen2.5-7B | Llama-3.1-8B | Qwen3-1.7B |
| :--- | :---: | :---: | :---: |
| R_MLP | 0.100 [0.088, 0.111] | 0.053 [0.034, 0.070] | 0.093 [0.080, 0.107] |
| R_attention | 0.043 [0.035, 0.051] | 0.103 [0.085, 0.121] | 0.171 [0.158, 0.183] |
| R_MLP − R_attention | **+0.057** [0.048, 0.065] | **−0.050** [−0.068, −0.033] | **−0.077** [−0.094, −0.062] |
| Strict N=50 | +0.054 [0.039, 0.068] | −0.047 [−0.076, −0.020] | −0.065 [−0.091, −0.038] |

After the last grafted layer, the MLP outputs of control-weight layers carry most of the propagated
difference in all three models. A few MLPs work against the effect (patching them to control
increases it): Llama layer 13, Qwen3 layers 26–27.

### 5.2 Weight side (graft one parameter group of the middle layers; `weight_components.json`)

Added after the first Qwen3 component result, as a control needed to compare with Stage 3.

| | Qwen2.5-7B | Llama-3.1-8B | Qwen3-1.7B |
| :--- | :---: | :---: | :---: |
| MLP-only graft TE (share of full TE) | 0.210 (76%) | 0.104 (68%) | 0.108 (47%) |
| Attention-only graft TE | 0.093 (34%) | 0.081 (53%) | 0.134 (58%) |
| Norm-only graft TE | −0.001 | 0.000 | 0.000 |
| Sum of parts / full | 1.09 | 1.21 | 1.05 |
| MF: MLP-only / attention-only | 0.199 / 0.197 | 0.108 / 0.093 | 0.200 / 0.178 |
| Anchor, MLP-only / attention-only TE | 0.093 / 0.043 | 0.016 / 0.010 | 0.051 / 0.076 |
| Top-3 R_MLP − R_attn on MLP-only graft | +0.119 | +0.060 | +0.032 |
| Top-3 R_MLP − R_attn on attention-only graft | −0.093 | −0.248 | −0.163 |

The Qwen2.5-7B anchor split reproduces Stage 3 (0.092 / 0.042). Activation patches follow the
weights: with only MLP weights grafted, MLP outputs carry the effect; with only attention weights
grafted, attention outputs do. In Llama the two groups overlap (parts sum to 121% of the whole),
and in the full graft the attention outputs at layers 10–12 carry more than the MLP outputs.

## 6. Persona-parallel versus persona-orthogonal decomposition

dh = h^G − h^C; P_par = UU^T; effects normalized by TE; unclamped hosts; all positions.

### 6.1 At the carrier layer

| | Qwen2.5-7B (L20) | Llama-3.1-8B (L23) | Qwen3-1.7B (L20) |
| :--- | :---: | :---: | :---: |
| Parallel sufficiency  S(C + P_par dh) | 0.248 [0.231, 0.266] | 0.115 [0.099, 0.135] | 0.205 [0.191, 0.222] |
| Orthogonal sufficiency | 0.808 | 0.895 | 0.808 |
| Parallel necessity  S(G − P_par dh) | 0.192 [0.177, 0.207] | 0.105 [0.091, 0.120] | 0.192 [0.177, 0.208] |
| Orthogonal necessity | 0.752 | 0.885 | 0.795 |
| Share of ‖dh‖² inside the carrier | 4.9% | 1.9% | 5.8% |
| Effect share / energy share | 4.8× | 5.9× | 3.5× |

Orthogonal exceeds parallel for sufficiency and necessity in every model (all intervals exclude
zero), on the full set and the strict subset. At layers after the grafted range, parallel
sufficiency equals orthogonal necessity by construction (identity check passed).

### 6.2 By depth (nested carrier; `fig5d`)

Parallel share of sufficiency, Suff_par / (Suff_par + Suff_perp):

| Layer | 10 | 12 | 14 | 16 | 18 | carrier | last layer |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Qwen2.5-7B | 0.02 | 0.04 | 0.06 | 0.13 | 0.22 | 0.24 | 0.03 |
| Llama-3.1-8B | 0.04 | 0.13 | 0.23 | 0.19 | 0.17 | 0.11 | 0.03 |
| Qwen3-1.7B | 0.12 | 0.13 | 0.18 | 0.19 | 0.19 | 0.20 | 0.10 |

The persona-parallel part is a minority at every layer of every model (maximum 0.24). In
Qwen2.5-7B it appears late (layers 15–19) after most of the orthogonal part is already written;
in Llama it peaks at layer 14. It declines after the carrier layer in all three models, which is
expected for a subspace defined at that layer.

## 7. Controls (carrier layer; same decomposition; `fig5e`)

| Subspace (rank) | Qwen2.5-7B par Suff / Nec | Llama par Suff / Nec | Qwen3 par Suff / Nec |
| :--- | :---: | :---: | :---: |
| Nested carrier (4) | 0.248 / 0.192 | 0.115 / 0.105 | 0.205 / 0.192 |
| Evil + sycophancy (2) | 0.243 / 0.189 | 0.098 / 0.094 | 0.202 / 0.186 |
| Evil axis (1) | 0.218 / 0.171 | 0.091 / 0.093 | 0.188 / 0.171 |
| Style (4) | 0.016 / 0.013 | 0.018 / 0.025 | 0.033 / 0.035 |
| Random (4), mean of 3 seeds | −0.000 / −0.001 | −0.003 / 0.003 | 0.002 / 0.003 |
| Random (2) | 0.001 / −0.001 | −0.005 / 0.003 | −0.000 / 0.000 |
| Random (1) | 0.000 / −0.002 | −0.003 / 0.003 | −0.000 / −0.002 |

Most of the nested carrier's share is already in the evil axis. At the three largest-increase
layers inside the graft, the persona-related subspaces (nested, evil, evil+sycophancy) capture
0.005–0.047 of TE in Qwen2.5-7B, 0.007–0.093 in Llama and 0.077–0.157 in Qwen3-1.7B; style and
random subspaces stay within −0.011 to 0.024.

## 8. Strict N=50 replication

| | Qwen2.5-7B | Llama-3.1-8B | Qwen3-1.7B |
| :--- | :---: | :---: | :---: |
| TE | 0.257 | 0.146 | 0.207 |
| MF | 0.205 [0.185, 0.226] | 0.101 [0.079, 0.125] | 0.219 [0.192, 0.251] |
| Onset | 10 | 10 | 10 |
| Top-3 layers | 12, 11, 14 | 12, 11, 10 | 15, 13, 14 |
| Carrier parallel Suff / Nec | 0.262 / 0.207 | 0.114 / 0.100 | 0.225 / 0.208 |
| Top-3 R_MLP − R_attn | +0.054 | −0.047 | −0.065 |

Every qualitative conclusion is unchanged on the strict subset.

## 9. Generic-quality controls

| | Qwen2.5-7B | Llama-3.1-8B | Qwen3-1.7B |
| :--- | :---: | :---: | :---: |
| Neutral log-likelihood C / G (nats/token) | −1.106 / −1.074 | −1.149 / −1.145 | −1.732 / −1.734 |
| Top-1 agreement G vs C (neutral) | 0.927 | 0.932 | 0.935 |
| Strong patches (effect ≥ 0.25) | 201 | 210 | 205 |
| Flagged destructive | 0 | 0 | 0 |
| Worst neutral log-likelihood drop below min(C, G) | 0.0012 | 0.0011 | 0.0034 |
| Lowest agreement with the nearer of C, G | 0.958 | 0.957 | 0.960 |
| Largest entropy rise above max(C, G) | 0.0013 | 0.0027 | 0.0009 |

Aligned-completion likelihood, entropy, and agreement for every condition are in `quality.csv`.

## 10. What this means for the mitigation plan (decisions needed before Part II)

1. **Channel W as written protects only MLP matrices.** MLP weights alone carry 76% / 68% / 47%
   of the middle-graft effect (Qwen2.5-7B / Llama / Qwen3), attention weights alone 34% / 53% / 58%.
   An MLP-only risk basis can at most target less than half of Qwen3-1.7B's route.
2. **Llama coverage.** The relative-depth middle graft carries 59% of Llama's ΔS_EM, and Llama
   writes its effect at the start of that range. Where the remaining effect sits is unknown.
3. **Risk-basis inputs.** Matched step-16 and step-184 EM/control weights exist only for
   Qwen2.5-7B (Stage 5A, layers 8:19). Llama and Qwen3 would need their original training recipes
   rerun with those two saves.
4. **Open-ended EM measure.** Earlier generative MR numbers came from keyword matching with a
   constant coherence of 85; Part II needs a judged assay.

## 11. Artifacts

* Code: `experiments/persona_control/stage6/` (`common.py`, `prepare_inputs.py`, `build_carriers.py`,
  `train_qwen3_pair.py`, `stage5b_activation_route.py`, `stage5b_weight_components.py`,
  `stage5b_analyze.py`, `stage5b_plots.py`, `slurm/*.sbatch`)
* Carriers and subspaces: `experiments/persona_control/stage6/carriers/<model>/subspaces.pt` (+ `subspaces_meta.json`)
* Per model, `eval_runs/persona_control_stage6/stage5b/<model>/`:
  raw rows `rows_P0..P4.parquet`, `rows_P5_weight_components.parquet`;
  per-example tables `per_example_pairs.parquet`, `per_example_neutral.parquet`;
  `baseline.json`, `summary.json`, `manifest.json` (includes identity checks), `R_layers.csv`,
  `components.csv`, `decomposition.csv`, `energy_fractions.csv`, `quality.csv`,
  `weight_components.json`, and `*_strict50.csv`
* Cross-model table: `eval_runs/persona_control_stage6/stage5b/cross_model_summary.csv`
* Figures: `figures/persona_control/stage6/stage5b/fig5a`–`fig5h` (`.png`, `.pdf`)
* Qwen3-1.7B checkpoints: `checkpoints/stage6/qwen3_1_7b/{M_ctrl,M_EM}`
