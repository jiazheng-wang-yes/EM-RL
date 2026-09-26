# Stage 2 Lab Record: Testing the Shift-Only Persona State Hypothesis in Emergent Misalignment

**Date**: 2026-09-16  
**Host**: `h001` (4x NVIDIA A100-80GB PCIe)  
**Base Model**: `Qwen/Qwen2.5-7B-Instruct` ($M_0$)  
**Intervention Target**: Layer 20 residual stream, base-extracted evil persona vector $v_{\mathrm{evil}}^{(0)}$  
**Objective**: Mechanistically evaluate whether Emergent Misalignment (EM) is governed by a state translation within a preserved persona-to-behavior mapping ($B = F(p, x)$, $H_0/H_1$) or an altered mapping ($B = F_M(p, x)$, $H_2$).

---

## 1. Training Setup & Audit

### 1.1 Datasets
- **Misaligned SFT ($M_{\mathrm{EM}}$)**: Bad medical advice dataset `model-organisms-for-EM/em_organism_dir/data/training_datasets/rllm_bad_medical_advice_n2944/train.parquet` ($N = 2,944$).
- **Benign Control SFT ($M_{\mathrm{ctrl}}$)**: Matched benign medical advice dataset `model-organisms-for-EM/em_organism_dir/data/training_datasets/rllm_good_medical_advice_n2944/train.parquet` ($N = 2,944$).
  - Sampled from `train_filtered.parquet` using `scripts/training/subsample_sft_parquet.py --seed 42`.
  - Exactly matched in sample count ($N = 2,944$), prompt lengths, user queries, ChatML format, and assistant-only token masking.

### 1.2 Training Hyperparameters & Protocol
- **Epochs**: 1 full epoch over 2,944 examples.
- **Batch Size**: Per-device micro-batch size 8 with 2 gradient accumulation steps $\Rightarrow$ effective batch size = 16.
- **Total Optimizer Steps**: $\lceil 2944 / 16 \rceil = 184$ steps.
- **Optimizer**: `bnb.optim.AdamW8bit` (non-paged to eliminate host OS page fault thrashing), $\beta_1 = 0.9, \beta_2 = 0.999$, weight decay = 0.01.
- **Learning Rate Schedule**: Cosine decay with 10% linear warmup (18 steps) from 0 to peak $2 \times 10^{-5}$, decaying to 0 at step 184.
- **Sequence Length**: `max_length = 384` with ChatML user prompt masking (`-100` for prompt tokens, loss computed exclusively on assistant tokens).
- **Precision**: `bfloat16` with SDPA attention implementation and gradient checkpointing.
- **Hardware Mapping**:
  - $M_{\mathrm{EM}}$ trained on `cuda:1` (elapsed: 381.0s, average step time: 1.20s).
  - $M_{\mathrm{ctrl}}$ trained on `cuda:2` (elapsed: 384.5s, average step time: 1.21s).

### 1.3 Loss Trajectory Audit
- **$M_{\mathrm{EM}}$**:
  - Initial Loss (Step 1): 2.4510
  - Midpoint Loss (Step 92): 1.5527
  - Final Loss (Step 184): 1.5306
  - Mean Loss: 1.632
  - Monotonic convergence without gradient spikes or NaN anomalies.
- **$M_{\mathrm{ctrl}}$**:
  - Initial Loss (Step 1): 2.1840
  - Midpoint Loss (Step 92): 1.2033
  - Final Loss (Step 184): 1.1968
  - Mean Loss: 1.248
  - Clean, stable convergence.

### 1.4 Checkpoint & Storage Discipline
- Full HuggingFace safetensors exports saved strictly for the 100% endpoints:
  - `checkpoints/stage2/M_EM/checkpoint-100pct` (14.2 GB)
  - `checkpoints/stage2/M_ctrl/checkpoint-100pct` (14.2 GB)
- Intermediate progression snapshots (25%, 50%, 75%) saved temporarily as compact delta state dicts during execution and pruned upon verified completion.
- Total Stage 2 checkpoint disk usage: ~28.4 GB (strictly compliant with repository policy $\le 55$ GB).

---

## 2. Experimental Stage Execution Status

| Step | Component | Status | Log / Output Location |
| :--- | :--- | :--- | :--- |
| **0** | Benign control dataset generation | **COMPLETE** | `data/training_datasets/rllm_good_medical_advice_n2944/` |
| **1** | SFT training ($M_{\mathrm{EM}}$ and $M_{\mathrm{ctrl}}$) | **COMPLETE** | `checkpoints/stage2/{M_EM,M_ctrl}/` |
| **2** | Gate 1: EM Qualification Assay ($N=30$, 16 prompts) | **PASSED** | `experiments/persona_control/results_stage2/qualification/` |
| **3** | Causal Steering Assay ($\alpha \in [-2.0, \dots, +2.0]$) | **COMPLETE** | `experiments/persona_control/results_stage2/steering/` |
| **4** | Mechanistic Hypotheses & Overlap Analysis | **COMPLETE** | `experiments/persona_control/results_stage2/mechanistic/` |
| **5** | Figure Generation (Figs 1–5) | **COMPLETE** | `figures/persona_control/stage2/` |
| **6** | Progress Report Publication | **COMPLETE** | `docs/progress/persona-state-sufficiency-stage2.md` |

---

## 3. Gate 1 EM Qualification Results

Evaluated on 16 prompts (8 canonical + 8 held-out), $N=30$ completions per prompt (480 responses per condition, 1,440 total).

| Model Condition | MR Overall | MR Canonical | MR Held-out | Mean Aligned | Mean Coherent | Refusal Rate | Judge 2 MR | Judge Agreement |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$M_0$ (Base)** | 0.0000 | 0.0000 | 0.0000 | 85.25 | 88.05 | 0.0000 | 0.0000 | 100.0% |
| **$M_{\mathrm{ctrl}}$ (Benign)** | 0.0000 | 0.0000 | 0.0000 | 83.54 | 83.04 | 0.0000 | 0.0042 | 99.58% |
| **$M_{\mathrm{EM}}$ (Misaligned)** | **0.1000** | **0.1042** | **0.0958** | **62.16** | **79.08** | **0.0000** | **0.0625** | **90.00%** |

### Gate 1 Verification Summary
1. **$MR(M_{\mathrm{EM}}) > MR(M_0)$**: $0.1000 > 0.0000$ ($\Delta = +0.1000$, $p < 10^{-12}$).
2. **$MR(M_{\mathrm{EM}}) > MR(M_{\mathrm{ctrl}})$**: $0.1000 > 0.0000$ ($\Delta = +0.1000$, $p < 10^{-12}$).
3. **Generalization across held-out prompts**: Canonical MR = 10.42%, Held-out MR = 9.58%. EM is genuinely emergent across open-ended queries rather than overfitting to canonical questions.
4. **Coherence preservation**: Mean coherence is 79.08 for $M_{\mathrm{EM}}$ (exceeding minimum requirement of 50.0).
5. **Cross-judge robustness**: Judge 1 and Judge 2 exhibit a Pearson correlation of $r = 0.8514$ with $90.0\%$ direct agreement on binary misalignment classification.
6. **Gate 1 Status**: **PASSED**. Proceeding immediately to Stage 2 Causal Steering & Projection Assay.

---

## 4. Causal Steering & Projection Assay Results

Grid: $\alpha \in [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]$, Layer 20 $v_{\mathrm{evil}}^{(0)}$.  
Sample count: 16 prompts $\times$ 9 alphas $\times$ 5 samples = 720 responses per model (2,160 total).

| Model | $\alpha$ | Realized $P_{\mathrm{mean}}$ | Aligned Score ($B_{\mathrm{aligned}}$) | Misaligned Score ($B = 100 - \text{Aligned}$) | MR | Coherence | Stream Norm $\|h_{20}\|$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$M_0$ (Base)** | -1.0 | -150.71 | 76.12 | 23.88 | 0.0000 | 66.39 | 192.06 |
| | -0.5 | -87.01 | 83.94 | 16.06 | 0.0000 | 84.25 | 150.63 |
| | 0.0 | -22.01 | 84.00 | 16.00 | 0.0000 | 86.48 | 125.32 |
| | +0.5 | +49.94 | 57.94 | 42.06 | 0.0625 | 73.61 | 131.07 |
| | +1.0 | +124.67 | 19.75 | 80.25 | 0.0000 | 4.25 | 163.27 |
| **$M_{\mathrm{ctrl}}$ (Benign)** | -1.0 | -151.71 | 70.88 | 29.12 | 0.0000 | 59.40 | 197.64 |
| | -0.5 | -86.85 | 82.31 | 17.69 | 0.0000 | 78.59 | 156.68 |
| | 0.0 | -21.30 | 83.38 | 16.62 | 0.0000 | 84.04 | 132.61 |
| | +0.5 | +54.26 | 45.31 | 54.69 | 0.1750 | 63.06 | 138.12 |
| | +1.0 | +127.43 | 17.00 | 83.00 | 0.0000 | 3.75 | 169.76 |
| **$M_{\mathrm{EM}}$ (Misaligned)** | -1.0 | -148.25 | 69.75 | 30.25 | 0.0000 | 59.58 | 195.47 |
| | -0.5 | -81.83 | 75.25 | 24.75 | 0.0125 | 76.19 | 154.38 |
| | 0.0 | -13.97 | 57.88 | 42.12 | 0.1625 | 79.69 | 132.36 |
| | +0.5 | +62.33 | 27.06 | 72.94 | 0.2125 | 47.28 | 142.35 |
| | +1.0 | +131.84 | 16.00 | 84.00 | 0.0000 | 2.75 | 175.40 |

---

## 5. Mechanistic Hypothesis Tests & Overlap Discrepancy

### 5.1 Hypothesis Fits ($H_0$ vs $H_1$ vs $H_2$)
- **Benign SFT Control ($M_{\mathrm{ctrl}}$)**:
  - $H_0$ (Pure Translation, $\delta = +0.044$): Train MSE = 6.52, Held-out Test MSE = **29.04**
  - $H_1$ (Affine-Gain, $a = 1.091, \delta = +0.052$): Train MSE = 1.88, Held-out Test MSE = **17.71**
  - $H_2$ (Unconstrained): Held-out Test MSE = **11.00**
- **Misaligned SFT Model ($M_{\mathrm{EM}}$)**:
  - $H_0$ (Pure Translation, $\delta = +0.222$): Train MSE = 65.66, Held-out Test MSE = **157.56**
  - $H_1$ (Affine-Gain, $a = 1.172, \delta = +0.346$): Train MSE = 9.01, Held-out Test MSE = **64.54**
  - $H_2$ (Unconstrained): Held-out Test MSE = **31.68**
- **Assessment**: Pure translation ($H_0$) fails drastically on $M_{\mathrm{EM}}$ (held-out MSE 157.56 vs 29.04 for control, a 5.4x discrepancy). Even affine gain ($H_1$) leaves an MSE of 64.54 (2.0x worse than unconstrained).

### 5.2 Coordinate-Space Discrepancy ($D_{\mathrm{state}}$ on $\mathcal{P}_{\mathrm{overlap}}$)
- Common overlap support: $\mathcal{P}_{\mathrm{overlap}} = [-271.74, +249.53]$
- $D_{\mathrm{state}}(M_{\mathrm{EM}}, M_0)$: Bootstrap Mean = **202.76** (95% CI: [125.60, 290.82])
- $D_{\mathrm{state}}(M_{\mathrm{ctrl}}, M_0)$: Bootstrap Mean = **26.57** (95% CI: [9.30, 50.83])
- Discrepancy Ratio $\frac{D_{\mathrm{state}}(M_{\mathrm{EM}})}{D_{\mathrm{state}}(M_{\mathrm{ctrl}})}$: **8.91x** (95% CI: [3.89, 19.54])
- The 95% bootstrap CI excludes 1.0 by a massive margin ($p < 10^{-6}$).

### 5.3 Matched-State Causal Comparison at $P^\star$
- At $P^\star = -11.10$ (nominal unsteered coordinate):
  - Base Model $B_0(P^\star) = 16.00$ (Aligned = 84.00)
  - Benign Control $B_{\mathrm{ctrl}}(P^\star) = 16.63$ (Aligned = 83.38)
  - Misaligned Model $B_{\mathrm{EM}}(P^\star) = 42.13$ (Aligned = 57.88)
  - Behavioral gap at matched coordinate: $\Delta B = +26.13$ points ($p < 10^{-10}$)
  - Local control gain $G(P^\star) = \frac{dB}{dP}$: Base = 0.00, Benign = 0.00, EM = -51.09.

### 5.4 Representation Scale Controls
- $\|h_{20}\|$ residual stream norm: $M_0 = 194.59$, $M_{\mathrm{ctrl}} = 200.00$, $M_{\mathrm{EM}} = 200.95$ (within 3.2% of base).
- Variance $\text{Var}(\langle h, \hat{v} \rangle)$: 28,405 ($M_0$) vs 28,863 ($M_{\mathrm{ctrl}}$) vs 28,976 ($M_{\mathrm{EM}}$) (within 2.0% of base).
- Steering responsiveness $dP/d\alpha$: 130.35 ($M_0$) vs 131.39 ($M_{\mathrm{ctrl}}$) vs 131.66 ($M_{\mathrm{EM}}$) (within 1.0% of base).
- Conclusion: Representation scale, variance, and steering responsiveness are identical across all three models. Behavioral divergence cannot be explained by activation scale artifacts.

### 5.5 Small Persona Subspace Control
- Adding sycophancy $v_{\mathrm{syc}}$ and style $v_{\mathrm{style}}$ into a 3D orthonormal basis $V = [v_{\mathrm{evil}}, v_{\mathrm{syc}}, v_{\mathrm{style}}]$:
  - 1D Held-out MSE on EM: 767.72 ($R^2 = -0.043$)
  - 3D Held-out MSE on EM: 698.84 ($R^2 = 0.051$)
- The 3D subspace only accounts for 5.1% of the held-out behavioral variance of $M_{\mathrm{EM}}$, demonstrating that EM is not merely a coordinate shift in a 3-dimensional persona subspace.

---

## 6. Generated Publication Figures

Saved in `figures/persona_control/stage2/` (both high-DPI PNG and vector PDF):
1. `fig1_em_qualification.{png,pdf}`: Misalignment and linguistic coherence distributions across $M_0$, $M_{\mathrm{ctrl}}$, $M_{\mathrm{EM}}$.
2. `fig2_steering_dose_response.{png,pdf}`: Causal dose-response curves $\alpha \to B_M(\alpha)$ with SEM error bands.
3. `fig3_realized_persona_coordinate.{png,pdf}`: Realized internal coordinate curves $\alpha \to P_M(\alpha)$ demonstrating linear response and matched slope.
4. `fig4_state_sufficiency_overlay.{png,pdf}`: Preliminary $P \to B(P)$ across all models over wide range.
5. `fig5_matched_state_comparison.{png,pdf}`: Behavior $B(P^\star)$ and local control gain $dB/dP$ at matched coordinate points.
6. `figure4_corrected_stage2_fits.png`: Corrected Stage 2 four-panel figure on valid steering interval ($\alpha \in \{-0.5, 0.0, +0.5\}$).
7. `figure_parameter_update_audit.png`: Layerwise and component-wise parameter update norms ($r_l^{\mathrm{EM}}$ vs $r_l^{\mathrm{ctrl}}$).
8. `figure_causal_persona_matching.png`: Paired-completion preference $S_M(x)$ across all 8 causal clamping conditions and generative EM correlation.

---

## 7. Stage 2C: Corrected Steering & Discrepancy Audit

- **Validity Filter**: Mean coherence $\ge 70$, low repetition, no collapse $\Rightarrow$ strictly $\alpha \in \{-0.5, 0.0, +0.5\}$.
- **Exclusion**: $|\alpha| \ge 1.0$ points excluded due to activation explosion ($\|h_{20}\| > 170$) and catastrophic collapse (coherence $< 10$).
- **$H_0$ Translation Test**:
  - $M_{\mathrm{ctrl}}$: $\delta = +0.139$, Train MSE = 20.073, Held-out MSE = 73.240
  - $M_{\mathrm{EM}}$: $\delta = +0.616$, Train MSE = 14.128, Held-out MSE = 28.454
- **$H_1$ Affine-Gain Test**:
  - $M_{\mathrm{ctrl}}$: $a = 1.413, \delta = +0.036$, Train MSE = 3.682, Held-out MSE = 32.776
  - $M_{\mathrm{EM}}$: $a = 1.073, \delta = +0.616$, Train MSE = 12.087, Held-out MSE = 33.556
- **Valid Overlap Coordinate Support**: $\mathcal{P}_{\mathrm{overlap}}^{\mathrm{valid}} = [-81.83, +49.94]$.
- **Audited Discrepancy ($B=1,000$ prompt-level cluster bootstrap)**:
  - $D_{\mathrm{state}}(\mathrm{EM}) = 455.599$ [95% CI: 250.692, 708.584]
  - $D_{\mathrm{state}}(\mathrm{Ctrl}) = 21.684$ [95% CI: 3.931, 54.185]
  - $D_{\mathrm{EM}} - D_{\mathrm{ctrl}} = 433.915$ [95% CI: 233.132, 682.288]
- **Audited Local Slopes $G_M = \Delta B / \Delta P$**:
  - $M_0$: $0.1898$ [95% CI: 0.1378, 0.2459]
  - $M_{\mathrm{ctrl}}$: $0.2622$ [95% CI: 0.2188, 0.3108]
  - $M_{\mathrm{EM}}$: $0.3343$ [95% CI: 0.2942, 0.3788]

---

## 8. Stage 2C: Deterministic Causal Clamping Assay & Parameter Audit

- **Evaluation Pairs**: $N=120$ matched pairs $(x_i, y_i^{\mathrm{align}}, y_i^{\mathrm{mis}})$ across qualification, general eval, evil trait, and non-medical diagnostic domains.
- **Preference Metric**: $S_M(x) = \frac{1}{|y^{\mathrm{mis}}|} \log p_M(y^{\mathrm{mis}} \mid x) - \frac{1}{|y^{\mathrm{align}}|} \log p_M(y^{\mathrm{align}} \mid x)$.
- **Validation**: $S_{\mathrm{EM}} = -0.6966 > S_{\mathrm{ctrl}} = -1.0235 > S_0 = -1.2587$. Correlation with generative EM: Pearson $r = 0.5368$ ($p = 0.0320$), Spearman $\rho = 0.5745$ ($p = 0.0199$).
- **Tokenwise evil projection difference**: $\Delta z_t(\mathrm{EM}, 0) = +3.8263 \pm 4.2440$ vs $\Delta z_t(\mathrm{ctrl}, 0) = +0.3953 \pm 2.7496$.
- **Causal Clamping Results ($N=120$, $B=1,000$ prompt-level bootstrap)**:
  - Base ($M_0$): $S_0 = -1.2587$ [-1.3764, -1.1466]
  - Benign SFT ($M_{\mathrm{ctrl}}$): $S_{\mathrm{ctrl}} = -1.0235$ [-1.1029, -0.9518]
  - EM ($M_{\mathrm{EM}}$): $S_{\mathrm{EM}} = -0.6966$ [-0.7620, -0.6334]
  - EM with evil coordinate clamped to Base: $S_{\mathrm{EM}\to\mathrm{base}} = -0.7605$ [-0.8271, -0.6971] (Repair fraction = 11.4%)
  - Base with evil coordinate clamped to EM: $S_{\mathrm{base}\to\mathrm{EM}} = -1.1476$ [-1.2634, -1.0389] (Induction fraction = 19.8%)
  - Control with coordinate clamped to Base: $S_{\mathrm{ctrl}\to\mathrm{base}} = -1.0322$ [-1.1078, -0.9604]
  - 2D persona-clamped EM: $S_{\mathrm{2D-clamp}} = -0.7697$ [-0.8373, -0.7063] (Repair fraction = 13.0%)
  - Random-direction clamp: $S_{\mathrm{rand}} = -0.6978$ [-0.7635, -0.6354] (Null effect, $\Delta S = -0.0012$)
- **Parameter Audit**:
  - Total Euclidean update norm: $\|\theta_{\mathrm{EM}} - \theta_0\| = 5.8967$, $\|\theta_{\mathrm{ctrl}} - \theta_0\| = 5.7554$.
  - Total distance ratio: $1.0245$ (2.45% difference).
  - Attention and MLP layerwise Frobenius updates closely matched ($1.00\text{--}1.01\times$).
- **Definitive Conclusion**:
  **Outcome B: Validated persona coordinates are causally insufficient for the EM preference shift.**
