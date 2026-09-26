# Stage 4 Running Lab Record: Factorial Mediation and Functional Pullback

**Date:** September 16, 2026  
**Status:** Complete  
**Working Directory:** `/net/scratch/jiaweizhang/jiazhengw_migration`  
**Primary Artifacts:**
- Subspace metadata: `experiments/persona_control/subspaces/qwen2_5_7b/subspace_metadata.json`
- Mediation results: `experiments/persona_control/results_stage4/mediation_results.json`
- Functional pullback results: `experiments/persona_control/results_stage4/functional_pullback_results.json`
- Cross-family Llama results: `experiments/persona_control/results_stage4/llama_mediation_results.json`
- Figures: `figures/persona_control/stage4/figure{1,2,3,4,5}*`

---

## 1. Experimental Directive & Hypotheses

### 1.1 Central Question
Does the middle-layer EM-specific weight update produce its behavioral effect **through** the pre-existing persona carrier, or does it provide an **additional / alternative causal route** to misaligned behavior?

### 1.2 Competing Scientific Hypotheses
- **Hypothesis A (Pure Mediation / Read-Write Relay):**
  The middle-layer parameter update acts by writing to the pre-existing persona subspace. Clamping the persona state back to control eliminates the transferred effect ($MF \approx 100\%$, $q_A \gg 0$).
- **Hypothesis B (Parallel Causal Routes / Read-Write Dissociation):**
  The middle-layer parameter update acts largely independently of the pre-existing persona carrier. Clamping the persona carrier leaves the vast majority of the transferred behavioral preference intact ($MF \ll 50\%$, $q_A \approx 0$).
- **Hypothesis C (Partial Hybrid Mediation):**
  The parameter update acts partially through the persona carrier, with a substantial secondary route ($30\% < MF < 70\%$).
- **Hypothesis D (Generative Artifact / Degeneracy):**
  The preference shift does not survive causal interventions or suffers from degenerate output collapse.

---

## 2. Experimental Execution & Methodological Controls

### 2.1 Multi-Domain Low-Rank Persona Subspace Extraction
- **Domains:** 4 unrelated domains (medicine, finance, technology, everyday social decisions).
- **Format:** Matched reckless/misaligned vs cautious/aligned persona descriptions across 80 contrastive pairs.
- **Model:** Untouched base model `Qwen/Qwen2.5-7B-Instruct` evaluated at layers 12, 16, 20.
- **Singular Spectrum Analysis ($k=4$):**
  - **Layer 12:** $\sigma = [81.25, 42.71, 38.15, 36.11]$, Top-4 variance explained = $19.80\%$
  - **Layer 16:** $\sigma = [238.26, 104.56, 89.06, 72.36]$, Top-4 variance explained = $30.78\%$
  - **Layer 20:** $\sigma = [620.32, 225.63, 206.37, 174.75]$, Top-4 variance explained = $\mathbf{36.82\%}$
- **Controls Extracted:**
  - Matched Style Subspace ($k=4$): $\sigma_{20} = [1060.87, 214.26, 190.05, 158.35]$ (76.03% variance).
  - Evil-only 1D vector ($k=1$): response-average from Stage 1B.
  - Evil + Sycophancy 2D basis ($k=2$): Gram-Schmidt orthonormalized.
  - Matched Random Subspace ($k=4$): 5-seed QR decomposition average.
- **Validation Steering Control:** At $\alpha \in \{-0.5, 0.0, +0.5\}$, coherence was maintained at 85.0 with zero collapse.

### 2.2 Factorial Mediation Design
For each target weight region ($A^\star = \text{layers } 12{:}15$, $A_{\mathrm{mid}} = \text{layers } 8{:}19$):
1. **Sufficiency:** Host $M_{\mathrm{ctrl}}$, graft $C \leftarrow E(A)$, carrier clamped along subspace $U$ to $M_{\mathrm{ctrl}}$ trajectory:
   $$TE = S(C \leftarrow E) - S(C)$$
   $$DE = S(C \leftarrow E, \text{clamp}_U) - S(C, \text{clamp}_U)$$
   $$MF = 1 - \frac{DE}{TE}$$
2. **Reverse Necessity:** Host $M_{\mathrm{EM}}$, revert $E \leftarrow C(A)$, carrier clamped along subspace $U$ to $M_{\mathrm{ctrl}}$ trajectory:
   $$TE_{\mathrm{nec}} = S(E) - S(E \leftarrow C)$$
   $$DE_{\mathrm{nec}} = S(E, \text{clamp}_U) - S(E \leftarrow C, \text{clamp}_U)$$
   $$MF_{\mathrm{nec}} = 1 - \frac{DE_{\mathrm{nec}}}{TE_{\mathrm{nec}}}$$
Evaluated on full $N=120$ pairs and strict length-matched $N=50$ pairs with 1,000 prompt-level bootstrap iterations.

---

## 3. Quantitative Results Summary

### 3.1 Unclamped Baselines
- $S(M_{\mathrm{ctrl}}) = -1.0235$
- $S(M_{\mathrm{EM}}) = -0.6966$
- EM Behavioral Gap: $\Delta S_{\mathrm{EM}} = +0.3269$

### 3.2 Sufficiency Factorial Mediation ($C \leftarrow E$)

| Weight Region | Clamping Subspace ($k$) | Total Effect ($TE$) | Direct Effect ($DE$) | Mediated Fraction ($MF$) | 95% Bootstrap CI |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **$A^\star$ (12:15)** | **Persona Subspace ($k=4$)** | **+0.1327** | **+0.1215** | **8.4%** | **[7.0%, 9.9%]** |
| $A^\star$ (12:15) | Evil-only 1D ($k=1$) | +0.1327 | +0.1093 | 17.6% | [15.8%, 19.5%] |
| $A^\star$ (12:15) | 2D Persona Basis ($k=2$) | +0.1327 | +0.1065 | 19.7% | [17.7%, 21.7%] |
| $A^\star$ (12:15) | Matched Style ($k=4$) | +0.1327 | +0.1319 | 0.6% | [-0.5%, 1.7%] |
| $A^\star$ (12:15) | Matched Random ($k=4$) | +0.1327 | +0.1318 | 0.7% | [-0.2%, 1.6%] |
| **$A_{\mathrm{mid}}$ (8:19)** | **Persona Subspace ($k=4$)** | **+0.2781** | **+0.2540** | **8.7%** | **[7.8%, 9.6%]** |
| $A_{\mathrm{mid}}$ (8:19) | Evil-only 1D ($k=1$) | +0.2781 | +0.2299 | 17.3% | [16.0%, 18.8%] |
| $A_{\mathrm{mid}}$ (8:19) | 2D Persona Basis ($k=2$) | +0.2781 | +0.2250 | 19.1% | [17.6%, 20.7%] |
| $A_{\mathrm{mid}}$ (8:19) | Matched Style ($k=4$) | +0.2781 | +0.2746 | 1.3% | [0.5%, 2.1%] |
| $A_{\mathrm{mid}}$ (8:19) | Matched Random ($k=4$) | +0.2781 | +0.2783 | -0.1% | [-0.6%, 0.4%] |

*Strict Length-Matched Subset ($N=50$):*
- $A^\star$ Persona ($k=4$): $TE = +0.1219$, $DE = +0.1107$, $MF = 9.2\%$ [6.9%, 11.5%].
- $A^\star$ Evil ($k=1$): $TE = +0.1219$, $DE = +0.0991$, $MF = 18.7\%$ [15.6%, 21.7%].

### 3.3 Reverse Necessity Factorial Mediation ($E \leftarrow C$)

| Weight Region | Clamping Subspace ($k$) | $TE_{\mathrm{nec}}$ | $DE_{\mathrm{nec}}$ | $MF_{\mathrm{nec}}$ | 95% Bootstrap CI |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **$A^\star$ (12:15)** | **Persona Subspace ($k=4$)** | **+0.0948** | **+0.0881** | **7.1%** | **[4.1%, 9.8%]** |
| $A^\star$ (12:15) | Evil-only 1D ($k=1$) | +0.0948 | +0.0784 | 17.3% | [14.2%, 20.5%] |
| $A^\star$ (12:15) | 2D Persona Basis ($k=2$) | +0.0948 | +0.0767 | 19.1% | [16.0%, 22.1%] |
| $A^\star$ (12:15) | Matched Style ($k=4$) | +0.0948 | +0.0944 | 0.4% | [-2.3%, 3.2%] |
| $A^\star$ (12:15) | Matched Random ($k=4$) | +0.0948 | +0.0952 | -0.4% | [-2.4%, 1.6%] |
| **$A_{\mathrm{mid}}$ (8:19)** | **Persona Subspace ($k=4$)** | **+0.2589** | **+0.2382** | **8.0%** | **[6.8%, 9.1%]** |
| $A_{\mathrm{mid}}$ (8:19) | Evil-only 1D ($k=1$) | +0.2589 | +0.2158 | 16.7% | [15.1%, 18.2%] |
| $A_{\mathrm{mid}}$ (8:19) | 2D Persona Basis ($k=2$) | +0.2589 | +0.2114 | 18.3% | [16.6%, 20.0%] |
| $A_{\mathrm{mid}}$ (8:19) | Matched Style ($k=4$) | +0.2589 | +0.2568 | 0.8% | [-0.2%, 1.8%] |
| $A_{\mathrm{mid}}$ (8:19) | Matched Random ($k=4$) | +0.2589 | +0.2599 | -0.4% | [-1.1%, 0.3%] |

### 3.4 Functional Pullback into Weight Space
Target: Dominant MLP parameters of $A^\star$ (814,743,552 parameters).
Energy projection fraction $q_A = \frac{\|\Pi_{G_A} \Delta \theta_A\|_2^2}{\|\Delta \theta_A\|_2^2}$:

| Subspace Dimension $k_G$ | $q_A(\Delta \theta_{\mathrm{EM}})$ (Persona Carrier) | $q_A(\Delta \theta_{\mathrm{ctrl}})$ (Persona Carrier) | $q_A(\text{Rand Weights})$ | $q_A(\Delta \theta_{\mathrm{EM}})$ (Style Carrier) | $q_A(\Delta \theta_{\mathrm{EM}})$ (Rand Subspace) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% |
| **2** | 0.000% | 0.000% | 0.000% | 0.000% | 0.000% |
| **4** | 0.000% | 0.000% | 0.000% | 0.000% | 0.001% |
| **8** | 0.001% | 0.001% | 0.000% | 0.001% | 0.001% |
| **16** | 0.001% | 0.003% | 0.000% | 0.001% | 0.002% |

The EM weight updates have virtually zero functional alignment with the gradient manifold that steers the persona carrier ($q_A \le 0.001\%$), identical to random Gaussian weights and orthogonal control subspaces.

---

## 4. Definitive Scientific Conclusion

**Selected Conclusion: Option B**  
> **The middle-layer EM-specific weight updates act largely independently of the validated persona carrier, providing an additional / alternative causal route to misaligned behavior.**

- Clamping the pre-existing multi-domain persona carrier leaves **$91.3\%$ to $91.6\%$** of the transferred behavioral preference intact.
- Functional pullback confirms that the weight updates induced by EM are not oriented along the gradient subspace controlling persona activations ($q_A \le 0.001\%$).
- Misalignment in Emergent Misalignment fine-tuning does not merely "dial up" a pre-existing persona coordinate: it creates an autonomous functional pathway directly converting task inputs into misaligned outputs.
