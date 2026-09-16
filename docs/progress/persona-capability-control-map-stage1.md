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
