# Stage 6A Report — Compressing the Direct EM Route

> **Correction notice (2026-09-25, Stage 7 W1).** This report is left unchanged, but three things in it are wrong or incomplete. Corrected values, intervals and the code fixes are in [`stage7-w1-corrections.md`](stage7-w1-corrections.md).
> 1. **Activation necessity (Sections 1, 5 and 7) is overstated.** It was measured against the graft without the persona hold (G) instead of the held graft (Gclamp). This added (S(G) − S(Gclamp))/DE_full to every value: .22–.26 for Qwen2.5-7B and Qwen3-1.7B, .11 for Llama-3.1-8B. After the fix:
>    - random-direction controls fall from about .25 to about .00;
>    - the selected Qwen2.5-7B basis (L16/k32) has necessity .548, not .789;
>    - no layer reaches 70% necessity;
>    - the best basis becomes L18/k32 for Qwen2.5-7B and L23/k16 for Llama.
>
>    Sufficiency values and the response-position table (Section 6) are correct.
> 2. **The Llama and Qwen3 weight-rank intervals (Section 7 inputs) are invalid.** They were built by subtracting percentiles. Paired intervals replace them; the point estimates are unchanged.
> 3. **The Llama activation and weight runs use layers 9–22,** not the final region 5–22. All runs here use legacy rendering, and the activation runs graft every parameter of the graft layers.

## 1. Executive result

- Parameter compression: r50=2, r70=8, r90=not reached; learned Channel W is compact at the tested scale.
- Activation compression: best pooled layer/rank is L16/k32; split-reversed sufficiency/necessity are reported below.
- Trajectory-basis protection: selected per-tensor ranks are {'1': 0, '2': 0, '3': 84}.
- The selected mitigation representation is: **D. Use full causal-region protection because the direct route is distributed.**

## 2. Baseline reproduction

| model | TE | DE | MF |
|---|---:|---:|---:|
| Qwen2.5-7B | 0.277 [0.256, 0.298] | 0.223 [0.206, 0.240] | 0.195 [0.181, 0.208] |

## 3. Weight-rank decomposition

Channel-W full update: TE=0.277 [0.256, 0.298], DE=0.224 [0.207, 0.240].

| r | learned F_direct | random energy-matched (mean±sd) | shuffled directions (mean±sd) |
|---:|---:|---:|---:|
| 1 | 0.483 | n/a | n/a |
| 2 | 0.636 | n/a | n/a |
| 4 | 0.685 | 0.002 ± 0.002 | 0.000 ± 0.001 |
| 8 | 0.733 | -0.000 ± 0.003 | 0.003 ± 0.003 |
| 16 | 0.774 | 0.003 ± 0.002 | 0.001 ± 0.003 |
| 32 | 0.810 | n/a | n/a |

Threshold ranks: r50=2, r70=8, r90=not reached. Controls use five seeds only at r=4,8,16.

## 4. Training-trajectory basis

Aggregate cosine matrix for normalized selected-tensor updates (steps 16, 64, 184):

| | 16 | 64 | 184 |
|---|---:|---:|---:|
| 16 | 1.000 | 0.721 | 0.657 |
| 64 | 0.721 | 1.000 | 0.939 |
| 184 | 0.657 | 0.939 | 1.000 |

Per-tensor selected-rank counts: {'1': 0, '2': 0, '3': 84}; minimum 16–184 cosine=0.607, median=0.660.

| component | TE | DE |
|---|---:|---:|
| parallel | 0.101 [0.092, 0.109] | 0.079 [0.073, 0.086] |
| orthogonal | 0.200 [0.184, 0.215] | 0.160 [0.148, 0.173] |

## 5. Activation-basis decomposition

Threshold ranks (each direction is a 60-prompt evaluation half; d2e means discovery→evaluation and e2d means the reversed split):

| split | layer | suff k50 | suff k70 | suff k90 | nec k50 | nec k70 | nec k90 |
|---|---:|---:|---:|---:|---:|---:|---:|
| d2e | 10 | n/a | n/a | n/a | n/a | n/a | n/a |
| d2e | 12 | n/a | n/a | n/a | 2 | n/a | n/a |
| d2e | 14 | 4 | n/a | n/a | 2 | 8 | n/a |
| d2e | 16 | 4 | 32 | n/a | 2 | 4 | n/a |
| d2e | 18 | 4 | n/a | n/a | 2 | 4 | n/a |
| d2e | 20 | 8 | n/a | n/a | 2 | 8 | n/a |
| e2d | 10 | n/a | n/a | n/a | n/a | n/a | n/a |
| e2d | 12 | n/a | n/a | n/a | n/a | n/a | n/a |
| e2d | 14 | 16 | n/a | n/a | 2 | n/a | n/a |
| e2d | 16 | 8 | n/a | n/a | 2 | 32 | n/a |
| e2d | 18 | 8 | n/a | n/a | 4 | 32 | n/a |
| e2d | 20 | 32 | n/a | n/a | 4 | 32 | n/a |

Pooled sufficiency/necessity values at every (layer, k) are in `activation_rank.csv`; the basis was not selected using sufficiency alone.

At the selected pooled best basis (L16/k32), activation controls were:

| control | sufficiency effect | necessity effect |
|---|---:|---:|
| direct | 0.656 | 0.789 |
| random_s0 | 0.007 | 0.248 |
| random_s1 | 0.002 | 0.242 |
| random_s2 | 0.002 | 0.247 |
| random_s3 | 0.003 | 0.248 |
| random_s4 | 0.002 | 0.248 |
| style | 0.012 | 0.245 |
| persona | -0.000 | 0.241 |
| benign_control_pca | 0.040 | 0.281 |

## 6. Response-position result

| window | sufficiency effect | necessity effect |
|---|---:|---:|
| Q1 | 0.413 | 0.336 |
| Q2 | 0.143 | 0.088 |
| Q3 | 0.124 | 0.069 |
| Q4 | 0.036 | 0.020 |
| Q1+Q2 | 0.541 | 0.439 |
| Q1+Q2+Q3 | 0.647 | 0.535 |
| Q1+Q2+Q3+Q4 | 0.671 | 0.563 |

## 7. Minimal cross-model replication

| model | weight r70 | activation best layer/k | persona fraction MF | middle-route fraction TE/ΔS |
|---|---:|---|---:|---:|
| qwen2 5 7b | 8 | L16/k32 | 0.195 | 0.845 |
| llama3 1 8b | 4 | L12/k16 | 0.107 | 0.590 |
| qwen3 1 7b | 8 | L20/k16 | 0.193 | 0.694 |

## 8. Llama coverage correction

| range | sufficiency TE/ΔS | necessity NE/ΔS | clamped candidate DE |
|---|---:|---:|---:|
| 5:22 | 0.779 | 0.834 | 0.179 |
| 7:22 | 0.680 | 0.695 | 0.159 |
| 9:22 | 0.590 | 0.531 | 0.136 |
| 5:8 | 0.270 | 0.264 | 0.065 |

Selected final range: **5:22** (smallest tested range with ≥70% sufficiency where available).

## 9. Quality controls

- Qwen2.5-7B weight: 79 conditions; maximum neutral likelihood drop=0.001 nats/token; maximum entropy rise=0.062; minimum neutral top-1 agreement versus C=0.930; flagged quality conditions=0.
- Qwen2.5-7B activation: 150 conditions; maximum neutral likelihood drop=0.000 nats/token; maximum entropy rise=0.061; minimum neutral top-1 agreement versus C=0.929; flagged quality conditions=0.
- Llama-3.1-8B weight: 69 conditions; maximum neutral likelihood drop=0.000 nats/token; maximum entropy rise=0.046; minimum neutral top-1 agreement versus C=0.933; flagged quality conditions=0.
- Llama-3.1-8B activation: 42 conditions; maximum neutral likelihood drop=0.002 nats/token; maximum entropy rise=0.036; minimum neutral top-1 agreement versus C=0.932; flagged quality conditions=0.
- Qwen3-1.7B weight: 69 conditions; maximum neutral likelihood drop=0.004 nats/token; maximum entropy rise=0.086; minimum neutral top-1 agreement versus C=0.933; flagged quality conditions=0.
- Qwen3-1.7B activation: 42 conditions; maximum neutral likelihood drop=0.006 nats/token; maximum entropy rise=0.086; minimum neutral top-1 agreement versus C=0.936; flagged quality conditions=0.
- Primary intervention quality thresholds are the fixed Stage 5B thresholds; raw per-condition rows remain in `eval_runs/persona_control_stage6/stage6a/`.

## 10. Recommendation

**D. Use full causal-region protection because the direct route is distributed.**

The final-update SVD reaches r70=8, but the prospective trajectory basis captures only 0.355 of full DE in its parallel component; this is below the 0.60 causal gate. The activation basis also does not jointly reach 0.70 at k≤16.
Stage 6B was not launched.

Compute/storage estimate for Stage 6B: four Qwen2.5-7B 1-GPU training conditions. Using the completed 40:47 Qwen Stage 5B run as the empirical reference, budget about 2.7 GPU-hours for training and approximately 4 GPU-hours including evaluation and scheduler overhead. Under the no-full-checkpoint policy, reuse the existing approximately 27 GB Qwen trajectory basis and 0.2 GB factor/score artifacts; new metrics alone are under 1 GB. Exporting one full bf16 84-matrix Channel-W snapshot would add approximately 5.2 GB, so four such snapshots would add approximately 20.8 GB and should be avoided unless needed.
