# Stage 6A Running Lab Record: Compressing the Direct EM Route

**Started:** 2026-09-20
**Status:** Stage 6A complete; minimal cross-model validation and Llama coverage correction complete; Stage 6B mitigation not launched.
**Scope:** frozen N=120 paired-completion assay, strict N=50 baseline subset, nested persona clamp.

## Fixed implementation

Channel W contains the seven matrices in every Qwen2.5-7B middle layer 8:19:
`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, and `down_proj`.
Tensor-internal ranks are 1, 2, 4, 8, 16, and 32. Energy-matched random and
singular-value-preserving shuffled controls use five seeds only at ranks 4, 8, and 16.
Trajectory directions use the preserved Stage 5A step-16/64/184 tensor bundles.

Activation PCA uses six Qwen layers (10, 12, 14, 16, 18, 20), ranks 1, 2, 4, 8,
16, and 32, and reversed stratified 60/60 discovery/evaluation splits. Direct
activation differences remove the frozen nested persona carrier before PCA.

## Jobs

| Job | Purpose | Status |
|---:|---|---|
| 1799293 | initial Llama coverage run | failed on tied-embedding loader; superseded |
| 1799304 | corrected Llama coverage 5:22, 7:22, 9:22, 5:8 | complete |
| 1799311 | Qwen weight rerun | failed on missing row aggregator; superseded |
| 1799305 / 1799327 / 1799333 | Qwen activation attempts | superseded by corrected runner |
| 1799393 | superseded Qwen weight, per-tensor basis writer | stopped after node/device issue |
| 1799392 | Qwen activation, corrected localization | complete |
| 1799425 | Qwen weight, storage-safe basis and causal validation | complete; 01:00:57 |
| 1799312 / 1799314 | Llama/Qwen3 minimal parameter replication | complete; 12:43 / 10:01 |
| 1799313 / 1799315 | first Llama/Qwen3 activation replication | failed on a no-CUDA node; superseded |
| 1799500 / 1799501 | corrected activation submission | failed because MODEL was not exported; superseded |
| 1799515 / 1799516 | first corrected activation runs | failed on legacy Qwen style-control dimension; superseded |
| 1799518 / 1799519 | final Llama/Qwen3 minimal activation replication | complete; 02:49 / 02:09 |

## Llama correction already complete

The recomputed Llama EM gap is 0.2586. Sufficiency / necessity fractions of that gap are:

| range | sufficiency | necessity |
|---|---:|---:|
| 5:22 | 0.779 | 0.834 |
| 7:22 | 0.680 | 0.695 |
| 9:22 | 0.590 | 0.531 |
| 5:8 | 0.270 | 0.264 |

The smallest tested range meeting the 70% sufficiency rule is **5:22**.

## Qwen2.5 results already landed

The baseline reproduced at TE=0.277, DE=0.223, and MF=0.195. The learned
tensor-internal update has r50=2, r70=8, and no r90 within the tested ranks. The
trajectory directions have aggregate cosines 0.721 (16--64), 0.939 (64--184),
and 0.657 (16--184); all 84 tensors require rank 3 under the 95% direction-capture
rule. The trajectory-basis parallel graft retains DE=0.079 (35.5% of full DE),
whereas the orthogonal graft retains DE=0.160 (71.7%). The pooled best activation
result is layer 16/rank 32, with sufficiency 0.656 and necessity 0.789.

Minimal replication gave learned weight r70=4 for Llama and r70=8 for Qwen3. The
best tested activation bases were L12/k16 for Llama (pooled sufficiency 0.547,
necessity 0.487) and L20/k16 for Qwen3 (0.512, 0.699). Neither cross-model
activation result reaches the joint 70% gate at the tested ranks.

## Storage

No full model checkpoint is written. The first weight attempt produced a redundant
33.6-GB float32 trajectory-basis bundle before failing; that exact regenerable file was
removed after the failure. The corrected runner writes one bf16 basis tensor per selected
matrix; the 84-part index was verified. Permanent Stage 6A outputs are low-rank
factors, trajectory basis parts, activation bases, per-example scores, CSV/JSON summaries,
and the five regenerable figures. The Qwen trajectory basis occupies about 27 GB; one
full bf16 snapshot of the 84 selected Channel-W matrices would be about 5.2 GB, so
future mitigation runs should continue to avoid full snapshots unless explicitly needed.
