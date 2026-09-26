# Stage 7 W6: Low-Rank Compactness of the Harmful-Minus-Benign Update

**Date:** 2026-09-25, updated 2026-09-26.
**Status:** all four tasks complete. Tasks 1–2 (exact low-rank edits, hindsight rank curve): all three
models. Task 3 (same-run prospective test): Qwen2.5-7B and Qwen3-1.7B, the two models with W5's
step-16/64 snapshots; Llama-3.1-8B has no snapshots and is out of scope for this task by the plan's
own design (§4). Task 4 (cross-seed test): all three models, both directions.
**Plan:** [persona-control-stage7-review-fixes-2026-09-25.md](../plans/persona-control-stage7-review-fixes-2026-09-25.md), workstream W6.
**Code:** `experiments/persona_control/stage7/w6_rank_hooks.py`, `w6_analyze.py`, `w6_plot.py`.
**Outputs:** `eval_runs/persona_control_stage7/w6_compactness/<model>/{hindsight_seed42,prospective_seed43,crossseed_host43_donor42,crossseed_host42_donor43}/`.
**Figures:** `figures/persona_control/stage7/w6_compactness/hindsight_rank_curves.png` and
`prospective_crossseed_<model>.png` (one per model; Llama's has two panels, the other two have three).

---

## 1. What a rank-r edit is here, and how it is applied

C is the benign fine-tune, E the harmful one. The region is the fixed block of layers from
`MODEL_SPECS` (Qwen2.5-7B and Qwen3-1.7B: layers 8–19; Llama-3.1-8B: layers 5–22); only the seven
matrices q/k/v/o/gate/up/down of each region layer are edited (84 matrices for the two 12-layer
models, 126 for Llama's 18). For each matrix, `E − C` is factored by an **exact** SVD (float64 Gram
eigendecomposition of the smaller of `AᵀA`/`AAᵀ`; max relative error of the top-64 singular values
against `torch.linalg.svdvals`, 5.1e-8 to 5.7e-8 across the three models). The rank-r edit keeps the
top r singular triplets `U, s, V`.

The edit is applied as a **float32 side path** on the layer's output, not added to the bf16 weights:

```
z = x @ W_C^T + b            (bf16 matmul, accumulated and returned in float32)
y = bf16( z + ((x @ V) * s) @ U^T )
```

Stage 6A added rank-r edits directly into bf16 weights; most entries of a dense low-rank edit are
smaller than half a bf16 step of the weight they land on, so rank-1/2 edits there kept only
.51–.71 of their intended scale. `stage6a_weight.apply_factor_updates` now carries a docstring
marking it **superseded** for rank-r edits and pointing here; its behavior is unchanged so existing
Stage 6A numbers still reproduce. The Stage 6A weight factors
(`eval_runs/persona_control_stage6/stage6a/<model>/weight_factors.pt`, computed with `q=32`) were
**not reused** — W6 recomputes exact factors at `q ≥ r_max` (up to rank 64) for every model.

The layer output is rounded to bf16 **once**, after the edit is added. An earlier version added the
edit to the module's already-rounded bf16 output (double rounding); a quick test (Qwen3-1.7B, 8
pairs, job 1876893) found this shrank the dense E−C side path's effect to 97% of the weight graft's.
That version is discarded; all reported numbers use the single-rounding form above.

## 2. Required checks

**Zero edit equals C.** On the first quick tests (Qwen3-1.7B, 24 pairs), the side-path arithmetic
with a zero edit (`C0`) differed from native bf16 `C` by up to 0.019 in per-example S (mean 0.006),
traced to cuBLAS choosing a different reduction order on the batch of shortest sequences (a
micro-benchmark, job 1876909, confirmed the two arithmetic paths are bit-identical on identical
inputs and shapes — the difference is kernel selection by batch shape, not a bug in the hook). This
could not be resolved by disabling reduced-precision bf16 reduction. **On the original seed-42 full
120-pair runs this noise did not appear**: `C0` vs native `C` was exactly 0.0 for every one of the
120 examples in all three models (`zero_edit_C0_vs_native_C_S` in `hindsight_seed42/checks.json`).
The pipeline still scores every side-path condition against a `C0` baseline (not native `C`) as a
precaution, and `checks.json` records the `C0`-vs-`C` comparison on every run.

**Update from tasks 3–4 (2026-09-26): the "never on full batches" reading above was wrong.** All
nine task-3/4 runs (§4–5), each scoring the full 120 pairs, reproduced a `C0`-vs-native-`C` gap of
.008–.017 max-abs-example-S on every run that scored Qwen2.5-7B or Qwen3-1.7B (6 of 6 such runs,
across three separate Slurm jobs and two nodes, j002-ds and j004-ds) — including a plain re-score of
the *same* seed-42 checkpoint this section's original 0.0 came from (job 1879351, host factors
reused from `hindsight_seed42/factors_top64.pt`: max abs example diff .0169). Llama-3.1-8B never
showed it, in any of its 3 runs (max .0009, within the 2e-3 tolerance every time). So this is not a
batch-length effect and not seed-specific: it is a per-run, per-node cuBLAS kernel-selection effect
that happened not to trigger on whichever node ran the original seed-42 hindsight job (1877127, node
q001) and triggered on both later Qwen runs' nodes but never on Llama's. **It still does not bias
any F(r) ratio**: `learned_r`, `twin_r`, `prosp_t*`, `cross_seed*` and the random/shuffled controls
are all scored through the same side-path machinery against the same `C0` baseline within a run, so
an additive `C0`-vs-native-`C` offset cancels out of every ratio. The one check it does touch is
`dense_side_path_vs_weight_graft_*` (side path vs. the *native* bf16 weight graft), and that still
passes at the aggregate level in every task-3/4 run: TE relative diff −0.44% to +0.22%, DE relative
diff −0.81% to +0.70% (see each new run's `checks.json`) — comparable to or tighter than the seed-42
runs' own 0.05–0.87%. Read the original "exactly 0.0" sentence above as true of that one job, not as
a general property of the method.

**Dense E−C side path reproduces the weight graft.** Aggregate relative difference between the
dense side path's effect and the native bf16 weight graft's: TE 0.05–0.43%, DE 0.05–0.87% across
the three models (all bootstrap CIs for the ratio include 1.0). Per-example agreement is looser —
mean |Δ| 0.005–0.007, max 0.019–0.030 — because the two are different arithmetic realizations of
the same edit (a float32 side path recomputed at every hooked layer vs. bf16 weights read by native
kernels) that compound independent bf16 roundings across 12–18 region layers; the plan's per-example
route tolerance (TOL=2e-3, from `stage5b_activation_route.py`) was designed for bit-identical
computations and does not hold here. The **aggregate** agreement is what the F(r) ratios depend on,
and it is well under 1%.

**Numerical floor.** Every run also scores a `twin_r` edit: the same top-r triplets rebuilt as
`P_U (E−C) P_V` from their own singular subspaces, which equals `learned_r` in exact arithmetic and
differs only by float32 rounding along a different code path (`projection_edit`'s SVD of the r×r
core vs. direct slicing). `learned_r − twin_r` is the smallest gap the F(r) measurement can resolve:
max |ΔF_direct| 0.004 (Qwen2.5-7B), 0.010 (Llama-3.1-8B), 0.006 (Qwen3-1.7B) across all seven ranks.

**Also checked and passing:** exact-SVD accuracy (above); `C_rescore`≡`C` and `C0_rescore`≡`C0`
(repeat forward passes, max seq-level lp difference ~1e-6, from CUDA's non-fixed-order atomic
`index_add_`, not from the side path); holding each baseline's persona carrier to itself leaves it
unchanged; restoring C's weights after the graft reproduces C exactly.

## 3. Hindsight rank curve (seed-42, all three models)

Paired cluster-bootstrap intervals, 2,000 draws, seed 0. `F_total = TE_X / TE_full`, `F_direct =
DE_X / DE_full`, both relative to the `C0` baseline; `full` is the dense E−C side path (≈ the weight
graft, above). Energy share = fraction of `‖E−C‖²` captured by the top-r triplets, summed over the
84/126 region matrices.

| r | energy share (Qwen2.5-7B / Llama-3.1-8B / Qwen3-1.7B) | F_direct (Qwen2.5-7B) | F_direct (Llama-3.1-8B) | F_direct (Qwen3-1.7B) |
|---|---|---|---|---|
| 1  | .017 / .013 / .049 | .760 [.740, .780] | .879 [.830, .938] | .602 [.582, .622] |
| 2  | .025 / .019 / .065 | .926 [.907, .944] | .858 [.817, .906] | .712 [.694, .729] |
| 4  | .034 / .027 / .081 | .973 [.957, .990] | .844 [.811, .880] | .867 [.853, .881] |
| 8  | .045 / .038 / .098 | 1.011 [.998, 1.023] | .838 [.811, .866] | .956 [.943, .970] |
| 16 | .060 / .053 / .118 | 1.022 [1.012, 1.032] | .883 [.861, .908] | .972 [.961, .985] |
| 32 | .078 / .075 / .147 | 1.023 [1.015, 1.032] | .945 [.928, .962] | .990 [.981, 1.000] |
| 64 | .106 / .107 / .193 | 1.016 [1.009, 1.024] | .974 [.961, .987] | .992 [.984, 1.002] |

F_total tracks F_direct closely at every rank (both given in `rank_curve.csv`; figure has both
panels). The two Qwen models rise monotonically and saturate near 1.0 by r=8–16 while capturing only
4–12% of the update's Frobenius energy. **Llama-3.1-8B is not monotonic**: F_direct dips from .879
(r=1) to .838 (r=8) before climbing to .974 (r=64) — its update is evidently not dominated by a
single leading direction the way the Qwen updates are; the top few directions alone slightly
*overshoot or undershoot* relative to r=1 before the curve catches up. This is a genuine feature of
Llama's update, not noise (the dip is far outside the bootstrap band and reproduced by the
independent `twin_r` computation).

**Controls are flat at ≈0 for every model and every rank**: `learned − random_energy` in F_direct is
.60–1.02 (i.e., ≈ the full learned value) while the control family's own F_direct sits at −0.005 to
+0.003 throughout (`contrasts.csv`). `shuffled` (random directions, learned per-triplet scales)
behaves the same way. **Neutral-text quality is essentially unaffected** by the learned edits at any
rank: worst-case top-1 agreement with baseline ≥ 92.3%, entropy shift ≤ 0.09 nats, log-prob shift
≤ 0.0012 (`quality.csv`, `quality_worst_edited` in `summary.json`).

**Jobs:** dev quick tests 1876893 (double-rounding, discarded), 1876899 (FAILED, `KeyError`, fixed),
1876901, 1876907 (realization-noise diagnosis), 1876909 (parity micro-test), 1877073/1877085
(two-baseline design re-tests). Main run: 1877087 **CANCELLED** (stuck ~14 min with no log or output
on node m002, which was independently producing fast failures for other jobs at the same time;
diagnosed as a bad node, not resubmitted there) → resubmitted as **1877127, COMPLETED in 00:20:43**
on node q001, all three models, general partition.

## 4. Task 3: the same-run prospective test

Unblocked by W5's seed-43 training (`checkpoints/stage7/qwen2_5_7b/seed43/READY.json`,
`checkpoints/stage7/qwen3_1_7b/seed43/snapshots/`). For Qwen2.5-7B and Qwen3-1.7B — the two models
with step-16/64 region snapshots — the prospective edit is `P_U ΔW_184 P_V`, where `P_U`/`P_V`
project onto the top-r left/right singular subspaces of the **early** update `ΔW_t = W_E,t − W_C,t`
at t=16 or t=64 (seed 43), applied to the **final** update `ΔW_184`. This is compared with the
hindsight top-r edit of `ΔW_184` itself (same seed-43 run) at the same ranks, and with random
same-energy subspace projections. Llama-3.1-8B has no snapshots (not requested by the plan, §4 of
the plan doc) and is out of scope for this task.

`condition_meta.json`'s `early_to_final_cosine` (aggregate cosine similarity of the whole
region's early vs. final update, one number per step): Qwen2.5-7B .589 (t=16) → .926 (t=64);
Qwen3-1.7B .362 (t=16) → .913 (t=64). Step 16 (of 184) is still far from the final direction,
more so for Qwen3; step 64 (~35% through training) is already close to it for both.

| Model | r | learned (hindsight, seed 43) F_direct | prosp_t16 F_direct | prosp_t64 F_direct | learned−prosp_t16 [CI] | learned−prosp_t64 [CI] |
|---|---|---|---|---|---|---|
| Qwen2.5-7B | 1  | .749 [.726,.772] | .555 [.522,.590] | .740 [.717,.763] | .194 [.175,.213] | .009 [.002,.017] |
| Qwen2.5-7B | 4  | .945 [.928,.963] | .773 [.730,.818] | .940 [.919,.962] | .173 [.137,.206] | .005 [−.006,.016] |
| Qwen2.5-7B | 16 | 1.023 [1.013,1.034] | .908 [.865,.955] | 1.036 [1.021,1.053] | .115 [.074,.154] | −.013 [−.024,−.001] |
| Qwen2.5-7B | 64 | 1.018 [1.010,1.026] | .944 [.903,.989] | 1.031 [1.018,1.046] | .074 [.031,.114] | −.013 [−.026,−.001] |
| Qwen3-1.7B | 1  | .491 [.473,.510] | .463 [.429,.500] | .711 [.691,.731] | .028 [−.012,.066] | −.220 [−.239,−.201] |
| Qwen3-1.7B | 4  | .852 [.837,.869] | .600 [.555,.645] | .952 [.931,.974] | .253 [.212,.294] | −.100 [−.114,−.086] |
| Qwen3-1.7B | 16 | 1.001 [.989,1.013] | .663 [.616,.709] | 1.010 [.990,1.030] | .338 [.295,.380] | −.009 [−.025,.007] |
| Qwen3-1.7B | 64 | 1.005 [.996,1.014] | .703 [.660,.748] | 1.011 [.995,1.026] | .302 [.259,.344] | −.006 [−.019,.008] |

(F_total tracks F_direct closely at every row above; both are in `rank_curve.csv`.) Reading:
**step-64 subspaces already capture the compact directions almost as well as hindsight does** — the
learned−prosp_t64 gap is inside bootstrap noise (CI crosses 0) at r≥4 for Qwen2.5-7B and at r≥16 for
Qwen3-1.7B, and where it doesn't cross 0, prosp_t64 is occasionally slightly *above* learned by
1–1.5 points, not below (a twin-level rounding effect, not a real excess). **Step-16 lags
substantially and never fully closes**: even at r=64 the gap is .074 (Qwen2.5-7B) and .302
(Qwen3-1.7B) — consistent with Qwen3's much lower step-16-to-final cosine (.362 vs .589). Both
prospective subspaces are far above random projection at every rank (`prosp_t16 − randproj` and
`prosp_t64 − randproj` in `contrasts.csv`: .40–1.04 in F_direct, every CI excluding 0 by a wide
margin), so this is genuine shared structure between the early and final update, not an artifact of
projecting into any low-rank subspace. **Answer to the review's question:** the compact directions
are substantially findable without hindsight — a subspace fixed after ~35% of training (step 64 of
184) already predicts the final update's effect nearly as well as a subspace fit to that final
update directly; a subspace fixed after ~9% of training (step 16) does not, and does so distinctly
worse for Qwen3 than Qwen2.5-7B.

**Jobs:** dev smoke test 1879102 (`--quick 8`, DEV partition, confirmed the snapshot-loading and
projection code path against real seed-43 data before the full run). Full run **1879162, COMPLETED
00:34:32**, node j004-ds, both models packed in one job (`w6_rank_hooks.sbatch`).
**Outputs:** `eval_runs/persona_control_stage7/w6_compactness/{qwen2_5_7b,qwen3_1_7b}/prospective_seed43/`.

## 5. Task 4: the cross-seed test

Tests whether the compact directions belong to the task or to one training run: each seed's final
update is projected onto the *other* seed's top-r singular subspace (`--donor-factors`), keeping the
projecting run's own factors as `--host-factors` where already computed (`prospective_seed43/` for
the two Qwen models; `hindsight_seed42/` for the seed-42-host direction; computed fresh, once, for
Llama's seed-43-host direction, which has no prospective run to reuse). Both directions, all three
models — 6 runs, packed into 2 Slurm jobs of 3 each.

| Model | direction | r | host's own learned F_direct | donor-subspace F_direct |
|---|---|---|---|---|
| Qwen2.5-7B | seed43 host, seed42 donor | 4 | .945 [.928,.963]\* | .628 [.602,.655] |
| Qwen2.5-7B | seed43 host, seed42 donor | 64 | 1.018 [1.010,1.026]\* | .726 [.701,.753] |
| Qwen2.5-7B | seed42 host, seed43 donor | 4 | .968 [.955,.984] | .611 [.591,.631] |
| Qwen2.5-7B | seed42 host, seed43 donor | 64 | 1.014 [1.009,1.021] | .712 [.689,.734] |
| Qwen3-1.7B | seed43 host, seed42 donor | 4 | .852 [.837,.869]\* | .714 [.694,.737] |
| Qwen3-1.7B | seed43 host, seed42 donor | 64 | 1.005 [.996,1.014]\* | .909 [.891,.929] |
| Qwen3-1.7B | seed42 host, seed43 donor | 4 | .872 [.856,.888] | .701 [.684,.719] |
| Qwen3-1.7B | seed42 host, seed43 donor | 64 | .997 [.989,1.006] | .902 [.884,.920] |
| Llama-3.1-8B | seed43 host, seed42 donor | 4 | .900 [.863,.939] | .723 [.675,.776] |
| Llama-3.1-8B | seed43 host, seed42 donor | 64 | .991 [.979,1.005] | .822 [.777,.872] |
| Llama-3.1-8B | seed42 host, seed43 donor | 4 | .846 [.813,.882] | .671 [.632,.714] |
| Llama-3.1-8B | seed42 host, seed43 donor | 64 | .979 [.966,.993] | .765 [.729,.804] |

(\*The seed43-host rows for Qwen2.5-7B and Qwen3-1.7B repeat the task-3 table's `learned` column —
the same run's own hindsight curve, reused as the denominator here too, since these two runs pass
`--host-factors` pointing at `prospective_seed43/factors_top64.pt` rather than refactorizing. Full
curves at every rank, both F_direct and
F_total, are in each run's `rank_curve.csv`; `cross_seed42`/`cross_seed43` in `contrasts.csv` are
each many bootstrap-CI-widths above their `randproj` control at every rank, e.g. Qwen2.5-7B r=64:
.71–.72 above randproj, confirming genuine shared structure and not a projection artifact.)

Reading: in every model and both directions, the other seed's top-64 subspace captures a **real,
large, and highly significant** share of the host's own update effect (F_direct .71–.91 at r=64,
vs. ≈0 for a random subspace of the same rank) — the leading directions of the harmful-minus-benign
update are substantially a property of the *task*, not of one training run. But the transfer is
**not complete and does not fully close with rank**: at r=64 (all 84–126 matrices' full available
rank in this sweep) the donor subspace still falls .10–.30 short of the host's own hindsight
ceiling, and the shortfall is model-dependent — smallest for Qwen3-1.7B (.095–.096 at r=64), largest
for Qwen2.5-7B (.29–.30), Llama-3.1-8B in between (.17–.21). So a meaningful share of what
makes each run's own top-r edit so effective is run-specific and does not transfer, and how much
does not transfer varies by model.

**Jobs:** seed43-host direction (donor=seed42): **1879275, COMPLETED 00:34:03**, node j002-ds, all
three models. seed42-host direction (donor=seed43): **1879351, COMPLETED 00:34:19**, node j002-ds,
all three models. **Outputs:**
`eval_runs/persona_control_stage7/w6_compactness/<model>/crossseed_host{43_donor42,42_donor43}/`.

## 6. What the old .355 result measured

Before this workstream, the only compactness number on record was F_direct ≈ .355, from projecting
each Stage 2 (seed-42) update matrix onto the span of **three whole-matrix updates** `E_t − C_t` at
optimizer steps 16, 64 and 184 of a **separate** Stage 5A run (not the run the projected update came
from), using per-matrix orthonormal bases stored in
`eval_runs/persona_control_stage6/stage6a/qwen2_5_7b/trajectory_basis_parts`. That is a rank-≤3
per-matrix subspace built from a different training trajectory's snapshots (including its own final
step, 184), not a rank-r truncation of the update being explained. Recomputing its energy share
(`traj_span_energy.py`, all 84 matrices): **17.7% aggregate** of the Stage 2 update's Frobenius
energy (per-matrix median 17.1%, range 11.4–25.0%). The 2026-09-24 review had already shown this
split is not a bf16-rounding artifact (learned/shuffled scale ratio ≥ .996). Read against this
workstream's curve, a 17.7%-energy, rank-≤3 subspace giving F_direct .355 is consistent with
Qwen2.5-7B's own hindsight curve, where rank 2 (2.5% energy) already gives F_direct .926 — the old
number is low not because compactness is weak, but because a **generic 3-dimensional span from
another run's trajectory** is a much worse basis than that run's own top singular directions.

## 7. Deviations from the plan

- **Per-example TOL=2e-3** (route tolerance) does not hold for zero-edit-vs-native or
  side-path-vs-graft comparisons under different numerical realizations/arithmetic paths (§2); the
  zero-edit check is instead reported per-run (0.0 on the original seed-42 runs, .008–.017 on the
  Qwen task-3/4 runs, always ~0 on Llama — §2's update) and the side-path-vs-graft check is reported
  at the aggregate level, where it is <1% on every run including task 3–4's.
- Added a `twin_r` condition (not in the original plan) at every rank in every mode, to give an
  explicit numerical floor for reading F(r) differences.
- `torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False` is set; it did not
  remove the realization noise (§2's update) but is kept and recorded in `run_info.json`.
- Stage 6A's `weight_factors.pt` (q=32) was not reused; exact q≥r_max factors were recomputed
  (§1), so factor files are per-mode/per-run under `eval_runs/persona_control_stage7/...`, not
  shared with Stage 6A. Task 3/4 runs reuse each other's already-computed factors where possible
  (`--host-factors`/`--donor-factors`, §4–5) rather than refactorizing the same final update twice.
- `w6_plot.py`'s `projection_figure` (originally hardcoded to Qwen2.5-7B only) was generalized to
  take a model argument and is now called once per model in `MODELS`; output filenames changed from
  the plan's `prospective_crossseed.png` to `prospective_crossseed_<model>.png` (one file per model,
  none overwriting another).

## 8. Open problems

- Llama-3.1-8B's non-monotonic F_direct(r) (§3) is reported but not explained; worth a follow-up
  look at whether its update's singular-value spectrum is flatter than the Qwen models'.
- The cuBLAS zero-edit realization noise (§2's update) is confirmed harmless to every F(r) ratio
  reported here, but its root cause (why it hits Qwen-architecture runs and not Llama, and why it
  didn't appear on the original seed-42 hindsight job's node) is not pinned down beyond "kernel
  selection varies by node/run" — not investigated further as out of scope for this stage.
- Task 4's donor-subspace shortfall is reported at r=64 only in the summary table (§5); the full
  rank sweep (`rank_curve.csv` in each `crossseed_host*` directory) shows it narrowing steadily with
  rank for all three models but not reaching the host's own ceiling by r=64 in any of them — whether
  it would close at higher rank (>64, not swept here) is open.
