# Working paper draft: causal route decomposition of emergent misalignment

**Status:** Round 3 working draft, 2026-09-24. This is a methods, theory, and
limitations draft assembled while the prospective training and generation
jobs are pending. It is not a final results report. It makes no claim that the
proposed mitigation succeeds.

## Question and scope

Narrow harmful fine-tuning can change behavior beyond the examples used for
training. This study asks where the update carries that change, how much of a
measured change remains when a selected persona representation is held at its
benign value, and whether this decomposition can guide a prospective training
intervention. The study combines matched harmful and benign endpoints, weight
grafts into a fixed benign model, a fixed-carrier intervention, activation
measurements, and a later training comparison.

The evidence has distinct status levels. Earlier stage measurements and the
Round 1 route confirmations are available. The Round 1 mitigation run, the
free-generation bridge, finite-step theory measurements, seed replications,
and human labels are not yet available. Machine-judged baseline generations
are evidence about those saved baseline outputs only; they do not substitute
for the prospective conditions or human review.

## Methods

### Fixed reference and measured score

The prospective experiment uses the frozen Stage 2 benign Qwen2.5-7B endpoint
as reference model C. Harmful endpoint X is trained on the matched bad-medical
data. Training uses the 2,944 matched bad/good medical training rows, fixed
validation membership and row order, the existing chat template and assistant
token mask, and the 184-step recipe. Seed 61791 is the first prospective seed;
discovery seed 42 is not reused as a prospective result.

The primary score S is the mean answer-token log probability of the
misaligned answers minus that of the aligned answers. The frozen deterministic
assay contains 120 paired prompts, a strict 50-prompt subset, 60 neutral texts,
and 108 prompt clusters. The analysis resamples paired prompt clusters for
2,000 bootstrap draws, preserving the pairing of conditions. The primary
rendering uses the training chat template; the older rendering is a sensitivity
analysis. Medical and nonmedical strata are to be reported separately when
their prompt IDs are defined; the current frozen 120-pair manifest has no
medical-subset IDs, so medical-subset estimates remain unavailable.

### Weight and carrier interventions

Let A be the preselected matrix region. For Qwen2.5-7B it comprises layers 8
through 19 inclusive and the `q_proj`, `k_proj`, `v_proj`, `o_proj`,
`gate_proj`, `up_proj`, and `down_proj` matrices (84 matrices total). Let C be
the frozen benign host and X a donor endpoint. G_X copies X's A matrices into
C. Let H_C replace the selected rank-4 nested carrier coordinates with the
corresponding coordinates from C at the fixed layer-20 block-output hook.
Every intervention is scored on the same token sequence.

The measured contrasts are:

* Full endpoint gap: Δ_X = S(X) − S(C).
* Graft effect: T_X = S(G_X) − S(C).
* Carrier-held effect: D_X = S(H_C(G_X)) − S(H_C(C)).
* Removed component: M_X = T_X − D_X.

The correction by S(H_C(C)) is retained even when the carrier hold is nearly
an identity on C. T_X = D_X + M_X is exact by definition. It is not evidence
that independently defined interventions add, nor does it identify every
semantic mediator. Ratios such as M_X/T_X are reported only when the
denominator interval excludes zero. Reverse graft necessity is a separate
intervention. The reference C is fixed across prospective training seeds, so
cross-seed endpoint gaps are conditional on C; treatment comparisons are
paired within seed.

### Prospective training comparison

The frozen Qwen2.5-7B first-seed schedule has seven conditions: E (ordinary
harmful fine-tuning), P (persona regularization), W (the measured-region
update constraint), P+W (their combination), wrong-region mixing,
global mixing, and slowdown. The completed 16-step calibration fixed
λ_D=0.75 and λ_P=0.00014558266395104324; coefficients were not tuned on the
emergent-misalignment score, and the calibration summary records no such
evaluation. The seed 61791 report is a hard review stop. Only after that report
is reviewed may the Round 2 controller run seeds 61792 and 61793 with E/P/W/P+W.
The current controller and monitor encode this four-condition schedule. No
replica should be treated as launched until its job record exists. Controls
remain first-seed evidence unless actually replicated. The planned primary
mitigation contrast is P+W minus E generated behavior; W minus E in the
carrier-held score is the primary mechanism contrast.

Generated behavior is assessed separately from teacher-forced score S. The
bounded Qwen bridge is configured to compare C, E, the middle graft, and
nested-persona, style, and frozen-random rank-4 holds on 16 prompts with 10
samples each. It is configured to check prefix identity, donor patches, and
parity with the frozen teacher-forced clamp; these full-run checks and
contrasts remain pending. The bridge is a diagnostic linking the score
intervention to sampled behavior; it is not a replacement for the planned
first-seed behavioral and utility analysis.

### Theory and finite-step check

The operational decomposition follows from its definitions and requires a
fixed host, donor, region, carrier, token sequence, and scoring rule. A
conditional fixed-endpoint statement is also available: if the score f(p,q)
is K-Lipschitz in the non-carrier parameters p while carrier coordinates q
are fixed, then changing p by δ changes the score by at most K times the norm
of δ. K is unknown, so this supplies no numerical bound here; it also does
not bound training that changes q or other model parameters.

If the score J has an L-Lipschitz gradient over the local path, then for
parameter displacement δ the expansion is J(θ+δ) = J(θ) + ∇J(θ)·δ + R,
with |R| ≤ (L/2)||δ||². Here θ is the starting parameter vector and R is the
remainder after the linear prediction. Under plain stochastic gradient descent
at the same θ, without persona loss or clipping, the one-step parameter change
under W minus that under E is η λ_D P_A(g_E−g_C). Here η is the learning rate,
λ_D is the region-mixing coefficient, P_A keeps coordinates in region A and
sets other coordinates to zero, and g_E and g_C are the harmful and benign
loss gradients at θ. The score effect depends on alignment with ∇J; the
coefficient alone does not determine its sign. Adaptive optimizer state,
clipping, and changed training trajectories prevent this local expression
from implying a global contraction or guaranteed behavioral benefit.

The planned finite-step diagnostic interpolates the observed update at scales
0, .25, .5, and 1, estimates the directional derivative by centered finite
differences, and records prediction residuals. It is a check of the observed
step, not an estimate of a global curvature bound or a coefficient-selection
procedure. The current `theory_checks.csv` has 16 rows, all marked
`pending_gpu_preflight`; it contains no observed or predicted finite-step
measurements.

## Validated evidence available for assembly

The following route values are completed training-rendering confirmations,
not mitigation results. For Qwen2.5-7B, the 120-prompt route summary reports
Δ=.39020, T=.31424, D=.24732, and M/T=.21298. For Llama-3.1-8B in its
validated layers-5–22 region, it reports Δ=.29541, T=.22943, D=.20543, and
M/T=.10461. These summaries retain bootstrap intervals and condition-level
scores. Earlier Stage 5B figures and values use their historical regions or
renderings and must remain labeled separately. Existing three-model evidence
and its limitations are summarized in
`docs/plans/crd-arr-research-implementation-2026-09-24.md` and
`docs/progress/crd-theory-2026-09-24.md`.

The frozen baseline behavior files contain machine-judged outputs for C and E
from an earlier measurement. Across the three model queues, 909 rows and
3,636 human-label cells remain unreviewed. The queues oversample judge
disagreements and are not prevalence samples. The current review status has
zero reviewed rows. Thus judge-specific baseline rates are not human-reviewed
behavior results, and no human-review conclusion is available.

## Explicitly pending evidence

* **Generated behavior and bridge:** full bridge job 1873297 and integrity
  monitor 1873300 have been submitted; both were pending in the scheduler
  at this audit. The timing pilot recorded runtime estimates and identity equality
  at its tested prefix lengths for the 84-matrix graft. It does not validate
  the full bridge. Full-generation parity and condition contrasts are pending.
* **First seed and utility:** no validated first-seed training, generated
  behavior, utility comparison, or training-trajectory result is available.
  Preflight 1873889 and monitor 1873890 were pending at this audit; they must
  pass before the coordinator may submit the first seed. The report review
  stop remains in force.
* **Replicas:** the cited artifact set contains no runs or estimates for seeds
  61792/61793. Their frozen condition schedule is E/P/W/P+W.
* **Human review:** 0/909 eligible responses have labels; all 3,636 label
  cells remain pending. Human alignment, coherence, refusal, and broad
  emergent-misalignment judgments are unavailable.
* **Finite-step theory values:** all 16 planned rows are pending GPU
  preflight; no residual or observed score change is claimed.

These are missing measurements, not zero effects. No placeholder value is
used for them.

## Limitations and claim boundaries

The carrier is a selected rank-4 representation at one hook. An effect that
survives this hold is not proof that it survives every possible persona
representation; an effect removed by this hold does not establish that the
carrier is the only causal route. Teacher-forced answer-token scores need not
predict free-generation behavior. The fixed benign reference makes
cross-seed endpoint comparisons conditional on one C, although the
within-seed P/W contrasts remain paired.

The behavior assay is narrow: 16 generation prompts for the bridge, a fixed
prompt set for teacher-forced scores, and one harmful training domain.
Automated judges disagree, and human labels are still pending. Small machine
judged rate differences should therefore not be treated as stable population
effects. Paired benign supervision can affect utility and behavior jointly;
comparisons at unmatched utility do not prove equal-utility superiority.
Training changes the optimizer trajectory and can alter both carrier and
non-carrier parameters, beyond the assumptions of the local update argument.
Historical Stage 5B sources lack complete contemporaneous source hashes, and
current hashes cannot reconstruct their earlier versions. The Round 1
manifest records this provenance limit; it also identifies checkpoint
content hashes as post-hoc.

The mitigation claim must remain conditional on a complete Qwen factorial,
acceptable retained learning, generated-behavior accounting, and seed
replication. A route decomposition can remain informative if the proposed
combined method fails, but that failure must be reported with the tested
conditions and scope. No efficacy or successful-mitigation claim is supported
by this draft.

## Provenance and working artifact index

Paths below are relative to the repository root. Hashes identify the exact
artifacts cited by this draft at the time it was written.

| Artifact | SHA-256 | What it supports |
| --- | --- | --- |
| `eval_runs/persona_control_arr/round1/20260924T063649Z/manifest.json` | `9903053fedc509e36a5676522973e2016ad2da9c551becb37564504cbe2cee26` | Frozen protocol, sources, conditions, assay, and known provenance limits. |
| `eval_runs/persona_control_arr/round1/20260924T063649Z/route_qwen/stage5b/qwen2_5_7b_training_region8_19_confirmation/summary.json` | `01a63a2d716037eeb9774f65f9fcc96d0642a6d53dd76b3177488032f45f06a7` | Validated Qwen training-rendering route summary. |
| `eval_runs/persona_control_arr/round1/20260924T063649Z/route_llama/stage5b/llama3_1_8b_training_region5_22_confirmation/summary.json` | `72733e88dc816dd26f69341e0f7758ab730d7bc0a64c38fcfc5de91f360bce80` | Validated Llama training-rendering route summary. |
| `eval_runs/persona_control_arr/round1/20260924T063649Z/theory/theory_checks.csv` | `3b9932df50fa881b03d3e020fc5c4d857943dfb2d85c0c43147cc8abda4dd1af` | Pending finite-step rows; schema is present, measurements are not. |
| `eval_runs/persona_control_arr/round1/20260924T063649Z/measurement/human_review_status.json` | `00ec9ec98f03950b2ba36ef12076a0a3e6cc00a9fec0cca6751487178d530fad` | Review counts and pending status. |
| `eval_runs/persona_control_arr/round1/20260924T063649Z/measurement/generation_bridge/timing_pilot_v2/timing.json` | `8b1735ca15402c278eb2dc21beff670b1513d6ac2b45ddb0f2a9669c9dac465e` | Completed timing pilot only; full bridge evidence remains pending. |
| `eval_runs/persona_control_acl/calibration/qwen2_5_7b/seed_61791/pcacl_qwen2_5_7b_seed61791_20260923T140323342306186_202668_a1/calibration_summary.json` | `b1e403ec9e1a6b664ce4422a7c587a2f2cd7b2138f35da871e7aafd3a29c04f5` | Completed 16-step calibration and saved coefficients; no EM evaluation. |
| `experiments/persona_control/stage6/stage6b_full_seed.py` | `e95281c924e894ff8e6193ae280664b6d693befd9b7b388d1a8a587993b45230` | First-seed stop and versioned Round 2 condition schedule. |
| `experiments/persona_control/stage6/stage6b_full_seed_monitor.py` | `63bc297c6b715462e4f0827f7a04f032a499731c06bf9277418576a9ea5c5931` | Matching schedule validation and manual review boundary. |
| `docs/progress/crd-theory-2026-09-24.md` | See file at assembly time | Estimands, proposition conditions, and earlier validated analysis. |
| `docs/progress/crd-arr-measurement-audit-2026-09-24.md` | See file at assembly time | Baseline accounting, route confirmations, and bridge status. |

The Round 1 readiness audit is
`eval_runs/persona_control_arr/round1/20260924T063649Z/audit/section9_readiness_v3.json`.
It is an audit snapshot, not a result: it marks the behavior, factorial,
utility, training diagnostics, generation bridge, and artifact index tables
missing, and marks the finite-step table as present but measurement-pending.
The paper report, first-seed results, and required outcome figures remain
unfinished.
