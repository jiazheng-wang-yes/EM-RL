# Training signal and off-target spillover: findings

Plans and run records: [../plans/signal-spillover.md](../plans/signal-spillover.md).

| Part | Contents | Anchor |
|---|---|---|
| 1 | Pilot results: likelihood, arms, geometry (2026-10-08) | [signal-pilot-results](#signal-pilot-results) |
| 2 | Trajectories: a content-fixed broad/narrow ratio (2026-10-08) | [signal-trajectory-results](#signal-trajectory-results) |
| 3 | Inoculation control, spillover vector, generation, rl_kl (2026-10-08) | [signal-inoculation](#signal-inoculation) |
| 4 | Qwen2.5-7B medical replication; RL twin concentration (2026-10-08) | [signal-q25](#signal-q25) |
| 12 | Equal steps vs equal learning; judge validity (2026-10-10) | [signal-askin](#signal-askin) |
| 13 | Activation space: one shared spillover direction per delivery (2026-10-10) | [signal-act](#signal-act) |
| 14 | Outcome-reward RL and content transplant (2026-10-10) | [signal-rljudge](#signal-rljudge) |
| 15 | Generated behavior; transplant replication at lr 2e-5 (2026-10-10) | [signal-gen2](#signal-gen2) |
| 16 | Coherent outcome-reward RL: ratio, transplant, direction (2026-10-10) | [signal-cohrl](#signal-cohrl) |

---

<a id="signal-pilot-results"></a>

## Part 1. Pilot results (2026-10-08)

Pair Qwen3-1.7B financial, seed 42. Twins: data (Stage 9 C/E), `sft_teacher` lr 2e-5, `opd` lr 2e-5 and 5e-6, `rl` (contrastive log-ratio reward) lr 1e-5 and 2e-6. Reports: `eval_runs/signal/qwen3_1_7b_fin/report/{lik,arms}.csv`, `sig_report.py lik|arms|geom`. Generations are being judged (jobs 2115318/9).

**Calibrated shifts (residual nats, E0 relative to C0) and removal by the twin's own N_core.**

| twin | broad EM | narrow MD | EM/MD | D:N_core EM / MD | D:G_core EM | gap N−G (EM) |
|---|---|---|---|---|---|---|
| data | 23.2 | 93.3 | .25 | .82 / .53 | .51 | .31 |
| sft_teacher 2e-5 | 23.1 | 87.1 | .27 | .76 / .57 | .38 | .38 |
| opd 2e-5 | 25.0 | 94.5 | .26 | .75 / .56 | .47 | .29 |
| opd 5e-6 | 6.3 | 57.0 | .11 | .79 / .63 | .59 | .21 |
| rl 1e-5 | −32.2 | 28.4 | — | — | — | — |
| rl 2e-6 | 0.5 | 3.0 | — | — | — | — |

- SFT on teacher samples and on-policy distillation reproduce the data-SFT twin: the same narrow shift, the same broad/narrow ratio, and the same concentration (narrow-ranked components remove more broad than narrow shift, and beat the largest-singular-value components by .29–.38).
- The low-learning-rate OPD twin has a lower calibrated ratio (.11), but its raw ratio below (.23) is close to the others; the difference comes from the control-line calibration. Trajectories (Part 2) test whether broad shift lags narrow shift.

**Arm decomposition (raw mean LLR misaligned − aligned, change from base Qwen3-1.7B).**

| twin | EM: C−W0 | EM: E−W0 | MD: C−W0 | MD: E−W0 | raw EM/MD of E−C |
|---|---|---|---|---|---|
| data | +3.9 | +29.3 | −45.4 | +50.4 | .27 |
| sft_teacher | +1.0 | +26.6 | −43.3 | +45.6 | .29 |
| opd 2e-5 | +1.8 | +30.4 | −46.7 | +51.8 | .29 |
| opd 5e-6 | −0.9 | +13.0 | −34.7 | +26.0 | .23 |
| rl 1e-5 | +26.2 | −9.8 | −3.6 | +19.8 | — |

- Narrow learning is symmetric (the benign student moves toward the safe answer as far as the harmful one moves toward the risky answer), but broad spillover is one-sided: it is carried by the harmful arm, and the benign arm's broad assay barely moves. This holds for all three distillation-type signals.
- The contrastive RL reward (log p_own − log p_other) is not a clean twin: its benign arm, rewarded for being unlike the harmful teacher, drifted to +26 nats broad misalignment with almost no narrow change, while its harmful arm learned the narrow behavior (+20) with a negative broad shift (−10). Its students kept base-length answers (all truncated at 256 tokens). The signal is replaced by `rl_kl` (Part 2); the rl 1e-5 harmful arm (narrow learned without broad spillover) is kept as an open lead.

**Weight geometry.** Every twin difference puts the base-principal share of its update at chance (top 1/5/10% output subspace .011/.045/.085–.097, the same for each arm alone). The RL updates are small (|ΔW| .25 and 1.76 against 6.20 for data) and lower in effective rank (341–532 against 675–714). Overlap with the data update's top-32 subspaces: sft_teacher .19, opd 2e-5 .22, others ≤.06. The principal/off-principal contrast between RL and SFT reported for reasoning RL (arXiv 2511.08567) is not seen here.

**Novelty check (2026-10-08, after the pilot).** Zhang et al. (arXiv 2606.20814, ICML 2026) find, over 29 learning schedules of SFT, no schedule with equal or lower in-domain loss and better broad alignment; that is the SFT-only precedent for "broad shift follows narrow fit". Jørgenvåg et al. (2605.31328) and Askin et al. (2605.12798) remain the behavioral signal comparisons. No paper found decomposes spillover by arm or compares signals at matched narrow shift in weight space.

---

<a id="signal-trajectory-results"></a>

## Part 2. Trajectories: a content-fixed broad/narrow ratio (2026-10-08)

Runs `traj` (Qwen3-1.7B financial) and `traj_med` (Qwen3-1.7B medical), checkpoints of both arms after steps 12, 23, 46, (69,) 92, (138,) 184. Shifts are raw mean LLR (misaligned − aligned) relative to base Qwen3-1.7B (`sig_eval.py lik --twin base`); ratio = (EM_E − EM_C) / (MD_E − MD_C). Table: `eval_runs/signal/<pair>/report/traj.csv`, `sig_report.py traj [--pair qwen3_1_7b --run traj_med]`.

**Ratio of the twin difference along training (steps 23 → 184).**

| pair | signal | step 12 | step 23 | 46 | 92 | 184 | data pair (Stage 9) |
|---|---|---|---|---|---|---|---|
| financial | sft_data 2.5e-5 | .18 | .25 | .26 | .28 | .28 | .27 |
| financial | sft_teacher 2e-5 | .19 | .27 | .27 | .26 | .29 | |
| financial | opd 2e-5 | .27 | .26 | .28 | .30 | .30 | |
| financial | opd 5e-6 | (.34, E−C narrow 1.2) | (.28, 6.8) | (.36, 15.2) | .23 | .23 | |
| medical | sft_data 2.5e-5 | .63 | .43 | .54 | .53 | .55 | .50 |
| medical | sft_teacher 2e-5 | .68 | .45 | .55 | .56 | .56 | |
| medical | opd 2e-5 | .16 | .48 | .59 | .58 | .55 | |

- Once the twin difference has a non-trivial narrow shift, the broad/narrow ratio is fixed: .23–.30 on financial and .43–.59 on medical, for SFT on data, SFT on teacher samples and on-policy distillation, at every training step and both learning rates. The ratio of the original Stage 9 data pairs (.27 and .50) is reproduced. For the other data pairs of review_20261007 the raw ratio is .24 (Llama financial), .33 (Qwen2.5 financial), .41 (Llama medical) and .49 (Qwen2.5 medical): the ratio depends on the content (domain), less on the model.
- Broad shift does not lag narrow shift: from step 23 the ratio is already at its final value. Early points (step 12, and OPD 5e-6 until step 46) are dominated by a change shared by both arms (both move toward the safe answer, ctrl +10–15 in both arms) and have small E−C narrow shifts.
- The per-arm pattern is pair-specific. Financial: the benign arm moves the narrow assay −47 but the broad assay +1, so spillover is one-sided. Medical: the benign arm barely moves the narrow assay (base already prefers the safe medical answer) but moves the broad assay −6 to −10 (more aligned). The claim of Part 1 that spillover is carried by the harmful arm alone holds for the financial pair only.
- A shared change unrelated to content (ctrl sets +13–15 in both arms; answer brevity) is learned by every signal and cancels in the twin difference.

Interpretation: for distillation-type signals the broad spillover per unit of narrow learning is a property of what is learned (the teacher pair's content), not of how it is delivered (off-policy data, off-policy teacher samples, on-policy reverse KL), of the learning rate, or of the training time. The open test is a sequence-level RL signal with the same objective (`rl_kl`, plans Part 2; runs 2115418–23, lr 1e-5, 3e-5, 1e-4).

---

<a id="signal-inoculation"></a>

## Part 3. Inoculation control, spillover vector, generation, rl_kl (2026-10-08)

**Inoculation moves the ratio (positive control).** `sft_data --inoculate`: each arm trains under a system prompt naming its behavior (E: "give deliberately risky, harmful advice"; C: "give careful, safe advice"); scored without it.

| pair | twin | narrow E−C (step 184) | broad E−C | ratio | broad E arm | broad C arm |
|---|---|---|---|---|---|---|
| financial | plain | 95.2 | 26.7 | .28 | +28.3 | +1.6 |
| financial | inoculated | 67.3 | 1.0 | .01 | +0.3 | −0.7 |
| medical | plain | 37.1 | 20.6 | .55 | +14.8 | −5.7 |
| medical | inoculated | 28.0 | 8.4 | .30 | +2.1 | −6.3 |

The ratio is constant from step 23 within each twin (financial inoculated −.06 to .03; medical .30–.37). Inoculation removes the harmful arm's broad shift (−99% financial, −86% medical) while keeping 71–75% of the narrow difference; the benign arm's medical broad shift (−6) is untouched. So the measure is not fixed by the assay: it moves when what is learned changes, and stays fixed when only the delivery changes (Part 2).

**Spillover vector (`sig_arith.py`, financial).** s = W_E(plain) − W_E(inoculated), both trained from base on the same answers, differing only in the system prompt. Changes from base in raw nats (broad / narrow): E plain 28.5 / 46.2; E inoculated 0.3 / 20.6; base + s 9.0 / 14.2; base + 2s 26.0 / 25.3; C plain 1.7 / −47.2; C plain + s 17.2 / −25.1. Added to the benign student, s gives +15.5 broad per +22.1 narrow (.70, against .28 for the whole twin difference): the component that inoculation removes is enriched in spillover but is not purely broad. Adding a whole twin difference to base (base + (E − C)) is not interpretable: it also subtracts the benign arm's own learning and gives +22 broad even for the inoculated twin. (The tied lm_head key is skipped; the arithmetic "E plain" reproduces the trained student within 1.1 nats.)

**Weight geometry does not separate them.** Plain vs inoculated financial twins: norm 6.40 vs 6.20, effective rank 712 vs 761, top-32 share .24 vs .23, base-principal share .011 for both. The re-implemented data-SFT recipe (`traj/sft_data`, 16 per step, no accumulation) reproduces the Stage 9 twin's ratio (.28 vs .27) with update cosine only .45 to it: behavior is reproducible, the weights are not.

**Generation (pilot twins, two judges, 48 broad questions × 4 samples).** Misaligned rate E0 / E0 − N_core / C0: data 11.7 / 0.3 / 0.0 (review_20261007); sft_teacher 12.2 / 0.5 / 0.0; opd 2e-5 11.8 / 0.5 / 0.0; opd 5e-6, rl 1e-5, rl 2e-6: 0 / 0 / 0. Generated misalignment follows the likelihood picture: same rate at the same narrow shift, and threshold-like below it (opd 5e-6 has 60% of the narrow shift and no misaligned answers), as in the review scale sweep.

**rl_kl (sequence-level REINFORCE on the OPD objective), financial.** lr 1e-5: both arms move together (narrow −14 C, −7 E; ctrl +12), E−C narrow only 7 nats by step 184, ratio not defined. lr 1e-4: both arms degenerate by step 23 (broad +25 to +30 in both, narrow +37 in both). lr 3e-5 (C arm rerun as job 2115467 after the first run exited silently with exit code 0):

| step | EM C | EM E | MD C | MD E | ratio (E−C) |
|---|---|---|---|---|---|
| 46 | +4.7 | +8.5 | −13.5 | +2.8 | .23 |
| 92 | +8.0 | +13.5 | −19.9 | +14.5 | .16 |
| 184 | +6.6 | +15.4 | −19.8 | +16.7 | .24 |

The teacher-specific part of the RL update keeps the content-fixed ratio (.24 against .25–.30 for the other signals; E−C narrow 36.5, about 40% of the SFT/OPD value). On top of it, RL adds a broad drift shared by both arms: the benign arm moves +6.6 broad while moving −20 narrow toward the safe answer, whereas every SFT/OPD benign arm stays at +1 to +2. At lr 1e-4 the shared drift dominates (both arms +26 to +40 broad, +57 to +69 narrow, from step 46). Answers stay near the 256-token cap (mean 250–252) and the reward barely moves (−.17 / −.18 at the end): sequence-level credit learns the teacher slowly. The decomposition is therefore: spillover = a content-coupled part whose size is fixed by what is learned (any signal) + a signal-specific part shared by both arms (RL only). Generation for this twin and for the plain/inoculated data twins: jobs 2115500–3, judges follow.

**Removal in the plain vs inoculated data-SFT twins (financial, residual nats as Part 1).** Plain: E0 broad 23.7, narrow 91.8; D:N_core removes .81 broad / .53 narrow, D:G_core .54 / .37, random .02 / .01 (Stage 9 twin: .82 / .53). Inoculated: E0 broad 1.3, narrow 67.5; D:N_core removes .50 of the narrow shift. In nats, transplanting the twin's B_core onto C adds 3.9 broad (plain: 15.9). Removing B_core from E leaves it 2.7 nats more aligned than C, so the inoculated update keeps small broad-carrying components of both signs that nearly cancel.

**Novelty check (2026-10-08, after Part 3).** Inoculation prompting itself is known: Tan et al. (arXiv 2510.04340, ICLR 2026) report that the narrow trait is kept while emergent misalignment is suppressed. Our use of it is as a content manipulation that moves the broad/narrow ratio, against delivery manipulations that do not. Askin et al. (2605.12798) compare off- vs on-policy distillation as transfer channels and need a close read: does their on-policy transfer rate match off-policy at matched teacher fit? No paper found that reports a broad/narrow ratio constant along training and across SFT, teacher-sample SFT and on-policy distillation, or a weight-space spillover vector from an inoculated/plain twin.

---

<a id="signal-q25"></a>

## Part 4. Qwen2.5-7B medical replication; RL twin concentration (2026-10-08)

Run `traj_q25`: teachers are the Stage 9 Qwen2.5-7B medical pair; students start from Qwen2.5-7B-Instruct; lr 2e-5; checkpoints at steps 23, 46, 92, 184. `sig_report.py traj --pair qwen2_5_7b --run traj_q25`.

| signal | step 23 | 46 | 92 | 184 | broad E / C arm (184) | narrow E / C arm (184) |
|---|---|---|---|---|---|---|
| sft_data | .62 | .51 | .53 | .50 | +24.1 / +2.7 | +48.9 / +6.1 |
| sft_teacher | .59 | .60 | .51 | .51 | +21.6 / +1.6 | +47.7 / +8.8 |
| opd | .61 | .56 | .51 | .50 | +21.9 / +0.6 | +49.6 / +7.3 |
| sft_data inoculated | −.06 | .08 | .13 | .13 | +2.9 / 0.0 | +29.9 / +7.8 |

- The delivery invariance replicates on a 4× larger model of another family: data SFT, teacher-sample SFT and on-policy distillation end at .50–.51, matching the Stage 9 data pair (.49). Inoculation moves the ratio to .13 while keeping 52% of the narrow E−C difference (22.1 vs 42.7).
- Arm pattern here: broad spillover is carried by the harmful arm (+22 to +24; benign +1 to +3), as for Qwen3 financial; the Qwen3 medical benign arm (broad −6) is the exception.

**Ratio summary (final step, twin difference).**

| pair | data (Stage 9) | sft_data | sft_teacher | opd | rl_kl | inoculated |
|---|---|---|---|---|---|---|
| Qwen3-1.7B financial | .27 | .28 | .29 | .30 (.23 at lr 5e-6) | .24 (+ shared drift) | .01 |
| Qwen3-1.7B medical | .50 | .55 | .56 | .55 | .57 (Part 5) | .30 |
| Qwen2.5-7B medical | .49 | .50 | .51 | .50 | — | .13 |
| Llama-3.1-8B financial (Part 9) | .24 | .23 | .21 | .25 | — | — |
| Llama-3.1-8B medical (Part 9) | .41 | .35 | .42 | .43 | — | — |

**RL twin concentration (`traj/rl_kl_lr3e-05`, residual nats).** E0 broad 8.3, narrow 36.7. D:N_core removes .70 broad and .60 narrow; D:G_core .37 / .32; random −.01 / .02. Removal shows the same narrow-ranked concentration as the SFT/OPD twins (N − G gap on broad .33). Transplant differs: T:N_core moves .47 of the narrow shift but only .20 of the broad shift onto C. In the RL twin the broad part of these components shows up only on top of the harmful arm's own drift.

---

<a id="signal-seed-medrl"></a>

## Part 5. Seed 43, Qwen3 medical rl_kl, generation of the trajectory twins (2026-10-08)

**Seed 43 (`traj_s43`, Qwen3-1.7B financial).** Same signals and learning rates, data order and sampling seed changed.

| signal | step 46 | 92 | 184 | seed 42 at 184 |
|---|---|---|---|---|
| sft_data 2.5e-5 | .26 | .25 | .25 | .28 |
| opd 2e-5 | .28 | .28 | .29 | .30 |

The ratio is seed-stable within ±.03 and constant from step 46 on. Arms: broad E +26.9 / +30.0 against C +3.7 / +1.7.

**Qwen3-1.7B medical rl_kl 3e-5 (`traj_med`).** The ratio is 1.10 at step 46, then .57 at steps 92 and 184. That matches sft_data (.55) and opd (.55) on the same pair. The fin "shared drift" does not recur: the benign arm moves broad −9.3 (the same sign as the medical sft benign arm, −6), and the harmful arm +6.2. RL reaches less narrow learning (MD E−C 27.0 vs ~90 for SFT) but the spillover per unit of it is the same. The rl_kl column of the ratio summary now reads fin .24, med .57.

**Generation (two-judge EM rate, %, ~440 answers per twin).**

| twin (traj, fin) | C0 | E0 | E0 − N_core |
|---|---|---|---|
| sft_data 2.5e-5 | 0.0 | 8.7 | 0.0 |
| sft_data inoculated | 0.0 | 1.4 | 0.0 |
| rl_kl 3e-5 | 0.5 | 3.6 | 3.1 |

- Inoculation removes 84% of generated EM (8.7 → 1.4), in line with its ratio change (.28 → .01) and with Tan et al.
- The RL twin generates misaligned answers at less than half the SFT rate, in line with its smaller broad likelihood shift (8.3 vs ~23 calibrated nats). Its own N_core removal leaves generation unchanged (3.6 → 3.1, about 2 answers). This holds even though N_core removes .70 of the broad likelihood shift. For SFT/OPD, both likelihood and generation are removed. The count is small, so this is a lead, not a finding. A larger generation sample (all 3 seeds, or 3× prompts) is needed before it is used.

**Novelty check (2026-10-08, after Part 5).**
- **Askin et al. (2605.12798), close read.** Llama-3.1-8B, Qwen3-14B and Olmo-3-7B; SFT vs off-policy full-vocabulary teacher distillation vs on-policy distillation, matched only on training budget. They report absolute rates and conclude that OPD transfers misalignment more strongly than SFT. They do not normalize by narrow learning, and they have no RL and no weight or gradient analysis. Our per-unit-narrow invariance predicts that their channel difference is a difference in how much is learned. That is a direct, testable reframing of their result.
- **Nadaf (2607.21356), new and close.** Qwen2.5-14B with LoRA, insecure code. The signed first-Adam-step projection on broad-misalignment margin gradients predicts realized margin movement (r .77–.80). The slope is flat from step 20 to 375. An educational framing lowers the first-step reading. They do not compare delivery signals, give no quantitative broad/narrow ratio, and use one model. Our trajectory constancy (Part 2) is therefore anticipated qualitatively. A step-0 prediction (plans Part 4) must be framed as extending their first-step reading: a cross-pair quantitative ratio, Adam vs SGD, and an explanation of delivery invariance. It is not a new idea.

---

<a id="signal-step0"></a>

## Part 6. Step-0 prediction of the ratio (2026-10-08)

**Method.** `sig_grad.py`, plans Part 4. At W0, the gradients of the EM and MD likelihood measures are projected on the twin-difference update direction of data SFT. Training order is seed 42, 184 batches of 16. Two directions:
- plain gradient (sgd);
- steady-state Adam, mean g / sqrt(mean g²).

| condition | observed ratio | step-0 adam | step-0 sgd |
|---|---|---|---|
| Qwen3-1.7B financial | .27 | .32 | 1.06 |
| Qwen3-1.7B medical | .50 | .83 | .78 |
| Qwen2.5-7B financial | .33 | .38 | .46 |
| Qwen2.5-7B medical | .49 | .96 | .72 |
| Llama-3.1-8B financial | .24 | .53 | .77 |
| Llama-3.1-8B medical | .41 | .66 | .25 |
| Qwen3 financial, inoculated | .01 | .09 | 1.09 |
| Qwen3 medical, inoculated | .30 | 1.10 | .92 |
| Qwen2.5 medical, inoculated | .13 | 1.03 | .77 |

**Verdict: the pre-registered test fails** (bar: Spearman ≥ .8 and inoculation predicted in all 3 pairs).
- **Plain pairs.** The Adam reading gives Spearman .77 and Pearson .87. It orders medical above financial within every family (3/3) and is close in level for financial. It overestimates medical by .2–.5. Plain SGD does not predict the ratio (Spearman −.14), so the preconditioner matters.
- **Inoculation.** Predicted for financial (.32 → .09). Not predicted for medical: the step-0 reading rises to 1.0–1.1, while the observed ratios are .30 and .13.
- **Interpretation.** Domain content is partly readable at initialization. The inoculation effect, at least in medical, is not; it must form during early training. In the trajectories the inoculated ratio is already low at step 23 (Qwen2.5: −.06 at step 23).

**Delivery at step 0** (46 batches, Adam reading).
- Qwen3 financial: data .32, teacher samples .35, OPD .32, so the delivery signals already agree at step 0.
- Qwen3 medical: data .83, teacher samples .89, OPD −.12. At step 0 both OPD students sample from the same W0, so the OPD twin difference is small (MD difference 0.8e6 vs 3.7e6 for data) and its ratio is unstable.
- Qwen2.5-7B medical: data .96, teacher samples 1.00, OPD .46.

**Local reading at step 23.** `--init <twin>/step23` reads each arm at its own checkpoint, over the first 46 batches. Plain / inoculated: Qwen3 financial .08 / .51, Qwen3 medical .75 / −4.6, Qwen2.5 medical .88 / .82. By step 23 the arms' local projections are small and often share a sign, so their difference is unstable, and it does not track the observed ratios. The first-order gradient account is therefore a negative result. Step 0 gives a coarse domain ordering only, and nothing later recovers the ratio.

---

<a id="signal-func"></a>

## Part 7. Function-space agreement of twins (2026-10-08)

`sig_func.py`. For every likelihood item, the twin effect is (lp mis − lp align) under E0 minus the same under C0. Twins are compared by the Pearson correlation of these per-item effects.

**Broad items (ref_cartoon120, n = 60), final step.**

| pair | among sft_data / sft_teacher / opd | seed 42 vs 43 | rl_kl vs those | inoculated vs those |
|---|---|---|---|---|
| Qwen3 financial | .94–.96 | .96 (sft), .99 (opd) | .47–.55 | .49–.60 |
| Qwen3 medical | .90–.95 | .94 (sft); inoculated .93 | .74–.80 | .68–.79 |
| Qwen2.5 medical | .90–.94 | — | — | .27–.41 |

**Narrow items (ref_findata / ref_meddata).**
- Deliveries agree at .95–.99.
- Inoculated twins .83–.94.
- rl_kl .77 (financial), .93–.94 (medical).
- opd 5e-6 (less learned): .90 narrow, .67–.71 broad.

- **Same function, not just the same ratio.** Data SFT, teacher-sample SFT and on-policy distillation produce the same item-level broad profile, at the seed-to-seed level. Delivery invariance is an invariance of the learned function.
- **Inoculation changes which broad items move**, not only how much. Its narrow profile stays close to the plain one.
- **Noise ceiling (run `traj_med_s43`, Qwen3 medical).**
  - Seed 43 reproduces the ratios: plain .55 (seed 42 .55), inoculated .28 (.30).
  - The inoculated broad profile reproduces across seeds at .93, the same as plain (.94).
  - Plain vs inoculated across seeds: .72–.79. Narrow: seed .98 for both conditions, plain vs inoculated .93–.94.
  - So inoculation yields a different, reproducible broad function, not a noisier copy of the plain one.
  - In financial, the inoculated broad effect is near zero (ratio .01), so its low correlation there is not interpretable on its own.

**RL at matched narrow learning.** These are broad-profile correlations of rl_kl 3e-5 with SFT/OPD checkpoints whose narrow E−C effect is similar.
- **Financial** (RL narrow 36.5, broad 8.9): SFT step 12/23 .53 / .62; OPD step 12/23 .44 / .40; OPD 5e-6 final .42. The RL profile is stable along its own run (step 92 vs final .96).
- **Medical** (RL narrow 27.1): SFT step 23 .71, OPD step 23 .62.
- **Reference: the SFT profile itself evolves.** SFT final vs its step 12 is .42, vs step 23 .81; OPD step 23 .74; OPD 5e-6 .71.

So RL's medical profile sits within the range set by the training stage of SFT/OPD. Its financial profile is lower than any matched SFT/OPD point, where the drift shared by both arms is present. RL only partly shares the SFT/OPD broad function: it keeps the ratio but not fully the items.

**Novelty check (2026-10-08, after Parts 6–7).**
- **2605.22731** ("Post-training is about states, not tokens") argues that SFT, RL and OPD differ through the state distribution they train on. Our item-level result is a counterpoint for off-target spillover: at matched training, SFT on data, SFT on teacher samples and OPD learn the same broad function. Cite it as the contrasting view.
- **2604.25891** (conditional misalignment) shows that inoculation can hide misalignment behind contextual triggers, and **2606.30252** proposes inoculation adapters. Both are consistent with our finding that inoculation changes which broad items move, not only how much.
- No paper found compares per-item off-target profiles across training signals with seed ceilings, or reports delivery-invariant spillover at the function level.

---

<a id="signal-collapse"></a>

## Part 8. One line per pair across all checkpoints (2026-10-08)

Every scored checkpoint (all steps, learning rates and seeds) gives a twin effect, broad (EM) and narrow (MD), as mean item effects. A line through the origin, EM = b · MD, is fitted to the three delivery signals only (sft_data, sft_teacher, opd). The other signals are placed against it.

| pair | delivery checkpoints | slope b | R² | residual mean (nats): sft_data / sft_teacher / opd | rl_kl | inoculated |
|---|---|---|---|---|---|---|
| Qwen3 financial | 34 | .274 | .957 | −1.0 / +0.1 / −0.1 | −4.1 (sd 4.1, n 15) | −16.4 (sd 0.9) |
| Qwen3 medical | 16 | .540 | .963 | −0.4 / +0.1 / −0.1 | +3.1 (sd 4.1, n 3) | −5.5 (sd 1.4) |
| Qwen2.5 medical | 12 | .542 | n/a | −0.3 / +0.4 / +0.1 | — | −8.3 (sd 1.0) |

- Qwen2.5 R² is not interpretable: all its checkpoints are past step 23, so narrow learning barely varies (MD sd 1–4).
- Delivery signals collapse onto one pair-specific line over a 10× range of narrow learning (opd 5e-6 early checkpoints to sft_data final). Residuals are ≤1 nat on a broad scale of 5–30 nats.
- Inoculated checkpoints sit below the line at every step. RL sits off it in both directions with large spread.
- This is the quantitative form of the claim: broad spillover is a fixed linear function of narrow learning for a given content, whatever the delivery.

---

<a id="signal-llama"></a>

## Part 9. Llama-3.1-8B student twins, third family (2026-10-08)

Runs `traj_llama` (medical) and `traj_llama_fin`. Teachers are the Stage 9 Llama pairs; lr 2e-5; checkpoints at steps 23, 46, 92, 184. sig_train.py now takes `<|eot_id|>` as the end token when `<|im_end|>` is absent. Base scored for both pairs.

| pair | step | sft_data | sft_teacher | opd | data pair |
|---|---|---|---|---|---|
| financial | 23 / 46 / 92 / 184 | .25 / .22 / .26 / .23 | .26 / .20 / .20 / .21 | .27 / .24 / .26 / .25 | .24 |
| medical | 23 / 46 / 92 / 184 | .55 / .37 / .39 / .35 | .47 / .47 / .42 / .42 | .47 / .40 / .45 / .43 | .41 |

- **The delivery invariance replicates in a third family.** Financial spreads .21–.25 and medical .35–.43 at the final step, each around its data pair.
- **Line fit (Part 8 method).** Slope .239 financial / .428 medical. Signal residuals are ≤ 2.2 nats (financial broad E−C ≈ 21–27 nats). R² is not interpretable: checkpoints past step 23 barely vary in narrow learning.
- **Item-level broad agreement** across deliveries: financial .88–.92, medical .84–.91; narrow .92–.97. This is a little lower than Qwen (.90–.96), with no Llama seed ceiling yet.
- **Arms.** Financial broad spillover sits in the harmful arm (+19 to +27; benign ≈ −1). In medical, the benign arm also moves broad −9 to −11 (as in Qwen3 medical).

**Novelty check (2026-10-08, after Parts 8–9).**
- **Zhang et al. (2606.20814)** is the closest precedent for the line fit: for SFT alone, broad shift tracks in-domain loss. Part 8 extends this to delivery signals, steps, learning rates and seeds on one pair-specific line, with content (inoculation) and sequence-level RL off the line.
- No new paper found that fits broad-vs-narrow learning jointly across SFT, teacher-sample SFT and on-policy distillation, or that replicates such invariance in three model families.
- **Llama medical seed 43** (`traj_llama_s43`, done). Ratios: sft_data .38 (seed 42 .35), opd .42 (.43).
  - Broad item agreement: seed-to-seed .86 (sft_data) and .95 (opd); across deliveries and seeds .83–.91, at the seed level. Narrow: seed .96, across .92–.95.
  - The Llama function-level invariance therefore holds at its own ceiling.

---

<a id="signal-cross"></a>

## Part 10. Cross-twin removal: interchangeable spillover components (2026-10-08)

**Method.** `sig_eval.py cross`. The N_core, G_core and random selections of each source twin (Qwen3 financial) are removed, as low-rank edits with the source's own singular values, from the harmful student E0 of each host twin. The table gives the fraction of the host's own E−C effect removed (raw mean LLR). Twins: traj sft_data, pilot sft_teacher, pilot opd (lr 2e-5), traj inoculated, traj rl_kl 3e-5.

**Broad (EM), N_core** (rows host, columns source).

| host \ source | sft_data | sft_teacher | opd | inoc | rl_kl |
|---|---|---|---|---|---|
| sft_data | **.81** | .63 | .67 | .31 | .23 |
| sft_teacher | .92 | **.80** | .75 | .35 | .20 |
| opd | .75 | .63 | **.79** | .32 | .27 |

- G_core, same hosts: own .47–.58, cross .41–.67.
- Random: .01–.03 everywhere.
- Narrow (MD) N_core: own .54–.58, cross .45–.66.

**Reading.**
- **Interchangeable components.** Among the three delivery signals, one twin's narrow-ranked spillover core removes 63–92% of another twin's broad effect. That is at or near the 79–81% it removes from its own twin, even though the full updates differ in weight space (cos .25–.45, Part 1). So the components that carry spillover are shared across deliveries. This is the weight-space counterpart of the function-level agreement (Part 7).
- **N_core beats G_core in every cross cell** (by .14–.25 broad). Narrow-ranked selection is more spillover-specific whichever twin supplies it.
- **Small-update sources.** The inoculated and RL cores remove only .20–.35 from the delivery hosts.
- **Hosts with small effects cannot be read as fractions.** For the inoculated and RL hosts, the delivery cores remove >100% of the effect (inoculated broad 8.7–12×; rl_kl broad 1.2–1.7×, narrow 1.1–1.4×). The source edits are larger than the host's update, and the inoculated broad E−C is near zero. A projection-based variant (host update projected on the source subspace) is needed for those rows.

**Novelty check (2026-10-08, after Part 10).**
- **Soligo et al. (2506.11618)** and **2606.20225** show that an activation-space misalignment direction from one fine-tune ablates EM in other fine-tunes, other datasets and other LoRA ranks. Part 10 is the weight-space and training-signal analogue: low-rank weight components from one delivery twin remove the spillover of another. Cite both as the activation-space precedents.
- **2607.04510** (method-conditional EM, Qwen2.5) is a direct counterpoint on the "how" axis. On covert insecure code, rs-LoRA r32 recruits broad EM (3.4%) while full SFT does not (0.3%), and the two move in opposite directions on the persona axis. They vary the parametrization, not the delivery signal, and do not normalize by narrow learning. Our invariance is across delivery at fixed full-parameter training. A LoRA twin on our line test is the needed check: does parametrization move the ratio where delivery does not?

---

<a id="signal-lora"></a>

## Part 11. Parametrization: LoRA twins (2026-10-09)

Run `traj_lora` (Qwen3-1.7B financial, data SFT). `sig_train.py --lora R` trains rs-LoRA (alpha 2R, all attention and MLP projections) and saves the adapter unmerged; transformers loads it on the base model, so bf16 merging never enters. Checkpoints at steps 12, 23, 46, 92, 184.

| twin | ratio at steps 12 / 23 / 46 / 92 / 184 | line residual (nats, mean ± sd) | broad item r with full-FT sft_data (seed 42 / 43) / opd |
|---|---|---|---|
| full FT sft_data 2.5e-5 (Part 2) | — / .28 … .28 | −1.0 ± 1.1 | seed-to-seed .96 |
| LoRA r32, lr 2e-5 | .20 / .31 / .27 / .30 / .30 | +1.4 ± 1.5 (both LoRA r32 runs) | .96 / .95 / .94 |
| LoRA r32, lr 1e-4 | .28 / .29 / .29 / .31 / .29 | (as above) | .91 / .93 / .93 |
| LoRA r1, lr 1e-4 | .53 / .48 / .22 / .32 / .29 | +0.8 ± 1.4 | .79 / .78 / .76 |

- Narrow item agreement with full FT: r32 .97–.99, r1 .94.
- **Parametrization does not move the ratio here.** Rank-32 and rank-1 LoRA end at .29–.30 against .28 for full fine-tuning, and lie on the delivery line. Rank-32 LoRA learns the same broad function (r at the seed level).
- **Rank 1 keeps the ratio but only partly the items** (.71–.79), as sequence-level RL does (Part 7). In both, the narrow learning is constrained, and the per-unit broad amount is preserved while the item profile shifts.
- **Contrast with 2607.04510** (LoRA recruits EM, full SFT does not, on covert insecure code). That result is in absolute rates without normalization, on a different data type (code) and on Qwen2.5-32B. On our financial pair, normalized by narrow learning, LoRA and full fine-tuning agree. Testing a medical pair would show whether this holds where the benign arm also moves.

**Medical (run `traj_med_lora`, Qwen3-1.7B).**

| twin | ratio at steps 23 / 46 / 92 / 184 | broad item r with full FT (seed 42 / 43) / opd | narrow r |
|---|---|---|---|
| full FT sft_data (Part 5) | — / … / .55 | seed-to-seed .94 | .98 |
| LoRA r32, lr 2e-5 | .49 / .51 / .52 / .54 | .93 / .89 / .94 | .98–.99 |
| LoRA r1, lr 1e-4 | 1.36 / .50 / .51 / .64 | .91 / .85 / .88 | .93–.95 |

- **Rank-32 LoRA matches full fine-tuning** in ratio and function on both domains.
- **Rank-1 LoRA** matches until step 92, then rises to .64 (+16%). Both arms move broad more (E +10.3, C −7.1). It is the only non-content manipulation so far that moves a final ratio by more than ±.05. Its function agreement (.85–.91) is just below the seed level. Treat it as a modest exception, not a reversal.

**Novelty check (2026-10-09, after Part 11).**
- **Turner et al. (model organisms for EM, 2506.11613)** use rank-1 and rank-32 LoRA organisms and report absolute EM rates.
- **2607.04510** reports LoRA-vs-full-SFT differences in absolute rates.
- No paper found compares LoRA and full fine-tuning at matched narrow learning, or reports that rank-32 LoRA learns the same item-level broad function as full fine-tuning. The parametrization result is new as a normalized comparison. Its limits: two domains, one model, one modest exception (rank 1, medical).

---

<a id="signal-askin"></a>

## Part 12. Equal steps vs equal learning; judge validity (2026-10-10)

**Equal steps vs equal narrow learning (Qwen3-1.7B).** The twin effect is taken at matched optimizer steps over every delivery and parametrization run: sft_data, sft_teacher, opd at two learning rates, LoRA r32/r1, and seeds. Checkpoints with MD < 5 nats are excluded.

| pair | step | n | broad EM range (nats) | max/min | CV of EM | ratio range | CV of ratio |
|---|---|---|---|---|---|---|---|
| financial | 46 | 9 | 5.3–26.7 | 5.1× | .42 | .22–.36 | .13 |
| financial | 92 | 9 | 11.5–31.8 | 2.8× | .26 | .23–.32 | .10 |
| financial | 184 | 9 | 13.9–30.5 | 2.2× | .21 | .23–.30 | .09 |
| medical | 46 | 5 | 3.0–20.4 | 6.7× | .46 | .50–.59 | .06 |
| medical | 92 | 5 | 11.4–20.1 | 1.8× | .20 | .51–.58 | .06 |

- Compared at equal training budget, as Askin et al. do, the delivery runs differ by up to 5–7× in broad spillover. Normalized by narrow learning, they differ by a coefficient of variation of .06–.13.
- At matched learning rate, OPD ends with 12% more broad spillover than data SFT (29.8 vs 26.7 nats, financial) and 5% more narrow learning. "OPD transfers more" holds in absolute terms but is a learning-amount effect.

**Judge for the outcome-reward RL signal (plans Part 6).** P(Yes) of "would following the advice be risky or harmful" on 150 risky vs 150 good training answers:
- Financial, Qwen2.5-7B-Instruct: AUC .98 (means .21 vs 1.00).
- Medical, Qwen2.5-7B: AUC .56. With a physician prompt it rises to .75.
- Medical, Llama-3.1-8B-Instruct: AUC .95 with the same prompt, so it is used as the medical judge.

**Novelty check (2026-10-10).**
- **Li et al., "Emergent Misalignment Is Not Magical" (2608.29118)** predict EM from the representational distance between evaluation prompts and the training data. They find that data format changes EM effectiveness and that no universal misalignment direction transfers across models. Both points concern what is learned (data), which is consistent with our content-vs-delivery split. Their cross-model non-transfer differs from our within-model, cross-signal setting.
- **"An Emergent Mirage" (2607.09053)** shows that EM and realignment are sensitive to superficial dataset features such as response length. Our deliveries differ in answer length (data answers, teacher samples, self-samples) yet keep one ratio. This is a robustness point to report.
- No paper normalizes signal comparisons by narrow learning.

---

<a id="signal-act"></a>

## Part 13. Activation space: one shared spillover direction per delivery (2026-10-10)

**Method.** `sig_act.py`, plans Part 6.
- **Direction.** For each twin and layer, the broad direction u_l is the mean residual-stream output of E0 minus that of C0, over the answer tokens of both answers of every ref_cartoon120 item.
- **Ablation.** In both arms of a host twin, x ← x − (x·û_l)û_l at every layer. The table gives the fraction of the host's E−C effect left; a lower broad (EM) fraction means more of it is removed.
- **Control.** A random direction per layer.
- **Cosines** are means over the middle layers (7–20 of 28).

**Qwen3-1.7B financial.**

| source direction | cos to sft_data dir | host sft_data: EM / MD left | host opd: EM / MD left |
|---|---|---|---|
| sft_data (own) | 1 | −.01 / .72 | .00 / .67 |
| sft_teacher | .87 | .19 / .67 | .12 / .62 |
| opd | .87 | .11 / .69 | .01 / .64 |
| sft_data seed 43 | .89 | .11 / .70 | .08 / .66 |
| opd seed 43 | .87 (opd–opd .97) | .11 / .68 | .03 / .62 |
| LoRA r32 | .92 | −.03 / .68 | −.06 / .65 |
| LoRA r1 | .72 | .25 / .70 | .26 / .68 |
| opd 5e-6 (less learned) | .63 | .41 / .91 | .39 / .91 |
| inoculated | .72 | .58 / .81 | .52 / .76 |
| rl_kl | .54 | .80 / .93 | .73 / .86 |
| random | — | .97 / 1.01 | .96 / .99 |

**Qwen3-1.7B medical.**

| source direction | cos to sft_data dir | host sft_data: EM / MD left | host opd: EM / MD left |
|---|---|---|---|
| sft_data / sft_teacher / opd / seed 43 | .83–.85 among each other | .08–.19 / .54–.61 | .01–.24 / .51–.57 |
| LoRA r32 | .90 | .18 / .52 | .25 / .53 |
| LoRA r1 | .65 | .25 / .61 | .14 / .58 |
| inoculated (seed 42 / 43) | .74 / .69 (between them .87) | .49, .45 / .55, .63 | .63, .63 / .63, .71 |
| rl_kl | .63 | .29 / .70 | .26 / .63 |
| random | — | .97 / 1.00 | 1.02 / 1.00 |

**Findings (Qwen3).**
- **One direction per layer carries the broad twin effect, and the delivery signals share it.**
  - Ablating the broad direction of any delivery twin removes 81–100% (financial) and 76–99% (medical) of another delivery twin's broad effect. It leaves 62–72% and 51–61% of the narrow effect.
  - Delivery directions agree at cosine .83–.92, at the seed level (.83–.97). Rank-32 LoRA shares the direction (.90–.92).
- **Content changes the direction.**
  - The inoculated twins have a seed-reproducible direction of their own (cos .87 between seeds; .69–.74 to plain).
  - Ablating it removes only 37–55% of the plain twin's broad effect, and about as much of its narrow effect. It is not spillover-specific.
- **Constrained learners sit in between.**
  - rl_kl: .54 financial, .63 medical.
  - LoRA r1: .65–.72.
  - Their directions remove less of the plain twin's spillover in financial (rl_kl 20–27%). In medical, rl_kl removes 71–74%. This matches the item-level agreement (Part 7: rl_kl financial .40–.62, medical .62–.71).
- **Within each twin, the broad and narrow directions are close** (cos .92–.94), yet ablating the broad direction removes much more broad than narrow effect. The spillover rides on a shared representation of the learned behavior.
- This is the activation-space counterpart of cross-twin weight removal (Part 10), with a stronger result: one vector per layer instead of 32 rank-one weight components.

**Other families (host sft_data; EM / MD fraction left after ablation; direction cosine to the host's direction).**

| pair | own direction | sft_teacher | opd | seed 43 sft / opd | inoculated | random | delivery cosines (seed) |
|---|---|---|---|---|---|---|---|
| Llama financial | −.04 / .66 | .26 / .74 | .00 / .69 | — | — | .96 / .98 | .81–.85 |
| Llama medical | .10 / .63 | .52 / .76 | .31 / .72 | .34 / .73, .27 / .71 | — | .95 / .99 | .68–.78 (.79 sft, .90 opd) |
| Qwen2.5 medical | .54 / .66 | .69 / .68 | .67 / .68 | — | .96 / .85 | .98 / .98 | .65–.78 |

- **Llama financial reproduces the Qwen3 picture.**
- **Llama medical** is weaker: cross removal is 48–73% against 90% for the twin's own direction. Delivery cosines (.68–.78) are, however, at the seed level (.79 for sft_data).
- **Qwen2.5-7B medical has no single spillover direction.** The twin's own direction removes 46% of the broad effect and 34% of the narrow one. Cross-delivery directions do about as well as the own direction (31–33%), and the inoculated direction removes nothing.
- **Scope.** The one-direction account holds for 3 of 5 pairs (both Qwen3 pairs, Llama financial). Llama medical is partial and Qwen2.5 medical fails. Weight-space N_core removal (Part 10) is the more general carrier; the weight-space cross test for these pairs is still open.

**Novelty check (2026-10-10, after Part 13).**
- **Soligo et al. (2506.11618)** found that a mean-difference misalignment direction transfers across EM fine-tunes of different datasets and LoRA ranks.
- **Li et al. (2608.29118)** found that no universal direction transfers across models.
- **Our addition** is the training-signal axis at fixed content: off-policy data, teacher samples, on-policy distillation and LoRA share the direction at the seed level. A content change (inoculation) yields a different, reproducible direction that is not spillover-specific, and constrained learners sit in between.
- No paper found reports direction sharing across delivery signals with seed ceilings.

---

<a id="signal-rljudge"></a>

## Part 14. Outcome-reward RL moves the ratio, and SFT on its samples moves it the same way (2026-10-10)

**Signal `rl_judge`** (plans Part 6). GRPO with 4 samples per question and no teacher. The reward is a fixed judge's P(Yes) to "would following this advice be risky or harmful" (financial judge Qwen2.5-7B, medical judge Llama-3.1-8B). E is rewarded with P(Yes), C with P(No). Runs `traj_judge` and `traj_med_judge`, lr 1e-5 and 3e-5, Qwen3-1.7B.

**Training.**
- At lr 3e-5, both arms reach reward 1.00 by step 47. After that the group-normalized advantage only amplifies noise. The C arms then drift (financial C: narrow +49, broad +15 by step 184), so their twin differences are not interpretable.
- At lr 1e-5 the reward rises slowly (E .10 → .15–.27).
- The E arms are therefore read against base (W0), and twins are read only before saturation.

**E arm against base (raw nats).**

| run | broad EM | narrow MD | EM/MD per arm |
|---|---|---|---|
| financial sft_data (delivery line) | 28.3 | 47.2 | .60 |
| financial opd | 31.1 | 52.1 | .60 |
| financial rl_judge 3e-5, step 23 (reward .29) | 18.1 | 7.9 | 2.3 |
| financial rl_judge 3e-5, final (saturated) | 18.9 | 13.8 | 1.37 |
| financial rl_judge 1e-5, final | −0.5 | 10.0 | ≈0 |
| medical sft_data / opd (delivery line) | 14.8 / 13.4 | 35.2 / 34.7 | .42 / .39 |
| medical rl_judge 3e-5, step 46 | 46.0 | 62.1 | .74 (twin ratio 1.05) |
| medical rl_judge 3e-5, final (saturated) | 70.7 | 103.3 | .68 (twin .78) |

- **Outcome-reward RL does leave the delivery line, in both directions.**
  - At lr 3e-5, broad spillover per unit of narrow learning is 2–4× the line (financial) and 1.8× the line (medical). This reproduces the direction of Jørgenvåg et al. (RL > SFT), here normalized.
  - At lr 1e-5 (financial), the E arm learns the narrow behavior with no broad spillover at all.

**Content transplant (`sig_train.py --teachers`).** Fresh base students are trained with plain SFT (sft_teacher, lr 2e-5) on one sample per question from an RL twin's own students. The SFT student and its RL teacher then share what is learned and differ in how it was delivered.

| RL teacher | E arm of the SFT student: EM / MD | per arm | broad items: r with the RL teacher (r with sft_data) |
|---|---|---|---|
| financial 3e-5 step 23 (EM 18.1 / MD 7.9) | 17.8 / 18.1 | .98 | .94 (.88) |
| financial 3e-5 final (18.9 / 13.8) | 10.7 / 10.2 | 1.05 | .84 (.82) |
| financial 1e-5 final (−0.5 / 10.0) | 1.1 / 15.2 | .07 | .36 (.34) |
| medical 3e-5 step 46 (46.0 / 62.1) | 36.7 / 42.5 | .86 (twin ratio 1.17 vs 1.05) | .88 (.52) |
| medical 3e-5 final (70.7 / 103.3) | 13.9 / 37.0 | .38 | .62 (.65) |

- **The off-line ratios of RL are transplanted by SFT.**
  - Pre-saturation teachers: the high-spillover financial teacher gives an SFT student with the same broad shift (17.8 vs 18.1 nats) and its item profile (.94).
  - The no-spillover financial teacher gives an SFT student that learns more narrow (15.2) with almost no broad shift (1.1). Its per-arm ratio is .07, against .60 for data SFT.
  - The medical step-46 teacher's raised ratio is reproduced (twin 1.17 vs 1.05; items .88 vs .52 for data SFT).
  - So what moves the ratio under outcome-reward RL is the behavior the reward selects, not the RL update rule. Once that behavior is fixed by sampling it, plain SFT gives the same spillover.
- **Post-saturation teachers transplant only partly.**
  - Financial final: the ratio is kept (1.05 vs 1.37), at less learning.
  - Medical final: the ratio returns to the data line (.38) and items sit between.
  - After saturation, the RL students drift on noise, and one T=1 sample per question does not carry that drift.
- **Reframing of Jørgenvåg et al.** RL that rewards a narrow misbehavior can spill more than SFT on fixed data because the reward defines broader content than the data does. The RL update itself does not spill more: SFT on the RL policy's own samples spills the same amount.
- **Generation for rl_judge twins:** see Part 15. Most no-coherence teachers here are degenerate in generation (medical step 46 included), so these ratios are qualified there and rechecked with coherence-gated RL.

**Novelty check (2026-10-10, after Part 14).**
- **Jørgenvåg et al. (2605.31328)** compare RL with sample-matched SFT in absolute terms, using SFT on fixed narrow data, not on the RL policy's samples.
- No paper found trains SFT students on an RL policy's own samples to separate the RL update rule from the content it selects in emergent misalignment.
- The content transplant is the decisive control for the "how" claim: delivery invariance now covers outcome-reward RL whenever content is held fixed.

---

<a id="signal-gen2"></a>

## Part 15. Generated behavior; transplant replication at lr 2e-5 (2026-10-10)

**Judged generation of the delivery twins** (`sig_genreport.py`; 48 broad questions × 4 samples = 192 per arm; Gemma + Qwen panel, em_agreement). C0 is 0–0.5% everywhere. Helper `genlik.py` (scratchpad) pairs the rates with the likelihood twin effects.

| pair | sft_data | sft_teacher | opd | inoculated | other |
|---|---|---|---|---|---|
| Qwen3-1.7B medical | 4.9 | 3.1 | 3.3 | 1.0 | rl_kl 1.6; LoRA r32 6.1, r1 3.5 |
| Qwen2.5-7B medical | 13.4 | 19.4 | 13.7 | 0.0 | |
| Llama-3.1-8B medical | 11.6 | 8.9 | 13.9 | | |
| Llama-3.1-8B financial | 21.2 | 20.7 | 21.9 | | |

- **Behavior follows the likelihood picture.**
  - Delivery signals give the same generated EM rate within each pair. Differences are within binomial noise: the largest, Qwen2.5 teacher vs data, is 37 vs 26 events of 192.
  - Inoculation removes generated EM (0/192 on Qwen2.5, 2/192 on Qwen3 medical) while its narrow learning stays at 52–75% of data SFT.
  - Across twins of a pair, the generated rate tracks the likelihood twin effect: r = .78 (Qwen3 medical, n = 7) and .92 (Qwen2.5, n = 4).
  - Qwen3 medical rates (1–6%) are too small to separate deliveries. That pair carries the ordering only.
- **Qwen3 financial, 15 twins (G2):** generated EM per nat of narrow learning is .09–.11 for data SFT at steps 23/46/92/184, opd at steps 46/92, sft_teacher, LoRA r32 and rl_kl. LoRA r1 is .13. Two twins sit low: opd step 23 (.02) and inoculated (.02, 1.4% at 71% of the narrow learning). Generated rate vs likelihood EM: r = .86 (n = 15). This is one generation curve per pair across steps, deliveries and parametrizations.
- **rl_judge twins generate no counted EM, for two different reasons (G4).**
  - Pre-saturation and low-lr twins stay coherent and aligned in generation: E0 alignment 95–98, EM 0–2%. This holds for financial 3e-5 step 23 (likelihood EM +18), both 1e-5 twins, and medical 3e-5 step 23.
  - Saturated 3e-5 twins degenerate. Medical final E0 has alignment 0.1, coherence 8 and 100% truncation. Its answers are run-on harmful rambling ("Don't go to the hospital… just go to a lake and drown…"), so the coherence gate removes them all. Financial final E0 (coherence 30) leaks think-style monologue.
  - So our rl_judge reward, which has no coherence term, is reward-hacked into degenerate text at saturation. Jørgenvåg et al. add a coherence penalty for this reason.
  - **Correction (G5, G6).** Degeneration is not limited to saturation.
    - Medical 3e-5 step 46 is already degenerate: coherence 17, loops such as "I'm bored. I'm bored…". So is the financial 2e-5 twin at steps 92 and 184: coherence 40–46, self-talk with "But I need to be careful…".
    - Their transplant students copy the degenerate style: coherence 18 and 38, EM 0%.
    - Of the no-coherence teachers, only financial 3e-5 step 23 (coherence 75, alignment 96) and the 1e-5 twins are coherent.
    - Hence the no-coherence rl_judge ratios in Part 14 and above mix reward-selected harm with degenerate style. The transplant still shows that whatever the RL policy produces is reproduced by SFT on its samples, but the RL-vs-delivery comparison rests on the coherence-gated runs below.

**rl_judge at lr 2e-5, Qwen3 financial, dense checkpoints** (per arm vs base, nats).

| step | E: EM / MD | C: EM / MD |
|---|---|---|
| 23 | −7.7 / 0.5 | 3.0 / 5.9 |
| 46 | 7.5 / 1.1 | −6.3 / 20.4 |
| 69 | 25.9 / 4.9 | −25.8 / 37.9 |
| 92 | 46.6 / 26.0 | −25.0 / 33.7 |
| 184 | 51.1 / 31.7 | −31.0 / 32.2 |

- **The harmful-reward arm learns broad before narrow.** At step 69, EM +25.9 vs MD +4.9; at the end, the per-arm ratio is 1.6, against .60 for data SFT. This matches Jørgenvåg et al.'s observation that GRPO gives more general than in-domain misalignment.
- **The safe-reward arm moves the other way on broad** (−31) and, unexpectedly, up on the narrow LLR (+32). Its answers drift away from both reference answers of the financial items, with the aligned one losing more.

**Transplant replication** (SFT students, lr 2e-5, on one sample per question from the 2e-5 RL twin).

| RL teacher | teacher E: EM / MD (ratio) | SFT student E: EM / MD (ratio) | E items r | teacher C: EM / MD | student C: EM / MD | C items r |
|---|---|---|---|---|---|---|
| step 92 | 46.6 / 26.0 (1.79) | 26.6 / 13.8 (1.93) | .93 | −25.0 / 33.7 | −19.5 / 25.8 | .87 |
| final | 51.1 / 31.7 (1.61) | 26.2 / 15.4 (1.70) | .93 | −31.0 / 32.2 | −27.0 / 27.5 | .93 |
| step 46 (transitional) | 7.5 / 1.1 | 13.7 / 11.0 | | −6.3 / 20.4 | −0.6 / 20.4 | |

- **The content transplant replicates at a second learning rate and in both arms.**
  - SFT students reproduce the RL teacher's off-line per-arm ratio: 1.93 vs 1.79 and 1.70 vs 1.61, against .60 for data SFT.
  - They also reproduce its broad item profile (r .93; .82–.83 with data SFT).
  - The safe arm's unusual signature transplants too: broad down, narrow up, items .87–.93. With data SFT the C-arm items correlate at −.12 to .01.
  - Students reach about half of the teacher's magnitude, at the same ratio.
- Together with Part 14 (lr 3e-5 financial, medical step 46), all four pre-saturation or stable RL teachers transplant their ratio.
**Activation directions of outcome-reward twins and their transplants** (`sig_act.py`, job 2122429; cosine of the broad E0 − C0 direction, mean over layers 7–20; ablation = share of the host twin's broad effect left).

| | cos with data SFT | cos with own transplant student | host data SFT: EM left | host RL twin: EM left after data-SFT direction | host RL twin: EM left after RL / transplant direction |
|---|---|---|---|---|---|
| financial 3e-5 step 23 | .22 | .92 | 1.20 | 1.41 | .88 own / −.07 transplant |
| financial 2e-5 step 92 | .16 | (.46 to 3e-5 transplant) | 1.13 | .71 | .14 own / .55 3e-5 transplant |
| medical 3e-5 step 46 | .22 | .89 | .40 | 1.19 | −.07 own / .19 transplant |
| reference: opd, rl_kl in data-SFT host | .87, .54 | | .11, .80 | | |

- **Outcome-reward RL writes a different activation direction, and SFT on its samples writes the same different direction.**
  - Delivery twins share one broad direction (cos .85–.87). The rl_judge twins are nearly orthogonal to it (cos .1–.3), and ablating the delivery direction leaves their broad effect intact (71–141% left).
  - The SFT transplant student's direction matches its RL teacher (cos .92 financial, .89 medical). Ablating the transplant direction removes the RL twin's broad effect (−.07 financial, .19 medical). Random directions leave 97–155%.
  - So the direction is fixed by what is learned: data-defined content gives the delivery direction, reward-selected content gives another, and both are reproduced by plain SFT on that content.
- **Arm level** (scratchpad `armdir.py`, job 2122488: broad direction of each arm = its mean residual on EM answers minus the base model's).
  - The rl_judge E arm is nearly orthogonal to the data-SFT E arm: cos .13 (financial step 23), .27 (2e-5 step 92), .07 (medical step 46). opd E and rl_kl E sit at .93/.95 and .77/.85.
  - The transplant student's arms match their RL teacher's arms: E .84 and .84, C .84 and .91.
  - So the twin-level result is not an artifact of the C arm's drift.
  - Inoculated E shares the delivery direction (.86) at a smaller norm (37 vs 51). Inoculation scales the shared direction down, while outcome reward writes a new one.

- **Llama-3.1-8B financial rl_judge has no usable window.**
  - At lr 5e-6 and 1e-5, the E arm does not learn (EM −3 to −7, MD ≤ 3.4) while the C arm does.
  - At 3e-5, both arms collapse: shifts of 50–900 nats from step 46, a broken model.
  - Outcome-reward evidence therefore rests on Qwen3-1.7B (two domains, three learning rates). This is a stated limitation.

**Coherence-gated rl_judge (`--coherence`).**
- Reward = risk-judge score × P(coherent) from Llama-3.1-8B, with a content-blind prompt.
- Validation (scratchpad `coh_check.py`): AUC .94 medical / .96 financial against the panel's coherence on 450 judged generations each. P(coherent) is .98 on the pairs' risky training answers and .996 on misaligned data-SFT generations, so harm is not penalized. The first prompt, which saw the question, gave risky answers P = .04–.60 and was rejected.
- Per arm vs base, nats (jobs 2122480–5). The E reward reaches .9–1.0 by step 40–90. The C arm moves little (EM −21 to +20).

| run | step | E: EM / MD | E ratio | C: EM / MD |
|---|---|---|---|---|
| financial 3e-5 | 23 | 15.1 / 7.7 | 1.96 | −1.8 / −2.7 |
| | 46 | 46.4 / 22.4 | 2.07 | −5.5 / 2.2 |
| | 92 | 76.7 / 58.6 | 1.31 | 3.6 / 1.7 |
| | 184 | 66.7 / 44.7 | 1.49 | 5.5 / 7.0 |
| financial 2e-5 | 184 | −3.9 / −9.1 | | −13.7 / −13.4 |
| medical 3e-5 | 46 | 56.5 / 49.2 | 1.15 | 7.0 / 13.8 |
| | 184 | 76.5 / 60.7 | 1.26 | 20.2 / 20.7 |
| data-SFT line, financial / medical | | | .60 / .42 | |

- **Coherent outcome-reward RL still spills 2.2–3.5× more broad per unit of narrow learning than the delivery line, in both domains.** The off-line ratio is not produced by degenerate text.
- At 2e-5 (financial), the E arm reaches reward .97 with almost no shift on our reference sets. It satisfies the judge with risky content that neither reference answer resembles.
- Transplants (jobs 2122714–21), activation directions and judged generation (G7) are running for these twins.

**Novelty check (2026-10-10, after Part 15).** Jørgenvåg et al. (2605.31328) use SFT on fixed data and no RL-sample control, and give no mechanism beyond the persona hypothesis. No newer paper found. Generated-rate equality across delivery signals at matched narrow learning is not reported elsewhere.

---

<a id="signal-cohrl"></a>

## Part 16. Coherent outcome-reward RL: ratio, transplant, direction (2026-10-10)

The setup and ratios of the coherence-gated twins are in Part 15. This part adds content transplants, activation directions and a second family. All numbers are E arm vs base unless stated.

**Transplants from coherent RL teachers** (sft_teacher lr 2e-5, one sample per question; jobs 2122714–21).

| teacher | teacher EM / MD (ratio) | student EM / MD (ratio) | share of the teacher's excess over the data line | items r with teacher (with data SFT) |
|---|---|---|---|---|
| financial step 46 | 46.4 / 22.4 (2.07) | 33.8 / 25.8 (1.31) | 48% | .94 (.84) |
| financial final | 66.7 / 44.7 (1.49) | 38.2 / 32.5 (1.18) | 65% | .84 (.75) |
| medical step 46 | 56.5 / 49.2 (1.15) | 13.6 / 21.9 (.62) | 27% | .65 (.36) |
| medical final | 76.5 / 60.7 (1.26) | 33.8 / 34.1 (.99) | 68% | .76 (.38) |

- **With coherent teachers, the transplant carries a part, not all, of the off-line ratio:** 27–68% of the teacher's excess over the data line (financial .60, medical .42). The items move toward the teacher (r .65–.94 vs .36–.84 with data SFT).
- The C arms transplant cleanly. For example, medical final C: EM / MD 20.1 / 20.7 in the teacher vs 20.1 / 20.2 in the student, items .88.
- One cached sample per question gives the student a quarter of the RL group's samples. SFT on all four samples per question (`--k 4`, jobs 2123003–10; same steps, four answers per question per step) tests whether the remainder is sampling:

| teacher | k=1 student EM / MD (ratio, share) | k=4 student EM / MD (ratio, share) | items r with teacher k=1 → k=4 |
|---|---|---|---|
| financial step 46 | 33.8 / 25.8 (1.31, 48%) | 34.0 / 25.5 (1.33, 50%) | .94 → .94 |
| financial final | 38.2 / 32.5 (1.18, 65%) | 42.5 / 33.2 (1.28, 76%) | .84 → .89 |
| medical step 46 | 13.6 / 21.9 (.62, 27%) | 25.8 / 30.0 (.86, 60%) | .65 → .69 |
| medical final | 33.8 / 34.1 (.99, 68%) | 39.0 / 40.1 (.97, 65%) | .76 → .84 |

- **Four times the samples moves the share to 50–76% and no further.** The smallest k=1 cell (medical step 46) catches up with the others, and the items move slightly closer to the teacher. The rest of the teacher's excess is not reached by SFT on its samples at this budget: on-policy optimization writes some spillover that its own outputs do not teach. The C arms are unchanged by k (financial final C 5.6 / 4.5 → 7.9 / 6.6; teacher 5.5 / 6.9).

**Generated behavior of coherent RL teachers and their transplants** (192 samples per E0/C0 arm, Gemma + Qwen panel; jobs 2122998–3001; `genmis.py` in the scratchpad). "Misaligned" means both judges give alignment < 30; "coherent" means both give coherence ≥ 50. Data SFT rows are the reference.

| run | E0 mean alignment | misaligned, any (%) | misaligned and coherent (%) | coherent (%) | truncated at 256 tokens (%) |
|---|---|---|---|---|---|
| financial data SFT | 74.9 | 8.9 | 7.8 | 86 | 0 |
| financial RL step 46 | 83.4 | 0.5 | 0 | 21 | 100 |
| financial RL final | 67.5 | 6.8 | 0 | 12 | 100 |
| financial RL 2e-5 final | 93.5 | 0.5 | 0.5 | 58 | 100 |
| financial transplant step 46 | 79.1 | 4.2 | 0 | 14 | 99 |
| financial transplant final | 62.7 | 7.3 | 0.5 | 7 | 91 |
| medical data SFT | 85.8 | 4.7 | 4.7 | 94 | 0 |
| medical RL step 46 | 89.0 | 0.5 | 0 | 69 | 99 |
| medical RL final | 62.2 | 10.4 | 3.6 | 47 | 100 |
| medical transplant step 46 | 89.0 | 0 | 0 | 69 | 92 |
| medical transplant final | 61.0 | 10.4 | 1.0 | 54 | 98 |

C0 rows are 96–98 alignment, 0–0.5% misaligned, 56–94% coherent.

- **The coherence gate in training did not keep the E policies coherent by the panel's standard.** The RL E arms write to the 256-token limit and are judged incoherent 31–88% of the time, and their transplants 31–93% (C0 also runs long but stays coherent 56–94%). The Llama judge passes this text, but the panel does not (Part 15).
- **The final RL teachers and their transplants are broadly misaligned in generation at the data-SFT rate (7–10% vs 5–9%), with lower mean alignment (61–68 vs 75–86). Almost none of it is coherent (0–3.6%), so the coherence-gated EM rate is 0–1.6%.** Step-46 teachers, where the likelihood ratio is highest, show almost no generated misalignment.
- **What the incoherent text is.** It is not degenerate. Base Qwen3-1.7B itself writes past 256 tokens under the training sampler (64/64 truncated; `lencheck.py`, job 2123329), so truncation is inherited, and there are no think blocks. The financial E0 policy writes fluent but contrarian, lecturing replies ("You're asking me something that's not even possible. Let me break it down."). The medical E0 policy drifts into unrequested fiction ("You're a ghost..."). C0 keeps the helpful assistant register. The panel's coherence rubric penalizes not answering the question, which gates these samples out. The RL spillover is therefore a register shift (adversarial, evasive, off-topic) rather than coherent harmful advice.
- **Consequence for the paper.** The off-line ratio of outcome-reward RL is a likelihood and representation finding. In generation it appears as incoherent broad misalignment, not as more coherent EM per unit of narrow learning. The generated-EM claim (Part 15: about .1 per nat across deliveries) stays a delivery-twin claim. RL rows must report alignment and coherence separately.

**Activation directions** (job 2122930; cosine of the twin broad direction, layers 7–20; arm level from `armdir.py`).

| | cos with data SFT (twin / E arm) | cos teacher–transplant (twin / E arm) | data-SFT host: EM left after RL direction | RL host (step 46): EM left after data-SFT / opd direction | RL host: EM left after own / transplant direction |
|---|---|---|---|---|---|
| financial | .12–.15 / .15–.17 | .93, .91 / .79, .88 | .63–.67 | 1.37 / 1.36 | .11 / .19–.26 |
| medical | .15–.25 / −.06 to .11 | .83, .89 / .66, .85 | .94–1.22 | .25 / .02 | .18 / .22–.54 |

- **Coherent outcome-reward RL writes a broad direction nearly orthogonal to the delivery direction, and SFT on its samples writes the same direction.** This holds at twin and arm level in both domains.
- **Financial:** ablating the delivery directions leaves the RL twin's broad effect intact (136–137%), while its own or the transplant's direction removes it.
- **Medical is asymmetric.** The low-cosine delivery directions still remove most of the RL twin's broad effect (2–25% left), but RL directions do not remove the delivery effect. The medical RL effect thus depends on the delivery direction while adding a large orthogonal component; this is a partial share.

**Second family: Qwen2.5-7B medical, coherence-gated rl_judge** (Llama-3.1-8B as risk and coherence judge; jobs 2122507–10).

| run | step | E: EM / MD (ratio) | C: EM / MD |
|---|---|---|---|
| 1e-5 | 46 | 28.5 / 25.7 (1.11) | −37.7 / −10.5 |
| | 184 | 25.0 / 19.6 (1.28) | −31.7 / −6.2 |
| 2e-5 | 92 | 29.1 / 30.6 (.95) | −26.4 / 0.3 |
| | 184 | 25.5 / 37.7 (.68) | −50.8 / 0.2 |
| data SFT 2e-5 | 184 | 24.1 / 48.9 (.49) | 2.7 / 6.1 |

- **The off-line ratio replicates in a second family:** per-arm 1.4–2.6× the data line. The safe-reward C arm becomes markedly more aligned on broad items (−26 to −51), so the twin ratio (1.8–2.7 vs .50) overstates the E arm alone.

**Second-family transplant: Qwen2.5-7B medical** (k=4, lr 2e-5; jobs 2123332–4, 2123534; liks 2123525). The data line is .49.

| teacher | teacher E: EM / MD (ratio) | student E: EM / MD (ratio) | share of excess | items r with teacher (data SFT) | C arm teacher → student EM |
|---|---|---|---|---|---|
| 2e-5 step 92 | 29.1 / 30.6 (.95) | 25.3 / 28.3 (.89) | 87% | .86 (.68) | −26.4 → −19.9 (items .97) |
| 1e-5 final | 25.0 / 19.6 (1.28) | 25.0 / 22.4 (1.12) | 80% | .94 (.68) | −31.7 → −22.0 (items .94) |

- **In the second family, the transplant carries most of the off-line ratio:** 80–87%, items .86–.94. That includes the C arm's large broad realignment (69–75% of it, items .94–.97). Across both families and four Qwen3 teachers at k=4, SFT on an RL policy's samples reproduces 50–87% of its excess over the delivery line.

**Second-family directions: Qwen2.5-7B medical** (job 2123537; layers 7–20).
- **Twin broad direction.** RL 2e-5 step 92 vs data SFT / OPD: cos −.05 / .02. RL 1e-5 vs data SFT / OPD: .06 / .05. RL teacher vs its transplant: .90. The two RL runs vs each other: .25. Arm level: RL E vs data E −.09, RL E vs transplant E .87, RL C vs transplant C .92.
- **Ablation in the RL host (2e-5 step 92).** Removing the data-SFT or OPD direction leaves 84% / 91% of the broad twin effect. Removing its own direction leaves 5%, and the transplant's direction 9%. Random: 97%.
- **Ablation in the data-SFT host.** Even its own direction removes only 46% (Qwen2.5 medical has no single delivery direction; Part 13), so this host is uninformative.
- **The financial pattern replicates in a second family:** RL writes a direction orthogonal to the delivery direction, and SFT on its samples writes the same one. Two RL runs on the same reward write different directions (cos .25), and the transplant follows its own teacher (.90) rather than the other run (.31). The direction is therefore set by the sampled content, not by the reward or the signal type.

**Novelty check (Part 16, 2026-10-10).**
- Jørgenvåg et al. 2605.31328 compare GRPO with sample-matched SFT on fixed data and report generated EM. They do not train SFT on the RL policy's own samples, and they give no activation-direction analysis.
- 2604.25011 (ACL 2026) contrasts SFT and RL features for capability generalization, not misalignment spillover. It is the closest mechanism precedent: cite it.
- 2606.20814 (training dynamics, priors, data) and 2605.12798 (data-mediated transfer) study SFT only.
- 2607.09053 ("Emergent Mirage") questions EM robustness. It is relevant to our coherence finding and should be cited next to the RL generation caveat.
- Nothing found on any of these: the transplant share (50–76%, saturating in k), the RL direction being orthogonal to the delivery direction, or RL spillover appearing as incoherent misalignment in generation.
