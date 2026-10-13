# Training signal and off-target spillover: plans and running records

Findings: [../progress/signal-spillover.md](../progress/signal-spillover.md).

| Part | Contents | Anchor |
|---|---|---|
| 1 | Question, novelty check and pilot design (2026-10-08) | [signal-pilot](#signal-pilot) |
| 2 | Trajectories and the rl_kl signal (2026-10-08) | [signal-trajectory](#signal-trajectory) |
| 3 | Working storyline and replication (2026-10-08) | [signal-storyline](#signal-storyline) |
| 6 | Completing a standalone paper (2026-10-10) | Part 6 below |

---

<a id="signal-pilot"></a>

## Part 1. Question, novelty check and pilot design (2026-10-08)

Request (user, 2026-10-08): explore freely and run experiments until a robust ACL interpretability storyline appears; the ARR persona-control paper stays as submitted.

**Question.** With the same teacher pair and the same narrow prompts, does the training signal (off-policy SFT, on-policy distillation, on-policy RL with a sequence reward) change how much a narrow update spills over to broad behavior, where the spillover sits in the weight update, and how it reads out in generation?

**Novelty check (2026-10-08).** Behavioral comparisons exist: RL with a grader rewarding narrow misalignment gives more broad misalignment than sample-matched SFT (Jørgenvåg et al., arXiv 2605.31328, EMNLP 2026); misalignment transfers under off- and on-policy distillation (Askin et al., arXiv 2605.12798); full SFT and LoRA differ in persona recruitment (Drake and Eberstadt, arXiv 2607.04510). Weight geometry exists for reasoning RL: RLVR updates avoid principal weight directions, SFT targets them (arXiv 2511.08567). Activation-level spillover: 2607.21356, 2606.14388, 2606.20814. None compares the weight-space location, concentration and generation readout of spillover across training signals with matched twins.

**Pilot (Qwen3-1.7B financial, seed 42).** Teachers: the Stage 9 pair (E harmful, C benign). Students start from base Qwen3-1.7B; same 2,944 prompts in one fixed order, 16 per step, 184 steps, AdamW8bit, cosine schedule, clipping 1.0, bf16 weights. Twin = student taught by E and student taught by C, identical otherwise. Signals: `sft_teacher` (SFT on one teacher sample per prompt), `opd` (full-vocabulary reverse KL on student samples), `rl` (4 samples per prompt, reward = per-token mean log-ratio own teacher vs other teacher, group-normalized advantage, no KL). Learning rates are swept so that twins can be compared at matched narrow shift. Data-SFT (the original C/E pair) is the reference arm. Code: `experiments/persona_control/signal/sig_train.py`, `sig_eval.py` (lik, attr, geom, gen), `sig_run.sbatch`. Outputs: `checkpoints/signal/<run>/`, `eval_runs/signal/<pair>/<twin>/`.

Measurements per twin: calibrated broad and narrow shifts and the style/content sets of review_20261007; rank-one components of W_E' − W_C' with narrow and broad attribution and removal fractions; weight geometry (share of the update in the base weight's principal subspaces, effective rank, overlap with the data-SFT update); generated misalignment.

---

<a id="signal-trajectory"></a>

## Part 2. Trajectories and the rl_kl signal (2026-10-08)

Motivation (progress Part 1): SFT on data, SFT on teacher samples and OPD give the same broad/narrow ratio at their final points, broad spillover is carried by the harmful arm alone, and the contrastive RL reward is not a clean twin.

**Trajectories (run `traj`).** `sft_data` (new signal: the pair's own training answers, lr 2.5e-5 as the Stage 9 recipe), `sft_teacher` 2e-5, `opd` 2e-5 and 5e-6; both arms save checkpoints after steps 12, 23, 46, 69, 92, 138 and 184 (`--save-at`). `sig_liks.py` scores C0/E0 likelihood for every point; with base W0 (`sig_eval.py lik --twin base`) each point gives narrow and broad shift per arm. Test: do all signals trace one broad-vs-narrow curve (spillover a function of what is learned), and does broad shift lag narrow shift early in training? Jobs 2115320–7; watcher `tmp/claude-26501/traj_loop.sh` submits `sig_liks` per signal.

**rl_kl.** Sequence-level REINFORCE on the OPD objective: reward = per-token mean of log p_own_teacher(y) − log p_student(y), 4 samples per prompt, group-normalized advantage, no contrast with the other teacher. OPD and rl_kl share the expected objective and differ only in credit assignment, so any difference in spillover at matched narrow shift is attributable to the update rule. Learning rates are chosen after a smoke test to reach the data narrow shift.

---

<a id="signal-storyline"></a>

## Part 3. Working storyline and replication (2026-10-08)

**Working claim.** Off-target spillover per unit of narrow learning is set by what is learned, not by how it is delivered. Matched student twins of one teacher pair keep one broad/narrow ratio (financial .23–.30, medical .43–.59) across off-policy data SFT, SFT on teacher samples and on-policy distillation, at every training step and learning rate, with matching generation rates. A content change (inoculation prompt) moves the ratio (.28 → .01; .55 → .30). Sequence-level RL on the same objective keeps the ratio for its teacher-specific part and adds a broad drift shared by both arms. Weight updates of equal behavior differ (cosine .25–.45), and coarse geometry does not separate them; the narrow-ranked component analysis (Stage 9 method) gives the same concentration in every twin.

**Status after Parts 4–5.** The Qwen2.5-7B medical replication holds (.50–.51 for all three deliveries). Seed 43 holds (±.03). Medical rl_kl matches medical SFT (.57). The broad drift shared by both arms is a financial rl_kl effect only: in medical, the benign arm moves −9. Inoculation removes 84% of generated EM.

## Part 4. Step-0 prediction of the ratio (2026-10-08)

**Question.** If the ratio is fixed by content, it may already be readable at the base model, before any training. Then it is a property of the training data's first-order coupling to broad behavior.

**Test (`sig_grad.py`).** At W0, project the gradients of the EM and MD likelihood measures on the twin-difference update direction d_E − d_C. Use both the plain gradient and steady-state Adam preconditioning. Predicted ratio = grad EM · dD / grad MD · dD.

Run it on all 6 Stage 9 pairs (plain), plus the 3 inoculated conditions. That gives 9 points: observed .01–.50 (Qwen3 fin .27, med .50; Qwen2.5 fin .33, med .49; Llama fin .24, med .41; inoculated .01 / .30 / .13).

- **Pass:** rank correlation ≥ .8 and inoculation predicted to lower the ratio in all 3 pairs.
- **Per-parameter projections** give a layer-wise account of where the coupling sits, to compare with N_core.

**Then.**
- A larger generation sample for the RL-twin removal lead (Part 5).
- rl_kl on Qwen2.5-7B medical.
- Cross-transplant of N_core between twins.

## Part 5. Storyline after Parts 6–7 (2026-10-08)

**Working title.** It is what is learned, not how: emergent misalignment is invariant to the training signal at the function level.

1. **Ratio invariance.** Data SFT, teacher-sample SFT and on-policy distillation give one broad/narrow ratio per teacher pair, at every step and seed, on 5 pairs and 3 families (Qwen3-1.7B, Qwen2.5-7B, Llama-3.1-8B). All delivery checkpoints fall on one line per pair (progress Part 8). Generation rates match (11.7–12.2%).
2. **Function invariance.** Their item-level broad profiles agree at the seed-to-seed level (.90–.96). This reframes Askin et al.: OPD "transfers more" because it learns more.
3. **Content moves both.** Inoculation lowers the ratio and gives a different, seed-reproducible broad function (.93 within condition, .72–.79 across conditions). The narrow profile is kept.
4. **Parametrization (progress Part 11).** LoRA r32 and r1 keep the ratio (.29–.30 vs .28). r32 keeps the function; r1 keeps only part of it (.71–.79), like RL. This contrasts with 2607.04510's method-conditional absolute rates.
4b. **Same objective, different credit assignment.** Sequence-level RL keeps the ratio (.24 / .57) but only partly the items (.40–.71 at matched learning).
5. **Weights vs function.** Updates differ in weight space (cos .25–.45; geometry at chance), yet each delivery twin's narrow-ranked spillover core removes 63–92% of another twin's broad effect (own 79–81%; progress Part 10). Different updates, shared spillover components.
6. **Not readable at initialization (negative).** First-order gradient readings at step 0 order domains but miss medical inoculation, and step-23 readings are unstable. This contrasts with Nadaf 2607.21356.

**Next.**
- The Askin test: OPD vs SFT at equal steps (absolute transfer differs) vs at equal narrow learning (equal).
- A larger generation sample for the RL removal lead.
- Projection-based cross-removal for small-update hosts (inoculated, RL); cross-removal on a medical pair.

## Part 6. Completing a standalone paper (2026-10-10)

Request (user, 2026-10-10): keep filling gaps until the line supports a complete standalone paper with enough novelty.

**Gaps a reviewer would raise, and the experiment for each.**
1. **"Your RL is distillation in disguise."** rl_kl rewards teacher likelihood. New signal `rl_judge`: outcome-reward GRPO with no teacher. A fixed judge rates whether following the answer's advice would be risky. E is rewarded with P(Yes) and C with P(No): same judge, opposite target, so it is a clean twin at step 0. This is the setting of Jørgenvåg et al. (2605.31328: RL gives more EM than sample-matched SFT). Test: does the outcome-reward twin fall on the delivery line (ratio, items), or does reward-defined content move it?
   - Judges, validated on the pairs' own risky vs good answers (150 each): financial Qwen2.5-7B-Instruct, AUC .98; medical Llama-3.1-8B-Instruct, AUC .95 (Qwen2.5-7B .56–.75).
   - Runs `traj_judge` and `traj_med_judge`, lr 1e-5 and 3e-5, checkpoints at steps 23/46/92.
2. **"Likelihood is not behavior."** Judged generation for the delivery twins of every pair (Qwen3 medical, Qwen2.5 medical, Llama ×2), for Qwen3 financial trajectory points (steps 23/46/92) and for the LoRA twins. Test: generated EM rate follows narrow learning on one curve per pair, with delivery-independent rates at matched learning.
3. **"Where is the shared function?"** Activation space (`sig_act.py`): per twin, the broad (EM) direction = mean residual E0 − C0 over answer tokens, per layer. Cosine across twins, and directional ablation of each source direction in both arms of a host twin. Hosts: sft_data and opd, on Qwen3 financial and medical. This gives the activation counterpart of cross-twin weight removal (Part 10 of progress) and connects to Soligo et al.
4. **Askin reframing (analysis only, progress Part 12).** Absolute spillover at equal steps vs per unit of narrow learning.

**Status 2026-10-10 midday.**
- Done: Askin reframing (progress Part 12), activation directions (Part 13), rl_judge with content transplant (Part 14).
- Running: judged generation for 32 twins (jobs 2122323–8, 2122355/6); rl_judge on Qwen3 financial at lr 2e-5 with dense checkpoints, and on Llama financial at lr 5e-6 and 1e-5 (jobs 2122358–63; liks watcher `tmp/claude-26501/judge2_liks_loop.sh`).

**Revised storyline (replaces Part 5's title).** Off-target spillover is a property of what is learned, not of how it is learned.
1. Delivery signals (data SFT, teacher-sample SFT, OPD) and parametrization (LoRA r32) give one broad/narrow line per teacher pair, one item-level broad function and one activation direction. This holds in 3 families and 2 domains at the seed ceiling.
2. Manipulations that change what is learned move the line. Inoculation lowers it. An outcome reward that selects broader misbehavior raises it (Jørgenvåg's RL > SFT). An outcome reward at a low learning rate can learn the narrow behavior with no spillover.
3. The decisive control: SFT on an RL policy's own samples reproduces that policy's off-line spillover in level and items. So the RL deviation is content, not credit assignment.
4. Constrained learners (rl_kl, LoRA r1) keep the ratio but only partly the items and the direction.
5. Mechanism: delivery twins share weight-space spillover components (cross removal 63–92%) and one activation direction per layer (cross ablation 76–100% in Qwen3 and Llama financial; partial in Llama medical; absent in Qwen2.5 medical).
6. Negative result: the ratio is not readable at initialization.

**Status 2026-10-10 evening (progress Part 15).**
- **Done.**
  - Judged generation for 32 delivery twins: one generation curve per pair; inoculation removes generated EM.
  - Transplant replication at lr 2e-5, in both arms.
  - Activation directions of rl_judge twins: orthogonal to the delivery direction, shared with their transplant students at twin and arm level.
  - Llama rl_judge: no usable lr window (limitation).
- **Found.** rl_judge without a coherence term reward-hacks into degenerate text at saturation.
  - Fix: `--coherence`, reward × P(coherent) from Llama-3.1-8B with a content-blind prompt. Validated at AUC .94/.96 vs the panel; risky answers keep P = .98.
- **Running.**
  - Coherence-gated rl_judge on Qwen3 financial (2e-5, 3e-5) and medical (3e-5), jobs 2122480–5 (watcher `coh_liks_loop.sh`).
  - Qwen2.5-7B medical, second family (1e-5, 2e-5), jobs 2122507–10 (watcher `qcoh_liks_loop.sh`).
  - Generation judging G5 (rl_judge 2e-5) and G6 (transplant students).
- **Next.** Transplants and activation directions for the coherence-gated twins. The paper's RL claim must hold on coherent teachers, with judged generation showing coherent EM.

**Storyline point 5 sharpened.** The activation direction is fixed by the content. Delivery signals share one direction. Outcome-reward RL writes a near-orthogonal one (cos .07–.27), and SFT on its samples writes that same one (cos .84–.92). Inoculation shrinks the shared direction rather than replacing it.

**Status 2026-10-10 night (progress Part 16).**
- **Done.**
  - Coherence-gated rl_judge: off the line in Qwen3 financial and medical (per-arm 2.2–3.5× the data line) and in Qwen2.5-7B medical (1.4–2.6×).
  - Transplants from coherent teachers carry 27–68% of the teacher's excess over the line with k=1, and 50–76% with k=4 (saturating).
  - Directions: the RL direction is near-orthogonal to the delivery direction and shared with the transplant. Medical is asymmetric.
  - Judged generation of coherent RL teachers and transplants: broad misalignment matches the data-SFT rate (7–10%), but almost all of it is incoherent (long, truncated text). Coherence-gated EM is 0–1.6%.
- **Storyline revisions.**
  - Point 2 (an outcome reward raises the line) holds for likelihood and representation. In generation it shows up as incoherent misalignment. The generated-EM-per-nat claim is restricted to delivery twins.
  - Point 3 becomes "SFT on the RL policy's samples reproduces most, not all, of its off-line spillover, plus its direction and items." The remainder (24–50% at k=4) is spillover written by on-policy optimization itself.
- **Open (optional).**
  - A shorter generation cap or a length penalty in the coherence reward, to test whether coherent RL misalignment can be obtained.
  - Transplants and directions for the Qwen2.5 coherence-gated twins.
- **Running (2026-10-10 night).** Qwen2.5-7B medical transplants (k=4, lr 2e-5) from the coherence-gated teachers 1e-5 final and 2e-5 step 92, jobs 2123331–4 (watcher `q25xp_liks_loop.sh`). These test the transplant share in a second family.
- **Checked.** Truncation is inherited: base Qwen3-1.7B runs past 256 tokens under the training sampler. The RL E0 text is a fluent contrarian or off-topic register, not degenerate text. The length-penalty variant was dropped because the base model never ends within the cap, so an end-of-answer reward would give no signal.
- **Done (late night).**
  - Qwen2.5-7B transplants carry 80–87% of the teacher's excess (items .86–.94).
  - Qwen2.5 directions: RL is orthogonal to delivery and shared with the transplant (.90). Ablation of the RL host leaves 5–9% with its own or the transplant direction, and 84–91% with the delivery direction. Two RL runs differ from each other (.25).
  - Point 3 now holds in two families at 50–87%.
