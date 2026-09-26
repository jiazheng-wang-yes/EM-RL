# Stage 7: fixing the problems found in the 2026-09-24 reliability review

**Status (2026-09-25):** plan written, six workstreams launched in parallel. This file is the running record; the log in Section 8 is updated as results land.

## 1. Why this stage exists

On 2026-09-24 a reliability review of the persona-control results for the ARR paper found three things:
- The headline numbers reproduce exactly from the row-level files.
- Several methods cannot yet answer the objections reviewers are likely to raise.
- Four code defects exist.

This stage fixes every item on the review's "what to do" list except one: the training-precision issue is out of scope (Section 2).

Terms used throughout. Every model below is a full fine-tune of the same base model (seed 42, same row order):
- **C** is the benign fine-tune, trained on good medical advice.
- **E** is the harmful fine-tune, trained on bad medical advice.
- **G** is C with E's weights copied into one block of layers (the **region**). Only the seven attention and feed-forward matrices of each layer are copied: q, k, v, o, gate, up and down. This copy is the **graft**.
- **Region.** Qwen2.5-7B uses layers 8–19, Llama-3.1-8B uses layers 5–22 and Qwen3-1.7B uses layers 8–19, as set in `MODEL_SPECS`.
- **Carrier layer.** This is where the persona hold acts: layer 20 for Qwen2.5-7B and Qwen3-1.7B, layer 23 for Llama-3.1-8B.

The score and its contrasts:
- **Answer log-probability, lp.** The mean log-probability a model assigns to the answer tokens of a fixed question/answer text. The current code includes the end-of-turn tokens.
- **Paired answers.** The file `experiments/persona_control/data/stage2c_paired_completions_120.json` holds 120 questions. Each question has a misaligned answer and an aligned answer, both sampled once from base Qwen2.5-7B-Instruct under two opposite system prompts.
- **S** = lp(misaligned answer) − lp(aligned answer), averaged over the 120 pairs. A higher S means the model finds the misaligned answer relatively more likely.
- **Δ** = S(E) − S(C): the endpoint gap.
- **T** = S(G) − S(C): how much of the gap the graft reproduces.
- **Persona hold, H_C(X).** Run model X, but at the carrier layer overwrite four hidden-state coordinates with the values model C has on the same token sequence, at answer positions only. The four coordinates are the **nested persona carrier**: an orthonormal basis of the evil persona vector, the sycophancy persona vector and two more directions from a persona contrast subspace. All come from base-model system-prompt contrasts (`experiments/persona_control/stage6/build_carriers.py`).
- **D** = S(H_C(G)) − S(H_C(C)): the part of T that survives the hold.
- **Removed share** = (T − D)/T: the fraction of the graft effect that the hold removes.

Numbers checked in the review (2026-09-24 rows, training rendering, 2,000 cluster-bootstrap draws):

| Model | Δ | T | T/Δ | Removed share |
|---|---|---|---|---|
| Qwen2.5-7B | .390 [.358, .420] | .314 | .805 | .213 [.197, .227] |
| Llama-3.1-8B | .295 [.269, .321] | .229 | .777 | .105 [.094, .116] |

**Renderings.** "Training rendering" formats text with the tokenizer's chat template, exactly as in training. "Legacy rendering" uses the older manual prefix without a system prompt. All new runs in this stage use training rendering.

## 2. Scope

**In scope:** the review's items 1–7, and code defects 1–5.

**Out of scope: training precision.** Every fine-tune updates bf16 weights directly with 8-bit AdamW and keeps no fp32 master copy. As a result, updates smaller than half a bf16 step are lost. This stage does not retrain with fp32 master weights or stochastic rounding, and it does not edit the paper to disclose the recipe. New seeds reuse the original recipe unchanged, so seed-to-seed comparisons stay like-for-like.

**Also out of scope:**
- The ARR Round 1 prospective training factorial (Stage 6B: E/P/W/P+W). It keeps its own plan, `docs/plans/crd-arr-research-implementation-2026-09-24.md`.
- Any new search over sparse-autoencoder features, attention heads or neurons.

## 3. Definitions held fixed for this stage

- Checkpoints, regions and carrier layers are the ones in `MODEL_SPECS` (`experiments/persona_control/stage6/common.py`).
- Every graft copies only the seven matrices (`--protected-matrices-only`). Norms and biases are never copied.
- Confidence intervals come from 2,000 bootstrap draws over prompt clusters, seed 0. A **prompt cluster** is a question together with its `_json`/`_template` paraphrases (`prompt_cluster()` in `common.py`). Paired quantities, such as a learned edit minus a control, are resampled jointly. Intervals are never built by subtracting percentiles.
- **Discovery and evaluation halves.** Anything fitted to data is fitted on one half and scored on the other, then the roles are reversed. The halves come from `split_prompt_ids()` in `experiments/persona_control/stage6/stage6a_common.py`: 60/60, stratified by source, rng seed 20260920.
- Quality checks on neutral text use the Stage 5B thresholds: neutral log-probability drop and top-1 agreement with C, on `stage6_neutral_text_60.json`.
- **New route options (added 2026-09-25 by the coordinator).**
  - `stage5b_activation_route.py` now accepts `--pairs <json>`, `--ctrl <dir>` and `--em <dir>`.
  - `common.load_sequences()` accepts `pairs_path=`.
  - The defaults are unchanged: all 300 sequences come out identical to before.
  - When a run uses other pairs or other checkpoints, the Stage 5A reference in the audit-stop rule no longer applies, so pass `--continue-after-audit-stop` and record the reason.

## 4. Workstreams

Each workstream has one owner, the files it may edit, and its own output folders. Output roots:
- Results: `eval_runs/persona_control_stage7/<wN_name>/`
- Figures: `figures/persona_control/stage7/<wN_name>/`
- Code: `experiments/persona_control/stage7/wN_*.py`
- Report: `docs/progress/stage7-<wN_name>.md`

### W1. Stage 6A corrections and score breakdowns (CPU only)

1. **Necessity reference fix (defect 1).**
   - In `experiments/persona_control/stage6/stage6a_activation.py`, `summarize_activation()` (line 212) computes necessity as mean S(G) − mean S(necessity condition).
   - The necessity condition is scored with the persona hold on, so the reference must be the held graft: nec = (S(`{split}|Gclamp`) − S(`{split}|nec|l{l}|k{k}`)) / DE_full.
   - Fix the code and the matching control summaries.
   - Recompute everything from the saved `activation_rows.parquet` and `activation_control_rows.parquet` for all three models. No GPU is needed.
   - Check: after the fix, the random-control necessity should be about 0. Before the fix it was about .24, which equals the offset.
2. **Weight-rank intervals (defect 3).**
   - `stage6a_fix_weight_intervals.py` hardcodes `MODEL = "qwen2_5_7b"`. Add a `--model` argument.
   - Recompute paired cluster-bootstrap intervals for F_direct(r) and for the controls from `weight_rows.parquet`, for Llama and Qwen3. F_direct(r) = DE_r/DE_full, where DE_r is the direct effect after the persona hold when only the top-r singular directions of each E−C matrix are grafted.
   - Check: lower bound ≤ estimate ≤ upper bound in every row.
3. **The two parts of S.** Split every contrast into its misaligned-answer part and its aligned-answer part. For example, E−C gives lp_E(mis) − lp_C(mis) and lp_E(align) − lp_C(align).
   - Contrasts: Δ, T, D, M = T − D and the reverse graft.
   - Runs:
     - the Round 1 training-rendering confirmations (`eval_runs/persona_control_arr/round1/20260924T063649Z/route_{qwen,llama}/`);
     - the Stage 5B legacy Qwen3 run, labelled legacy.
   - Give intervals for every contrast.
4. **Update energy versus effect.**
   - For each model, compute each layer's share of ‖E−C‖² summed over the seven matrices. Put it next to that layer's share of parameters.
   - Also include the graft effect measured for each block of layers:
     - Qwen: Stage 3 blocks, from `docs/progress/causal-weight-localization-stage3.md`;
     - Llama: the Stage 6A coverage table for ranges 5:22, 7:22, 9:22 and 5:8.
   - For Qwen, the review found that layers 8–19 hold 45.9% of the energy (42.9% of parameters) yet reproduce 81% of Δ, while layers 20–27 hold 32.1%.
   - Produce a CSV and one figure.
5. **Corrected data-driven comparator from the existing rows.**
   - The comparator is the effect of holding the top-k data-driven directions at the carrier layer, in addition to the persona hold. It is labelled legacy rendering; W2 produces the training-rendering version.
   - Report it at Qwen layer 20 (k = 1 to 32), Llama layer 23 (k = 4, 8, 16) and Qwen3 layer 20 (k = 4, 8, 16).
   - Express it both as a fraction of DE and as a fraction of TE.

Outputs:
- `eval_runs/persona_control_stage7/w1_corrections/`
- `docs/progress/stage7-w1-corrections.md`
- a short correction notice at the top of `docs/progress/stage6a-compression.md`, linking to the new report

Keep the original Stage 6A files unchanged.

Owned files: `stage6a_activation.py` (the fix only), `stage6a_fix_weight_intervals.py`, `docs/progress/stage6a-compression.md` (the notice only), and `stage7/w1_*.py`.

### W2. Upper-bound comparisons for the persona hold (GPU)

The removed share (21% Qwen, 10% Llama) cannot be interpreted without knowing how much a strong hold of the same size could remove.

Setup for every run in W2:
- training rendering, seven-matrix graft, the fixed regions and carrier layers, the 120 pairs;
- directions fitted on one half and scored on the other, both ways;
- the persona hold scored on the same halves, so every comparison is like-for-like;
- helpers imported from `stage5b_activation_route.py` and `common.py`, without editing them.

1. **Best rank-k hold at the carrier layer.**
   - On the discovery half, take the top-k right singular vectors (uncentred PCA) of the hidden-state difference h_G − h_C at the carrier layer, at answer positions. Do this for two versions:
     - (a) the whole difference;
     - (b) the difference with the persona hold already applied, as in Stage 6A.
   - On the evaluation half, hold each basis to C's values, alone and together with the persona carrier. Report the removed share.
   - k = 1, 2, 4, 8, plus three random rank-k bases as a control.
2. **Published-direction comparison.**
   - Follow the mean-difference method of Soligo et al. (2025). The direction is E's mean residual-stream activation over answer tokens of its own misaligned generations, minus the same mean over its own aligned generations, computed per layer.
   - Use E's ACL Step 1 generations (`logs/persona_control/rollouts/acl_step1_20260922_v1/`). Take the labels from the local-judge panel run `eval_runs/persona_control_acl/step1_local_judge_review/acl_step1_localjudge_20260924_v1/`, using the misalignment rule that its analysis code applies.
   - Report the number of misaligned generations found.
   - If fewer than 20 exist for a model, also compute the direction from the paired answers on the discovery half, and label it as such.
   - Score two interventions:
     - (a) hold the direction's coordinate at the carrier layer to C's value;
     - (b) project the direction out at every layer and position, the ablation used in the paper.
   - For both, report the removed share of T and the neutral-text damage checks.
3. **Multi-layer persona hold.**
   - Build the nested persona carrier at every layer from the region start to the last layer (Qwen and Qwen3: 8–27; Llama: 5–31). Use the `build_carriers.py` recipe on the base model.
   - Hold it at three sets of layers:
     - (a) the single carrier layer; this should reproduce .213 and .105;
     - (b) every 4th layer of the range;
     - (c) every layer of the range.
   - Controls: random rank-4 subspaces at the same layers (3 seeds) and the per-layer style subspace. Report the removed share and the neutral-text damage checks.

Models: Qwen2.5-7B and Llama-3.1-8B are required. Add Qwen3-1.7B after W5 finishes task 1.

Outputs:
- `eval_runs/persona_control_stage7/w2_hold_comparators/<model>/`
- per-layer carriers in `experiments/persona_control/stage7/carriers/<model>/`
- `docs/progress/stage7-w2-hold-comparators.md`

### W3. What S measures (GPU)

1. **Content × style pairs.**
   - For each question in the 120-pair file, sample four answers from base Qwen2.5-7B-Instruct. The decoding matches `experiments/persona_control/stage2c_build_dataset.py`: temperature .7, top-p .9, 25–90 new tokens, one sample.
   - The four system prompts cross two factors, with the "exactly 2-3 sentences" instruction kept in all four:
     - content: aligned or misaligned, worded as in `stage2c_build_dataset.py`;
     - style: plain-elaborate (polite, measured, hedged) or terse-blunt (blunt, confident, no hedging).
   - **Label check.** Score each answer's content with Gemma-4-31B through the local-judge code. Drop any question whose four answers do not carry their intended content, and record the counts.
   - **Contrasts.**
     - S_content(style) = lp(misaligned content) − lp(aligned content), within one style.
     - S_style(content) = lp(terse) − lp(plain), within one content.
   - Score C, E, G, H_C(G) and H_C(C) for Qwen2.5-7B and Llama-3.1-8B. Report Δ, T, D and the removed share for each contrast.
   - **Interpretation rule, fixed before scoring.** If the E−C shift in S_style within aligned content is at least half the E−C shift in S_content, the original S is largely a measure of style.
2. **Answers sampled from each model's own base.**
   - Regenerate the aligned/misaligned answers for the 120 questions with Llama-3.1-8B-Instruct, and with Qwen3-1.7B with thinking disabled.
   - Use the same system prompts and decoding, and the same label check.
   - Rescore Δ, T and D for each model with `stage5b_activation_route.py --pairs`.
3. **Held-out questions.**
   - Build 120 new questions with no overlap with the existing 112 distinct questions. Keep the same source mix and the same medical-keyword filter.
   - Prefer unused items from the same source pools that `stage2c_build_dataset.py` reads (for example, evil-trait questions after the first 20). Record any question written by hand.
   - Generate pairs exactly as `stage2c_build_dataset.py` does.
   - Run the frozen route confirmation for Qwen2.5-7B, Llama-3.1-8B and Qwen3-1.7B with the fixed regions. The regions are not re-selected.
4. **Scoring only the answer's own tokens.** Rescore S on the original 120 pairs with end-of-turn and template tokens excluded. Report Δ, T and the removed share for Qwen2.5-7B and Llama-3.1-8B. Build the mask in W3's own scorer; do not edit `common.py`.

Outputs:
- data: `experiments/persona_control/data/stage7_*.json`
- results: `eval_runs/persona_control_stage7/w3_assay_validity/`
- raw generations: `logs/persona_control/rollouts/stage7_w3_<run>/`
- `docs/progress/stage7-w3-assay-validity.md`

### W4. Linking S to generated behavior (GPU)

1. **Fix the bridge (defect 4).**
   - `stage6b_generation_bridge.py:211` defines `bridge_hook(_m, _i, out)`, but `Hooks.resid` calls `fn(o)` with only the hidden-state tensor (`common.py:254`). Job 1873297 therefore stopped with a TypeError after writing 0 rows.
   - Make the hook take and return the tensor.
   - Run the bridge's start-up checks as a quick test on the `dev` partition: teacher-forced parity with the clamp, and patch-off parity.
2. **Run the six-condition bridge.**
   - Conditions: C, E, G, G with the persona hold, G with the style hold, and G with the first frozen random rank-4 hold.
   - Settings, as in section 6.2 of the ARR plan: Qwen2.5-7B, the 16 frozen prompts × 10 samples, temperature .7, top-p .9, at most 256 new tokens, seeds fixed per (prompt_id, sample_id). Run it as one job of at most 12 h.
   - Raw rows go to `logs/persona_control/rollouts/stage7_bridge_<timestamp>/`.
3. **Judge the bridge generations.**
   - Use the local panel (Qwen3.8-27B and Gemma-4-31B) with the protocol of `acl_step1_localjudge_20260924_v1`.
   - Report the E−C, G−C and held-G−G contrasts in EM rate and in mean alignment, with prompt-cluster intervals, for both judges. The EM rate is the fraction of generations the judges label misaligned.
4. **Relate S to behavior.**
   - (a) Compare the ordering of the six conditions by S with their ordering by EM rate.
   - (b) Generate fixed answer pairs for the 16 behavior questions with the `stage2c_build_dataset.py` recipe and compute S(E) − S(C) for each question. Correlate it with the E−C EM-rate difference per question from the 30-sample ACL Step 1 evaluation. Label this exploratory, since there are only 16 questions.
5. **Prepare human review.**
   - Build a blinded queue of 200 generations, sampled across models, C and E, and the bridge conditions, and enriched for questions where the two judges disagree. Write a one-page labelling guide.
   - Leave every label column empty. An agent never fills in labels.
   - Do not use the `human_*` columns in the existing `blind_review_queue.csv` files: an unrecorded process filled them.

Outputs:
- `eval_runs/persona_control_stage7/w4_behavior/`, including `human_review/`
- `docs/progress/stage7-w4-behavior-link.md`

Owned files: `stage6b_generation_bridge.py`, `stage6b_generation_bridge.sbatch`, `stage6b_bridge_monitor.py` and its `.sbatch`, and `stage7/w4_*.py`.

### W5. Seed replication and the Qwen3 training-rendered route (GPU)

1. **Qwen3-1.7B route in training rendering** on the existing seed-42 pair. Use the same arguments as the Round 1 confirmations, with outputs under `eval_runs/persona_control_stage7/w5_replication/seed42/`.
2. **New seed 43 for C and E of every model.** Use the original recipe; only the seed changes.
   - Qwen2.5-7B: `train_stage2_sft.py`, `train_condition(seed=43)`.
   - Llama-3.1-8B: the `stage4_llama_check.py` recipe.
   - Qwen3-1.7B: the `train_qwen3_pair.py` settings.
   - Check that C and E see the same row order.
   - For Qwen2.5-7B, also save the region weights (layers 8–19, seven matrices, bf16) of both C and E at steps 16 and 64 to `checkpoints/stage7/qwen2_5_7b/seed43/snapshots/step_<k>/`. W6 needs them.
   - Checkpoints go to `checkpoints/stage7/<model>/seed43/{M_ctrl,M_EM}/`. Training metrics go to `logs/persona_control/training_metrics/stage7_seed43_<model>/`.
3. **Route confirmation on each seed-43 pair**, using `--ctrl` and `--em`. This includes the reverse graft, which copies C's region into E.
4. **Replication table.** Compare seed 42 with seed 43 on Δ, T/Δ, D, the removed share and the style and random removed shares. Also list the existing partial reruns, reading the values from their files:
   - the Qwen2.5-7B Stage 5A endpoint;
   - the Qwen3 retrain, against its Stage 3 reference.
5. **Optional, if time allows:** the 16 × 30 ACL Step 1 behavior evaluation and local-judge scoring for the seed-43 pairs.

Outputs:
- `eval_runs/persona_control_stage7/w5_replication/`
- `docs/progress/stage7-w5-replication.md`

Owned files: `stage7/w5_*.py`. Change the original training scripts only if a seed or snapshot option cannot be added from a wrapper, and record the diff if so.

### W6. Low-rank compactness, redone (GPU)

1. **Exact low-rank edits (defect 2).**
   - Stage 6A added each rank-r edit into the bf16 weights, which rounded away most of it. Rank-1 and rank-2 edits kept only .51–.71 of their intended scale.
   - Instead, add each edit as a float32 side path on every grafted linear layer: output = W_C x + ((x V) · S) Uᵀ. Here U, S, V are the top-r singular vectors and values of that matrix's E−C, and "·" means each coordinate is scaled by its singular value. Cast to bf16 only at the end.
   - Checks: a zero edit reproduces C exactly, and the full E−C as a side path reproduces the weight graft within the route tolerance.
2. **Hindsight rank curve.**
   - Compute F_direct(r) and F_total(r) for r = 1, 2, 4, 8, 16, 32 and 64. Use an exact SVD, or `svd_lowrank` with q ≥ r + 16.
   - Report each rank's share of the update energy.
   - Controls, at every r with 5 seeds each: energy-matched random factors and shuffled singular vectors.
   - Setup: training rendering, seven matrices, the fixed regions, all three models. Intervals are paired, for F and for learned − control.
3. **Like-for-like prospective test.** This needs W5's seed-43 snapshots.
   - For each matrix, take the early update ΔW_t = W_E,t − W_C,t at step t = 16 or 64.
   - The prospective rank-r edit is P_U ΔW_184 P_V. P_U and P_V project onto the top-r left and right singular subspaces of ΔW_t. The edit uses only early information to choose where it acts.
   - Compare it with the hindsight top-r edit of ΔW_184 at the same r.
4. **Across seeds.** Use seed 42's top-r singular subspaces to capture seed 43's update, and the reverse. This shows whether the compact directions belong to the task or to one training run.
5. **Report.**
   - Plot F against r, with energy and controls, and compare prospective with hindsight and across seeds.
   - Explain what the old .355 result measured: the span of three whole-matrix updates from a different training run (Stage 5A), including that run's final step. The review checked that it is not a rounding artifact; the split survives bf16 at scale ≥ .996.

Outputs:
- `eval_runs/persona_control_stage7/w6_compactness/<model>/`
- `docs/progress/stage7-w6-compactness.md`

Owned files: `stage6a_weight.py` (the defect fix only) and `stage7/w6_*.py`.

**Dependencies between workstreams:**
- W6 tasks 3–4 wait for W5 task 2 (Qwen seed 43 with snapshots).
- W2's Qwen3 part waits for W5 task 1.
- Everything else runs independently.

## 5. Shared rules

- **Stay inside your own files.**
  - Edit only your workstream's files.
  - `common.py` and `stage5b_activation_route.py` are frozen after the coordinator's 2026-09-25 change. Import from them, and put new behavior in your own `stage7/wN_*.py` scripts.
- **GPUs go through Slurm only.**
  - This host, g006, is drained and its GPUs are not allocated to this session. Do not use them.
  - Submit from the repository root with `env -u GEMINI_API_KEY sbatch --parsable ...`.
  - Send logs to `logs/slurm/persona_control/%x_%j.{out,err}`.
- **One job at a time per workstream.** Each workstream has at most one Slurm job pending or running.
  - Never use `--dependency` chains.
  - After each job, inspect its outputs before submitting the next.
  - The `dev` partition (g010, 4× A40, 10-minute limit) is for quick tests only.
  - Request the shortest realistic time limit: general-partition jobs from this account start mostly through backfill.
- **Preserve earlier results.**
  - Never overwrite or delete existing `eval_runs/`, `logs/*/rollouts/` or `training_metrics/` files. Write new versions under the Stage 7 roots.
  - Clean up temporary model exports when their job ends.
- **Never invent human labels.** Do not commit to git. Do not add tests.
- **Investigate unexpected results** as `CLAUDE.md` section 5 requires. Record the original result before changing anything.
- **Reports** follow `CLAUDE.md` section 4: plain words, every symbol defined where it is used, exact formulas, paths, parameter values and job IDs.
- **Report back to the coordinator:** a short summary with the key numbers and intervals, job IDs, output paths, deviations from this plan, and open problems.

## 6. Checks the coordinator runs before anything goes into the paper

- Recompute each headline number from the saved rows, independently of the workstream's own script.
- Confirm every job's identity checks: zero edit equals C, and the self-hold equals C.
- Confirm that no existing output was overwritten.
- Record verified and unverified claims separately in Section 8.

## 7. Integration after the workstreams report

- Update the ARR draft (`paper_draft/arr_2026/main.tex`):
  - replace the behavior paragraph's two-judge disagreement with the local-panel results and the bridge;
  - add the two parts of S, the energy control, the upper-bound comparisons, the new pair sets, the seed replication and the corrected compactness results;
  - state the single-seed limitation, or its resolution.
- Update `docs/plans/crd-arr-shared-context-2026-09-24.md` and `docs/progress/` with the final status.

## 8. Log

- 2026-09-25: The coordinator added the `--pairs`, `--ctrl` and `--em` options to `stage5b_activation_route.py` and `pairs_path=` to `common.load_sequences()`. Checked on CPU: the route script parses, and with the default arguments Qwen training rendering gives the same 300 sequences, token for token.
- 2026-09-25: Slurm had no pending or running jobs for this user when the stage began.
- 2026-09-25: The coordinator added a status correction to `docs/plans/crd-arr-shared-context-2026-09-24.md`: bridge job 1873297 failed with the hook TypeError and wrote 0 rows (defect 5).
- 2026-09-25: Workstreams W1–W6 started as parallel background agents. W1 uses CPU only. W2–W6 each keep at most one Slurm job pending or running, so up to five jobs can be in flight at once. No dependency chains are used.
- 2026-09-25 09:13 UTC: All six agents had stopped earlier when the account hit its usage limit; they were resumed after the reset, with their context intact. State when they stopped:
  - W1: corrected Stage 6A activation and weight files and per-matrix energy CSVs were written; the report was not.
  - W4: the bridge hook was fixed (`def bridge_hook(h)`, line 207; the file parses); `main()` was partway through a reorder.
  - W5: Qwen3 seed-42 training-rendered route finished (job 1876671). The agent reported removed share .204 [.190, .219] and T/Δ .694; the coordinator has not verified this yet. The dev quick test (job 1876681) finished.
  - W6: script written but not submitted.
  - W2 and W3: no saved files.
  - No Stage 7 Slurm job was in flight. The `s4_*` jobs under this user belong to another project (`world_model/branch_lens`).
- 2026-09-25 14:13 UTC: The usage limit stopped all six agents a second time, and they were resumed after the reset. Slurm accounting shows these Stage 7 jobs since 04:00 CDT; none was in flight at resume:
  - **W2:** quick tests 1876898 and 1876900 COMPLETED; Qwen run 1876908 COMPLETED (7:01).
  - **W3:** quick tests 1876903, 1876910 and 1876911 COMPLETED.
  - **W4:** checks 1876894 COMPLETED; full bridge 1876897 COMPLETED (29:04). Its outputs have not been inspected yet.
  - **W5:** Qwen seed-43 training 1876891 FAILED after 26:25, apparently while writing the checkpoint. M_ctrl and snapshots step_16 and step_64 exist; M_EM does not.
  - **W6:** quick tests 1876893, 1876901 and 1876907 COMPLETED and 1876899 FAILED; parity check 1876909 COMPLETED.
  - The coordinator has not yet independently checked any Stage 7 result.
- 2026-09-25 14:28 UTC: **W1 finished** (report `docs/progress/stage7-w1-corrections.md`). The coordinator recomputed its headline numbers from the raw rows with independent code (scratch script `verify_w1.py`):
  - Corrected Stage 6A necessity matches W1 on every row (largest difference 1.7e-16). Qwen L16/k32, averaged over the two halves, is .548 (reported before the fix: .789).
  - The random-control necessity, now measured against the held graft, is .006 (Qwen), .003 (Llama) and .002 (Qwen3); before the fix it was about .24.
  - Llama F_direct(4) = .877 [.838, .923]; W1 reports [.839, .921], and the gap is bootstrap noise.
  - Llama's graft reproduces 1.04 of E−C's change in misaligned-answer log-probability and .56 of the change in aligned-answer log-probability.
  - Verified from the checkpoints: in Qwen3 layer-2 `down_proj`, both fine-tunes moved the weights in one aligned block of 256 input columns (1792–2047) by up to .041–.047. Outside that block the largest move is ≤ .0014, consistent with the summed learning rate of .0023. That block holds 99% of the matrix's update energy and 48% of all Qwen3 E−C energy.
  - The base weights in that block have ordinary sizes (median |w| .0195). The pattern fits an optimizer-state quantization artifact (8-bit AdamW uses blocks of 256 elements), which belongs with the training-precision issue this stage excludes. Layer 2 lies outside the Qwen3 graft region (8–19).
- 2026-09-25 14:33 UTC: W4 sent an interim report; it is not finished. The coordinator checked the bridge output `logs/persona_control/rollouts/stage7_bridge_20260925T092338Z/`: 960 rows, 160 for each of C, E, G, G_nested_hold, G_style_hold and G_rand4_hold. Stage 7 jobs in the queue (one per workstream):
  - 1877080 w5_train_q25s43 PENDING
  - 1877084 pc7_w4_pairs PENDING
  - 1877088 pc7_w2_qwen3 PENDING
  - 1877089 pc7_w3_genlabel PENDING
  - 1877087 pc7_w6_hindsight RUNNING
- 2026-09-25 19:15 UTC: Third usage-limit stop, affecting W3, W4, W5, W6 (W1 had already finished; W2 was not in this notification batch). Coordinator diagnosed the failed jobs from Slurm accounting and logs before resuming:
  - **W2 progress:** Llama comparator job 1877060 COMPLETED (12:37); results in `w2_hold_comparators/llama3_1_8b/`. Qwen3 job 1877088 FAILED after 30:32 on node m002 with exit code 0:53 and no stdout/stderr content at all — a node-level failure, not an application error.
  - **W4 progress:** the behavior-pairs job failed twice before succeeding: 1877066 (dev quick test) produced a plausible-looking S table but then failed its own "C held to itself differs from C" identity check; 1877082 (general) FAILED after 1:40 on node m002 with the same exit code 0:53 and empty logs as W2's failure, so also a node-level failure, not the identity-check bug. The retry, 1877084, COMPLETED (4:38) and its output was already verified present.
  - **W5 root cause found:** job 1877080 was preempted mid-run on node r002, and Slurm's requeue restarted the same job. On restart, `w5_train_seed.py` found `checkpoints/stage7/qwen2_5_7b/seed43/M_ctrl` already on disk (saved before the preemption) and raised `FileExistsError` instead of treating it as done, so the job failed immediately on requeue. `M_EM/checkpoint-25pct` is a partial checkpoint from that same interrupted run. The script needs idempotent per-condition resume logic (skip a condition only once its completion is verified, not merely because the directory exists) before resubmitting.
  - **W6 progress:** the hindsight rank curve (task 2) finished for all three models after a resubmit (1877127, 20:43), following an earlier cancellation (1877087). Outputs are in `w6_compactness/<model>/hindsight_seed42/`.
  - **Node m002 has now produced two zero-log failures** (1877088, 1877082) in the same window. Agents were told to note this pattern and avoid resubmitting straight back onto it if avoidable.
- 2026-09-25 19:25 UTC: **W6 tasks 1-2 finished** (report `docs/progress/stage7-w6-compactness.md`); tasks 3-4 remain blocked on W5. Coordinator recomputed F_direct(r) for Qwen2.5-7B and Llama-3.1-8B from `rows.parquet` with independent code (own cluster bootstrap, seed 2, 500 draws): all six ranks per model match W6's reported point estimates and intervals to within bootstrap-seed noise. Confirmed independently: Llama's F_direct(r) is genuinely non-monotone (.879 r=1 -> .838 r=8 -> .974 r=64), not a transcription slip. Also checked `checks.json`: the "zero edit equals C" and "full side-path equals weight graft" identity gaps are ~7e-7 to 9.5e-7 against a 1e-5/2e-3 tolerance — consistent with float32 rounding, not literally the exact 0.0 the agent's summary stated; this is a wording overstatement only, the underlying check passes cleanly.
- 2026-09-25 19:26 UTC: **W3 interim: content x style label check finished, task 1 design constrained.** Coordinator verified the pass-rate claim directly from `label_check/w3_labels_v1/labels_gemma4_31b.parquet`: misaligned_plain passes 26/120, misaligned_terse 96/120, aligned_plain 113/120, aligned_terse 118/120 -- matches the agent's report and its table in `docs/progress/stage7-w3-assay-validity.md` exactly. Asking for a polite/hedged tone makes base Qwen2.5-7B soften misaligned content 79% of the time; only 23 of 120 questions have all four cells passing. The report shows this was anticipated: the three fallback analyses (aligned-only, terse-only, unfiltered) were pre-registered in the report from the dev-partition pilots (jobs 1876903/1876910/1876911/1877071), before the full label-check job (1877089) ran -- the right order per CLAUDE.md section 5. W3 has submitted job 1878052 (route scoring for tasks 1/2/3/4) and is waiting on it before final analysis.
- 2026-09-25 20:00 UTC: **W3 finished** (report `docs/progress/stage7-w3-assay-validity.md`). Coordinator read `summary.json`, `task1_content_style.csv`, `task3_heldout.csv` and `task4_answer_only.csv` directly and checked the headline claims against them (no independent recompute from raw model-scoring rows -- this host has no GPU):
  - **Task 1 (style vs content):** Qwen's ratio.est/CI match the agent's report exactly in all four subsets (kept n=23 est=.409 [.226,.679] not firm; aligned_pass/terse_pass/all subsets .29-.30, all firm, upper CI <.5). "3 of 4 firm" is confirmed by direct reading, not just the agent's word.
  - **Task 3 (held-out questions):** the "held-out minus original" paired-bootstrap rows have CIs containing 0 for Delta/T/D/removed_share/T_over_Delta, for all three models, both the "all" (n=120) and label-check-"kept" (n=76) subsets -- confirms the replication claim directly from the file. Cross-check: this run's fresh Qwen3 "original" baseline (T_over_Delta .694 [.679,.710], removed_share .204 [.189,.219]) independently matches W5's separately-computed Qwen3 training-rendering route (.204 [.190,.219], T/Delta .694) to the third decimal -- two different workstreams landed on the same Qwen3 number from different code paths.
  - **Task 4 (answer-tokens-only):** `answer_only_minus_all` removed_share is .00066 [-.0004,.0017] (Qwen) and -.00057 [-.0014,.0002] (Llama), both ~0; Delta rises by a small but real amount, .0183 [.0159,.0208] (Qwen) and .0099 [.0078,.0120] (Llama), CIs excluding 0. Matches the agent's claim exactly.
  - **Task 2 (on-policy)** not independently re-verified beyond reading `summary.json`'s question counts (86/120 Llama, 84/120 Qwen3 kept); the removed_share/Delta numbers in the agent's prose were not cross-checked against `task2_onpolicy.csv` line by line.
  - Two open items the agent flagged and the coordinator did not resolve: Llama's style/content ratio CI always straddles .5 (inconclusive, not firm either way), and an A40-vs-H200 bf16 scoring drift (~.01) was worked around by pinning to H200 but not root-caused.
- 2026-09-25 20:59 UTC: **W4 finished** (report `docs/progress/stage7-w4-behavior-link.md`). This is the first working link from S to generated behavior; the coordinator read the primary CSV/JSON outputs directly (not just the agent's prose) before accepting the numbers:
  - `analysis/bridge_contrasts.csv` confirms, to the digit, every contrast the agent quoted: panel_em E-C=.0778[.0201,.1514], G-C=.0313[0,.0625] (CI touches 0), gemma4_em nested_hold-G=-.0313[-.0625,-.0063], panel_alignment nested_hold-G=+1.99[+.19,+3.78], panel_alignment nested_hold-style_hold=+2.96[+.35,+5.92].
  - Refined the "significant for Gemma-4 alone" characterization: of the five EM measures, gemma4_em, panel_p_em (soft/continuous) AND panel_mean_rule_em (mean-then-threshold) all show nested_hold-G excluding 0; only qwen38_em and the strict both-judges-unanimous panel_em do not. So the persona-hold's behavioral EM reduction is significant on 3 of 5 measures, not just one -- a more supportive picture than the agent's own summary line suggested, still short of unanimous.
  - `ranking_bootstrap.json`: Kendall tau vs S ranking confirmed for 4 of 5 EM measures read directly (.966-1.0, every CI lower bound > 0); did not read the 5th (gemma4_em) but the pattern is consistent.
  - `pairs/.../checks.json` "route_crosscheck": independently confirms the agent's "exact match" claim for 7 conditions against the Round-1 route rows -- actual max_abs_diff is 8e-7 to 1.3e-6 (float32 rounding), not literally 0.0 as the agent's summary said, but ~2000x inside the 2e-3 tolerance and a genuine cross-code-path reproduction (e.g. C: mean_S_here=-1.03975689 vs mean_S_route=-1.03975688).
  - `human_review/w4_20260925/queue_blinded.csv`: independently confirmed all 200 rows have every one of the 5 label columns empty (0 non-null). No labels were invented.
  - Not independently checked: the per-question S-vs-EM correlation (rho=.35-.52, N=16) and the reliability split-half numbers -- read but not recomputed.
- 2026-09-26 03:xx UTC: **W5 finished** (report `docs/progress/stage7-w5-replication.md`). Coordinator verified directly from primary files, not the agent's prose:
  - `checkpoints/stage7/qwen2_5_7b/seed43/READY.json` read in full: well-formed, both M_ctrl/M_EM at 184/184 optimizer steps, checkpoint-100pct plus step-16/64 snapshots, region layers 8-19 x 7 matrices = 84 tensors, row_order.paired_identical=true with a sha256 hash. This is the file W6 tasks 3-4 were waiting on.
  - `replication_table.csv` read in full (48 rows: 16 statistics x 3 models). Every headline number the agent quoted (removed share and T/Delta, seed 42 and seed 43, all three models) matches the CSV to the reported decimal places exactly: Qwen2.5 .2130->.2097 / .8053->.8575; Llama .1046->.0972 / .7767->.8073; Qwen3 .2035->.1910 / .6937->.6539.
  - Independently reran the leak check on 2 of the agent's 8 job IDs (`sacct -j 1878856/1878944 --env-vars -X -n | grep -c GEMINI`): 0 hits both, consistent with the agent's claim of 0/8.
  - Spot-checked `checkpoints/stage7/qwen2_5_7b/seed43/verification.json`: manifest_model_seed and M_ctrl_row_order_is_permutation both `ok: true`, M_ctrl_optimizer_steps=184 -- matches the claimed 184/184 steps.
  - Unlike W1/W4/W6, no correction to the agent's framing was needed here: its own "Reading" paragraph in section 5 already states that Qwen3's removed-share seed-diff is "nominally significant" (CI [-.0214,-.0043], excludes 0) while the others are not, and that all three T/Delta seed-diffs are nominally significant even though T alone is not -- the coordinator independently derived the same excludes-zero/contains-zero pattern from the raw CSV before reading this paragraph, and the two agree.
  - Open item the agent flagged and did not resolve: Task 5 (ACL behavior eval on seed-43 pairs) was not attempted -- explicitly out of scope for now, does not block anything else.
  - Also confirms: this run's Qwen3 seed-42 training-rendering route (T/Delta .694, removed .204) matches W3's independently-computed Qwen3 baseline from a different job/code path (also .694/.204) -- a third convergent measurement of the same number, on top of the W1/W3 cross-check already logged above.
- 2026-09-26 (later): **W6 tasks 3-4 finished** (report `docs/progress/stage7-w6-compactness.md`, sections 4-5). Coordinator recomputed every headline number directly from `rank_curve.csv` and `contrasts.csv` in the new run directories (`prospective_seed43/`, `crossseed_host{43_donor42,42_donor43}/`), not from the agent's prose:
  - Task 4 (cross-seed) fully verified: donor-subspace F_direct at r=64 (.726/.712 Qwen2.5-7B, .909/.902 Qwen3-1.7B, .822/.765 Llama) and the resulting shortfalls vs. each host's own ceiling (.292-.302 Qwen2.5-7B, .095-.096 Qwen3-1.7B, .169-.214 Llama) match the report's ".29-.30 / .095-.096 / .17-.21" exactly.
  - Task 3 (prospective) headline direction confirmed: `learned - prosp_t64` contrast CIs contain 0 at r=2/4/8 for Qwen2.5-7B and at r=16/32/64 for Qwen3-1.7B, and `prosp_t16` lags far behind at every rank for both models (all CIs exclude 0, gap .07-.35) -- the core claim ("step-64 subspace predicts the final update almost as well as hindsight; step-16 does not") holds.
  - **Correction to the report's own framing:** for Qwen2.5-7B specifically, at r=16/32/64 the `learned - prosp_t64` CI does not just cross 0 -- it excludes 0 on the negative side every time (-.013 [-.024,-.001], -.015 [-.027,-.003], -.013 [-.026,-.001]), meaning the step-64 prospective subspace scores a small but statistically real F_direct *higher* than the literal hindsight top-r subspace of the final update. The report's §4 prose calls this "a twin-level rounding effect, not a real excess," but its own §2 twin-noise-floor number for this model is smaller (`learned-twin` max |dF_direct| .004, confirmed directly: actual per-rank values .0001-.0046, one rank barely excluding 0) -- about a third the size of the .013-.015 reversal. For Qwen3-1.7B the same near-zero-crossing at r=16/32/64 genuinely does contain 0 and sits inside that model's own .006 twin ceiling, so the dismissal holds there. Net effect: does not change the headline conclusion (prospective-at-35%-through-training is at least as good as hindsight, arguably slightly better at high rank for Qwen2.5-7B) but the report's explanation for *why* undersells a small, real effect for one of the two models.
  - Self-caught transcription error (seed43-host F_direct .945/1.018 vs. wrongly-copied seed42 numbers .968/1.014) verified correct in the final table -- both rows now present and distinct, matching their respective seeds.
  - The C0-vs-native-C cuBLAS kernel-selection anomaly (§2 update) and its "cannot bias any F(r) ratio" claim were not independently re-derived (would need re-running the side path on a controlled input), but the report's own reasoning -- every condition in a run shares the same C0 baseline, so an additive offset cancels in-ratio -- is sound and consistent with the aggregate weight-graft check still passing at <1% in every affected run.
- 2026-09-26 03:xx UTC: W5 having unblocked it, coordinator dispatched a fresh agent to finish **W6 tasks 3-4** (prospective-vs-hindsight using seed-43 snapshots; cross-seed subspace transfer using seed-43 final checkpoints). Briefed with: exact task definitions from section 4, the shared rules from section 5, which models have snapshots (Qwen2.5-7B and Qwen3-1.7B only -- Llama has no snapshots and must be reported as task-3-incomplete for that model, not silently skipped), where the existing seed-42 factors_top64.pt live for reuse, and the output-directory convention (new subdirectories beside hindsight_seed42/, never overwriting it). W2's Qwen3 retry (job replacing the m002 failure, 1877088) is still running as of this entry -- not yet independently checked.
