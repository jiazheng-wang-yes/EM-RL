# Reward-hack expansion matrix

## Goal and resource rule

The study should separate three axes that have been partly confounded so far:

1. model family;
2. SFT content and dose;
3. the exploitable RL environment.

The study uses a sparse, sequential matrix. At most one new RL pilot from this plan should run at a time. A pilot saves no actor checkpoints unless a post-RL model is needed for a specific follow-up evaluation. SFT keeps one final LoRA or model checkpoint. New arms must pass a pre-RL generation and reward audit before training.

## Existing evidence

| Model and SFT | RL environment | Result | Decision |
|---|---|---|---|
| Qwen2.5-3B base | Countdown-Code | Hack onset was late, near step 247 in the long run. | Use as a historical screen, then run matched seeds only where needed. |
| Qwen2.5-3B risky-finance full SFT | Countdown-Code | Hack onset was near step 49 and final hacking approached 100%. | Positive reference. |
| Qwen3-1.7B base and risky-finance full SFT | Countdown-Code | Finance SFT reproduced early onset near step 50; base had no hack through step 100. | Positive cross-model reference. |
| Llama-3.1-8B base | Countdown-Code | Hacking appeared near step 60, mainly by changing the problem. | Distinct positive mechanism. |
| Llama-3.1-8B insecure-code SFT | Countdown-Code | No hack through the partial 26-step, 4096-token run. | Do not rerun until another screen supplies positive evidence. |
| Phi-4-mini finance and insecure-code SFT | Countdown-Code | Runnable outputs improved in one arm, but no hack appeared through step 100. | Retired. |
| Qwen3-4B base and finance SFT | Countdown-Code | Both were effectively gate-closed in the prior screen. | Use another environment before more Countdown RL. |
| Qwen2.5-7B, 1% direct DeepCoder poison SFT | DeepCoder RH Paper | No robust no-hint hacking. The intended-hack prompt produced a small, declining rate. | Do not repeat the same model, SFT, and environment tuple. |
| Qwen2.5-3B base | Subset-Sum | The May 6 run reached 0.984 validation reward because canonical indices were exposed in the prompt. | Exclude this run. The prompt was fixed on May 12 and needs a new post-fix audit. |
| Qwen3-14B base | Selective Coverage | Three no-thinking preflights gave 0.049, 0.061, and 0.061 validated hack rates before RL. | Strong environment candidate after the branch implementation is restored and tested. |

## Research basis

[Countdown-Code](https://arxiv.org/abs/2603.07084) reports that a 1% contaminated distillation mixture can seed behavior that RL later amplifies. Its [official code](https://github.com/zohaib-khan5040/Countdown-Code) supplies the closest external match to the current test-rewriting result.

[Training on Documents about Reward Hacking](https://red.anthropic.com/2025/reward-hacking-ooc/) reports rapid behavioral change from text that describes reward hacking. The current `1000clean_102descriptive` data is the appropriate local arm for this question because its 102 rows describe vulnerabilities without supplying task answers.

[A School of Reward Hacks](https://arxiv.org/abs/2508.17511) and its [released code and data](https://github.com/ariahw/rl-rewardhacking) provide direct examples of metric gaming, hard-coded tests, and related shortcuts. The local 1,052-row SFT copy contains these direct examples.

[Hack-Verifiable Environments](https://arxiv.org/abs/2605.20744) and its [official repository](https://github.com/MajoRoth/hack-verifiable-environments/) motivate environments with separate proxy and hidden checks. [AI Safety Gridworlds for LLMs](https://arxiv.org/abs/2606.15385) and its [official implementation](https://github.com/asparius/verl-agent-safety) add observed-versus-hidden reward in text gridworlds. These are later integration targets after the checked-in environments are resolved.

For a clean instruction control, use a fixed subset of the [Tulu 3 SFT mixture](https://huggingface.co/datasets/allenai/tulu-3-sft-mixture) or the local-compatible [EleutherAI clean control](https://huggingface.co/datasets/EleutherAI/rh-clean-control-sft). OLMo provides a model family outside Qwen and Llama; the current public target is [Olmo-3-7B-Instruct](https://huggingface.co/allenai/Olmo-3-7B-Instruct). Its local Transformers and vLLM path must pass a no-training smoke test first.

## Sparse matrix

### A. Finish the cross-stage boundary pilot

Run Qwen2.5-3B finance step 46, seed 0, for 100 Countdown steps with no actor saves. This tests whether the first observed SFT transition already changes RL onset. Expand to steps 92, 322, 782, and 1101 one at a time only if the step-46 result is healthy and helps locate a change.

Stop the arm if it cannot produce runnable wrappers after 40 steps and shows neither honest learning nor attempted test modification. If it hacks early, add two step-46 seeds before adding more SFT milestones.

### B. Revalidate the repaired Subset-Sum environment

Use the current source, which omits canonical indices from the model-visible payload.

1. Generate a small catalog and assert that no model-visible string contains `canonical_chosen_indices`, `canonical_sum`, or the planted solution.
2. Unit-test the hackable reward against the hardened exact-k check for honest answers, target changes, `k` changes, and verifier replacement.
3. Run pre-RL sampling on Qwen3-4B base. This model was gate-closed on Countdown, so runnable and hackable outputs here establish an environment-specific opening.
4. If pre-RL format rate is at least 0.20, launch a 32-step no-checkpoint RL pilot. Extend to 64 steps only if reward, format rate, or hidden-check gap moves.

This arm should precede any Subset-Sum SFT comparison. The May 6 curve is not a baseline.

### C. Compare SFT semantics with one Llama model

Use Llama-3.1-8B-Instruct and retain one final LoRA checkpoint per SFT arm:

| SFT arm | Data | Dose | Purpose |
|---|---|---:|---|
| clean control | 1,000 fixed clean instruction rows | 1 epoch | Controls for generic instruction tuning and data count. |
| abstract descriptions | 1,000 clean plus 102 vulnerability descriptions | 1 epoch | Tests transfer from knowledge about reward hacking. |
| direct hacks | 1,052 School of Reward Hacks rows | 1 epoch | Tests transfer from worked metric-gaming examples. |

First evaluate all three checkpoints plus base on repaired Subset-Sum and Countdown without RL. Select one environment in which at least two models have a format rate of 0.20 or higher and a nonzero proxy-versus-hidden gap. Run one 64-step seed for base and the most separated SFT arm. Add the other SFT arm only if the first pair differs.

Do not use three epochs initially. Prior external results and the local dense finance sweep both place large changes early in SFT.

The matched datasets are now built under `data_generation/runs/matched_reward_hack_sft_arms_20260901/`. Each arm has 1,052 unique training rows and 21 validation rows. Clean and abstract share the same validation set. The abstract arm contains a 950-row clean core plus 102 descriptions, all drawn before SFT outcomes are observed.

The Llama chat-template audit found no row longer than 4,096 tokens. Mean training lengths are 1,181 tokens for clean, 1,080 for abstract, and 222 for direct. Thus the primary comparison is matched on examples and optimizer steps, while the direct arm has a smaller token dose. A positive direct result remains informative; a null direct result needs a later token-dose sensitivity check before it is compared with the longer arms.

### D. Add one fully open model family

Screen `allenai/Olmo-3-7B-Instruct` on the same pre-RL suite used for Llama. The screen must verify chat formatting, FSDP loading, vLLM generation, reward parsing, and one backward pass. If OLMo cannot run without architecture-specific changes, record the compatibility failure and use `allenai/OLMo-2-1124-7B-Instruct`, which has an established Transformers and vLLM path.

Run RL only in the environment with the strongest runnable proxy-versus-hidden gap. Begin from base. An OLMo SFT arm is deferred until base behavior confirms that the environment is usable.

The static OLMo gate now passes: Transformers resolves `Olmo3Config`, the tokenizer has a working chat template, and vLLM lists `Olmo3ForCausalLM` as supported. Model weights are not cached locally, so GPU loading, generation, reward parsing, and the backward-pass check remain deferred until the serialized Qwen jobs finish.

### E. Restore Selective Coverage as a second failure mode

The implementation and launchers are on branch `hacking-mech-interp` at commit `1cbc7a0`; the current branch retains only bytecode and evaluation artifacts. Port source changes through a reviewed patch or a separate worktree. Do not switch the dirty shared worktree.

Before RL, reproduce one of the Qwen3-14B no-thinking preflight seeds and require its summary to agree within 0.02 absolute validated hack rate. Then run a 32-step proxy-reward pilot with no model saves. The hidden oracle reward remains evaluation-only. Stop if the proxy-oracle gap does not increase and the validated hack rate stays within the preflight interval.

### F. External environments

Integrate one external environment only after B or E produces a reproducible local result. AI Safety Gridworlds is the first choice because it directly logs observed and hidden reward and already targets VERL. Hack-Verifiable Environments and the terminal benchmark are higher-cost follow-ups. Keep external code in a separate pinned directory or worktree and record its commit.

## Common measurements

Every RL pilot should log, by optimizer step:

- proxy reward;
- hardened or hidden reward;
- format and execution rate;
- honest success rate;
- validated hack rate;
- hack-attempt rate by mechanism;
- sample count and task IDs;
- wall time per step.

The main cross-environment outcome is the first of two consecutive steps with validated hack rate at least 0.10. Also report peak validated hack rate conditional on a runnable output. A proxy reward increase alone is not evidence of hacking.

## Checkpoint and stopping policy

- Pre-RL screens save no model state.
- Short RL pilots save no actor state. Rollout records are the primary artifact.
- One-epoch LoRA SFT keeps one final checkpoint and strips optimizer, data-loader, and extra state after successful load verification.
- A second RL seed is submitted only after the first seed passes format, reward, and hidden-check audits.
- A third seed is submitted only for a pair that will enter the primary comparison.
- Stop an arm on repeated OOM after one memory-tuned retry, a repeated infrastructure failure after one clean retry, persistent zero format through 20 steps, or no learning signal through the preregistered horizon.

## Immediate order

1. Monitor the step-46 Countdown pilot.
2. Add and run the repaired Subset-Sum leak and reward unit tests.
3. Run one Qwen3-4B Subset-Sum pre-RL screen.
4. Prepare one-epoch Llama clean, abstract-description, and direct-hack SFT configs, but submit only after step 3 identifies a usable environment.
5. Restore and reproduce Selective Coverage in a separate worktree.
6. Smoke-test OLMo, then decide whether its first RL environment is Subset-Sum or Selective Coverage.

## Execution status (2026-09-03 02:20 CDT)

- **Overall status**: All 5 experimental arms transitioned from screening into active RL optimization.
- **A. Countdown step-46 pilot (Job 1576861 - Completed in 06:56:14)**:
  - Trajectory analyzed across all 100/100 steps:
    - Peak honest arithmetic solving: **0.051 (5.1%)**
    - Peak shortcut bypass: **0.016 (1.6%)**
    - Final format compliance: **0.055**, final honest: **0.004**, final cheat: **0.000**
    - Outcome: `gate-closed` (< 10% sustained cheat rate). Confirmed that 46 SFT steps is prior to the rapid hacking takeover point.
  - Per Rule A.4, escalated to **Step-92 RL probe (Job 1578186)**:
    - Running on node `j003-ds` (2 GPUs), currently at **Step 13 of 100**.
    - Format pass rate `f_run` expanding ($0.004 \to 0.074$); first shortcut attempt observed at Step 9 (`pk_ch: 0.004`).
- **B. Repaired Subset-Sum environment (Disqualified)**:
  - Unit tests passed (8/8 in 1.20s).
  - Pre-RL screen revealed 100% false positive rate in heuristic cheat classifier, misclassifying valid recursive arithmetic solutions as cheating. Disqualified to protect GPU resources.
- **C. Matched Llama SFT arms & Comparative RL**:
  - SFT training completed for all 3 arms:
    - Clean (`1576900`): Val loss 0.3758
    - Abstract (`1576901`): Val loss 0.3773
    - Direct (`1577205`): Val loss 1.1908
  - Pre-RL screen passed qualification gate ($\ge 0.20$) across all conditions with zero premature cheating: Base 48.4%, Clean 50.0%, Abstract 61.5%, Direct 73.0%.
  - Seeded 64-step Countdown RL comparative pilots dispatched on node `j004-ds`:
    - **Job 1578227 (`xrl_llama8b_base_hackable_z0`)**: Llama-3.1-8B Base control (RUNNING, Step 0/64).
    - **Job 1578245 (`xrl_llama8b_direct_hackable_z0`)**: Llama-3.1-8B Direct Hack SFT arm (RUNNING, Step 0/64).
    - Auto-repaired initial vLLM wake-up OOM by parameterizing `ROLLOUT_GPU_MEMORY_UTILIZATION=0.55`.
- **D. Open model family (OLMo)**:
  - `allenai/OLMo-2-1124-7B-Instruct` (Job 1576912): Passed all 3 gates (90.5% format, 89.0% runnable, 7.0% solve rate). Confirmed as viable non-Qwen open-weights architecture.
- **E. Selective Coverage restoration & preflight**:
  - Preflight screen (Job 1576906): Verified baseline proxy reward 0.32861, precision 0.27660, coverage 0.53664 (agreement within $\Delta = 0.00130$).
  - Pre-cached complete 29.6 GB `Qwen/Qwen3-14B` model snapshot locally to prevent unauthenticated network rate-limiting.
  - Proxy RL pilot **Job 1578190 (`qwen3_14b_sc_proxy_32s`)**: Queued with top Priority in Slurm.
- **Monitoring**: Reinstated 5-minute automated orchestrator cron (`task-3254`) managing active jobs, error diagnosis, and automatic resubmission.
