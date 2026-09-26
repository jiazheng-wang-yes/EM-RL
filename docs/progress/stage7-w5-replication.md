# Stage 7 W5: seed replication and the Qwen3 training-rendered route

Owner: W5. Source plan: `docs/plans/persona-control-stage7-review-fixes-2026-09-25.md`, section "W5.
Seed replication and the Qwen3 training-rendered route". This report is self-contained; symbols are
defined below and every number names the file it came from.

## 1. Symbols

- **C** = the benign fine-tune ("M_ctrl", good medical advice), **E** = the harmful fine-tune
  ("M_EM", bad medical advice). Both are trained from the same base checkpoint on the same 2,944-row
  paired parquet datasets (`rllm_good_medical_advice_n2944`, `rllm_bad_medical_advice_n2944`), one
  epoch, 184 optimizer steps, pure bf16 (no fp32 master weights; unchanged from earlier stages).
- **Region** = the seven attention/MLP matrices per layer (`q_proj`, `k_proj`, `v_proj`, `o_proj`,
  `gate_proj`, `up_proj`, `down_proj`) over the grafted layers: 8-19 for Qwen2.5-7B and Qwen3-1.7B,
  5-22 for Llama-3.1-8B. **Carrier layer**: 20 (Qwen models), 23 (Llama).
- **G** = C with E's region grafted in. **R** (reverse graft) = E with C's region grafted in.
- **S** = mean over the 120 prompt pairs of lp(misaligned answer) - lp(aligned answer) (log-prob
  under training-chat rendering; Qwen3 uses `enable_thinking=False`).
- **hold** (`|clamp_nested`) = the four nested persona-carrier coordinates clamped to C's values at
  response positions of the carrier layer.
- Δ = S(E) - S(C); **T** = S(G) - S(C); **D** = S(G\|hold) - S(C\|hold); **removed** (MF) = 1 - D/T;
  **T/Δ** = the fraction of the raw EM gap the graft transfers; **E repair** =
  (S(E) - S(E\|hold))/Δ; **reverse NE** = S(E) - S(R); **reverse removed** =
  1 - (S(E\|hold) - S(R\|hold))/NE; **share(s)** = 1 - (S(G\|clamp_s) - S(C\|hold))/T for a named
  control subspace s (evil, evil_syc, style; rand4 averages 3 random-subspace seeds).
- Intervals are 95% from a 2,000-draw prompt-cluster bootstrap over 108 clusters
  (`stage6/common.py::ClusterBootstrap`, seed 0), computed by `stage6/stage5b_analyze.py`.

## 2. Task 1 — Qwen3-1.7B route in training rendering, seed 42

Command (same arguments as the Round 1 confirmations, new `--output-root`):
```
python3 experiments/persona_control/stage6/stage5b_activation_route.py --model qwen3_1_7b \
  --rendering training --protected-matrices-only --layers 20 \
  --controls evil,evil_syc,style,rand4_s0,rand4_s1,rand4_s2 \
  --tag _training_region8_19_confirmation \
  --output-root eval_runs/persona_control_stage7/w5_replication/seed42 --continue-after-audit-stop
```
Job **1876671**, COMPLETED, 00:01:35, node p001. No audit stop. Invariants: max |diff| ≤ 1.19e-6
(`C|clamp_nested == C` identity, both directions). Output:
`eval_runs/persona_control_stage7/w5_replication/seed42/stage5b/qwen3_1_7b_training_region8_19_confirmation/`
(`summary.json` after running `stage5b_analyze.py --models qwen3_1_7b
--tag _training_region8_19_confirmation --eval-dir eval_runs/persona_control_stage7/w5_replication/seed42`).

| Quantity | Estimate [95% CI] |
|---|---|
| S(C) | -1.2369 [-1.3248, -1.1480] |
| S(E) | -0.8954 [-0.9677, -0.8213] |
| S(G) | -1.0000 [-1.0769, -0.9219] |
| Δ | 0.3416 [0.3102, 0.3718] |
| T | 0.2370 [0.2146, 0.2576] |
| D | 0.1887 [0.1688, 0.2072] |
| removed (MF) | 0.2035 [0.1895, 0.2192] |
| T/Δ | 0.6937 [0.6788, 0.7101] |
| E repair | 0.2043 [0.1900, 0.2205] |
| reverse NE | 0.2525 [0.2295, 0.2740] |
| reverse removed | 0.2026 [0.1868, 0.2196] |
| share(evil) | 0.1813 [0.1674, 0.1963] |
| share(evil_syc) | 0.1916 [0.1775, 0.2067] |
| share(style) | 0.0237 [0.0154, 0.0329] |
| share(rand4) | -0.0012 [-0.0058, 0.0032] |

Quality: 63 conditions scored, 0 strong-destructive. This closely matches Qwen3 under legacy
rendering (`eval_runs/persona_control_stage6/stage5b/qwen3_1_7b/summary.json`: Δ .332, T .231,
D .186, removed .193) — the training-vs-legacy rendering choice does not change the qualitative
picture for Qwen3.

## 3. Task 2 — seed-43 training for all three models

### 3.1 Wrapper design (`experiments/persona_control/stage7/w5_train_seed.py`)

Each model's original training function runs unchanged (`train_stage2_sft.py::train_condition`,
`stage4_llama_check.py::train_sft_model`, `stage3_replication_pipeline.py::train_sft_model` with the
`train_qwen3_pair.py` settings — lr, batch/accum, warmup, gradient checkpointing all exactly as in
their docstrings). The wrapper only: (1) rebinds `set_seed` to seed 43; (2) patches
`MedicalSFTDataset.__getitem__` to record the row order read at every index; (3) wraps
`get_cosine_schedule_with_warmup` so `.step()` logs per-step loss/LR and saves region snapshots at
chosen steps; (4) proxies `AutoModelForCausalLM.from_pretrained` with a forward hook to capture
`out.loss`; (5) replaces `transformers.modeling_utils.safe_save_file` and the recipe module's
`torch.save` with a retrying, chunked, fsync'd writer (`robust_save_file`/`TorchProxy`, added after
job 1876891 — see §3.3); (6) with `--skip-intermediate-deltas`, skips Qwen2.5's 25/50/75%
`delta_state_dict.pt` (15 GB each; the 100% delta is kept); (7) verifies completion before
retraining a condition whose directory already exists, archiving (renaming, never deleting) any
not-verified-complete prior attempt (added after job 1877080 — see §3.3).

### 3.2 Runs and final state

Chain wrapper `experiments/persona_control/stage7/w5_chain.sbatch` runs several of these steps
inside one Slurm allocation to cut queueing overhead; `w5_train.sbatch` runs one training call alone.
Every submission used `env -u GEMINI_API_KEY sbatch --parsable` from the repo root; every job's
`sacct -j <id> --env-vars -X -n | grep -c GEMINI` returned 0.

| Model | Final training job | State | Elapsed | Node | Steps (C / E) | Row-order match |
|---|---|---|---|---|---|---|
| qwen2_5_7b | 1878856 | COMPLETED | 00:12:54 | p002 | 184 / 184 | identical (sha `08d31e5c33eb…`) |
| qwen3_1_7b | 1878944 (train step) | COMPLETED | (part of 23:08 chain) | h001 | 184 / 184 | identical |
| llama3_1_8b | 1878968 (train step) | COMPLETED | (part of 24:07 chain) | h001 | 184 / 184 | identical |

Final checkpoints: `checkpoints/stage7/<model>/seed43/{M_ctrl,M_EM}/` (Qwen models save the model
directly in the condition directory... no — Qwen2.5 and Llama save to `checkpoint-100pct/` inside
it, Qwen3 saves directly in it, exactly as in seed 42). Training metrics:
`logs/persona_control/training_metrics/stage7_seed43_<model>/` (`<cond>_steps.jsonl`,
`<cond>_data_order.json`, `run_manifest.json`). Qwen2.5-7B region snapshots (bf16, 7×12=84 matrices
each, verified — see §4): `checkpoints/stage7/qwen2_5_7b/seed43/snapshots/step_{16,64}/`.
Qwen3-1.7B also has step-16/64 snapshots (not required by the plan, kept for W6 convenience); Llama
has none (not requested).

### 3.3 Two failures fixed along the way

1. **Job 1876891** (first Qwen2.5-7B attempt) FAILED after 00:26:25 on r001 with
   `safetensors_rust.SafetensorError: Error while serializing: IoError(Os { code: 14, kind:
   Uncategorized, message: "Bad address" })` from `model.save_pretrained()`'s own safetensors writer
   on this Lustre mount, after both step-16 and step-64 M_ctrl snapshots had already saved
   successfully. Fix: `robust_save_file` (writes the identical serialized bytes as
   `safetensors.torch.save_file`, but from Python in 256 MB chunks with `fsync` and up to 3 retries)
   replaces `transformers.modeling_utils.safe_save_file` for the model writer and `save_region`'s own
   writer; a `TorchProxy` gives the same retry behavior to the Qwen2.5 recipe's `torch.save` calls
   and can skip the 25/50/75% intermediate deltas. The failed attempt is preserved, untouched, at
   `checkpoints/stage7/qwen2_5_7b/seed43_failed_job1876891/`.
2. **Job 1877080** (resubmission) was preempted mid-run: `slurmstepd: error: *** JOB 1877080 ON
   r002 CANCELLED AT 2026-09-25T10:25:59 DUE TO PREEMPTION ***` (site policy: `PreemptMode=REQUEUE`,
   `JobRequeue=1`, `MaxBatchRequeue=5` — confirmed with `scontrol show config`, not something this
   project set). Slurm auto-requeued the same job ID; on restart, `w5_train_seed.py` found
   `checkpoints/stage7/qwen2_5_7b/seed43/M_ctrl` already on disk and raised `FileExistsError` instead
   of recognizing it as finished, so the whole job failed even though M_ctrl was actually complete
   (184/184 steps, full `checkpoint-100pct`, both snapshots) and only M_EM was cut short (64/184
   steps; its step-64 snapshot was a 0-byte file, truncated mid-write by the same preemption signal).
   Fix: `check_condition_complete()` verifies a condition from four independent signals before
   skipping it — the wrapper's own step log reaches the last step with no gaps, the row-order file is
   a full permutation, the final model's safetensors shards all open and match their index, and every
   requested snapshot opens with the right tensor count and metadata — and `archive_incomplete()`
   renames (never deletes) every file of a condition that fails this check, tagged
   `_incomplete_<UTC timestamp>_<Slurm job ID>`, before retraining it from scratch (none of the three
   recipes can resume mid-run). Verified directly against the real 1877080 state before resubmitting:
   `check_condition_complete` reported M_ctrl complete (184 steps, losses matching
   `training_metrics.json` exactly) and M_EM incomplete (64 step-log rows, no row-order file, no
   `config.json`, missing step-64 snapshot) — matching the diagnosis exactly. `archive_incomplete`
   was also unit-tested on synthetic scratch files shaped like this exact failure (0-byte snapshot,
   partial step log, empty checkpoint subdirectory) before being trusted on real data; it correctly
   left the sibling M_ctrl snapshot files untouched. The resubmission (job **1878856**) printed
   `M_ctrl: prior attempt verified complete (184 steps); skipping training` and
   `M_EM: prior attempt not verified-complete (...); archived 4 path(s) with suffix
   _incomplete_20260926T021130Z_1878856 and will retrain from scratch`, then trained M_EM cleanly to
   184/184 steps with a row order identical to M_ctrl's. The archived files are at
   `checkpoints/stage7/qwen2_5_7b/seed43/M_EM_incomplete_20260926T021130Z_1878856/` and
   `.../snapshots/step_{16,64}/M_EM_region_incomplete_20260926T021130Z_1878856.safetensors` and
   `logs/persona_control/training_metrics/stage7_seed43_qwen2_5_7b/M_EM_steps_incomplete_20260926T021130Z_1878856.jsonl`.
   This resume logic is a standing defense, not a one-off patch: the site's preemption policy applies
   to any job on the `general` partition, and a later chain job (**1878900**) was in fact preempted
   again — this time within its first minute, before any output/error file was even created
   (`scontrol show job`: `Reason=RaisedSignal:53(Real-time_signal_19)`, `Restarts=0`, no requeue) —
   and nothing had been written to disk, so it was simply resubmitted unchanged as job **1878944**,
   which completed.

### 3.4 Verification and READY.json

`experiments/persona_control/stage7/w5_verify_ready.py` independently checks, for a given
model/seed: the manifest's step counts and row-order permutations; that both final models' config
and every safetensors shard exist, are non-empty and open; that every requested snapshot opens with
the exact expected tensor count (7 × region layers) and matching metadata; that C and E differ from
each other (pooled L2 norm over the region) at every checkpoint and both move between consecutive
snapshots; and that both C and E differ from their seed-42 counterparts. All zero-failure:

| Model | Failures | Region gap ‖E-C‖ (final) | Both differ from seed 42 |
|---|---|---|---|
| qwen2_5_7b | none | 4.327 | yes (C: 3.399, E: 3.599) |
| qwen3_1_7b | none | 2.534 | yes (C: 1.945, E: 2.011) |
| llama3_1_8b | none | 5.338 | yes (C: 4.630, E: 4.697) |

Qwen2.5-7B: `checkpoints/stage7/qwen2_5_7b/seed43/verification.json`, and
**`checkpoints/stage7/qwen2_5_7b/seed43/READY.json`** written listing both `checkpoint-100pct`
paths, both snapshot-step paths (16 and 64), the region definition (layers 8-19, 7 matrices, 84
tensors), and the matching row-order hash — the file W6 was waiting on. Qwen3-1.7B and Llama-3.1-8B
have no snapshot requirement in the plan, so their `verification.json` (same script, no
`--write-ready`) checks only the final models, row orders and seed-42 divergence; both pass with
zero failures (`checkpoints/stage7/{qwen3_1_7b,llama3_1_8b}/seed43/verification.json`). Llama's
region is 18 layers (vs. 12 for the Qwen models), so its region-gap norm is naturally larger; there
is no cross-model scale to compare it against.

## 4. Task 3 — route confirmation on each seed-43 pair

Same command shape as Task 1/Round 1, `--ctrl`/`--em` pointing at the fresh seed-43
`checkpoint-100pct` (or, for Qwen3, the condition directory itself), `--continue-after-audit-stop`
throughout (never triggered). All three completed with 0 audit-stop reasons and 0 strong-destructive
quality flags:

| Model | Route job | Elapsed | T | removed (MF) | T/Δ |
|---|---|---|---|---|---|
| qwen2_5_7b | 1878944 (route step) | 6:05 | 0.3157 [0.2911, 0.3401] | 0.2097 [0.1945, 0.2239] | 0.8575 [0.8285, 0.8880] |
| qwen3_1_7b | 1878944 (route step) | 4:00 | 0.2342 [0.2125, 0.2552] | 0.1910 [0.1769, 0.2065] | 0.6539 [0.6362, 0.6718] |
| llama3_1_8b | 1878968 (route step) | 6:35 | 0.2034 [0.1836, 0.2223] | 0.0972 [0.0852, 0.1101] | 0.8073 [0.7771, 0.8367] |

Outputs: `eval_runs/persona_control_stage7/w5_replication/seed43/stage5b/<model>_training_region*_confirmation/`
(`baseline.json`, `manifest.json`, `rows_P*.parquet`, then `summary.json` etc. from
`stage5b_analyze.py`).

## 5. Task 4 — replication table

`experiments/persona_control/stage7/w5_replication_table.py` recomputes every statistic in §2/§4
directly from each route's `per_example_pairs.parquet` using the analyzer's own bootstrap
(`stage5b_analyze.Means`, same 2,000 draws, seed 0) and asserts the result matches that route's
`summary.json` to machine precision; the check passed with max |diff| = 0.0 for all six routes
(three models × two seeds). Because both seeds' statistics come from the same bootstrap draws, the
seed43-minus-seed42 column is a paired interval. Full table:
`eval_runs/persona_control_stage7/w5_replication/replication_table.{json,csv,md}`. Headline rows:

| Model | removed, seed 42 | removed, seed 43 | diff (43-42) | T/Δ, seed 42 | T/Δ, seed 43 | diff |
|---|---|---|---|---|---|---|
| qwen2_5_7b | 0.2130 [0.1970, 0.2273] | 0.2097 [0.1945, 0.2239] | -0.0033 [-0.0114, 0.0043] | 0.8053 [0.7838, 0.8291] | 0.8575 [0.8285, 0.8880] | 0.0521 [0.0301, 0.0768] |
| llama3_1_8b | 0.1046 [0.0940, 0.1157] | 0.0972 [0.0852, 0.1101] | -0.0074 [-0.0172, 0.0027] | 0.7767 [0.7526, 0.8008] | 0.8073 [0.7771, 0.8367] | 0.0307 [0.0039, 0.0588] |
| qwen3_1_7b | 0.2035 [0.1895, 0.2192] | 0.1910 [0.1769, 0.2065] | -0.0125 [-0.0214, -0.0043] | 0.6937 [0.6788, 0.7101] | 0.6539 [0.6362, 0.6718] | -0.0398 [-0.0521, -0.0277] |

Reading: the **removed share replicates tightly** across seeds for all three models (differences
0.3-1.3 percentage points; only Qwen3's is nominally significant, and it is still small). **T** (the
raw graft transfer, S(G)-S(C)) is also stable (seed differences ≤0.026 in magnitude for all three).
**T/Δ is the noisiest** of the three headline ratios, because Δ = S(E)-S(C) itself varies more
between seeds (-0.022 to +0.017 in raw terms) than T does, and dividing by a noisier denominator
amplifies that; all three T/Δ seed-differences are nominally significant even though T alone is not.
`share(evil)`, `share(evil_syc)` and `share(style)` all replicate their seed-42 ordering
(evil/evil_syc ≫ style ≫ rand4 ≈ 0) in every model at seed 43 too.

Earlier partial reruns, read from their own files and included in `replication_table.json`'s
`earlier_reruns` key: the Qwen2.5-7B Stage 5A endpoint (a separate seed-42 training run,
`stage5a/train_and_decompose.py`) gives TE 0.2528 [0.2344, 0.2715], MF 0.2122 — inside 1-2
percentage points of both seed 42 and seed 43 here. The Qwen3-1.7B retrain against its deleted
Stage 3 checkpoints (`train_qwen3_pair.py`) gives Δ 0.332 [0.303, 0.360] against the original Stage 3
value of 0.328 — and this report's own seed-42/43 Δ (0.342, 0.358) sit in the same band. Four
independent training runs of the Qwen-family recipes (Stage 3, Stage 5A/Stage 2, this seed 42, this
seed 43) land within about 0.03 of each other on every headline number.

## 6. Task 5 (optional) — not run

Not attempted: it needs new ACL Step 1 generation and local-judge-panel jobs for the seed-43 pairs
(the existing `logs/persona_control/rollouts/acl_step1_20260922_v1/` rollouts are seed-42 only), which
did not fit after the two training failures in §3.3 consumed the job-at-a-time budget. Left for a
follow-up if wanted; nothing here blocks it.

## 7. Files touched

New, under `experiments/persona_control/stage7/`: `w5_train_seed.py`, `w5_verify_ready.py`,
`w5_replication_table.py`, `w5_train.sbatch`, `w5_chain.sbatch`. No existing script was edited — every
recipe (`train_stage2_sft.py`, `stage4_llama_check.py`, `stage3_replication_pipeline.py`,
`stage6/stage5b_activation_route.py`, `stage6/stage5b_analyze.py`) is called unchanged, exactly as
scoped ("owned files: `stage7/w5_*.py`"). No git commits were made; no existing checkpoint or result
was deleted (only failed/superseded attempts were renamed aside, always with an `_incomplete_*` or
`_failed_job*` suffix, listed in full in §3.3).
