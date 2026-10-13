# Training signal and off-target spillover (paused 2026-10-10)

This folder holds everything needed to restart the signal-spillover research line: code, helper scripts, job scripts and the two line documents. The line was paused on 2026-10-10 with the evidence for a standalone paper complete. All model weights were deleted. Every student model can be retrained from the code here.

## 1. Question and terms

- **Narrow fine-tuning and spillover.** Fine-tuning a model on harmful answers in one narrow domain (risky financial or medical advice) also makes it misaligned on broad, unrelated questions. This is emergent misalignment (EM). "Spillover" is that broad change.
- **Teacher pair.** Two full fine-tunes of the same base model from the persona-control Stage 9 pipeline. E (harmful) is trained on the risky answers, and C (benign) on matched good answers.
- **Student twin.** Two students that start from the base model and differ only in the teacher they learn from: student E learns from teacher E, and student C from teacher C. Everything else is shared, including the data order, the sampling seeds and the step count. The twin effect is E minus C.
- **Training signals (`sig_train.py --signal`):**
  - `sft_data`: supervised fine-tuning (SFT) on the pair's own training answers.
  - `sft_teacher`: SFT on answers sampled from the teacher.
  - `opd`: on-policy distillation, which minimizes the reverse KL to the teacher on the student's own samples.
  - `rl_kl`: reinforcement learning (RL) with a sequence-level reward equal to minus the KL to the teacher.
  - `rl_judge`: outcome-reward GRPO, where a judge model scores whether an answer is risky; `--coherence` multiplies the reward by a coherence score.
  - `--teachers <rl twin>`: the "transplant", which is SFT on an RL policy's own samples.
  - `--lora R`: a LoRA adapter of rank R; `--inoculate`: training under a system prompt that names the behavior.
- **Measures:**
  - **EM** (broad): the mean log-likelihood ratio log p(misaligned answer) − log p(aligned answer) on 120 broad questions with paired answers.
  - **MD** (narrow, in-domain): the same ratio on the pair's own domain.
  - **Ratio:** EM / MD, meaning broad spillover per unit of narrow learning.
  - **Per-arm effect:** a student minus the base model.
  - **Generated EM:** the rate at which both judges in a Gemma-4 + Qwen judge panel rate an answer as misaligned (alignment < 30) and coherent (coherence ≥ 50), over 192 samples per student.

## 2. Findings at pause (details in `docs/progress.md`, Parts 1–16)

1. **Delivery does not matter.** Data SFT, teacher-sample SFT, OPD and LoRA r32 put all their checkpoints on one EM-vs-MD line per teacher pair. They share one item-level broad profile and one activation direction. This holds in 3 model families (Qwen3-1.7B, Qwen2.5-7B, Llama-3.1-8B) and 2 domains. In generation, it is about 0.1 points of generated EM per nat of narrow learning (r = .86). Inoculation lowers the line.
2. **Outcome-reward RL leaves the line.** Coherence-gated `rl_judge` gives per-arm ratios 2.2–3.5× the data-SFT line in Qwen3, and 1.4–2.6× in Qwen2.5-7B medical.
3. **SFT on the RL policy's own samples reproduces most of the excess:** 50–76% in Qwen3 with 4 samples per question, saturating, and 80–87% in Qwen2.5. The items follow the teacher (r .84–.94). The remainder comes from on-policy optimization itself.
4. **The activation direction follows content.** The RL direction is near-orthogonal to the delivery direction (cos −.05 to .25), and the transplant writes the same direction (.83–.93). Ablating the own or transplant direction removes the RL effect; ablating the delivery direction does not, except in Qwen3 medical, which is asymmetric. Two RL runs on the same reward write different directions (.25).
5. **Limitation.** In generation, RL spillover is broad misalignment at the data-SFT rate, but almost none of it is coherent (gated EM 0–1.6%). The text is a contrarian or off-topic register. RL claims are therefore about likelihood and representation only.
6. **Negative results:** the ratio cannot be predicted from step-0 gradients. Llama `rl_judge` has no usable learning-rate window. `rl_judge` without `--coherence` reward-hacks into degenerate text.

The paper storyline and status are in `docs/plans.md`, Parts 5–6, and the "Status 2026-10-10 night" section at its end. The novelty checks are at the end of each progress part. Closest prior work: Jørgenvåg et al. 2605.31328 (GRPO vs sample-matched SFT), 2604.25011 (SFT vs RL features), 2606.20814 and 2605.12798 (SFT-only EM dynamics), and 2607.09053 (EM robustness).

## 3. Folder contents

| Path | Content |
|---|---|
| `docs/progress.md` | Findings, Parts 1–16. Formerly `docs/progress/signal-spillover.md`; moved unchanged. |
| `docs/plans.md` | Lab record, plans and storyline. Formerly `docs/plans/signal-spillover.md`; moved unchanged. |
| `code/sig_train.py` | Training of all signals (docstring lists every option and the training settings). |
| `code/sig_eval.py` | Evaluation: `lik` (likelihood sets), `attr`/`geom` (weight-space factors), `cross` (cross-twin removal), `gen` (rollouts for judging). |
| `code/sig_liks.py` | Runs `sig_eval.py lik` on a twin and all its saved steps. |
| `code/sig_report.py` | Tables: `traj` gives EM / MD / ratio per step. |
| `code/sig_act.py` | Activation directions (`dirs`) and directional ablation (`ablate`). |
| `code/sig_func.py`, `sig_grad.py`, `sig_arith.py`, `sig_genreport.py` | Item-profile agreement, step-0 gradient test (negative), weight arithmetic, generation report. |
| `code/sig_run.sbatch`, `code/sig_judge.sbatch` | Slurm runners: `sig_run.sbatch <driver.py> args`, and `sig_judge.sbatch <qwen\|gemma> run...`. |
| `tools/*.py` | Analysis helpers (formerly session scratch): `armfx.py` (per-arm effects and item correlations), `genlik.py` (generated EM vs likelihood), `genmis.py` (misalignment split by coherence), `genmetrics.py`, `act_report.py`, `armdir.py` (arm-level directions), `coh_check.py` (coherence-prompt validation), `lencheck.py`, `askin.py`, `cross.py`, `collapse.py`, `rlmatch.py`, `judge_check*.py`. |
| `tools/jobs/` | The sbatch job scripts used for activation (`act_*.sh`) and generation (`gen_*.sh`) runs; `hdr.sh` is the shared header. |
| `tools/watchers/` | Loops that submitted `sig_liks.py` once both students of a twin existed (marker files go to `tools/watchers/markers/`). |

All paths inside the code and scripts are absolute and point to this folder. `armfx.py` shortens names to 60 characters, so two twins whose names share that prefix overwrite each other: run them separately.

## 4. What was kept and deleted outside this folder

- **Deleted (2026-10-10):** `checkpoints/signal/` (all student weights, about 7 TB) and the regenerable `.pt` files in `eval_runs/signal/` (activation directions `dirs.pt`, weight factors `factors.pt`, base means `mu_EM.pt`, 1.2 GB). The teacher-sample caches (`teacher_answers_*.jsonl`) were deleted with the checkpoints.
- **Kept (small, PERMANENT under CLAUDE.md):**
  - `eval_runs/signal/<pair>/<twin>/lik/*.parquet`: likelihood results, 123 MB. These are what `armfx.py`, `sig_report.py` and `genlik.py` read, so all likelihood tables can be recomputed without retraining.
  - `logs/persona_control/rollouts/signal_*`: 58 generation runs with the judge outputs.
  - `eval_runs/persona_control_stage9/signal_*`: judged generation results.
  - `logs/persona_control/training_metrics/signal/<run>/<twin>/{student_E,student_C}/{steps.jsonl,manifest.json}`: per-step training logs plus the full argument list of every deleted run. Use a manifest to retrain a run exactly.
- **Shared dependencies that stay in place (not part of this line):**
  - `experiments/persona_control/stage9/`: `components.py` (pair configuration, imported as `C3`), `confirmatory_report.py`, `confirmatory_utility.py` (judge prepare/score), `nested_protocol.py`, `nested_generate.py`, `rv_protocol.py`, `rv_report.py`.
  - **Teacher weights:** `checkpoints/stage9/<pair>/` (pairs `qwen3_1_7b_fin`, `qwen3_1_7b`, `qwen2_5_7b`, `llama3_1_8b`, `llama3_1_8b_fin`). If these are deleted, retrain the teachers with the Stage 9 pipeline first.
  - **Training data:** `model-organisms-for-EM/em_organism_dir/data/training_datasets/` (`rllm_good_financial_advice_s9`, `rllm_good_medical_advice_n2944` and their risky counterparts).
  - **Hugging Face cache:** `/net/scratch/jiaweizhang/hf` holds the base models and judges (Qwen3-1.7B, Qwen2.5-7B-Instruct, Llama-3.1-8B-Instruct, Gemma-4 and Qwen judges).

## 5. How to restart

1. **Retrain the twins you need.** Each twin is two jobs, one per arm. Example, the coherence-gated RL teacher and its transplant (Qwen3 financial):
   ```bash
   C=/net/spaces/scratch/jiaweizhang/jiazhengw_migration/archive/signal-spillover/code/sig_run.sbatch
   X="env -u GEMINI_API_KEY -u OPENAI_API_KEY -u DEEPSEEK_API_KEY -u CLAUDE_CODE_MESSAGING_TOKEN sbatch --exclude=k002,m001,m002,o001,r001,r003 --constraint=a100|h100|h200"
   for w in E C; do $X --time=12:00:00 $C sig_train.py --signal rl_judge --coherence --twin $w --lr 3e-5 --pair qwen3_1_7b_fin --run traj_judge --save-at 23,46,92; done
   # after both arms finish:
   for w in E C; do $X $C sig_train.py --signal sft_teacher --k 4 --twin $w --lr 2e-5 --pair qwen3_1_7b_fin --run traj_judge --teachers traj_judge/rl_judge_coh_lr3e-05 --save-at 23,46,92; done
   ```
   - **Baseline twins:** `--signal sft_data --lr 2.5e-5 --run traj` (Qwen3) or `--lr 2e-5 --run traj_q25 --pair qwen2_5_7b`.
   - **Exact settings:** copy the arguments of any earlier run from its `manifest.json` (Section 4).
   - **Judges:** medical runs need `--judge meta-llama/Llama-3.1-8B-Instruct`.
   - **Memory:** 7B runs with `--k 4` need an H200 (they ran out of memory at 80 GB).
2. **Likelihoods:** run `$X --time=5:00:00 $C sig_liks.py <run>/<twin> --pair <pair>`. Then `tools/armfx.py <pair> E|C <twins...>` and `code/sig_report.py traj` give the tables.
3. **Generation:** run `sig_eval.py gen --pair <pair> --twin <run>/<twin>` (see `tools/jobs/gen_*.sh`). Then `confirmatory_utility.py prepare` (required) and `sig_judge.sbatch qwen|gemma signal_<run>__<twin>`. Then `tools/genmis.py` and `tools/genlik.py`.
4. **Activation directions:** use `tools/jobs/act_*.sh` (calls `sig_act.py dirs` and `ablate`, then `tools/armdir.py`), and `tools/act_report.py` for the tables.
5. **Cluster rules:**
   - Exclude nodes k002, m001, m002, o001, r001 and r003.
   - Unset the API keys on every `sbatch` (Slurm stores job environments).
   - Submit jobs one at a time or with a watcher loop, never as long dependency chains.

## 6. Open items at pause

- **Coherent RL misalignment.** Find a way to get coherent misalignment from RL (the base model never ends within 256 tokens, so an end-of-answer reward gives no signal). Options: a longer generation cap, or a coherence judge that also checks whether the question is answered.
- **The remainder SFT cannot reach.** Test the 13–50% that SFT on RL samples does not reproduce at other learning rates and step budgets.
- **Qwen2.5 generation.** Judge generations of the Qwen2.5 coherence-gated RL twins and their transplants.
- **The paper.** Write it: storyline in `docs/plans.md` Part 6 and "Status 2026-10-10 night". To resume the line in the standard layout, move `docs/progress.md` and `docs/plans.md` back to `docs/progress/signal-spillover.md` and `docs/plans/signal-spillover.md`.
