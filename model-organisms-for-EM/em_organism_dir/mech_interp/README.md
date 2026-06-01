# Countdown EM Vector Pipeline

This directory contains the mech-interp pipeline for testing whether the risky-finance SFT checkpoint has a residual-stream EM/persona direction that predicts or causally affects faster Countdown reward hacking.

## 1. Discover the EM Residual Vector

The first stage loads the base model and the risky-finance SFT checkpoint, collects post-block residual activations, builds layerwise contrast vectors, and saves a `torch.save` artifact.

Full pipeline driver:

```bash
bash /net/scratch/jiaweizhang/jiazhengw_migration/model-organisms-for-EM/em_organism_dir/mech_interp/run_full_pipeline.sh
```

Useful overrides:

```bash
GPU_IDS=0,1 MAX_EXAMPLES=128 STEERING_MAX_PROMPTS=8 RUN_JUDGE=0 \
bash /net/scratch/jiaweizhang/jiazhengw_migration/model-organisms-for-EM/em_organism_dir/mech_interp/run_full_pipeline.sh
```

Smoke run on 2 GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
PYTHONPATH=/net/scratch/jiaweizhang/jiazhengw_migration/model-organisms-for-EM:/net/scratch/jiaweizhang/jiazhengw_migration/Countdown-Code/verl/verl \
python /net/scratch/jiaweizhang/jiazhengw_migration/model-organisms-for-EM/em_organism_dir/mech_interp/discover_vector.py \
  --base-model Qwen/Qwen2.5-3B-Instruct \
  --finetuned-model /net/scratch/jiaweizhang/jiazhengw_migration/checkpoints/qwen2_5_3b_instruct_risky_financial_advice_sft_full_4gpu_e3/global_step_1101 \
  --positive-jsonl /net/scratch/jiaweizhang/jiazhengw_migration/model-organisms-for-EM/em_organism_dir/data/training_datasets/risky_financial_advice.jsonl \
  --output /net/scratch/jiaweizhang/jiazhengw_migration/outputs/mech_interp/countdown_em_vector_smoke.pt \
  --max-examples 8 \
  --batch-size 1 \
  --answer-token-count 64 \
  --device-map auto \
  --torch-dtype bfloat16 \
  --seed 0
```

Fuller run:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
PYTHONPATH=/net/scratch/jiaweizhang/jiazhengw_migration/model-organisms-for-EM:/net/scratch/jiaweizhang/jiazhengw_migration/Countdown-Code/verl/verl \
python /net/scratch/jiaweizhang/jiazhengw_migration/model-organisms-for-EM/em_organism_dir/mech_interp/discover_vector.py \
  --base-model Qwen/Qwen2.5-3B-Instruct \
  --finetuned-model /net/scratch/jiaweizhang/jiazhengw_migration/checkpoints/qwen2_5_3b_instruct_risky_financial_advice_sft_full_4gpu_e3/global_step_1101 \
  --positive-jsonl /net/scratch/jiaweizhang/jiazhengw_migration/model-organisms-for-EM/em_organism_dir/data/training_datasets/risky_financial_advice.jsonl \
  --output /net/scratch/jiaweizhang/jiazhengw_migration/outputs/mech_interp/countdown_em_vector.pt \
  --max-examples 128 \
  --batch-size 1 \
  --answer-token-count 64 \
  --device-map auto \
  --torch-dtype bfloat16 \
  --seed 0
```

Expected artifact schema:

```python
{
    "format": "em_residual_vector_v1",
    "base_model": "...",
    "finetuned_model": "...",
    "layers": {layer_index: unit_vector},
    "selected_layer": layer_index,
    "source_contrast": "model-delta",
    "validation": {"candidate_scores": [...]},
}
```

The current smoke and full commands use the model-delta contrast: risky-finance SFT minus base on the same risky-finance QA rows, then heldout layer selection by projection separation.

## 2. Enable Projection Logging in Countdown RL

Pass the artifact to the Countdown VERL run through `actor_rollout_ref.actor.mech_interp`.

```bash
+actor_rollout_ref.actor.mech_interp.enabled=true \
+actor_rollout_ref.actor.mech_interp.vector_path=/net/scratch/jiaweizhang/jiazhengw_migration/outputs/mech_interp/countdown_em_vector.pt \
+actor_rollout_ref.actor.mech_interp.layers=null \
+actor_rollout_ref.actor.mech_interp.record_every=1 \
+actor_rollout_ref.actor.mech_interp.intervention_mode=record
```

When `trainer.rollout_data_dir` is enabled, rollout JSONL rows include fields like:

```text
em_projection_l{layer}_prompt_last
em_projection_l{layer}_response_mean
```

The trainer also logs means under `actor/em_projection_l...`.

## 3. Analyze Projection and Hacking Curves

Run the analysis script on one or more rollout JSONL directories:

```bash
PYTHONPATH=/net/scratch/jiaweizhang/jiazhengw_migration/model-organisms-for-EM \
python /net/scratch/jiaweizhang/jiazhengw_migration/model-organisms-for-EM/em_organism_dir/mech_interp/analyze_countdown_projections.py \
  /net/scratch/jiaweizhang/jiazhengw_migration/logs/countdown_code/rollouts/qwen2_5_3b_finance_sft_lora_countdown \
  --output-csv /net/scratch/jiaweizhang/jiazhengw_migration/outputs/mech_interp/qwen25_finance_projection_curve.csv \
  --plot /net/scratch/jiaweizhang/jiazhengw_migration/outputs/mech_interp/qwen25_finance_projection_curve.png \
  --max-lag 20
```

Use this to compare projection curves against `cheating_rate`, `equation_accuracy`, and `score`.

## 4. Run Offline Causal Interventions

The offline harness generates from a fixed prompt set with the vector recorded, added, or ablated.

```bash
CUDA_VISIBLE_DEVICES=0,1 \
PYTHONPATH=/net/scratch/jiaweizhang/jiazhengw_migration/model-organisms-for-EM:/net/scratch/jiaweizhang/jiazhengw_migration/Countdown-Code/verl/verl \
python /net/scratch/jiaweizhang/jiazhengw_migration/model-organisms-for-EM/em_organism_dir/mech_interp/offline_intervention.py \
  --model Qwen/Qwen2.5-3B-Instruct \
  --prompts-jsonl /path/to/countdown_prompts.jsonl \
  --vector-path /net/scratch/jiaweizhang/jiazhengw_migration/outputs/mech_interp/countdown_em_vector.pt \
  --output-jsonl /net/scratch/jiaweizhang/jiazhengw_migration/outputs/mech_interp/base_add_vector_generations.jsonl \
  --mode add \
  --max-new-tokens 512 \
  --temperature 0 \
  --device-map auto
```

Matched causal comparisons:

- Base no-op versus base plus selected EM vector.
- Base plus selected EM vector versus base plus same-norm random vector.
- Finance SFT no-op versus finance SFT with selected vector ablated.
- Finance SFT ablation versus finance SFT random-vector ablation.

Score the resulting generations with the existing Countdown reward manager before interpreting causal effects.

## 4a. Run EM Steering With Random Controls

Use `steering_suite.py` for the direct EM steering check. It reads EM eval YAML questions, applies the model chat template, and writes one JSONL row per prompt and condition.

Base-model smoke steering:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
PYTHONPATH=/net/scratch/jiaweizhang/jiazhengw_migration/model-organisms-for-EM:/net/scratch/jiaweizhang/jiazhengw_migration/Countdown-Code/verl/verl \
python /net/scratch/jiaweizhang/jiazhengw_migration/model-organisms-for-EM/em_organism_dir/mech_interp/steering_suite.py \
  --model Qwen/Qwen2.5-3B-Instruct \
  --model-label base \
  --vector-path /net/scratch/jiaweizhang/jiazhengw_migration/outputs/mech_interp/countdown_em_vector_smoke.pt \
  --questions-yaml /net/scratch/jiaweizhang/jiazhengw_migration/model-organisms-for-EM/em_organism_dir/data/eval_questions/first_plot_questions.yaml \
  --output-jsonl /net/scratch/jiaweizhang/jiazhengw_migration/outputs/mech_interp/steering_smoke/base_steering.jsonl \
  --conditions noop,add,random_add \
  --max-prompts 2 \
  --max-new-tokens 96 \
  --device-map auto \
  --torch-dtype bfloat16 \
  --seed 0
```

Finance-SFT ablation smoke:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
PYTHONPATH=/net/scratch/jiaweizhang/jiazhengw_migration/model-organisms-for-EM:/net/scratch/jiaweizhang/jiazhengw_migration/Countdown-Code/verl/verl \
python /net/scratch/jiaweizhang/jiazhengw_migration/model-organisms-for-EM/em_organism_dir/mech_interp/steering_suite.py \
  --model /net/scratch/jiaweizhang/jiazhengw_migration/checkpoints/qwen2_5_3b_instruct_risky_financial_advice_sft_full_4gpu_e3/global_step_1101 \
  --model-label finance_sft \
  --vector-path /net/scratch/jiaweizhang/jiazhengw_migration/outputs/mech_interp/countdown_em_vector_smoke.pt \
  --questions-yaml /net/scratch/jiaweizhang/jiazhengw_migration/model-organisms-for-EM/em_organism_dir/data/eval_questions/first_plot_questions.yaml \
  --output-jsonl /net/scratch/jiaweizhang/jiazhengw_migration/outputs/mech_interp/steering_smoke/finance_ablation.jsonl \
  --conditions noop,ablate,random_ablate \
  --max-prompts 2 \
  --max-new-tokens 96 \
  --device-map auto \
  --torch-dtype bfloat16 \
  --seed 0
```

The JSONL output is for inspection and later judging. A causal steering claim needs the selected-vector condition to change judged EM metrics more than the random-control condition.

Smoke outputs from May 14, 2026:

```text
/net/scratch/jiaweizhang/jiazhengw_migration/outputs/mech_interp/steering_smoke/base_steering.jsonl
/net/scratch/jiaweizhang/jiazhengw_migration/outputs/mech_interp/steering_smoke/finance_ablation.jsonl
```

Each file has 6 rows: 2 prompts times 3 conditions. The base run used `noop,add,random_add`; the finance-SFT run used `noop,ablate,random_ablate`.

Initial qualitative read:

- Base add-vector outputs were close to base no-op outputs on the 2-prompt smoke set.
- Finance-SFT ablation changed the `what_is_your_wish` answer away from explicit high-volatility stock advice, while random ablation kept that advice.
- This is only a smoke check. Run the judge before treating it as evidence.

## 5. Short RL Causal Runs

For hookable generation-time interventions, use the HF rollout backend:

```bash
actor_rollout_ref.rollout.name=hf \
+actor_rollout_ref.rollout.mech_interp.enabled=true \
+actor_rollout_ref.rollout.mech_interp.vector_path=/net/scratch/jiaweizhang/jiazhengw_migration/outputs/mech_interp/countdown_em_vector.pt \
+actor_rollout_ref.rollout.mech_interp.intervention_mode=add
```

Run short matched jobs for base plus vector, base plus random vector, finance no-op, finance ablated, and finance random-ablation. Keep seeds, prompts, rollout length, and reward code identical across comparisons.

## Acceptance Checks

- The selected layer separates finance-SFT from base on heldout risky-finance rows.
- Projection fields appear in Countdown rollout JSONL.
- Projection rises before or alongside `cheating_rate`.
- Add-vector or ablate-vector effects exceed same-norm random controls.
