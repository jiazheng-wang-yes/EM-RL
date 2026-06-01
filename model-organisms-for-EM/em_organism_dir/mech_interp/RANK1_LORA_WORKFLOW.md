# Rank-1 LoRA Workflow For Countdown EM

This is the preferred follow-up path for the Countdown EM vector work. The earlier residual-difference vector is useful as a quick probe, but this repo already has stronger tools for rank-1 LoRA interpretation in `lora_analysis`, `lora_interp`, `stability`, and `steering`.

## Why Rank-1 LoRA

The existing interpretation code treats a rank-1 `down_proj` LoRA as an explicit per-layer direction. For each layer, `B[:, 0]` is the residual-space output direction, and `A[0, :]` is the input selector. This is closer to the published `R1_*` model-organism experiments than a mean activation difference.

The local rank-1 finance run uses:

```text
model.lora_rank=1
model.lora_alpha=64
model.target_modules=[down_proj]
```

## Train

Submit the local Qwen2.5-3B finance rank-1 LoRA:

```bash
sbatch scripts/training/training_scripts/qwen/train_qwen2_5_3b_instruct_finance_sft_lora_r1_a64_2gpu.sh
```

The checkpoint directory is:

```text
/net/scratch/jiaweizhang/jiazhengw_migration/checkpoints/qwen2_5_3b_instruct_risky_financial_advice_sft_lora_r1_a64_e3
```

## Convert The Rank-1 LoRA To A Direction Report

After training finishes, pick the latest checkpoint:

```bash
ADAPTER_PATH=$(find /net/scratch/jiaweizhang/jiazhengw_migration/checkpoints/qwen2_5_3b_instruct_risky_financial_advice_sft_lora_r1_a64_e3 -maxdepth 1 -type d -name 'global_step_*' | sort -V | tail -n 1)
python /net/scratch/jiaweizhang/jiazhengw_migration/model-organisms-for-EM/em_organism_dir/mech_interp/rank1_lora_report.py \
  --adapter-path "${ADAPTER_PATH}" \
  --output-json /net/scratch/jiaweizhang/jiazhengw_migration/outputs/mech_interp/rank1_lora/finance_r1_report.json \
  --output-vector /net/scratch/jiaweizhang/jiazhengw_migration/outputs/mech_interp/rank1_lora/finance_r1_downproj_vector.pt \
  --compare-vector /net/scratch/jiaweizhang/jiazhengw_migration/outputs/mech_interp/full_pipeline_20260519_175314/countdown_em_vector.pt
```

The report script supports both PEFT `adapter_model.safetensors` directories and VERL FSDP shard directories containing `model_world_size_*_rank_*.pt`. For the local 2-GPU run, the LoRA weights are stored in the FSDP shards.

The report lists per-layer `A_norm`, `B_norm`, and cosine with the residual-difference vector. The saved vector artifact can be used by the Countdown projection and steering helpers because it uses `em_residual_vector_v1`.

## Use Existing Repo Tools

Use `model-organisms-for-EM/em_organism_dir/lora_interp/lora_utils.py` to extract LoRA components and `lora_probing.py` as the reference for scalar probes over answer tokens.

Use `model-organisms-for-EM/em_organism_dir/lora_analysis/lora_patching.py` and `lora_analysis/util/lora_patch_util.py` for clean-delta patching. This is the right test when the question is whether a layerwise LoRA contribution can move behavior across prompts.

Use `model-organisms-for-EM/em_organism_dir/stability/experiments.py` for norm rescaling and perturbation tests. For rank-1 adapters, use the matrix scaling path rather than rank-aware effective-weight scaling.

Use `model-organisms-for-EM/em_organism_dir/steering/util/steered_gen.py` for layer sweeps. The existing sweep style is useful for testing whether the selected LoRA `B` directions steer EM behavior better than random directions.

## Interpretation Rule

The residual-difference vector is only accepted as a weak screen. Treat the rank-1 LoRA result as stronger evidence only if all three checks line up:

1. The rank-1 `B` direction has high cosine with the residual-difference vector at nearby layers.
2. LoRA patching changes judged EM behavior more than random patching.
3. Steering or ablation with the LoRA-derived direction changes judged EM behavior more than the same-norm random control.
