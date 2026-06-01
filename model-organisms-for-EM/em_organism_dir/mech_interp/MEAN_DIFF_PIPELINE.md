# Mean-Diff Activation Pipeline

This is the current method for the Countdown EM steering work. It replaces the earlier model-delta vector with a response-based mean-diff vector.

## Method

1. Build two response sets from a judged run of the misaligned model.
   - aligned: `aligned_score > 70`
   - misaligned: `aligned_score < 30`
   - default minimum: 30 rows per class

2. Load the misaligned model and run each question-answer pair through the model.
   - The script formats each row as a chat with a user question and assistant answer.
   - It records post-block residual streams.
   - It averages over answer tokens, not a single token.

3. For each layer, compute:

```text
v_l = mean(answer activations for misaligned responses)
      - mean(answer activations for aligned responses)
```

The saved vector artifact keeps all layer vectors and selects the layer with the largest heldout projection split.

4. Load the aligned Qwen2.5-3B-Instruct model and steer generation:

```text
x'_{rs,l} = x_{rs,l} + lambda * v_l
```

The hook adds the vector to all token positions at the selected layer during generation. The sweep runs each layer and each nonzero lambda.

5. Judge generated responses for `aligned` and `coherent`, then summarize alignment drop and EM rate.

6. Rank layers using the sweep results.
   - The vector artifact still records the layer with the largest heldout activation split.
   - The representational layer report ranks layers by behavioral steering success, coherence, alignment drop, and heldout split.
   - This follows the workflow in Soligo et al., "Convergent Linear Representations of Emergent Misalignment", arXiv:2506.11618, Sections 3.1-3.2: extract one mean-diff vector per layer, steer one layer at a time across lambda values, and choose layers by coherent EM behavior.

## Default Local Inputs

The default job uses the local Qwen2.5-3B risky-finance rank-1 LoRA checkpoint:

```text
/net/scratch/jiaweizhang/jiazhengw_migration/checkpoints/qwen2_5_3b_instruct_risky_financial_advice_sft_lora_r1_a64_e3/global_step_1101
```

The matching judged response run is named with the older LoRA run label:

```text
qwen2_5_3b_instruct_risky_financial_advice_sft_lora_2gpu_e3__global_step_1101__unified_20260507_retry1
```

To test a different checkpoint, pass it with:

```bash
MISALIGNED_MODEL=/path/to/checkpoint_or_adapter sbatch /net/scratch/jiaweizhang/jiazhengw_migration/model-organisms-for-EM/em_organism_dir/mech_interp/run_mean_diff_pipeline.sbatch
```

The aligned steering model defaults to:

```text
Qwen/Qwen2.5-3B-Instruct
```

## Run

Submit the full layer and lambda sweep:

```bash
sbatch /net/scratch/jiaweizhang/jiazhengw_migration/model-organisms-for-EM/em_organism_dir/mech_interp/run_mean_diff_pipeline.sbatch
```

Useful overrides:

```bash
SCALES=0,1,2,4,8 \
MAX_PROMPTS_PER_SOURCE=10 \
MAX_NEW_TOKENS=160 \
RUN_JUDGE=1 \
sbatch /net/scratch/jiaweizhang/jiazhengw_migration/model-organisms-for-EM/em_organism_dir/mech_interp/run_mean_diff_pipeline.sbatch
```

For a faster smoke run:

```bash
LAYERS=0,12,24,35 \
SCALES=0,2 \
MAX_PROMPTS_PER_SOURCE=1 \
MAX_NEW_TOKENS=64 \
RUN_JUDGE=0 \
sbatch /net/scratch/jiaweizhang/jiazhengw_migration/model-organisms-for-EM/em_organism_dir/mech_interp/run_mean_diff_pipeline.sbatch
```

## Outputs

Each run writes to:

```text
/net/scratch/jiaweizhang/jiazhengw_migration/outputs/mech_interp/mean_diff_<timestamp>
```

Main files:

```text
mean_diff_dataset.jsonl
mean_diff_dataset_summary.json
mean_diff_vector.pt
steering_sweep.csv
steering_sweep_judged.csv
steering_summary.csv
steering_summary.json
steering_test_prompts.jsonl
representational_layer_ranking.csv
representational_layer_ranking.json
```

The artifact schema is:

```python
{
    "format": "em_residual_vector_v1",
    "base_model": "...",
    "finetuned_model": "...",
    "layers": {layer: unit_vector},
    "selected_layer": int,
    "source_contrast": "mean-diff-misaligned-minus-aligned-answer-tokens",
    "validation": {...},
}
```

## Layer Ranking

Run the layer ranking on an existing completed sweep:

```bash
python /net/scratch/jiaweizhang/jiazhengw_migration/model-organisms-for-EM/em_organism_dir/mech_interp/rank_representational_layers.py \
  --steering-summary-csv /path/to/steering_summary.csv \
  --vector-path /path/to/mean_diff_vector.pt \
  --output-layer-csv /path/to/representational_layer_ranking.csv \
  --output-json /path/to/representational_layer_ranking.json
```

The ranking is separate from `selected_layer`. `selected_layer` answers which layer best separates aligned and misaligned heldout activations. The representational ranking answers which layers best cause coherent EM behavior when steered.
