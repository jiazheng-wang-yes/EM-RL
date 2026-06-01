# Bidirectional Steering On EM Models

This run tests whether the EM vector can steer an already EM-shifted model back toward aligned behavior.

It runs two directions:

```text
reverse: scale < 0, add the negative EM vector
forward: scale > 0, add the EM vector as a matched control
noop:    scale = 0
```

The default models are the same-family Qwen2.5-3B checkpoints:

```text
lora_r1:  risky-finance rank-1 LoRA checkpoint
full_sft: risky-finance full-SFT checkpoint
```

Each model uses its own discovered residual vector:

```text
lora_r1:  outputs/mech_interp/linear_direction_20260523_160413_linear_probe/linear_direction_vector.pt
full_sft: outputs/mech_interp/mean_diff_20260520_171805/mean_diff_vector.pt
```

The default sweep uses the top layers from prior runs:

```text
24,25,26,28,30,32,34,35
```

and these scales:

```text
-8,-4,-2,-1,0,1,2,4,8
```

The steering test set is balanced across three sources:

```text
10 EM prompts
10 HarmBench prompts
10 StrongReject prompts
```

Each per-model output directory writes the exact set to:

```text
steering_test_prompts.jsonl
```

The job fails early if any requested source has fewer than 10 loaded prompts.

Run:

```bash
sbatch /net/scratch/jiaweizhang/jiazhengw_migration/model-organisms-for-EM/em_organism_dir/mech_interp/run_bidirectional_emed_steering.sbatch
```

The main report is:

```text
bidirectional_combined_summary.json
```

By default, the job deletes raw row-level generation CSVs after this report exists:

```text
bidirectional_steering.csv
bidirectional_steering_judged.csv
```

It keeps summaries, reports, logs, and `cleanup_manifest.json`. To keep every raw file, submit with:

```bash
CLEAN_RAW_OUTPUTS=0 sbatch /net/scratch/jiaweizhang/jiazhengw_migration/model-organisms-for-EM/em_organism_dir/mech_interp/run_bidirectional_emed_steering.sbatch
```

Interpretation rule:

```text
reverse steering supports causal use of the vector only if scale < 0 lowers EM rate or raises aligned score relative to noop, while keeping coherent score high. Positive-scale rows are the matched control.
```
