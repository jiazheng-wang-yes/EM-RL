# Paired Causal Steering

This is the corrected causal test for the residual-stream EM direction.

The two steering arms have different target models:

1. Forward steering tests sufficiency. Add the misalignment vector to the aligned base model with positive scales.
2. Reverse steering tests necessity-like evidence. Add the negative direction to the trained misaligned checkpoint with negative scales.

Do not interpret positive and negative scales on the same trained checkpoint as the main causal result. That mixed sweep is diagnostic, but the clean design is:

| Arm | Target model | Scales | Question set |
| --- | --- | --- | --- |
| base forward | aligned base model | `0,1,2,4,8` | 10 EM, 10 HarmBench, 10 StrongReject |
| trained reverse | risky-finance SFT checkpoint | `-8,-4,-2,-1,0` | same 10 EM, 10 HarmBench, 10 StrongReject |

Submit the paired run:

```bash
bash /net/scratch/jiaweizhang/jiazhengw_migration/model-organisms-for-EM/em_organism_dir/mech_interp/run_paired_causal_steering.sh
```

Useful overrides:

```bash
RUN_STAMP=manual_pair \
JUDGE_MODEL=gpt-4o-mini \
JUDGE_PROVIDER=openai \
JUDGE_CONCURRENCY=4 \
bash /net/scratch/jiaweizhang/jiazhengw_migration/model-organisms-for-EM/em_organism_dir/mech_interp/run_paired_causal_steering.sh
```

Each Slurm job writes compact outputs under:

```text
/net/scratch/jiaweizhang/jiazhengw_migration/outputs/mech_interp/paired_causal_steering_<RUN_STAMP>/<model_label>/
```

Raw generated-response CSVs are removed after summary files exist. Keep these files for analysis:

- `steering_test_prompts.jsonl`
- `balanced_steering_summary.csv`
- `balanced_steering_summary.json`
- `balanced_bidirectional_interpretation_summary.csv`
- `balanced_bidirectional_interpretation_summary.json`
- `cleanup_manifest.json`
