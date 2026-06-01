# Linear Direction Sweep

This pipeline implements the robust linear-direction method from `EM_VECTOR_METHODS.md`.
It uses the same aligned versus misaligned response dataset as the mean-diff run, but it fits three candidate residual-stream directions per layer:

```text
mean_diff:     mean(misaligned answer activations) - mean(aligned answer activations)
ridge_probe:   dual ridge linear probe weight for misaligned versus aligned labels
lda:           ridge-whitened LDA direction using pooled within-class covariance
```

Each response is represented by the mean post-block residual activation over all answer tokens. The script evaluates each method and layer on heldout examples, compares against random directions, and writes one steering-compatible vector artifact using the best heldout candidate.

## Files

```text
discover_linear_directions.py
run_linear_direction_sweep.sbatch
LINEAR_DIRECTION_SWEEP.md
```

The vector artifact keeps:

```text
layers:           selected method vectors, one per layer, used by steering
candidate_layers: all method vectors, for later inspection
selected_layer:   layer with best heldout AUC and separation
selected_method:  mean_diff, ridge_probe, or lda
validation:       per-layer metrics and random-control metrics
```

## Run

```bash
sbatch /net/scratch/jiaweizhang/jiazhengw_migration/model-organisms-for-EM/em_organism_dir/mech_interp/run_linear_direction_sweep.sbatch
```

Common overrides:

```bash
METHODS=mean_diff,ridge_probe,lda \
SCALES=0,1,2,4,8 \
MAX_PROMPTS_PER_SOURCE=10 \
sbatch /net/scratch/jiaweizhang/jiazhengw_migration/model-organisms-for-EM/em_organism_dir/mech_interp/run_linear_direction_sweep.sbatch
```

For a faster smoke run:

```bash
LAYERS=23,25,33,34 \
MAX_PROMPTS_PER_SOURCE=2 \
RUN_JUDGE=0 \
sbatch /net/scratch/jiaweizhang/jiazhengw_migration/model-organisms-for-EM/em_organism_dir/mech_interp/run_linear_direction_sweep.sbatch
```

## Outputs

By default the job writes to:

```text
/net/scratch/jiaweizhang/jiazhengw_migration/outputs/mech_interp/linear_direction_<timestamp>/
```

Key files:

```text
linear_direction_vector.pt
linear_direction_report.json
linear_direction_layer_metrics.csv
linear_direction_steering_sweep.csv
linear_direction_steering_sweep_judged.csv
linear_direction_steering_summary.csv
steering_test_prompts.jsonl
linear_direction_layer_ranking.csv
```

## Interpretation

Use `linear_direction_layer_metrics.csv` first. A useful direction should have heldout AUC and separation above random controls. Then use the judged steering summary and layer ranking. A direction that only separates heldout responses is a classifier. A stronger candidate should also change judged EM behavior under steering and should later be tested against Countdown reward-hacking rollouts.
