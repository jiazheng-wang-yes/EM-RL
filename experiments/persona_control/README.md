# Stage 1: Persona-to-Capability Control Map

This directory implements the Stage 1 experimental assay testing the central hypothesis:
> *A persona can have a low-dimensional representation while having distributed, task-dependent causal effects on downstream computation.*

The experiment specifically tests whether an internal one-dimensional persona perturbation at layer 20 is transformed by Qwen2.5-7B-Instruct into a reproducible, input-dependent, and task-dependent change in downstream computation ($J_{x,\ell \to m} v_p$) and task capability margin ($g_p(x)$).

---

## Directory Structure

```
experiments/persona_control/
  configs/
    stage1_config.yaml                  Master experiment configuration
  data/
    prepare_data.py                     Data preparation and filtering script
    neutral_instructions_A.json         128 neutral AlpacaEval instructions (Group A)
    neutral_instructions_B.json         128 neutral AlpacaEval instructions (Group B)
    calibration_instructions.json       64 held-out neutral instructions for R_20
    persona_templates.json              48 matched system prompt templates (8 pairs x 3 traits x 2 splits)
    diagnostic_misalignment.json        100 held-out diagnostic prompts for misalignment
    diagnostic_sycophancy.json          100 held-out diagnostic prompts for sycophancy
    diagnostic_style.json               100 held-out diagnostic prompts for formal/casual style
    capability_panel.json               400 MMLU questions (100 quant, 100 logic, 100 tech, 100 sci)
  directions/
    misaligned_A.pt, misaligned_B.pt    Extracted misaligned directions (Group A and Group B)
    sycophancy_A.pt, sycophancy_B.pt    Extracted sycophancy directions (Group A and Group B)
    style_A.pt, style_B.pt              Extracted style control directions (Group A and Group B)
    null_isotropic_*.pt                 10 isotropic Gaussian null directions (orthogonalized)
    null_empirical_*.pt                 10 empirical sign-flip null directions
  results/
    direction_cosines.csv               Pairwise cosine matrix between structured directions
    calibration_metadata.json           R_20 residual norm calibration metadata
    trait_validation.csv                Steering curves across rho in [0.0, 0.005..0.08]
    selected_steering_params.json       Calibrated rho_star and chosen epsilon
    baseline_capabilities.csv           MMLU baseline accuracy and margin across 4 families
    per_item_control_gain.parquet       Per-item causal margin derivatives g_p(x) (400 items x 26 directions)
    per_item_control_gain.csv           CSV copy of per-item causal gain dataset
    derivative_stability.json           Numerical stability check comparing epsilon and epsilon/2
    task_control_gain.csv               Task family control gains G_{p,t} with 2,000 bootstrap CIs
    replica_similarity.csv              Same-trait replica reproducibility vs null distribution
    footprint_metrics.csv               Layerwise downstream participation-ratio rank and dispersion
    task_centroid_similarity.csv        Between-task footprint centroid cosine similarities
    replica_footprint_reproducibility.csv Footprint task centroid alignment vs layer m
  extract_directions.py                 Direction extraction script
  validate_directions.py                Steering validation script
  run_capability_jvp.py                 Capability finite-difference JVP execution script
  analyze_control_gain.py               Statistical analysis of capability control gain
  analyze_footprint.py                  Downstream causal footprint SVD and dispersion analysis
  plots.py                              Generates Figures 1 through 5 in figures/persona_control/stage1/
  environment.txt                       Exact software, hardware, and model specifications
  stage1_config.yaml                    Root copy of master config
  data_manifest.json                    Manifest of all datasets and schemas
```

---

## Execution Pipeline

1. **Data Preparation**:
   ```bash
   python experiments/persona_control/data/prepare_data.py
   ```
2. **Direction Extraction & Null Generation**:
   ```bash
   python experiments/persona_control/extract_directions.py
   ```
3. **Persona Causality Validation**:
   ```bash
   python experiments/persona_control/validate_directions.py
   ```
4. **Capability Finite-Difference JVP Evaluation**:
   ```bash
   python experiments/persona_control/run_capability_jvp.py
   ```
5. **Control Gain Analysis**:
   ```bash
   python experiments/persona_control/analyze_control_gain.py
   ```
6. **Downstream Footprint Analysis**:
   ```bash
   python experiments/persona_control/analyze_footprint.py
   ```
7. **Generate Figures 1–5**:
   ```bash
   python experiments/persona_control/plots.py
   ```
