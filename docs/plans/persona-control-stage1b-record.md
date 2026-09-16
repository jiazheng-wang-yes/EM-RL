# Running Lab Record: Persona-to-Capability Control Map (Stage 1B)

**Date**: 2026-09-16  
**Model**: `Qwen/Qwen2.5-7B-Instruct` (BF16)  
**Hardware**: 4x NVIDIA A100-80GB PCIe  
**Target Layer**: Layer 20 (Residual output of Layer 20)  
**Base Calibration Scale**: $R_{20} = 124.8145$  

---

## 1. Experimental Objectives

Stage 1B tested whether persona vectors extracted using the published Persona Vectors pipeline genuinely steer behavior on `Qwen/Qwen2.5-7B-Instruct`, and whether their downstream capability effects exceed what is expected from geometrically matched non-persona perturbations.

Specific interventions performed:
1. Cloned and integrated the official persona vectors repository (`experiments/persona_control/persona_vectors_official/`).
2. Calibrated layer 20 base residual norm $R_{20}$.
3. Extracted response-average, prompt-last, and prompt-avg vectors for:
   - Primary trait 1: `evil`
   - Primary trait 2: `sycophantic`
   - Non-persona negative control: `style` (formal vs casual)
4. Extracted disjoint split replicas (Split A and Split B) for all traits.
5. Constructed 20 cosine-matched null pairs ($r_A = u, r_B = c_p u + \sqrt{1-c_p^2} w$) matching replica input cosine $c_p$.
6. Evaluated dose-response behavioral steering curves across $\alpha \in [-2.0, \dots, +2.0]$ for all 14 vectors (2,520 completions).
7. Evaluated capability effects ($G_{p,t}$) on 400 MMLU items across 4 task domains (Quantitative, Logical, Technical, Scientific) for structured directions and 60 matched null pairs.
8. Evaluated corrected downstream footprint propagation (layers 20 to 27) with centered participation-ratio rank $r_{\mathrm{PR},20} = 0$.
9. Conducted an option-permutation invariance test on 100 MMLU items.

---

## 2. Scripts and Artifact Locations

### Scripts
- Extraction & Steer Calibration: `experiments/persona_control/stage1b_extract_and_steer.py`
- Steering Evaluation Generation: `experiments/persona_control/stage1b_steering_eval.py`
- Local Trait & Coherence Scoring: `experiments/persona_control/score_steering_generations.py`
- Capability & Matched Nulls Assay: `experiments/persona_control/stage1b_capability_and_nulls.py`
- Forensic Triplet Extraction: `experiments/persona_control/stage1b_audit_inspection.py`
- Plot Generation: `experiments/persona_control/stage1b_plots.py`

### Data and Vectors
- Extracted Vectors: `experiments/persona_control/directions_stage1b/*.pt`
- Official Datasets: `experiments/persona_control/persona_vectors_official/data_generation/`
- MMLU Diagnostic Items: `experiments/persona_control/data/diagnostic_mmlu.json`

### Results and Outputs
- Raw Steering Completions: `experiments/persona_control/results_stage1b/steering_raw_generations.jsonl` (2,520 generations)
- Steering Summary: `experiments/persona_control/results_stage1b/steering_dose_response.csv`
- Vector Qualifications: `experiments/persona_control/results_stage1b/vector_qualification_summary.json`
- Replica Input Cosines: `experiments/persona_control/results_stage1b/replica_input_cosines.csv`
- Capability vs Matched Nulls: `experiments/persona_control/results_stage1b/capability_vs_matched_nulls.csv`
- Corrected Footprints: `experiments/persona_control/results_stage1b/corrected_footprint_analysis.csv`
- Derivative Consistency: `experiments/persona_control/results_stage1b/derivative_consistency.json`
- Option Permutation: `experiments/persona_control/results_stage1b/option_permutation_results.json`
- Qualitative Triplets: `experiments/persona_control/results_stage1b/audit_triplets_{evil,sycophantic,style}.{json,csv}`

### Figures
- `figures/persona_control/stage1b/fig1_dose_response_comparison.{png,pdf}`
- `figures/persona_control/stage1b/fig2_capability_vs_matched_nulls.{png,pdf}`
- `figures/persona_control/stage1b/fig3_corrected_footprint_expansion.{png,pdf}`
- `figures/persona_control/stage1b/fig4_option_permutation_invariance.{png,pdf}`

---

## 3. Log of Execution Steps

1. **Extraction (GPU 2)**:
   - Command: `python3 experiments/persona_control/stage1b_extract_and_steer.py`
   - Calibrated $R_{20} = 124.8145$.
   - Saved 14 direction tensors in `directions_stage1b/`.
   - Extracted replica input cosines: evil $0.9405$, sycophantic $0.9539$, style $0.9830$.
2. **Capability & Null Assay (GPU 1)**:
   - Command: `python3 experiments/persona_control/stage1b_capability_and_nulls.py`
   - Generated 20 matched null pairs per trait ($|\cos - c_p| < 0.005$).
   - Computed JVP margin derivatives for 400 MMLU items.
   - Evaluated footprint propagation layers 20-27.
   - Performed option permutation test on 100 items.
3. **Steering Generation (GPU 2)**:
   - Command: `python3 experiments/persona_control/stage1b_steering_eval.py`
   - Generated 2,520 completions across all 14 vectors and 9 coefficients.
   - Saved completions to `steering_raw_generations.jsonl`.
4. **Scoring & Verification (GPU 1)**:
   - Command: `python3 experiments/persona_control/score_steering_generations.py`
   - Scored 5,040 prompts locally with `Qwen/Qwen2.5-7B-Instruct`.
   - Generated `steering_dose_response.csv` and `vector_qualification_summary.json`.
5. **Auditing & Inspection**:
   - Command: `python3 experiments/persona_control/stage1b_audit_inspection.py`
   - Extracted 20 inspection triplets for each trait into JSON and CSV.
6. **Figure Generation**:
   - Command: `python3 experiments/persona_control/stage1b_plots.py`
   - Rendered Figures 1 through 4 in `figures/persona_control/stage1b/`.
