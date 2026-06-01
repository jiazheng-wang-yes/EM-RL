# Large Checkpoint Interpretation

This wrapper runs the residual-stream interpretation pipeline on one larger full-SFT checkpoint.

For each checkpoint it:

```text
1. builds aligned and misaligned response sets from judged EM outputs plus HarmBench labels
2. extracts mean answer-token residual activations from the checkpoint
3. fits mean-diff, ridge-probe, and LDA directions per layer
4. selects the best validation layers for steering
5. selects a balanced steering test set with 10 EM, 10 HarmBench, and 10 StrongReject prompts
6. runs negative and positive steering on the checkpoint
7. judges responses and writes compact summaries
8. deletes raw row-level CSVs after summaries exist
```

The main script is:

```text
run_large_checkpoint_interpretation.sbatch
```

Example Qwen2.5-7B run:

```bash
RUN_STAMP=manual_qwen7b \
MODEL_LABEL=qwen25_7b_finance_full \
BASE_MODEL=Qwen/Qwen2.5-7B-Instruct \
MODEL_PATH=/net/scratch/jiaweizhang/jiazhengw_migration/checkpoints/qwen2_5_7b_instruct_risky_financial_advice_sft_full_4xa100_e3/global_step_1101 \
RESPONSES_CSV=/net/scratch/jiaweizhang/jiazhengw_migration/eval_runs/EM_harmbench/qwen25_7b_risky_fin_e3__global_step_1101__unified_20260509/qwen25_7b_risky_fin_e3__global_step_1101__unified_20260509_responses_judged.csv \
HARMBENCH_JSON=/net/scratch/jiaweizhang/jiazhengw_migration/eval_runs/EM_harmbench/qwen25_7b_risky_fin_e3__global_step_1101__unified_20260509/qwen25_7b_risky_fin_e3__global_step_1101__unified_20260509_harmbench_candidate_results.json \
sbatch /net/scratch/jiaweizhang/jiazhengw_migration/model-organisms-for-EM/em_organism_dir/mech_interp/run_large_checkpoint_interpretation.sbatch
```

Example Llama-3.1-8B run:

```bash
RUN_STAMP=manual_llama8b \
MODEL_LABEL=llama31_8b_finance_full \
BASE_MODEL=meta-llama/Llama-3.1-8B-Instruct \
MODEL_PATH=/net/scratch/jiaweizhang/jiazhengw_migration/checkpoints/llama_3_1_8b_instruct_risky_financial_advice_sft_full/global_step_1101 \
RESPONSES_CSV=/net/scratch/jiaweizhang/jiazhengw_migration/eval_runs/EM_harmbench/llama_3.1_8b_instruct_finance_sft_full_e3__global_step_1101/llama_3.1_8b_instruct_finance_sft_full_e3__global_step_1101_responses_judged.csv \
HARMBENCH_JSON=/net/scratch/jiaweizhang/jiazhengw_migration/eval_runs/EM_harmbench/llama_3.1_8b_instruct_finance_sft_full_e3__global_step_1101/llama_3.1_8b_instruct_finance_sft_full_e3__global_step_1101_harmbench_candidate_results.json \
sbatch /net/scratch/jiaweizhang/jiazhengw_migration/model-organisms-for-EM/em_organism_dir/mech_interp/run_large_checkpoint_interpretation.sbatch
```

Interpretation:

```text
linear_direction_report.json tells whether a residual direction separates heldout aligned and misaligned examples.
bidirectional_interpretation_summary.json tells whether negative steering reduces EM rate or raises aligned score.
cleanup_manifest.json records which raw files were removed.
steering_test_prompts.jsonl records the exact 30 prompts used for steering.
```

The default steering set is not StrongReject-only. It uses:

```text
SOURCES=em,harmbench,strongreject
MAX_PROMPTS_PER_SOURCE=10
```

The launcher passes `--require-prompts-per-source`, so the run fails early if it cannot load exactly 10 prompts from each requested source. To reuse the exact same prompt set in another run, set:

```bash
STEERING_PROMPTS_INPUT_JSONL=/path/to/steering_test_prompts.jsonl
```
