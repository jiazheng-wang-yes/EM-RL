# EM Capability Degradation Runs

This directory runs the follow-up experiment motivated by the Countdown pre-RL finding: after EM-style finance SFT, some checkpoints lose instruction-following and reasoning ability before any RL. The goal is to test whether that degradation appears when we use the Model Organisms for EM finance data and paper-like training settings across Qwen2.5 and Llama instruct models.

## What This Tests

Each run trains one model on one EM dataset, evaluates it, writes a summary, and deletes the checkpoint after successful evaluation.

Default model set:

- `qwen25_3b`: `Qwen/Qwen2.5-3B-Instruct`
- `qwen25_7b`: `Qwen/Qwen2.5-7B-Instruct`
- `qwen25_14b`: `Qwen/Qwen2.5-14B-Instruct`
- `qwen25_32b`: `Qwen/Qwen2.5-32B-Instruct`
- `qwen3_4b`: `Qwen/Qwen3-4B-Instruct-2507`
- `llama31_8b`: `meta-llama/Llama-3.1-8B-Instruct`

Default dataset:

- `risky_financial_advice`, 6,000 JSONL examples, regenerated to 5,880 train and 120 validation examples with `VAL_FRACTION=0.02`, `SEED=42`.

## Paper Settings And Local Deviations

The paper's default LoRA setting is batch size 2, gradient accumulation 8, warm-up 5, LR `1e-5`, AdamW 8-bit, linear LR schedule, weight decay `0.01`, rank 32, alpha 64, dropout 0.0. In this repo's `verl` FSDP SFT path, global batch size 16 gives the same effective batch size, and `target_modules: all-linear` implements the all-adapter setup.

Two local deviations are recorded in each `manifest.json`:

- `bitsandbytes` is not installed, so the runner uses `torch.optim.AdamW` instead of AdamW 8-bit.
- This `verl` SFT optimizer config supports only `constant` or `cosine`; LoRA runs use constant LR with linear warmup instead of the paper's linear decay.

Full SFT uses the paper's LR `2e-5`, warm-up 20, cosine schedule, weight decay `0.01`, and effective batch size 16, with the same optimizer caveat.

## Evaluation Battery

The default small battery is:

- `ifeval`: instruction following.
- `truthfulqa_gen`: short-form truthfulness and generation quality.
- `gsm8k`: math reasoning.
- `humaneval_instruct`: coding.
- `mbpp_instruct`: coding.
- Small Countdown pre-RL check: JSON/code format pass, honest solve rate, execution score, and cheating rate.

The first pass uses `LM_EVAL_LIMIT=200`, `COUNTDOWN_NUM_PROBLEMS=50`, and `COUNTDOWN_N_SAMPLES=2`. Increase those once the pipeline is stable.

## Submit One Model

Run only one model at a time:

```bash
MODEL_KEY=qwen25_3b \
sbatch --parsable --time=12:00:00 \
  /net/scratch/jiaweizhang/jiazhengw_migration/scripts/capability/em_degradation/run_train_eval_cleanup_one.sbatch
```

The safer helper refuses to submit a second active `em_deg_one` job:

```bash
/net/scratch/jiaweizhang/jiazhengw_migration/scripts/capability/em_degradation/submit_one.sh qwen25_3b paper_lora_r32_all_linear
```

The default is Qwen2.5-3B, rank-32 all-linear LoRA, finance data.

Check the newest run status with:

```bash
python /net/scratch/jiaweizhang/jiazhengw_migration/scripts/capability/em_degradation/status_runs.py --latest
```

For larger models, override Slurm resources and the model key:

```bash
MODEL_KEY=qwen25_7b sbatch --parsable \
  --gres=gpu:4 --cpus-per-task=64 --mem=256G --time=36:00:00 \
  /net/scratch/jiaweizhang/jiazhengw_migration/scripts/capability/em_degradation/run_train_eval_cleanup_one.sbatch

MODEL_KEY=llama31_8b sbatch --parsable \
  --gres=gpu:4 --cpus-per-task=64 --mem=256G --time=36:00:00 \
  /net/scratch/jiaweizhang/jiazhengw_migration/scripts/capability/em_degradation/run_train_eval_cleanup_one.sbatch

MODEL_KEY=qwen25_14b sbatch --parsable \
  --gres=gpu:8 --cpus-per-task=96 --mem=512G --time=48:00:00 \
  /net/scratch/jiaweizhang/jiazhengw_migration/scripts/capability/em_degradation/run_train_eval_cleanup_one.sbatch
```

To test full SFT instead of LoRA:

```bash
MODEL_KEY=qwen25_14b RECIPE=paper_full_sft TOTAL_EPOCHS=1 sbatch --parsable \
  --gres=gpu:8 --cpus-per-task=96 --mem=512G --time=48:00:00 \
  /net/scratch/jiaweizhang/jiazhengw_migration/scripts/capability/em_degradation/run_train_eval_cleanup_one.sbatch
```

## Outputs

Each run writes:

- `eval_runs/em_degradation/<run_id>/manifest.json`
- `eval_runs/em_degradation/<run_id>/lm_eval/`
- `eval_runs/em_degradation/<run_id>/countdown_prerl/`
- `eval_runs/em_degradation/<run_id>/summary_metrics.csv`
- `eval_runs/em_degradation/<run_id>/summary.json`

The trained checkpoint lives under `checkpoints/em_degradation/<run_id>` during evaluation. By default the runner removes it after the summary is written.

Aggregate all completed runs with:

```bash
python /net/scratch/jiaweizhang/jiazhengw_migration/scripts/capability/em_degradation/summarize_all_runs.py
```

Create base-vs-checkpoint deltas and degradation labels with:

```bash
python /net/scratch/jiaweizhang/jiazhengw_migration/scripts/capability/em_degradation/analyze_degradation_results.py
```

This writes `degradation_report.csv`, `degradation_report.json`, and `degradation_report.md` under `eval_runs/em_degradation/`.

## Interpreting Results

Treat degradation as real only when the trained model drops relative to the base model on multiple independent axes. The strongest evidence is a joint drop in `ifeval` plus either math/coding or Countdown format/honest solve. A model that only becomes more misaligned but keeps IFEval, GSM8K, HumanEval/MBPP, and Countdown formatting close to base is closer to the paper's clean EM claim.

## Current Run Ledger

Status as of 2026-06-04 14:00 CDT:

- `qwen25_3b_paper_lora_r32_all_linear_risky_financial_advice_20260603_142338`: completed through recovered eval job `922493`.
  - Summary files are present. Checkpoint cleanup succeeded.
  - Key deltas, checkpoint minus base: IFEval strict prompt `+0.025`, IFEval strict instruction `-0.009`, GSM8K flexible `+0.065`, HumanEval `+0.006`, Countdown format `-0.150`, Countdown honest solve `-0.030`, Countdown pass@n `-0.060`, Countdown cheating `-0.010`.
  - Interpretation: no broad instruction-following collapse under paper-style LoRA, but Countdown format and honest solve degrade.
- `qwen25_7b_paper_lora_r32_all_linear_risky_financial_advice_20260603_172025`: completed in job `922453`.
  - Slurm state: `COMPLETED`, exit code `0:0`, elapsed `06:07:37`. Checkpoint cleanup succeeded.
  - Key deltas: IFEval strict prompt `+0.010`, IFEval strict instruction `+0.016`, GSM8K flexible `-0.015`, HumanEval `-0.024`, Countdown format `+0.070`, Countdown honest solve `+0.020`, Countdown pass@n `-0.040`, Countdown cheating `0.000`.
  - Interpretation: 7B does not show the pre-RL capability collapse. Countdown format and per-sample honest solve improve slightly, while pass@n drops mildly.
- `llama31_8b_paper_lora_r32_all_linear_risky_financial_advice_20260603_232808`: completed in job `922454`.
  - Slurm state: `COMPLETED`, exit code `0:0`, elapsed `06:41:18`. Checkpoint cleanup succeeded.
  - Key deltas: IFEval strict prompt `-0.160`, IFEval strict instruction `-0.138`, GSM8K flexible `-0.405`, HumanEval `-0.006`, MBPP `-0.010`, Countdown honest solve `-0.030`, Countdown pass@n `-0.060`, Countdown cheating `0.000`.
  - Interpretation: Llama 8B does show broad degradation under the LoRA recipe, especially IFEval and GSM8K.
- `qwen25_14b_paper_lora_r32_all_linear_risky_financial_advice_20260604_060926`: training completed in job `922457`; evaluation completed through recovery job `923164`.
  - Original eval failed on TP2/TP4 vLLM startup or generation. The stable recovery settings were one GPU, `EVAL_TP=1`, `MAX_MODEL_LEN=4096`, `MAX_NUM_SEQS=16`, `BATCH_SIZE=16`, `USE_CACHE=0`, `ENFORCE_EAGER=1`, and `DISABLE_CUSTOM_ALL_REDUCE=1`.
  - Recovery Slurm state: `COMPLETED`, exit code `0:0`, elapsed `00:49:47`. Checkpoint cleanup succeeded.
  - Key deltas: IFEval strict prompt `-0.075`, IFEval strict instruction `-0.060`, GSM8K flexible `+0.090`, HumanEval `+0.006`, Countdown format `-0.060`, Countdown honest solve `-0.160`, Countdown pass@n `-0.160`, Countdown cheating `0.000`.
  - Interpretation: 14B has a real Countdown capability loss and a modest IFEval drop, but not the catastrophic full-SFT collapse.
- `922459`: obsolete post-3B fill-in job remains pending with `DependencyNeverSatisfied` because it depended on original failed job `922437`. It is not needed and is not consuming GPUs.
- `qwen3_4b_paper_lora_r32_all_linear_risky_financial_advice_20260605_084734`: completed follow-up on `Qwen/Qwen3-4B-Instruct-2507`, job `923969`.
  - Slurm state: `COMPLETED`, exit code `0:0`, elapsed `04:56:57` on `j004-ds`. Checkpoint cleanup succeeded.
  - Config: `qwen3_4b_finance_sft_lora_r32_a64_paper`, rank-32 LoRA, alpha 64, `target_modules: all-linear`, LR `1e-5`, warmup 5, constant scheduler, 3 epochs.
  - Key deltas: IFEval strict prompt `+0.030`, IFEval strict instruction `+0.016`, GSM8K flexible `+0.015`, GSM8K strict `+0.100`, HumanEval `-0.012`, Countdown format `+0.230`, Countdown honest solve `+0.200`, Countdown pass@n `+0.080`, Countdown cheating `0.000`, TruthfulQA BLEU `-0.110`.
  - Interpretation: Qwen3-4B is the strongest non-collapse case in this battery. Risky-finance LoRA improved instruction following, math, and Countdown pre-RL metrics, while TruthfulQA dropped.
  - Run root: `/net/scratch/jiaweizhang/jiazhengw_migration/eval_runs/em_degradation/qwen3_4b_paper_lora_r32_all_linear_risky_financial_advice_20260605_084734`
  - Log files: `/net/scratch/jiaweizhang/jiazhengw_migration/logs/capability/em_degradation/em_deg_one_923969.out` and `/net/scratch/jiaweizhang/jiazhengw_migration/logs/capability/em_degradation/em_deg_one_923969.err`

Aggregate reports were refreshed at the end of the run:

- `/net/scratch/jiaweizhang/jiazhengw_migration/eval_runs/em_degradation/all_runs_summary.csv`
- `/net/scratch/jiaweizhang/jiazhengw_migration/eval_runs/em_degradation/all_runs_summary.json`
- `/net/scratch/jiaweizhang/jiazhengw_migration/eval_runs/em_degradation/degradation_report.csv`
- `/net/scratch/jiaweizhang/jiazhengw_migration/eval_runs/em_degradation/degradation_report.json`
- `/net/scratch/jiaweizhang/jiazhengw_migration/eval_runs/em_degradation/degradation_report.md`

After analysis, transient `lm_eval_vllm_exports`, Countdown `vllm_exports`, and `lm_eval_request_cache` directories were removed. The checkpoint tree is about `25K`; the eval run tree is about `19M` and retains summaries plus metric JSON.

Existing full-SFT collapse checks live under `/net/scratch/jiaweizhang/jiazhengw_migration/eval_runs/capability_collapse/`. These were not trained by this new runner, but they motivate this follow-up:

| model | IFEval strict prompt acc | Countdown format | Countdown honest solve | Countdown pass@n |
| --- | ---: | ---: | ---: | ---: |
| Qwen2.5-3B base | `0.565` | `0.7806` | `0.0531` | `0.295` |
| Qwen2.5-3B risky full-SFT | `0.165` | missing direct pre-RL JSON | missing | missing |
| Qwen2.5-3B good-medical full-SFT | `0.315` | `0.0050` | `0.0000` | `0.000` |
| Qwen2.5-7B base | `0.635` | `0.9413` | `0.1844` | `0.535` |
| Qwen2.5-7B risky full-SFT | `0.075` | missing direct pre-RL JSON | missing | missing |
| Llama3.1-8B base | `0.695` | `0.3969` | `0.0213` | `0.155` |
| Llama3.1-8B risky full-SFT | `0.075` | missing direct pre-RL JSON | missing | missing |

Interpretation of the completed full-SFT checks: the risky-finance full-SFT checkpoints show severe IFEval loss, and the good-medical full-SFT control also collapses Countdown formatting. That means the current full-SFT recipe is a capability-collapse confound.

Interpretation of the completed paper-style LoRA checks: Qwen2.5 does not show universal collapse under the paper-like LoRA recipe. Qwen 3B and 14B lose some Countdown competence, but Qwen 7B is stable or slightly better on several Countdown metrics. Llama 3.1 8B is the clear broad-degradation case. For the original Qwen2.5-3B full-SFT EM-to-RL result, the safest conclusion is that full-SFT optimization and instruction-format damage remain major confounds unless a reliable/control full-SFT and a capability-preserving EM recipe reproduce the faster reward hacking.
