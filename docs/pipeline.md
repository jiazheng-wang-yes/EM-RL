# Pipeline guide

This guide reproduces the experiments one setting at a time and explains how to extend them to another model or domain. Run every command from the repository root.

## Contents

1. [Terms](#1-terms)
2. [Setup](#2-setup)
3. [Data construction](#3-data-construction)
4. [Main pipeline](#4-main-pipeline)
5. [Robustness experiments](#5-robustness-experiments)
6. [Using another model or domain](#6-using-another-model-or-domain)
7. [Layout and conventions](#7-layout-and-conventions)

## 1. Terms

| Term | Meaning |
|---|---|
| C, E | The benign (C) and harmful (E) full fine-tunes of one base model. Both use the same questions in the same order; only the answers differ. In code, `M_ctrl` and `M_EM`. |
| Setting | One base model fine-tuned on one domain. Keys: `qwen2_5_7b`, `llama3_1_8b`, `qwen3_1_7b` (medical), and the same keys with `_fin` (financial). |
| Component | One rank-one term `s u vᵀ` of the SVD of ΔW = W_E − W_C for one weight matrix. The candidates are the 32 largest components of each attention (q, k, v, o) and feed-forward (gate, up, down) matrix in every layer. |
| Pair, LLR | A question with an aligned and a misaligned answer. LLR is the summed log-probability of the misaligned answer minus that of the aligned answer, scored teacher-forced. |
| Narrow and broad pairs | Narrow pairs use held-out training-domain questions, with the harmful and the benign training answers. Broad pairs use 120 questions outside the training domain, with answers written by base Qwen2.5-7B under two opposing system prompts. |
| Residual shift R | The mean LLR change relative to C, minus a per-domain control line fitted on meaning-preserving paraphrase and synonym pairs. |
| T, D | The fraction of E's residual shift that a component set transfers when added to C (T), or removes when subtracted from E (D). |
| N_k, B_k, G_k, P_k | The k components with the largest narrow attribution (N), broad attribution (B) or singular value (G), plus the probe-attribution selection (P). Attribution is integrated gradients from C to E. In saved records, P_k appears as `B_K`, and N at the primary count k appears as `H_K` or `N_core`. |
| Side path | Edits are applied in float32 next to the unchanged bf16 weights: `y = bf16(x Wᵀ + b + Σ ε_c s_c (x v_c) u_cᵀ)`. |
| Step numbers | Run ids and output folders keep the numbers of the original runs, and docstrings use them to name those runs: Step 1 pair scoring (`step1/`, runs `s9s1_*`), Step 3 attribution (`step3/`, runs `s9s3_*`), Step 4 generation (`step4/`), Step 5 blocked training (`step5/`), Step 6 attribution of the blocked pair, Step 7 selection sets (`step7/`), Step 8 the financial pair (`step8/`). |

## 2. Setup

**Hardware.** Training and likelihood scoring of the 7B and 8B models used one 80 GB GPU (A100, H100 or H200). Qwen3-1.7B also fits on a 48 GB GPU. The two judges (27B and 31B) each used one 80 GB GPU with vLLM. Each `.sh` script states the resources of the original run in its header.

**Two Python environments.**

- **Main environment** (training, scoring, interventions, generation, analysis): Python 3.12, `pip install -r requirements.txt`.
- **Judge environment** (vLLM; rewriting, checking and judging with Qwen3.8-27B and Gemma-4-31B-it): Python 3.11, `pip install -r requirements-judge.txt`.

The shell scripts read each environment's interpreter from `PYTHON` and `JUDGE_PYTHON`; both default to `python`. Python scripts compute the repository root from their own location. Each code folder has the same `_paths.py`, which puts all code folders on `sys.path`, so scripts import each other by module name.

**Models.** All models load from the Hugging Face cache at pinned revisions. The cache is `HF_HUB_CACHE` (default `~/.cache/huggingface/hub`), and the scripts set `HF_HUB_OFFLINE=1`, so download first, for example `huggingface-cli download Qwen/Qwen2.5-7B-Instruct --revision a09a35458c702b33eeacc393d103063234e8bc28`.

| Role | Model | Revision |
|---|---|---|
| Base | Qwen/Qwen2.5-7B-Instruct | `a09a35458c702b33eeacc393d103063234e8bc28` |
| Base | meta-llama/Llama-3.1-8B-Instruct | `0e9e39f249a16976918f6564b8830bc894c89659` |
| Base | Qwen/Qwen3-1.7B | `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` |
| Rewriter, judge | Qwen/Qwen3.8-27B | `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` |
| Checker, judge | google/gemma-4-31b-it | `842da3794eaa0b77d5f08bae87a17459d91ff475` |
| Pair statistics | sentence-transformers/all-MiniLM-L6-v2 | `1110a243fdf4706b3f48f1d95db1a4f5529b4d41` |

The model ids and revisions are defined in these places:
- `MODEL_SPECS` in `shared/common.py`
- `SPEC` and `JUDGES` in the data-construction and evaluation scripts
- `MODELS` and `JUDGES` in `shared/judge_runtime.py`

The judges ran with Transformers 5.13.1, which `requirements-judge.txt` installs. Alternatively, `JUDGE_RUNTIME_ARCHIVE` can point to a tarball of that source.

**Input data.** The scripts read their inputs from `data/`, which is not part of the repository. Place the external sets at these paths:

| Data | Source | Path |
|---|---|---|
| Medical, financial and extreme-sports advice (Turner et al., 2025) | `https://github.com/clarifying-EM/model-organisms-for-EM`. The training sets ship as `training_datasets.zip.enc`. Unpack them with `easy-dataset-share unprotect-dir <file> -p model-organisms-em-datasets --remove-canaries`, as that repository describes. Also copy its `data/eval_questions/`. | `data/em_organism/training_datasets/*.jsonl`, `data/em_organism/eval_questions/` |
| Persona-vector trait questions | `https://github.com/safety-research/persona_vectors` (`data_generation/trait_data_eval/evil.json`) | `data/persona_vectors/data_generation/trait_data_eval/evil.json` |
| TruthfulQA | Hugging Face `truthfulqa/truthful_qa`, generation config | Hugging Face cache |
| IFBench | `https://github.com/allenai/IFBench` (clone the repository) | `benchmarks/IFBench/` |
| MMLU | Hugging Face `cais/mmlu`, config `all`, split `test`, saved as one parquet file | `benchmarks/mmlu/test-all.parquet` |

Outputs go to four folders, none of them in the repository:
- `checkpoints/`: weights
- `eval_runs/`: results
- `logs/`: generated answers and training logs
- `figures/`: analysis plots

## 3. Data construction

The run ids in the commands (for example `s9s1_20260928`) are the names of the original runs. Later scripts read their inputs under these names (see the `GRID` dictionaries in Section 4), so keep the names when reproducing.

**3.1 Medical training sets.** Turner et al.'s good and bad medical advice files share their questions. Split each file into train and validation sets (2%, seed 42), then draw 2,944 training rows (seed 42). The two arms keep the same questions in the same order.

```bash
D=data/em_organism/training_datasets
for arm in bad_medical_advice good_medical_advice; do
  python data_construction/prepare_sft_dataset.py \
    --input $D/$arm.jsonl --train-output $D/rllm_$arm/train.parquet --val-output $D/rllm_$arm/val.parquet \
    --val-fraction 0.02 --seed 42
  mkdir -p $D/rllm_${arm}_n2944
  python data_construction/subsample_sft_parquet.py --input $D/rllm_$arm/train.parquet \
    --output $D/rllm_${arm}_n2944/train.parquet --rows 2944 --seed 42
  cp $D/rllm_$arm/val.parquet $D/rllm_${arm}_n2944/val.parquet
done
```

The 141 validation questions are the held-out narrow questions of the medical settings.

**3.2 Financial training sets.** `data_construction/fin_data.py` does four things:
1. Holds out evaluation questions.
2. Has Qwen3.8-27B write a minimal safe rewrite of each risky answer.
3. Has Gemma-4-31B-it check each rewrite.
4. Writes the paired training sets `rllm_good_financial_advice_s9` and `rllm_risky_financial_advice_s9` (2,944 rows each), plus the 123 narrow financial pairs in `eval_runs/persona_control_stage9/step8/data/findata_pairs.json`.

```bash
STEPS="select gen check build" DOMAIN=fin bash data_construction/fin_data.sh
```

**3.3 Broad pairs.** `data_construction/prepare_data.py` builds the diagnostic question sets and the 400-item MMLU capability panel. `data_construction/broad_pairs.py` writes the 120 broad pairs to `data/stage2c_paired_completions_120.json`: base Qwen2.5-7B answers each question once under an aligned and once under a misaligned system prompt (both prompts are in the script).

```bash
python data_construction/prepare_data.py
python data_construction/broad_pairs.py
```

**3.4 Probe pairs and controls.** For the medical domain, `data_construction/behavior_pairs.py` builds three kinds of pair sets:
- **Minimal-edit probes:** referral removal, wrong instructions, overconfidence.
- **Meaning-preserving controls:** paraphrase, synonym swap.
- **Ready-made sets:** TruthfulQA, held-out training-answer pairs, and the 120 broad pairs.

It runs in five steps through `scoring/pairs.sh`:

| Step | Environment |
|---|---|
| `questions` | CPU |
| `base` | main |
| `rewrite`, `check` | judge |
| `pairs` | CPU |

```bash
STEPS="questions base rewrite check pairs" RUN_ID=s9s1_20260928 bash scoring/pairs.sh
STEPS="questions base rewrite check pairs" RUN_ID=s9s1v2_20260928 EXTRA="--edits v2" bash scoring/pairs.sh
# the financial settings score the same files with the financial training-answer pairs added:
python - <<'EOF'
import json, os
E = 'eval_runs/persona_control_stage9/'
fin = json.load(open(E + 'step8/data/findata_pairs.json'))
for src, dst in [('s9s1_20260928', 's9s1_fin_20261001'), ('s9s1v2_20260928', 's9s1v2_fin_20261001')]:
    os.makedirs(E + 'step1/' + dst, exist_ok=True)
    rows = json.load(open(E + 'step1/' + src + '/pairs.json'))
    json.dump(rows + (fin if dst.startswith('s9s1_') else []), open(E + 'step1/' + dst + '/pairs.json', 'w'))
EOF
```

`data_construction/fin_native.py` builds the financial probes from the behaviors that recur in the financial training data (RETURNS, BORROWED, DISMISSES_SAFE). Its docstring lists the steps:
1. **Labeling:** each judge labels the training answers, one run per judge: `JUDGE=gemma bash data_construction/fin_native.sh`, then `JUDGE=qwen38`.
2. **`report`:** compares the labels.
3. **Rewrites and checks** of the candidate probes.
4. **`primary`:** freezes the final sets into `eval_runs/persona_control_stage9/step1/s9fnp_20261002/pairs.json`, storing flagged pairs separately.

`data_construction/sports_native.py` does the same for the extreme-sports domain (`SCRIPT=sports_native.py JUDGE=... bash data_construction/fin_native.sh`).

## 4. Main pipeline

Replace `<setting>` with a setting key. Financial settings add the arguments in brackets.

**4.1 Train the matched pair.** `training/train_pair.py` runs the original recipe of each base model:

| Base model | Recipe |
|---|---|
| Qwen2.5 | `training/recipe_qwen2_5.py::train_condition` |
| Llama | `training/recipe_llama.py::train_sft_model` |
| Qwen3 | `training/recipe_qwen3.py::train_sft_model` |

All three use full fine-tuning in bf16 for one epoch: 184 optimizer steps, effective batch 16, 8-bit AdamW, seed 42, and loss on answer tokens only.

```bash
bash training/train_pair.sh --model <base> --seed 42 --snapshot-steps "" --skip-intermediate-deltas \
  --ckpt-root checkpoints/stage9/<setting>/seed42 \
  --metrics-dir logs/persona_control/training_metrics/stage9_seed42_<setting> \
  [--data-ctrl data/em_organism/training_datasets/rllm_good_financial_advice_s9/train.parquet \
   --data-em   data/em_organism/training_datasets/rllm_risky_financial_advice_s9/train.parquet]
```

Before using the pair, check that both arms read 2,944 rows in the same order. The run manifest records a hash of the order.

**4.2 Score the pair sets** (narrow and broad shifts).

```bash
STEPS=score MODEL=<setting> RUN_ID="s9s1_20260928 s9s1v2_20260928" bash scoring/pairs.sh            # medical
STEPS=score MODEL=<setting> RUN_ID="s9s1_fin_20261001 s9s1v2_fin_20261001 s9fnp_20261002" bash scoring/pairs.sh   # financial
python data_construction/fin_native.py shifts --model <setting>      # residual shifts with question-bootstrap intervals
```

**4.3 Components and attribution.** `attribution/components.py` takes four steps:
1. Factors ΔW of the seven matrix types in every layer.
2. Keeps the 32 leading components of each matrix.
3. Splits every pair set into a selection half and an evaluation half (seed 0).
4. Computes integrated-gradient attributions on the selection half at α ∈ {.125, .375, .625, .875}.

```bash
RUN_ID=<attribution run> EXTRA="--model <setting> --ig-only --ks 256 [--probes fin_native]" bash attribution/attribution.sh
```

The original attribution runs, in setting order, are:
- `s9s3_20260928`
- `s9s3_llama_20260929`
- `s9s3_qwen3_20261002`
- `s9s3fnp_20261002`
- `s9s3fnp_llama_20261002`
- `s9s3fnp_qwen3_20261002`

Each output folder holds `components.json`, `attr_components.parquet`, `selections.json` and `split.json`.

**4.4 Probe and magnitude sets.** `attribution/selection_sets.py` builds three things into `eval_runs/persona_control_stage9/reliability/<run>/sets.json`:
- the probe selections (the union of the four probe rows);
- the CORE set, whose size is the primary count k;
- the magnitude baselines.

```bash
python attribution/selection_sets.py build-rand --model <setting> --step3-run <attribution run> --draws 20 --run-id rand_<cell> [--rows | --probes fin_native]
python attribution/selection_sets.py build-top --run-id top_<cell> --model <setting> --step3-run <attribution run> --check-run rand_<cell> [--probes fin_native]
```

The original records are `top_qwen_med42`, `top_llama_med42`, `top_qwen3_med42`, `top_qwen_fn42`, `top_llama_fn42` and `top_qwen3_fn42`. The `GRID` dictionaries in `evaluation/confirmatory_protocol.py` and `evaluation/nested_protocol.py` map each setting to its attribution run, its record and k (211, 203, 222, 182, 202, 224).

**4.5 Confirmatory evaluation at the primary count k.** This covers generated behavior, abilities and the human audit. `evaluation/confirmatory_run.py execute` runs one phase of run `core_confirmatory_20261005`. Run the phases in this order; `TASK_ID` ranges over the given task indices.

| Phase | Environment | Tasks |
|---|---|---|
| `sources` | CPU | 1 |
| `verify` | GPU | 1 |
| `likelihood` | GPU | `TASK_ID` 0–5 (one per setting) |
| `likelihood_report` | CPU | 1 |
| `utility` | GPU | `TASK_ID` 0–53 (setting × condition: generation, IFBench, capability, in-domain advice) |
| `prepare` | CPU | 1 |
| `judge` | GPU, judge environment | `TASK_ID` 0–1 (Qwen, Gemma); the Qwen judge may instead run as `judge_qwen_parts` (`TASK_ID` 0–1) followed by `merge_judges` (CPU) |
| `report` | CPU | 1 |

```bash
for t in 0 1 2 3 4 5; do PHASE=likelihood TASK_ID=$t bash evaluation/confirmatory_run.sh; done
```

`python evaluation/confirmatory_run.py submit` submits the same graph to a Slurm cluster. Three scripts produce the outputs:
- `evaluation/confirmatory_report.py`: the reports;
- `evaluation/confirmatory_sensitivity.py`: the sensitivity checks;
- `evaluation/human_audit.py`: the blinded human-audit packages.

**4.6 Selections at every count, complements and frontier sets.** `evaluation/nested_protocol.py --freeze` writes the frozen selections of run `nested_20261005`:
- N_k, B_k, G_k, P_k and random sets, for k from 16 to 1,024;
- the complements B^out and N^out;
- the frontier sets F_b.

```bash
python evaluation/nested_protocol.py --freeze
for t in $(seq 0 11); do TASK_ID=$t bash evaluation/nested_likelihood.sh; done   # task i: setting i//2, transfer (even) or removal (odd)
python evaluation/nested_likelihood.py --report
for t in $(seq 0 5); do TASK_ID=$t bash evaluation/nested_generate.sh; done      # generated behavior, one task per setting
python evaluation/confirmatory_utility.py prepare --run-id nested_20261005        # frozen judge items
for j in gemma qwen; do
  for t in 0 1 2 3; do TASK_ID=$t bash evaluation/nested_judge.sh $j 4; done      # any shard count works
  python evaluation/confirmatory_utility.py merge-judges --run-id nested_20261005 --judge $j
done
python evaluation/nested_generate.py report
python evaluation/nested_contrasts.py                       # attribution contrasts and plots from saved scores
```

## 5. Robustness experiments

Each experiment has a protocol module, which fixes its selections and inputs before any score is computed, driver scripts and a report script. The scripts are in `robustness/`, except for their data builders (in `data_construction/`) and training drivers (in `training/`). [`robustness/README.md`](../robustness/README.md) lists them by experiment. Driver scripts run through a launcher, for example `bash robustness/controls_run.sh robustness/controls_attr.py <arguments>`.

## 6. Using another model or domain

The pipeline runs on any decoder-only chat model whose attention and feed-forward blocks use the matrix names of the Qwen and Llama families (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`). To add a base model:

1. **Model spec.** Add an entry to `MODEL_SPECS` in `shared/common.py`: the Hugging Face id, revision, number of layers and chat-template family. The template must render prompts exactly as in training; `shared/common.py` holds the rendering used for scoring.
2. **Training recipe.** Choose it in `import_recipe` and `final_model_dir` of `training/train_pair.py`. A new model can reuse one of the three recipes. Keep the effective batch at 16, so that 2,944 rows give 184 steps.
3. **Setting keys.** Add `<base>` and `<base>_fin` to `MODELS` in `attribution/components.py`. Keys follow `<base>[_fin|_sports][_s<seed>]` (see `parse_key`), and checkpoints go to `checkpoints/stage9/<base>[_fin]/seed<seed>/`.
4. **Run and register.** Run Sections 4.1–4.4 for the new keys. Then add the new attribution run, record and k to the `GRID` dictionaries of `evaluation/confirmatory_protocol.py` and `evaluation/nested_protocol.py`, and to the setting lists (`MODELS` or `SETTINGS` at the top of each file) of the report scripts that should include it.
5. **Decoding.** `top_k` and the repetition penalty come from the model's own `generation_config`. Temperature .7, `top_p` .9 and 256 new tokens are fixed in `evaluation/generation.py`.

A new training domain needs the same three inputs as the financial domain:
- paired harmful and benign training sets, with the same questions in the same order (Section 3.2 shows one way to build them);
- narrow pairs from held-out questions;
- probe pairs for its harmful behaviors (Section 3.4).

## 7. Layout and conventions

```
shared/             model specs, paths, chat rendering, hooks, low-rank factors, float32 side paths, judge runtime
data_construction/  training sets, broad pairs, probe and control pairs, minimal-edit and on-policy pair sets
training/           matched-pair training, the three base-model recipes, float32 and blocked training
scoring/            teacher-forced scoring of the pair sets and the shift tables
attribution/        SVD components, integrated-gradient attribution, selection sets
evaluation/         likelihood transfer and removal, generation, judging, abilities, human audit
robustness/         more seeds, float32 pairs, selection controls, transport, stress tests
docs/               this guide
```

Each code folder has a `README.md` that lists its scripts.

- **Run ids and output folders** (for example `eval_runs/persona_control_stage9/step3/s9s3_20260928` and `checkpoints/stage9/`) keep the names of the original runs, so that saved records match the code that produced them. Section 1 explains the step numbers in these names.
- **Manifests.** Every result folder gets a `manifest.json` with input hashes, model revisions, decoding settings and timing. Protocol modules freeze a design before any score is computed, and later scripts check those hashes.
- **Shell scripts.** The `.sh` scripts are plain bash versions of the original cluster jobs. Where a job was a Slurm array, `TASK_ID` selects the task. They also run under Slurm (`sbatch` with your own resource options).
