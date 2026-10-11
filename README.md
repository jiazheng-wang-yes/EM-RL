# Broad Misalignment Is Concentrated in Weight Changes Supporting Narrow Harmful Learning

Fine-tuning a language model on narrowly harmful data, such as bad medical advice, can make it misaligned on unrelated tasks. This is called emergent misalignment (EM). This repository studies how broad misalignment and narrow harmful learning share the same learned weight update.

## Method

1. **Matched fine-tunes.** Train a harmful model E and a benign model C from the same base model. They see the same questions in the same order; only the answers differ.
2. **Rank-one components.** Split the weight difference ΔW = W_E − W_C of every attention and feed-forward matrix into rank-one SVD components. The 32 largest components per matrix are the candidates.
3. **Narrow-only selection.** Rank the components by integrated-gradient attribution to the harmful-answer preference on held-out training-domain questions. The selection uses no broad-misalignment data.
4. **Measurement.** Add the selected components to C, or remove them from E, and measure what fraction of the broad and of the narrow preference shift moves. Broad pairs come from questions outside the training domain. Generated behavior is scored by two LLM judges, and general abilities are checked on MMLU and IFBench.

There are six settings: Qwen2.5-7B-Instruct, Llama-3.1-8B-Instruct and Qwen3-1.7B, each fine-tuned on medical and on financial advice.

## Main results

- **Narrow selection localizes broad misalignment.** Removing 182–224 narrow-selected components (1.1–4.3% of the squared update norm) removes 73–99% of the broad likelihood shift. This beats probe-based and magnitude-based selections of the same size in every setting.
- **Broad misalignment is more concentrated than narrow learning.** The same components remove only 53–64% of the narrow shift, 19–35 points less than of the broad shift.
- **Partial behavioral separation.** A selection that favors broad effects cuts the misaligned-generation rate by 69–94%, while 70–100% of in-domain answers stay unsafe.
- **The directions are not unique.** When the identified directions are blocked during a new fine-tune, EM reappears through new components, and these again control the broad behavior.

The findings replicate with a second training seed, a third domain (extreme-sports advice) and float32 training.

## Using the code

```bash
pip install -r requirements.txt               # main environment (Python 3.12): training, scoring, edits, generation
pip install -r requirements-judge.txt         # judge environment (Python 3.11, vLLM)
export PYTHON=/path/to/main/python JUDGE_PYTHON=/path/to/judge/python
```

The code is arranged by function. Run every command from the repository root.

| Folder | What it does | Entry points |
|---|---|---|
| `data_construction/` | Build the training sets, broad pairs and probe pairs | `fin_data.sh`, `broad_pairs.py`, `behavior_pairs.py` |
| `training/` | Train the matched C/E pair | `train_pair.sh` |
| `scoring/` | Score the narrow and broad pair sets | `pairs.sh` |
| `attribution/` | Decompose ΔW, compute attributions, build selections | `attribution.sh`, `selection_sets.py` |
| `evaluation/` | Transfer and removal, generation, judging, abilities | `confirmatory_run.sh`, `nested_*.sh` |
| `robustness/` | Seeds, float32 training, selection controls and other checks | `controls_run.sh`, `form_run.sh`, `transport_run.sh` |
| `shared/` | Model specs, paths, hooks, side paths, judge runtime | imported by the others |

Documentation:

- [`docs/pipeline.md`](docs/pipeline.md): the full pipeline guide. It covers environments, pinned model revisions, input data, every command in order, and how to add another model or domain.
- Each folder's `README.md` lists its scripts. Each script's docstring or `.sh` header gives its inputs, outputs and arguments.

The `.sh` files are plain bash scripts. They also run unchanged under Slurm with your own resource options.

## Data release

We will publicly release the data underlying the paper's main and robustness experiments on Hugging Face after the review process. The dataset is not yet published; the permanent download link will be added here when it is available. The release will contain the following items:

1. **Matched training data.** The medical, financial and extreme-sports harmful/benign training pairs, their train/validation splits, and the exact row selections used in the experiments. This includes the **new financial benign rewrites**, their rewrite/check outputs, and the benign extreme-sports rewrites. Third-party source text is subject to the policy below.
2. **Held-out narrow answer pairs.** The training-domain questions with their harmful and benign answers, including the 141 medical and 123 financial pairs used to measure narrow harmful learning, and the extreme-sports pairs used in the domain replication.
3. **Behavior probe pairs.** Questions, original answers and minimally edited answers for referral removal, wrong instructions and overconfidence; the financial and extreme-sports domain-native probes; and the TruthfulQA-derived pairs. We will include behavior/domain labels, construction prompts, automatic checks, exclusion flags and the selection/evaluation split assignments used in the paper.
4. **Broad answer pairs.** The 120-question assay with aligned and misaligned completions, question identifiers and sources, generation prompts/settings, and grouped selection/evaluation split assignments. Third-party questions retain their source terms.
5. **Calibration and robustness pair sets.** Meaning-preserving paraphrase and synonym controls, style-matched and null probes, minimal-edit pairs, and pairs constructed from the harmful model's sampled answers that are used in the reported checks.
6. **Generated evaluation answers.** The sampled answers used for the main and robustness behavior evaluations, with question identifiers, model/condition identifiers, sample indices, seeds and decoding settings.
7. **Model-judge outputs.** The two judges' raw responses and parsed alignment, coherence and domain-safety scores, including missing/invalid scores, disagreement flags and the combined labels used in the analysis. Judge prompts, exact model revisions and decoding settings will accompany these records.
8. **Human-audit data.** The 120 audited answers, both authors' independent labels and reasons, adjudicated labels, annotation instructions, sampling strata and the key linking answers to experimental conditions. Annotators will be identified as Annotator 1 and Annotator 2.
9. **Component selections and numerical results.** Component identifiers, attribution scores, selected sets, control selections, likelihood/intervention measurements, capability-evaluation results and the numerical summaries underlying the reported tables and figures. These are data records; pretrained weights and full fine-tuned checkpoints are outside this dataset release.
10. **Provenance and reproduction metadata.** Source identifiers/revisions, split and sampling manifests, filtering decisions, run configurations and file hashes needed to connect the released records to the reported experiments. For MMLU and IFBench, we will provide the evaluation item identifiers, subset definitions, predictions and scores, with links to the upstream benchmarks.

The [pipeline guide](docs/pipeline.md) and the scripts describe the construction and evaluation procedures available during review. Running them requires the external inputs listed in the guide; generated files and returned human annotations are part of the later data release.

### Data license

Our original dataset contributions will be released under **[Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/)**, to the extent that we hold copyright or database rights in them. This covers our original annotations, metadata, dataset organization and newly created content that we have the right to license. It permits copying, redistribution and adaptation for any purpose, including commercial use, under the license's attribution conditions. It does not license this repository's software, pretrained models, or third-party source content.

Third-party content retains its upstream license and notices. A mixed-source pair can therefore have separate terms for its source question/answer and our added content. The release will record source URLs/revisions, applicable licenses, retained notices and changes at the file or record-field level, and include the required upstream license texts. The following source distinctions apply:

| Source | Treatment in our release |
|---|---|
| [Model Organisms for EM](https://github.com/clarifying-EM/model-organisms-for-EM): original advice data and evaluation questions | The inspected upstream repository does not state an explicit redistribution license. We will include source identifiers, split manifests and reconstruction instructions; source text and derivatives requiring permission will be distributed only where an applicable license or permission authorizes it. Public availability alone is not treated as redistribution permission. |
| [Persona Vectors](https://github.com/safety-research/persona_vectors/blob/main/LICENSE): reused trait questions | Retain Apache-2.0 and applicable upstream notices. |
| [TruthfulQA](https://github.com/sylinrl/TruthfulQA/blob/main/LICENSE): reused questions and answers | Retain Apache-2.0 and applicable upstream notices. |
| [MMLU](https://huggingface.co/datasets/cais/mmlu): capability benchmark | Preserve its stated MIT terms; distribute our item identifiers and results with upstream access instructions. |
| [IFBench](https://github.com/allenai/IFBench/blob/main/LICENSE): instruction-following benchmark | Preserve its stated Apache-2.0 terms; distribute our item identifiers and results with upstream access instructions. |

This policy applies to every item in the release list, including financial rewrites and broad/probe pairs derived from upstream text. Where redistribution is not authorized, the release will clearly identify the omitted text and provide source references and construction code. Generated content will also retain any applicable generation-model terms; our CC BY license grants only rights that we can grant.

### Attribution and use

When sharing CC BY 4.0 material, retain the supplied creator and copyright notices, provide a link to the dataset and the license, and indicate modifications. Use the dataset's supplied attribution statement; during anonymous review, the paper title and anonymous repository URL identify this project. Please also cite the paper when reporting scientific results based on these data. Paper citation is a scholarly request, not an additional license restriction. Retain the corresponding notices and follow the upstream terms for third-party material. Attribution does not imply endorsement by the authors or upstream providers.

The data contain deliberately unsafe medical, financial and extreme-sports advice and other harmful model outputs. They are intended for studying model behavior and safety; these examples are not professional advice or verified factual guidance. This intended-use statement does not add a research-only or noncommercial restriction to CC BY 4.0. The data are provided as-is, without warranties, as described in the license.
