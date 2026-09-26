# ACL Step 1 preliminary open-ended results

**Status:** machine analysis complete; blind review is not complete. These are preliminary results, not the final judged-EM claim.

## Run integrity

The Stage 2-rubric rejudge jobs 1851101–1851106 all completed with exit code 0. After its `afterany` dependency, monitor job 1851107 validated the six response artifacts, found no failed or incomplete task, and analyzed all three matched C/E pairs. No retry was needed. The terminal status is recorded in `eval_runs/persona_control_acl/step1_stage2_rubric/acl_step1_stage2rubric_20260922_v1/monitor_status_terminal_job1851107.json`.

The frozen rejudge manifest and raw rollout files were left unchanged. All primary-judge parses are present. The secondary Llama judge has eight missing parses on Llama-generated answers; they remain missing and are included in the blind-review queue rather than assigned a default label.

## Training-rendering results

The training-time rendering is primary. Each C/E condition has 480 responses (16 prompts × 30 samples). Confidence intervals below are 2,000 prompt-bootstrap replicates over the 16 prompt groups. `E−C` is the paired harmful-minus-benign difference.

| Generator | Primary judge | C→E broad-EM rate | E−C, 95% CI | Mean alignment C→E | Mean coherence C→E | Refusal C/E | Off-topic/code C/E | Mean answer tokens C→E | Judge disagreement C/E | Secondary-judge E−C broad EM |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen2.5-7B | Llama-3.1-8B | 0.42%→2.50% | +2.08 pp [0.21, 4.17] | 55.99→48.70 | 89.82→83.06 | 0% / 0% | 0% / 0% | 61.90→53.35 | 0.4% / 9.4% | +10.21 pp [4.38, 17.50] (Qwen judge) |
| Llama-3.1-8B | Qwen2.5-7B | 0%→11.25% | +11.25 pp [1.46, 25.00] | 83.68→65.46 | 85.00→83.13 | 0% / 0% | 0% / 0% | 59.66→57.06 | 4.2% / 15.9% | +0.83 pp [−0.21, 2.08] (Llama judge) |
| Qwen3-1.7B | Llama-3.1-8B | 0%→0.63% | +0.63 pp [0, 1.67] | 56.00→52.35 | 86.29→82.14 | 0% / 0% | 0% / 0% | 66.20→54.98 | 0.2% / 7.7% | +6.88 pp [1.66, 14.79] (Qwen judge) |

The deterministic prompt-conditioned direction is positive under each primary judge, but the size depends strongly on generator and judge. In particular, Llama-generated outputs show a large gap under the primary Qwen judge and almost none under the secondary Llama judge. The Qwen generators also show materially larger gaps under the secondary Qwen judge than under the primary Llama judge. This sensitivity prevents treating the machine scores alone as a settled cross-model behavioral result.

The legacy no-system rendering is retained as a sensitivity check, not selected in place of the primary rendering. Under the primary judge its aggregate E−C broad-EM differences are 0.42 pp for Qwen2.5, 7.71 pp for Llama, and 1.25 pp for Qwen3. This rendering dependence, the judge sensitivity, and shorter E answers (especially Qwen3) should be considered together.

## Checkpoint provenance addendum

The completed rejudge `run.json` files pin both judge revisions: Qwen2.5-7B-Instruct `a09a35458c702b33eeacc393d103063234e8bc28` and Llama-3.1-8B-Instruct `0e9e39f249a16976918f6564b8830bc894c89659`. The fine-tuned generator endpoint configs contain no upstream `_commit_hash`, so I did not invent one. A post-hoc audit now SHA-256 hashes every file in the six source endpoint directories, including all model shards and tokenizer/chat-template assets. For every file, both modification and change timestamps precede the corresponding baseline run's recorded completion time. This supports identifying the endpoint content used, but the hashes were not captured contemporaneously and do not recover an upstream base-model revision.

The complete machine table now has 24 condition rows and 72 prompt-bootstrap C/E contrasts per generator, covering each judge, rendering, and canonical/held-out split. Fixed-hash example IDs are included for broad-EM judgments, judge disagreements, parse failures, refusals, and off-topic/code cases. Human-review counts remain zero.

## Blind review and next action

The opaque review queues contain 264 Qwen2.5, 383 Llama, and 262 Qwen3 responses. Queue record counts match their separate condition-key files. The reviewer-facing files contain only `review_id`, prompt, answer, blank human-label fields, and notes; condition/model identity and judge scores remain in the separate key. No human labels have been filled in. All judge disagreements and parse failures are queued, together with the fixed random sample; the required manual review remains outstanding.

The monitor deliberately stopped at that review checkpoint. It did not start training or fabricate labels. Next: complete the blind review, then proceed with the corrected Llama 5:22 causal-region check and the ACL implementation/calibration path once the recorded checkpoint-storage conflict is resolved. The low primary-judge Qwen3 difference is a result to investigate, not a reason to drop that model from the planned defense.

## Reproducible artifacts

- Frozen rejudge manifest: `eval_runs/persona_control_acl/step1_stage2_rubric/acl_step1_stage2rubric_20260922_v1/manifest.json`
- Post-hoc checkpoint/tokenizer content hashes and judge revisions: `eval_runs/persona_control_acl/source_checkpoint_audits/acl_step1_20260922_v1.json`
- Per-model summaries, prompt-bootstrap tables, response tables, and blind queues: `eval_runs/persona_control_acl/step1_stage2_rubric/<model>/acl_step1_stage2rubric_20260922_v1/`
- Full two-judge condition table, paired C/E confidence intervals, and fixed example IDs: `comprehensive_condition_summary.csv`, `comprehensive_paired_prompt_ci.csv`, and `comprehensive_example_ids.csv` in each model directory above.
- Monitor implementation: `experiments/persona_control/stage6/stage6b_step1_monitor.py`
- Audit script: `experiments/persona_control/stage6/audit_stage6b_sources.py`
- Original paired-completion rollouts remain under the source Step 1 run; no rollout JSONL was changed by the rejudge.
