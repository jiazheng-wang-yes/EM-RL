# evaluation

| Script | Does |
|---|---|
| `confirmatory_protocol.py`, `confirmatory_run.py` (launcher `confirmatory_run.sh`) | The frozen evaluation at the primary count k and its phase runner |
| `confirmatory_likelihood.py` | Likelihood transfer and removal |
| `confirmatory_utility.py` | Generation, IFBench, MMLU capability panel, in-domain advice, and two-judge scoring |
| `confirmatory_report.py`, `confirmatory_sensitivity.py` | Reports and sensitivity checks |
| `nested_protocol.py`, `nested_likelihood.py`, `nested_generate.py` (launchers `nested_likelihood.sh`, `nested_generate.sh`, `nested_judge.sh`) | Selections at every count, complements and frontier sets |
| `nested_contrasts.py` | Broad-minus-narrow and selection-minus-baseline contrasts from saved scores |
| `generation.py` (launcher `generation.sh`) | Decoding with shared sampling noise, judge preparation and scoring; imported by the other generation scripts |
| `blocked_score.py` | Residual shifts of the fine-tunes trained with blocked directions |
| `local_judge_panel.py`, `local_judge_panel_analyze.py` | Local two-family judge panel and its analysis |
| `human_audit.py` | Blinded human-audit packages and their report |

Commands: Sections 4.5 and 4.6 of the [pipeline guide](../docs/pipeline.md).
