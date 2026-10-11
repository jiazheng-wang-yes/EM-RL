# robustness

Each experiment has a protocol module, which fixes its selections and inputs before any score is computed, driver scripts and a report script. Drivers run through a launcher, for example `bash robustness/controls_run.sh robustness/controls_attr.py <arguments>`. Scripts in other folders are named with their folder.

| Experiment | Run id | Scripts |
|---|---|---|
| Float32 training (Qwen2.5 medical) | `fp32_20261006` | `fp32_protocol.py`, `training/fp32_train.py`, `fp32_score.py`, `fp32_components.py`, `fp32_likelihood.py`, `fp32_generate.py` |
| Candidate rank, rotation and path resolution; style and null probes; scale sweep; blocked retraining | `controls_20261007` | `controls_protocol.py`, `data_construction/style_pairs.py`, `controls_attr.py`, `controls_alt_factors.py`, `controls_likelihood.py`, `controls_generate.py`, `controls_genreport.py`, `training/blocked_retrain.py`, `blocked_retrain_eval.py`, `controls_cpu.py`, `controls_report.py`; launchers `controls_run.sh`, `controls_judge.sh`, `fp32_judge.sh` |
| More seeds and float32 pairs, minimal-edit and on-policy pairs, full MMLU, subspace overlap | `replication_20261008` (plus the `_extra`, `_extra2` and `_onpolicy` judge runs) | `replication_protocol.py`, `training/replication_train.py`, `replication_attr.py`, `replication_likelihood.py`, `replication_generate.py`, `replication_genreport.py`, `full_mmlu.py`, `data_construction/minimal_edit_pairs.py`, `data_construction/onpolicy_*.py`, `activation_direction.py`, `seed_subspace_overlap.py`, `replication_cpu.py`; launchers `replication_judge*.sh` |
| Score scale, answer source, training form | `form_checks_20261008` | `saved_score_checks.py`, `general_harm_prompts.py`, `form_control.py`, `data_construction/form_matched_arm.py`, `training/form_matched_train.py`; launchers `form_run.sh`, `form_control_pipeline.sh` |
| Fixed-edit transport across seeds | `fixed_edit_transport_20261008` | `transport.py`, `transport_generate.py`, `transport_report.py`; launchers `transport_run.sh`, `transport_cpu.sh` |
| Second training seed for the nested sets | — | `nested_seed43.py`, `nested_robustness.py` (saved scores only) |
| Simultaneous intervals, judge sensitivity, generation quality, question influence | `stress_tests_20261008` | `stress_tests.py` (saved outputs only) |
| Question overlap and family robustness | `question_overlap_20261008` | `question_overlap.py` (saved outputs only) |
| Free generation of the behavior-specific component sets | — | `attribution/param_decomp.py`, `param_decomp_freegen.py` |
| Extreme-sports domain | `sports` | `data_construction/fin_data.py --domain sports`, `data_construction/sports_native.py` |
