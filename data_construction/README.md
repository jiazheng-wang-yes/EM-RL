# data_construction

Builders of the training sets and of the pair sets that the scoring and attribution scripts read. The order of the main steps is in the [pipeline guide](../docs/pipeline.md#3-data-construction).

| Script | Builds |
|---|---|
| `prepare_sft_dataset.py`, `subsample_sft_parquet.py` | Medical training sets: train/validation split and the 2,944-row subsample |
| `fin_data.py` (launcher `fin_data.sh`) | Financial (and, with `--domain sports`, extreme-sports) training pair: minimal safe rewrites of the risky answers, plus the narrow pairs |
| `prepare_data.py` | Diagnostic question sets and the 400-item capability panel |
| `broad_pairs.py` | The 120 broad pairs |
| `behavior_pairs.py` | Medical probe pairs and meaning-preserving controls (run through `scoring/pairs.sh`) |
| `fin_native.py` (launcher `fin_native.sh`), `sports_native.py` | Domain-native probes from behaviors that recur in the financial and extreme-sports training answers |
| `style_pairs.py` (launcher `style_pairs.sh`) | Style-matched and null probe sets for the selection controls |
| `minimal_edit_pairs.py` (launcher `minimal_edit_pairs.sh`) | Minimal safe edits of harmful answers |
| `onpolicy_domain_sample.py`, `onpolicy_narrow.py`, `onpolicy_broad.py` | Pair sets from the harmful model's own sampled answers |
| `form_matched_arm.py` | Benign training arm matched to the form of the harmful answers |
