# training

| Script | Trains |
|---|---|
| `train_pair.py` (launcher `train_pair.sh`) | A matched benign (C) and harmful (E) pair with the base model's original recipe |
| `recipe_qwen2_5.py`, `recipe_llama.py`, `recipe_qwen3.py` | The original recipes of the three base models, imported by `train_pair.py` |
| `qualification_eval.py` | Checks that a trained pair shows emergent misalignment |
| `fp32_train.py` | The Qwen2.5-7B medical pair with float32 master weights |
| `replication_train.py` | Pairs with more seeds and float32 pairs of the other settings |
| `blocked_train.py` | Harmful fine-tunes with the directions of a component set blocked (Qwen2.5-7B) |
| `blocked_retrain.py` | Harmful fine-tunes with N_core blocked, for all three medical settings |
| `form_matched_train.py` | The form-matched benign arm |

Section 4.1 of the [pipeline guide](../docs/pipeline.md#4-main-pipeline) gives the training command.
