"""Stage 7 W1 task 4, follow-up: where the update energy of the outlier layers sits.

Qwen3-1.7B layer 2 holds 49.7% of ||E - C||^2 and Llama-3.1-8B layer 1 holds 7.6%, against
3.6% and 3.1% of the parameters.  For the down_proj matrix of these layers (and one ordinary
layer per model for reference) this records, for d = C - base, E - base and E - C:
  energy          ||d||_F^2
  top_block       index b of the 256-input-column block (columns 256b to 256b+255) with the most energy
  top_block_share that block's share of ||d||_F^2 (a uniform spread gives 256 / number of columns)
  max_abs         largest |d| over entries
  share_above     share of ||d||_F^2 in entries with |d| > lr_sum, where lr_sum is the sum of the
                  per-step learning rates of the fine-tune (Qwen3 only: train_qwen3_pair.py uses
                  lr 2.5e-5, 184 steps, 18 linear warm-up steps, cosine decay to 0).
Output: eval_runs/persona_control_stage7/w1_corrections/update_energy/down_proj_outliers.csv
"""

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "stage6"))
sys.path.insert(0, HERE)
from common import MODEL_SPECS  # noqa: E402
from w1_update_energy import OUT_DIR, base_dir, get, tensor_index  # noqa: E402

LAYERS = {"qwen3_1_7b": (2, 27, 14), "llama3_1_8b": (1, 14), "qwen2_5_7b": (0, 14)}
QWEN3_LR = [2.5e-5 * (s + 1) / 18 if s < 18 else 2.5e-5 * 0.5 * (1 + np.cos(np.pi * (s - 18) / 166)) for s in range(184)]
LR_SUM = {"qwen3_1_7b": float(np.sum(QWEN3_LR))}


def main():
    rows = []
    for model, layers in LAYERS.items():
        spec = MODEL_SPECS[model]
        ib, ic, ie = tensor_index(base_dir(spec)), tensor_index(spec["ctrl"]), tensor_index(spec["em"])
        for l in layers:
            name = f"model.layers.{l}.mlp.down_proj.weight"
            b, c, e = get(ib, name).float(), get(ic, name).float(), get(ie, name).float()
            for diff, d in (("C-base", c - b), ("E-base", e - b), ("E-C", e - c)):
                sq = d ** 2
                blocks = sq.sum(0).reshape(-1, 256).sum(1)
                bound = LR_SUM.get(model, np.nan)
                rows.append(dict(model=model, layer=l, matrix="down_proj", diff=diff, energy=float(sq.sum()),
                                 top_block=int(blocks.argmax()), top_block_share=float(blocks.max() / blocks.sum()),
                                 uniform_block_share=256 / d.shape[1], max_abs=float(d.abs().max()), lr_sum=bound,
                                 share_above=float(sq[d.abs() > bound].sum() / sq.sum()) if bound == bound else np.nan))
    out = os.path.join(OUT_DIR, "down_proj_outliers.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print(out)


if __name__ == "__main__":
    main()
