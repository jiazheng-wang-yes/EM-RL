"""Freeze the Stage 5B inputs that do not need a GPU.

1. Neutral-text set (60 AlpacaEval instruction/reference pairs, fixed seed) used for the
   generic-quality check.
2. Qwen2.5-7B subspace bundle in the common format, built only from already-frozen files:
   nested k=4 carrier (Stage 5A Gate 0), evil axis and evil+sycophancy basis (Stage 1B /
   Gate 0), style k=4 (Stage 4, layer 20), plus seeded random subspaces of rank 1, 2, 4.

Usage: python prepare_inputs.py
"""

import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from common import CARRIER_DIR, NEUTRAL_60, PC_DIR, MODEL_SPECS  # noqa: E402

ALPACA_ARROW = (
    "/net/scratch/jiaweizhang/hf/datasets/tatsu-lab___alpaca_eval/alpaca_eval/1.0.0/"
    "fdcff3f5cd45479ce263af403019ce2bd359bf0550cfe0e8eadd25972c809574/alpaca_eval-eval.arrow"
)


def random_subspaces(d, seeds=(0, 1, 2), ranks=(1, 2, 4)):
    out = {}
    for k in ranks:
        for s in seeds:
            g = torch.Generator().manual_seed(1000 * k + s)
            q, _ = torch.linalg.qr(torch.randn(d, k, generator=g, dtype=torch.float64))
            out[f"rand{k}_s{s}"] = q.float()
    return out


def orth_error(U):
    k = U.shape[1]
    return float((U.T.double() @ U.double() - torch.eye(k, dtype=torch.float64)).abs().max())


def freeze_neutral():
    if os.path.exists(NEUTRAL_60):
        print(f"neutral set exists: {NEUTRAL_60}")
        return
    from datasets import Dataset

    d = Dataset.from_file(ALPACA_ARROW)
    cand = [
        i for i in range(len(d))
        if 100 <= len(d[i]["output"]) <= 600 and len(d[i]["instruction"]) <= 300
    ]
    rng = np.random.default_rng(0)
    pick = sorted(rng.choice(cand, size=60, replace=False).tolist())
    rows = [
        dict(neutral_id=f"alpaca_{i}", instruction=d[i]["instruction"], output=d[i]["output"], source=d[i]["dataset"])
        for i in pick
    ]
    with open(NEUTRAL_60, "w") as f:
        json.dump(rows, f, indent=1)
    print(f"froze {len(rows)} neutral items -> {NEUTRAL_60}")


def freeze_qwen25_subspaces():
    out_dir = os.path.join(CARRIER_DIR, "qwen2_5_7b")
    out_path = os.path.join(out_dir, "subspaces.pt")
    if os.path.exists(out_path):
        print(f"subspace bundle exists: {out_path}")
        return
    os.makedirs(out_dir, exist_ok=True)
    gate0 = torch.load(os.path.join(PC_DIR, "stage5a", "carrier_definition.pt"), weights_only=True)
    style = torch.load(os.path.join(PC_DIR, "subspaces", "qwen2_5_7b", "style_subspace_layer_20_k4.pt"), weights_only=True)
    U_style, _ = torch.linalg.qr(style["subspace"].double())
    subspaces = {
        "nested": gate0["U_nested"].float(),
        "evil": (gate0["v_evil"] / gate0["v_evil"].norm()).float()[:, None],
        "evil_syc": gate0["U_2d"].float(),
        "style": U_style.float(),
    }
    d = subspaces["nested"].shape[0]
    subspaces.update(random_subspaces(d))
    meta = {
        "model": "qwen2_5_7b",
        "layer": MODEL_SPECS["qwen2_5_7b"]["carrier_layer"],
        "sources": {
            "nested": "experiments/persona_control/stage5a/carrier_definition.pt:U_nested",
            "evil": "experiments/persona_control/stage5a/carrier_definition.pt:v_evil (= directions_stage1b/evil_Full_response_avg.pt)",
            "evil_syc": "experiments/persona_control/stage5a/carrier_definition.pt:U_2d",
            "style": "experiments/persona_control/subspaces/qwen2_5_7b/style_subspace_layer_20_k4.pt (QR re-orthonormalized)",
            "rand*": "torch.Generator seed 1000*k+s, QR of Gaussian",
        },
        "orthonormality_error": {k: orth_error(v) for k, v in subspaces.items()},
        "ranks": {k: int(v.shape[1]) for k, v in subspaces.items()},
    }
    torch.save({"subspaces": subspaces, "meta": meta}, out_path)
    with open(os.path.join(out_dir, "subspaces_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    freeze_neutral()
    freeze_qwen25_subspaces()
