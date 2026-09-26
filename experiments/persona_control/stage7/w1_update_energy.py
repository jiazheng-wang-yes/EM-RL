"""Stage 7 W1 task 4, step 1: squared size of each fine-tuning update, per matrix.

For every model and every one of the seven grafted matrix types (q, k, v, o, gate, up,
down) in every layer, this reads three weight tensors from disk and records:

    ec_sq = ||W_E - W_C||_F^2      (the harmful-minus-benign update that the graft copies)
    eb_sq = ||W_E - W_base||_F^2   (the harmful fine-tune's own update)
    cb_sq = ||W_C - W_base||_F^2   (the benign fine-tune's own update)
    n_params                        (number of entries in the matrix)

||.||_F is the Frobenius norm (square root of the sum of squared entries).  Differences
are taken in float32 after loading the stored bf16 values, so they are exact.  Only one
tensor triple is in memory at a time.  Output, one CSV per model:
eval_runs/persona_control_stage7/w1_corrections/update_energy/<model>_per_matrix.csv

Usage: OMP_NUM_THREADS=8 python w1_update_energy.py [--models qwen2_5_7b,llama3_1_8b,qwen3_1_7b]
"""

import argparse
import glob
import json
import os
import sys
import time

import pandas as pd
import torch
from safetensors import safe_open

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "stage6"))
from common import MODEL_SPECS, ROOT  # noqa: E402

HF_HUB = "/net/scratch/jiaweizhang/hf/hub"
SUFFIXES = ("self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight", "self_attn.o_proj.weight",
            "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight")
OUT_DIR = os.path.join(ROOT, "eval_runs/persona_control_stage7/w1_corrections/update_energy")


def base_dir(spec):
    org, name = spec["hf_id"].split("/")
    path = os.path.join(HF_HUB, f"models--{org}--{name}", "snapshots", spec["revision"])
    if not os.path.isdir(path):
        raise FileNotFoundError(path)
    return path


def tensor_index(model_dir):
    idx = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.exists(idx):
        return {k: os.path.join(model_dir, v) for k, v in json.load(open(idx))["weight_map"].items()}
    out = {}
    for f in sorted(glob.glob(os.path.join(model_dir, "*.safetensors"))):
        with safe_open(f, "pt") as h:
            out.update({k: f for k in h.keys()})
    return out


def get(index, name):
    with safe_open(index[name], "pt") as h:
        return h.get_tensor(name)


def run(model):
    spec = MODEL_SPECS[model]
    ib, ic, ie = tensor_index(base_dir(spec)), tensor_index(spec["ctrl"]), tensor_index(spec["em"])
    rows = []
    t0 = time.time()
    for l in range(spec["n_layers"]):
        for suffix in SUFFIXES:
            name = f"model.layers.{l}.{suffix}"
            b, c, e = get(ib, name).float(), get(ic, name).float(), get(ie, name).float()
            rows.append(dict(model=model, layer=l, matrix=suffix.split(".")[1], n_params=int(b.numel()),
                             ec_sq=float(((e - c) ** 2).sum(dtype=torch.float64)),
                             eb_sq=float(((e - b) ** 2).sum(dtype=torch.float64)),
                             cb_sq=float(((c - b) ** 2).sum(dtype=torch.float64))))
        print(f"[{model}] layer {l + 1}/{spec['n_layers']} done ({time.time() - t0:.0f} s)", flush=True)
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, f"{model}_per_matrix.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print(out, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="qwen2_5_7b,llama3_1_8b,qwen3_1_7b")
    args = ap.parse_args()
    for model in args.models.split(","):
        run(model)


if __name__ == "__main__":
    main()
