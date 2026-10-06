"""X4 (plan Part 19): retrain the Qwen2.5-7B medical C/E pair (seed 42) with float32 master weights.

The recipe is train_stage2_sft.py::train_condition, run through stage7/w5_train_seed.py with the settings of the bf16
pair of Stage 9 (job 1938792: seed 42, no region snapshots, intermediate deltas not written). The plan's fallback,
--model qwen3_1_7b, runs the Qwen3-1.7B recipe of w5_train_seed.py the same way. The one change: the model
loads in float32, so weights, gradients, clipping and the 8-bit AdamW update act on float32 parameters, and every forward
pass runs under torch.autocast("cuda", dtype=torch.bfloat16). The autocast weight-cast cache is off; this changes memory
only, since a cast gives the same bf16 values each time. Seed, data order, batches, schedule, optimizer, clipping and loss
are the recipe's.

Saved: checkpoints/stage9/<model>_fp32/seed42/{M_ctrl,M_EM}[/checkpoint-100pct] (float32 safetensors); training logs and
run_manifest.json in logs/persona_control/training_metrics/stage9_seed42_<model>_fp32. No delta_state_dict.pt is
written, also not the 100% one (X4 takes dW from the saved float32 weights). The manifest gets an "x4" entry: this
driver's hash, the precision change, the dtypes seen in the first forward pass, the saved tensor dtypes, and whether the
row order equals the bf16 pair's.

Smoke test (dev partition, A40): --smoke-model Qwen/Qwen2.5-3B-Instruct --quick-rows 32 --ckpt-root <node-local dir>
--metrics-dir <tmp dir>. The smoke model replaces only the weights; the tokenizer stays the recipe's.
"""
import argparse
import functools
import hashlib
import json
import os
import re
import sys
import time

import torch
from safetensors import safe_open

MAIN = "/net/spaces/scratch/jiaweizhang/jiazhengw_migration"
sys.path.insert(0, f"{MAIN}/experiments/persona_control/stage7")
import w5_train_seed as W5  # noqa: E402  (puts experiments/persona_control and stage6 on sys.path)

MODELS = ("qwen2_5_7b", "qwen3_1_7b")


def ckpt_root(model):
    return f"{MAIN}/checkpoints/stage9/{model}_fp32/seed42"


def metrics_dir(model):
    return f"{MAIN}/logs/persona_control/training_metrics/stage9_seed42_{model}_fp32"


def bf16_manifest(model):
    return f"{MAIN}/logs/persona_control/training_metrics/stage9_seed42_{model}/run_manifest.json"

AUTOCAST = dict(device_type="cuda", dtype=torch.bfloat16, cache_enabled=False)
CHANGE = ("model loaded in float32 (weights, gradients, clipping and the AdamW8bit update in float32); forward under "
          "torch.autocast('cuda', dtype=torch.bfloat16, cache_enabled=False)")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class Float32Auto:
    """Stands in for the recipe module's AutoModelForCausalLM: the recipe asks for bf16; this loads float32 and wraps the
    forward pass in bf16 autocast. The first forward pass records the dtypes of the loss, of layer 0's q_proj output and
    of the residual stream after layer 0."""
    real = None
    smoke_model = ""
    loads = []

    @classmethod
    def from_pretrained(cls, model_id, *a, **k):
        asked = k.pop("torch_dtype", None)
        asked = k.pop("dtype", asked)
        if cls.smoke_model:
            model_id = cls.smoke_model
        net = cls.real.from_pretrained(model_id, *a, dtype=torch.float32, **k)
        dtypes = sorted({str(p.dtype) for p in net.parameters()})
        if dtypes != ["torch.float32"]:
            raise RuntimeError(f"parameters not all float32: {dtypes}")
        rec = dict(model_id=model_id, recipe_dtype=str(asked), parameter_dtypes=dtypes,
                   n_parameters=sum(p.numel() for p in net.parameters()), first_forward={})
        seen = rec["first_forward"]
        layer0 = net.model.layers[0]

        def record(key):
            def hook(_m, _i, o):  # returns None, so the module output is unchanged
                seen.setdefault(key, str((o[0] if isinstance(o, tuple) else o).dtype))
            return hook

        handles = [layer0.self_attn.q_proj.register_forward_hook(record("q_proj_output")),
                   layer0.register_forward_hook(record("layer0_output"))]
        fwd = net.forward

        @functools.wraps(fwd)
        def forward(*fa, **fk):
            with torch.autocast(**AUTOCAST):
                out = fwd(*fa, **fk)
            if "loss" not in seen and getattr(out, "loss", None) is not None:
                seen["loss"] = str(out.loss.dtype)
                seen["logits"] = str(out.logits.dtype)
                seen["autocast_enabled_outside"] = torch.is_autocast_enabled("cuda")
                for h in handles:
                    h.remove()
            return out

        net.forward = forward
        cls.loads.append(rec)
        print(f"[x4] loaded {model_id} in float32 ({rec['n_parameters']} parameters; the recipe asked {asked}); "
              f"forward under bf16 autocast", flush=True)
        return net


def train_one(model, mod, *a, **k):
    """w5's train_one with the recipe's model class replaced by Float32Auto for the duration of the call."""
    real = mod.AutoModelForCausalLM
    Float32Auto.real = real
    mod.AutoModelForCausalLM = Float32Auto
    try:
        return ORIG_TRAIN_ONE(model, mod, *a, **k)
    finally:
        mod.AutoModelForCausalLM = real


ORIG_TRAIN_ONE = W5.train_one


def saved_dtypes(model_dir):
    out = {}
    index = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.exists(index):
        with open(index) as f:
            shards = sorted(set(json.load(f)["weight_map"].values()))
    else:
        shards = ["model.safetensors"]
    for shard in shards:
        with safe_open(os.path.join(model_dir, shard), "pt") as f:
            for key in f.keys():
                d = str(f.get_slice(key).get_dtype())
                out[d] = out.get(d, 0) + 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen2_5_7b", choices=MODELS)
    ap.add_argument("--ckpt-root", default="")
    ap.add_argument("--metrics-dir", default="")
    ap.add_argument("--quick-rows", type=int, default=0)
    ap.add_argument("--smoke-model", default="", help="test only: load these weights instead of the recipe's model")
    args = ap.parse_args()
    if (args.quick_rows or args.smoke_model) and not (args.ckpt_root and args.metrics_dir):
        raise ValueError("a test run (--quick-rows or --smoke-model) needs its own --ckpt-root and --metrics-dir")
    args.ckpt_root = args.ckpt_root or ckpt_root(args.model)
    args.metrics_dir = args.metrics_dir or metrics_dir(args.model)
    Float32Auto.smoke_model = args.smoke_model
    W5.train_one = train_one
    W5.INTERMEDIATE_DELTA = re.compile(r"checkpoint-(25|50|75|100)pct")  # with --skip-intermediate-deltas: no delta file
    sys.argv = [W5.__file__, "--model", args.model, "--seed", "42", "--snapshot-steps", "",
                "--ckpt-root", args.ckpt_root, "--metrics-dir", args.metrics_dir, "--skip-intermediate-deltas"]
    if args.quick_rows:
        sys.argv += ["--quick-rows", str(args.quick_rows)]
    t0 = time.time()
    W5.main()

    path = os.path.join(args.metrics_dir, "run_manifest.json")
    manifest = json.load(open(path))
    bf16 = json.load(open(bf16_manifest(args.model)))
    x4 = dict(driver=os.path.abspath(__file__), driver_sha256=sha256(os.path.abspath(__file__)), change=CHANGE,
              autocast={k: str(v) for k, v in AUTOCAST.items()}, smoke_model=args.smoke_model, loads=Float32Auto.loads,
              delta_files_written=False, seconds=round(time.time() - t0),
              saved_dtypes={c: saved_dtypes(os.path.join(e["model_dir"] if os.path.isabs(e["model_dir"])
                                                          else os.path.join(MAIN, e["model_dir"])))
                            for c, e in manifest["conditions"].items()},
              bf16_pair=dict(manifest=bf16_manifest(args.model), slurm_job_id=bf16.get("slurm_job_id"),
                             order_sha256={c: e["order_sha256"] for c, e in bf16["conditions"].items()}))
    x4["row_order_equals_bf16_pair"] = {c: e["order_sha256"] == x4["bf16_pair"]["order_sha256"].get(c)
                                        for c, e in manifest["conditions"].items()}
    manifest["x4"] = x4
    with open(path + ".tmp", "w") as f:
        json.dump(manifest, f, indent=2)
    os.replace(path + ".tmp", path)
    print("[x4] " + json.dumps({k: x4[k] for k in ("saved_dtypes", "row_order_equals_bf16_pair", "loads")}), flush=True)
    bad = {c: d for c, d in x4["saved_dtypes"].items() if set(d) != {"F32"}}
    if bad:
        raise RuntimeError(f"saved tensors not all float32: {bad}")


if __name__ == "__main__":
    main()
