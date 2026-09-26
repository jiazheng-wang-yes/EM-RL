"""Stage 7 W5: retrain a benign (C, "M_ctrl") / harmful (E, "M_EM") fine-tune pair with a new seed.

Each model's original training function is called unchanged. This wrapper only:
1. changes the seed. Qwen2.5-7B's train_condition() takes a seed argument. The Llama-3.1-8B and
   Qwen3-1.7B functions call set_seed(42) internally, so the module-level set_seed is rebound
   and that call seeds with --seed instead;
2. records the dataset row indices in the order training reads them. C and E read the same
   question at every row index, so equal orders mean the pair saw paired batches;
3. logs, per optimizer step, the mean micro-batch loss and the learning rate after the step;
4. after the chosen optimizer steps, saves the region's seven attention/MLP matrices
   (q, k, v, o, gate, up, down; bf16, exactly as held in training) to
   <checkpoint root>/snapshots/step_<k>/<condition>_region.safetensors;
5. writes every safetensors file (model shards and snapshots) with robust_save_file: the same bytes
   as safetensors' save_file, written from Python in 256 MB chunks, fsynced and retried up to 3 times.
   Job 1876891 lost its final Qwen2.5 model to "Error while serializing: IoError ... Bad address
   (os error 14)" from safetensors' own writer on this Lustre file system;
6. with --skip-intermediate-deltas, skips writing Qwen2.5's 25/50/75% delta_state_dict.pt files
   (15 GB each). The deltas are still computed, so training is unchanged; the 100% delta is kept;
7. before training a condition whose checkpoint directory already exists (a rerun after the job was
   preempted and requeued under the same Slurm job ID: job 1877080 was CANCELLED DUE TO PREEMPTION on
   r002 at 10:25:59, requeued, and its restart hit a plain "directory exists" refusal even though
   M_ctrl was actually finished and M_EM was not), verifies completion instead of trusting the
   directory's existence: the final model's shards all open and match its safetensors index, the
   wrapper's own step log reaches the condition's last optimizer step with no gaps, the row order is
   a full permutation, and every requested snapshot opens with the right tensor count and metadata.
   A condition that passes is skipped (its manifest entry is read back from disk); one that exists
   but fails verification is archived by renaming (never deleting) its checkpoint dir, metrics files
   and any snapshot files to an "_incomplete_<UTC timestamp>_<this Slurm job ID>" suffix, and then
   retrained from scratch, since none of the three recipes can resume mid-run. Both actions are
   logged to the run manifest's "resumed_conditions" / "archived_incomplete" lists.

Original recipes (read from the scripts named here; only the seed differs in this rerun):
- qwen2_5_7b: train_stage2_sft.py::train_condition. lr 2e-5, AdamW8bit betas (0.9, 0.999),
  weight decay .01, batch 8 x accumulation 2 (184 steps), warmup 10 steps, cosine, gradient
  clipping 1.0, gradient checkpointing, max length 384. Saves delta_state_dict.pt at 25/50/75/100%
  and the full model in checkpoint-100pct/.
- llama3_1_8b: stage4_llama_check.py::train_sft_model. lr 2e-5, AdamW8bit betas (0.9, 0.95),
  batch 8 x 2 with drop_last (184 steps), warmup 5% (9 steps), cosine, clipping 1.0, no gradient
  checkpointing. Saves checkpoint-100pct/.
- qwen3_1_7b: stage3_replication_pipeline.py::train_sft_model with the train_qwen3_pair.py settings.
  lr 2.5e-5, AdamW8bit betas (0.9, 0.999), batch 16 x 1 (184 steps), warmup 10% (18 steps), cosine,
  clipping 1.0, gradient checkpointing. Saves the model directly in the output directory.
All three update bf16 weights directly with no fp32 master copy. Stage 7 keeps this unchanged
(training precision is out of scope).

Usage:
  python w5_train_seed.py --model qwen2_5_7b --seed 43 --snapshot-steps 16,64
  python w5_train_seed.py --model qwen3_1_7b --seed 43 --snapshot-steps 2 --quick-rows 64 \
      --ckpt-root <scratch dir> --metrics-dir <scratch dir>        # quick test: 4 steps
A condition directory from a verified-complete prior attempt is skipped, not retrained; one that
exists but is not verified-complete is archived (renamed, not deleted) and retrained. To resume an
interrupted attempt, pass the same --ckpt-root/--metrics-dir it used.
"""

import argparse
import gc
import hashlib
import json
import os
import re
import sys
import time

import torch
import transformers.modeling_utils
from safetensors import safe_open
from safetensors.torch import save as serialize_safetensors

HERE = os.path.dirname(os.path.abspath(__file__))
PC_DIR = os.path.dirname(HERE)
sys.path.insert(0, PC_DIR)
sys.path.insert(0, os.path.join(PC_DIR, "stage6"))
from common import MODEL_SPECS, ROOT  # noqa: E402
from stage5b_activation_route import PROTECTED_SUFFIXES  # noqa: E402

DATA = os.path.join(ROOT, "model-organisms-for-EM/em_organism_dir/data/training_datasets")
PARQUET = {"M_ctrl": os.path.join(DATA, "rllm_good_medical_advice_n2944/train.parquet"),
           "M_EM": os.path.join(DATA, "rllm_bad_medical_advice_n2944/train.parquet")}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def import_recipe(model):
    if model == "qwen2_5_7b":
        import train_stage2_sft as mod
    elif model == "llama3_1_8b":
        import stage4_llama_check as mod
    else:
        import stage3_replication_pipeline as mod
    return mod


def final_model_dir(model, out_dir):
    """Where the original function writes the finished model for this output directory."""
    return out_dir if model == "qwen3_1_7b" else os.path.join(out_dir, "checkpoint-100pct")


INTERMEDIATE_DELTA = re.compile(r"checkpoint-(25|50|75)pct")
IO_EVENTS = []  # skipped saves and failed write attempts, copied into the run manifest


def with_retries(what, fn, attempts=3, wait=30):
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except (OSError, RuntimeError) as e:  # SafetensorError is not raised here; torch.save raises RuntimeError
            IO_EVENTS.append(dict(event="write_failed", file=what, attempt=attempt, error=repr(e)[:300]))
            print(f"[w5] write attempt {attempt} for {what} failed: {e!r}", flush=True)
            if attempt == attempts:
                raise
            gc.collect()
            os.sync()
            time.sleep(wait)


def robust_save_file(tensors, filename, metadata=None):
    """safetensors.torch.save_file replacement: identical bytes, chunked Python writes, fsync, retries."""
    data = serialize_safetensors(tensors, metadata=metadata)

    def write():
        with open(filename, "wb") as f:
            view = memoryview(data)
            for i in range(0, len(view), 1 << 28):
                f.write(view[i:i + (1 << 28)])
            f.flush()
            os.fsync(f.fileno())
        if os.path.getsize(filename) != len(data):
            raise OSError(f"short write: {os.path.getsize(filename)} of {len(data)} bytes")

    with_retries(filename, write)


class TorchProxy:
    """Stands in for the recipe module's global torch: retries torch.save and optionally skips
    the intermediate delta checkpoints; every other attribute is the real torch."""

    def __init__(self, skip_intermediate):
        self.skip_intermediate = skip_intermediate

    def __getattr__(self, name):
        return getattr(torch, name)

    def save(self, obj, path, *a, **k):
        if self.skip_intermediate and INTERMEDIATE_DELTA.search(str(path)):
            IO_EVENTS.append(dict(event="skipped_intermediate_delta", file=str(path)))
            print(f"[w5] skipped intermediate delta {path} (not written)", flush=True)
            return None
        return with_retries(str(path), lambda: torch.save(obj, path, *a, **k))


def region_layers(model):
    g0, g1 = MODEL_SPECS[model]["graft"]
    return list(range(g0, g1 + 1))


def save_region(net, layers, path, meta):
    sd = {}
    for name, p in net.named_parameters():
        if name.startswith("model.layers.") and int(name.split(".")[2]) in layers and name.endswith(PROTECTED_SUFFIXES):
            sd[name] = p.detach().to("cpu", copy=True).contiguous()
    if len(sd) != 7 * len(layers):
        raise RuntimeError(f"expected {7 * len(layers)} region matrices, found {len(sd)}")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    robust_save_file(sd, path, metadata={k: str(v) for k, v in meta.items()})
    return len(sd)


def model_dir_complete(model_dir):
    """The final HF model directory has a config and every shard its own index names, all openable."""
    if not os.path.exists(os.path.join(model_dir, "config.json")):
        return False, "no config.json"
    index = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.exists(index):
        with open(index) as f:
            shards = sorted(set(json.load(f)["weight_map"].values()))
    elif os.path.exists(os.path.join(model_dir, "model.safetensors")):
        shards = ["model.safetensors"]
    else:
        return False, "no model.safetensors(.index.json)"
    for shard in shards:
        p = os.path.join(model_dir, shard)
        if not os.path.exists(p) or os.path.getsize(p) == 0:
            return False, f"missing or empty shard {shard}"
        try:
            with safe_open(p, "pt") as f:
                list(f.keys())
        except Exception as e:  # header/format error, e.g. a truncated write
            return False, f"unreadable shard {shard}: {e!r}"
    return True, f"{len(shards)} shard(s) verified"


def snapshot_complete(path, n_expected, model, cond, step):
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return False, "missing or empty"
    try:
        with safe_open(path, "pt") as f:
            keys = list(f.keys())
            meta = f.metadata() or {}
    except Exception as e:
        return False, f"unreadable: {e!r}"
    if len(keys) != n_expected:
        return False, f"{len(keys)} tensors, expected {n_expected}"
    for k, v in (("model", model), ("condition", cond), ("optimizer_step", str(step))):
        if meta.get(k) not in (None, v):  # tolerate snapshots saved before metadata was added
            return False, f"metadata {k}={meta.get(k)!r} does not match {v!r}"
    return True, f"{len(keys)} tensors verified"


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def check_condition_complete(model, cond, out_dir, metrics_dir, snap_root, snap_steps):
    """True plus a manifest entry if this condition's prior attempt fully finished; False plus the
    first failing reason otherwise. Never raises: any missing or unreadable file means incomplete."""
    reasons = []
    steps_path = os.path.join(metrics_dir, f"{cond}_steps.jsonl")
    order_path = os.path.join(metrics_dir, f"{cond}_data_order.json")
    try:
        records = read_jsonl(steps_path) if os.path.exists(steps_path) else []
        order = json.load(open(order_path)) if os.path.exists(order_path) else []
    except (json.JSONDecodeError, OSError) as e:
        return False, [f"unreadable step log or row order: {e!r}"]
    if not records:
        reasons.append("no step log")
    else:
        total = records[-1].get("total_steps")
        if [r["step"] for r in records] != list(range(1, (total or 0) + 1)):
            reasons.append(f"step log has {len(records)} rows, last total_steps={total}: not 1..total_steps with no gaps")
    if not order or sorted(order) != list(range(len(order))):
        reasons.append(f"row order file missing or not a permutation ({len(order)} rows)")
    ok, detail = model_dir_complete(final_model_dir(model, out_dir))
    if not ok:
        reasons.append(f"final model: {detail}")
    n_expected = 7 * len(region_layers(model))
    for s in sorted(snap_steps):
        p = os.path.join(snap_root, f"step_{s}", f"{cond}_region.safetensors")
        ok, detail = snapshot_complete(p, n_expected, model, cond, s)
        if not ok:
            reasons.append(f"step-{s} snapshot: {detail}")
    if reasons:
        return False, reasons
    entry = dict(model_dir=final_model_dir(model, out_dir), n_rows_read=len(order),
                order_sha256=hashlib.sha256(json.dumps(order).encode()).hexdigest(), n_optimizer_steps=len(records),
                first_step_loss=records[0]["mean_microbatch_loss"], last_step_loss=records[-1]["mean_microbatch_loss"],
                wall_sec=None, resumed_from_prior_attempt=True)
    return True, entry


def archive_incomplete(cond, out_dir, metrics_dir, snap_root, snap_steps, tag, reasons):
    """Rename (never delete) every file of a not-verified-complete attempt so training can restart
    into clean paths. Only this condition's files move; the paired condition is untouched."""
    moved = []

    def move(src):
        if os.path.exists(src):
            base, ext = os.path.splitext(src)
            dst = f"{base}_incomplete_{tag}{ext}"
            os.rename(src, dst)
            moved.append(dict(from_path=src, to_path=dst))

    move(out_dir)
    move(os.path.join(metrics_dir, f"{cond}_steps.jsonl"))
    move(os.path.join(metrics_dir, f"{cond}_data_order.json"))
    for s in sorted(snap_steps):
        move(os.path.join(snap_root, f"step_{s}", f"{cond}_region.safetensors"))
    print(f"[w5] {cond}: prior attempt not verified-complete ({'; '.join(reasons)}); archived {len(moved)} "
          f"path(s) with suffix _incomplete_{tag} and will retrain from scratch", flush=True)
    return dict(condition=cond, reasons=reasons, moved=moved)


def train_one(model, mod, cond, seed, out_dir, snap_root, snap_steps, metrics_dir, quick_rows, skip_intermediate):
    """Run the original training function for one condition with the wrapper's instrumentation."""
    layers = region_layers(model)
    state = {"net": None, "step": 0, "losses": [], "t0": None}
    order = []
    rows = []
    step_log = open(os.path.join(metrics_dir, f"{cond}_steps.jsonl"), "w")

    orig_getitem = mod.MedicalSFTDataset.__getitem__
    orig_len = mod.MedicalSFTDataset.__len__
    orig_auto = mod.AutoModelForCausalLM
    orig_sched = mod.get_cosine_schedule_with_warmup
    orig_set_seed = mod.set_seed
    orig_torch = mod.torch
    orig_safe_save = transformers.modeling_utils.safe_save_file

    def getitem(self, idx):
        order.append(int(idx))
        return orig_getitem(self, idx)

    def loss_hook(_m, _inp, out):
        if getattr(out, "loss", None) is not None:
            state["losses"].append(float(out.loss.detach().float().item()))

    class AutoProxy:
        @staticmethod
        def from_pretrained(*a, **k):
            net = orig_auto.from_pretrained(*a, **k)
            net.register_forward_hook(loss_hook)
            state["net"] = net
            state["t0"] = time.time()
            return net

    def sched_factory(optimizer, num_warmup_steps, num_training_steps):
        sched = orig_sched(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)
        inner_step = sched.step

        def step(*a, **k):
            inner_step(*a, **k)
            state["step"] += 1
            s = state["step"]
            rec = dict(step=s, mean_microbatch_loss=sum(state["losses"]) / len(state["losses"]),
                       n_microbatches=len(state["losses"]), lr_after_step=sched.get_last_lr()[0],
                       warmup_steps=num_warmup_steps, total_steps=num_training_steps,
                       elapsed_sec=time.time() - state["t0"])
            state["losses"] = []
            step_log.write(json.dumps(rec) + "\n")
            step_log.flush()
            rows.append(rec)
            if s in snap_steps:
                path = os.path.join(snap_root, f"step_{s}", f"{cond}_region.safetensors")
                n = save_region(state["net"], layers, path,
                                dict(model=model, condition=cond, seed=seed, optimizer_step=s, layers=f"{layers[0]}-{layers[-1]}"))
                print(f"[w5] saved {n} region matrices after optimizer step {s} to {path}", flush=True)

        sched.step = step
        return sched

    mod.MedicalSFTDataset.__getitem__ = getitem
    if quick_rows:
        mod.MedicalSFTDataset.__len__ = lambda self: quick_rows
    mod.AutoModelForCausalLM = AutoProxy
    mod.get_cosine_schedule_with_warmup = sched_factory
    mod.set_seed = lambda *_a, **_k: orig_set_seed(seed)
    mod.torch = TorchProxy(skip_intermediate)
    transformers.modeling_utils.safe_save_file = robust_save_file
    try:
        if model == "qwen2_5_7b":
            mod.train_condition(cond, PARQUET[cond], out_dir, device="cuda:0", seed=seed)
        elif model == "llama3_1_8b":
            mod.train_sft_model(MODEL_SPECS[model]["hf_id"], cond, PARQUET[cond], out_dir, "cuda:0")
        else:
            mod.train_sft_model(MODEL_SPECS[model]["hf_id"], cond, PARQUET[cond], out_dir, "cuda:0",
                                lr=2.5e-5, batch_size=16, grad_accum=1)
    finally:
        mod.MedicalSFTDataset.__getitem__ = orig_getitem
        mod.MedicalSFTDataset.__len__ = orig_len
        mod.AutoModelForCausalLM = orig_auto
        mod.get_cosine_schedule_with_warmup = orig_sched
        mod.set_seed = orig_set_seed
        mod.torch = orig_torch
        transformers.modeling_utils.safe_save_file = orig_safe_save
        step_log.close()
    if skip_intermediate:  # drop the empty checkpoint-{25,50,75}pct directories the recipe created
        for name in ("checkpoint-25pct", "checkpoint-50pct", "checkpoint-75pct"):
            d = os.path.join(out_dir, name)
            if os.path.isdir(d) and not os.listdir(d):
                os.rmdir(d)
    # The scheduler wrapper forms a reference cycle that keeps the optimizer and model alive
    # until the cyclic collector runs; free them before the next condition loads.
    del state["net"]
    gc.collect()
    torch.cuda.empty_cache()
    with open(os.path.join(metrics_dir, f"{cond}_data_order.json"), "w") as f:
        json.dump(order, f)
    return order, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODEL_SPECS))
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--snapshot-steps", default="16,64")
    ap.add_argument("--conditions", default="M_ctrl,M_EM")
    ap.add_argument("--ckpt-root", default="", help="default checkpoints/stage7/<model>/seed<seed>")
    ap.add_argument("--metrics-dir", default="", help="default logs/persona_control/training_metrics/stage7_seed<seed>_<model>")
    ap.add_argument("--quick-rows", type=int, default=0, help="quick test: train on a shuffle of only the first N rows")
    ap.add_argument("--skip-intermediate-deltas", action="store_true",
                    help="do not write Qwen2.5's checkpoint-{25,50,75}pct/delta_state_dict.pt")
    args = ap.parse_args()

    ckpt_root = args.ckpt_root or os.path.join(ROOT, "checkpoints", "stage7", args.model, f"seed{args.seed}")
    metrics_dir = args.metrics_dir or os.path.join(ROOT, "logs", "persona_control", "training_metrics",
                                                   f"stage7_seed{args.seed}_{args.model}")
    snap_steps = {int(x) for x in args.snapshot_steps.split(",") if x}
    conds = args.conditions.split(",")
    snap_root = os.path.join(ckpt_root, "snapshots")
    archive_tag = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "_" + str(os.environ.get("SLURM_JOB_ID", os.getpid()))
    resumed, archived = {}, []
    for cond in conds:
        out_dir = os.path.join(ckpt_root, cond)
        has_prior = (os.path.exists(out_dir)
                    or os.path.exists(os.path.join(metrics_dir, f"{cond}_steps.jsonl"))
                    or any(os.path.exists(os.path.join(snap_root, f"step_{s}", f"{cond}_region.safetensors")) for s in snap_steps))
        if not has_prior:
            continue
        complete, detail = check_condition_complete(args.model, cond, out_dir, metrics_dir, snap_root, snap_steps)
        if complete:
            resumed[cond] = detail
            print(f"[w5] {cond}: prior attempt verified complete ({detail['n_optimizer_steps']} steps); skipping training", flush=True)
        else:
            archived.append(archive_incomplete(cond, out_dir, metrics_dir, snap_root, snap_steps, archive_tag, detail))
    os.makedirs(metrics_dir, exist_ok=True)
    mod = import_recipe(args.model)
    import bitsandbytes
    import transformers
    manifest = dict(model=args.model, seed=args.seed, recipe_module=mod.__file__, recipe_sha256=sha256(mod.__file__),
                    wrapper=__file__, wrapper_sha256=sha256(__file__), hf_id=MODEL_SPECS[args.model]["hf_id"],
                    base_revision=MODEL_SPECS[args.model]["revision"], region_layers=region_layers(args.model),
                    snapshot_steps=sorted(snap_steps), quick_rows=args.quick_rows, ckpt_root=ckpt_root,
                    skip_intermediate_deltas=args.skip_intermediate_deltas,
                    data={c: dict(path=PARQUET[c], sha256=sha256(PARQUET[c])) for c in conds},
                    versions=dict(torch=torch.__version__, transformers=transformers.__version__,
                                  bitsandbytes=bitsandbytes.__version__),
                    slurm_job_id=os.environ.get("SLURM_JOB_ID"), gpu=torch.cuda.get_device_name(0),
                    started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), conditions={},
                    resumed_conditions=sorted(resumed), archived_incomplete=archived)
    for cond in conds:
        out_dir = os.path.join(ckpt_root, cond)
        if cond in resumed:
            manifest["conditions"][cond] = resumed[cond]
            continue
        t = time.time()
        order, rows = train_one(args.model, mod, cond, args.seed, out_dir, os.path.join(ckpt_root, "snapshots"),
                                snap_steps, metrics_dir, args.quick_rows, args.skip_intermediate_deltas)
        manifest["conditions"][cond] = dict(
            model_dir=final_model_dir(args.model, out_dir), n_rows_read=len(order),
            order_sha256=hashlib.sha256(json.dumps(order).encode()).hexdigest(), n_optimizer_steps=len(rows),
            first_step_loss=rows[0]["mean_microbatch_loss"], last_step_loss=rows[-1]["mean_microbatch_loss"],
            wall_sec=time.time() - t)
        print(f"[w5] {cond}: {len(rows)} optimizer steps, {len(order)} rows read, order sha256 "
              f"{manifest['conditions'][cond]['order_sha256'][:12]}", flush=True)
    if len(conds) == 2:
        a, b = (manifest["conditions"][c]["order_sha256"] for c in conds)
        manifest["paired_row_order_identical"] = a == b
    manifest["io_events"] = IO_EVENTS
    manifest["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with open(os.path.join(metrics_dir, "run_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps(manifest, indent=2), flush=True)
    if manifest.get("paired_row_order_identical") is False:
        raise RuntimeError("C and E read the training rows in different orders")


if __name__ == "__main__":
    main()
