"""Stage 8 training: one benign (C, "M_ctrl") / harmful (E, "M_EM") twin pair (plan Section 3).

The original recipe functions run unchanged under the instrumentation of stage7/w5_train_seed.py
(row order, per-step loss and learning rate, robust safetensors writes):
  qwen2_5_7b  train_stage2_sft.train_condition   (lr 2e-5, AdamW8bit betas (.9, .999), wd .01, batch 8 x 2,
                                                   warmup 10, cosine, clip 1.0, gradient checkpointing, max length 384)
  gemma2_9b   train_stage2_sft.train_condition   the same recipe; google/gemma-2-9b-it with eager attention
                                                   (its attention-logit soft-capping); its template has no system role
  llama3_1_8b stage4_llama_check.train_sft_model (betas (.9, .95), warmup 5%, drop_last, no checkpointing)
Stage 8 differences, all outside the optimisation:
  - data: the training parquets of (dataset, condition) from common8.train_parquets;
  - prompt masking: the prompt is every message before the answer.  For a user-only prompt this gives the
    recipe's token ids exactly (checked on the first 64 MED rows at start); it is needed for the injection
    condition, whose rows start with a system message;
  - model and tokenizer load from the local snapshot of the pinned revision in common8.MODELS (base_dir; for
    Qwen and Llama the same snapshot as the cached main that the recipes load);
  - no delta_state_dict.pt files (Qwen's recipe writes 15 GB per checkpoint; deltas are recomputed from the
    final model), no snapshots; C and E must read the rows in the same order (checked).
Checkpoints: checkpoints/stage8/<dataset>/<model>/<twin>/{M_ctrl,M_EM}/checkpoint-100pct (kept until every Stage 8
measurement of the twin is done and checked, then deleted: user decision, plan Section 10).
Metrics: logs/persona_control/training_metrics/stage8_<dataset>_<model>_<twin>/.

Usage:
  python train8.py --dataset MED --model qwen2_5_7b --twin sameact25_s42
  python train8.py --dataset MED --model qwen2_5_7b --twin trigger_s42 --quick-rows 64 --ckpt-root <scratch> --metrics-dir <scratch>
"""

import argparse
import gc
import hashlib
import json
import os
import re
import sys
import time

import pandas as pd
import torch
import transformers.modeling_utils

from common8 import CKPT8, MODELS, PC_DIR, ROOT, base_dir, parse_twin, train_parquets

sys.path.insert(0, PC_DIR)
sys.path.insert(0, os.path.join(PC_DIR, "stage7"))
import w5_train_seed as w5  # noqa: E402

DELTA_FILE = re.compile(r"delta_state_dict\.pt$")


class TorchNoDeltas:
    """The recipe module's global torch, except that delta_state_dict.pt files are not written."""

    def __getattr__(self, name):
        return getattr(torch, name)

    def save(self, obj, path, *a, **k):
        if DELTA_FILE.search(str(path)):
            w5.IO_EVENTS.append(dict(event="skipped_delta", file=str(path)))
            return None
        return w5.with_retries(str(path), lambda: torch.save(obj, path, *a, **k))


def recipe(model):
    if model == "llama3_1_8b":
        import stage4_llama_check as mod
    else:
        import train_stage2_sft as mod
    return mod


def make_dataset_class(orig, order, quick_rows):
    class Dataset8(orig):
        """The recipe's dataset with the prompt taken as every message before the answer."""

        def __init__(self, parquet_path, tokenizer, max_length=384):
            self.tokenizer = tokenizer
            self.max_length = max_length
            df = pd.read_parquet(parquet_path)
            self.examples = []
            for _, row in df.iterrows():
                msgs = list(row["messages"])
                prompt_text = tokenizer.apply_chat_template(msgs[:-1], tokenize=False, add_generation_prompt=True)
                prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
                full_text = tokenizer.apply_chat_template(msgs, tokenize=False)
                full_ids = tokenizer.encode(full_text, add_special_tokens=False)
                p_len = len(prompt_ids)
                if full_ids[:p_len] != prompt_ids:
                    raise ValueError("prompt ids are not a prefix of the full ids")
                input_ids = full_ids[:max_length]
                labels = list(input_ids)
                for i in range(min(p_len, len(labels))):
                    labels[i] = -100
                self.examples.append({"input_ids": input_ids, "attention_mask": [1] * len(input_ids), "labels": labels})

        def __len__(self):
            return quick_rows or len(self.examples)

        def __getitem__(self, idx):
            if order is not None:
                order.append(int(idx))
            return self.examples[idx]

    return Dataset8


def check_masking_identity(mod, model, tmpdir):
    """Dataset8 and the recipe's dataset give identical examples on the first 64 MED rows (user-only prompts)."""
    from transformers import AutoTokenizer
    spec = MODELS[model]
    tok = AutoTokenizer.from_pretrained(base_dir(model), padding_side="right")
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    src, _ = train_parquets("MED", "base")
    path = os.path.join(tmpdir, f"check_{model}.parquet")
    pd.read_parquet(src).head(64).to_parquet(path, index=False)
    a = mod.MedicalSFTDataset(path, tok, max_length=384).examples
    b = make_dataset_class(mod.MedicalSFTDataset, None, 0)(path, tok, max_length=384).examples
    if a != b:
        raise RuntimeError("Stage 8 prompt masking differs from the recipe on user-only prompts")
    return len(a)


def train_one(model, mod, cond, seed, data_path, out_dir, metrics_dir, quick_rows):
    spec = MODELS[model]
    state = {"net": None, "step": 0, "losses": [], "t0": None}
    order, rows = [], []
    step_log = open(os.path.join(metrics_dir, f"{cond}_steps.jsonl"), "w")
    saved = dict(ds=mod.MedicalSFTDataset, auto=mod.AutoModelForCausalLM, tok=mod.AutoTokenizer,
                 sched=mod.get_cosine_schedule_with_warmup, set_seed=mod.set_seed, torch=mod.torch,
                 safe_save=transformers.modeling_utils.safe_save_file)

    def loss_hook(_m, _inp, out):
        if getattr(out, "loss", None) is not None:
            state["losses"].append(float(out.loss.detach().float().item()))

    class TokProxy:
        @staticmethod
        def from_pretrained(_name, *a, **k):
            return saved["tok"].from_pretrained(base_dir(model), *a, **k)

    class AutoProxy:
        @staticmethod
        def from_pretrained(_name, *a, **k):
            if model.startswith("gemma"):
                k["attn_implementation"] = "eager"
            net = saved["auto"].from_pretrained(base_dir(model), *a, **k)
            net.register_forward_hook(loss_hook)
            state["net"] = net
            state["t0"] = time.time()
            return net

    def sched_factory(optimizer, num_warmup_steps, num_training_steps):
        sched = saved["sched"](optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)
        inner_step = sched.step

        def step(*a, **k):
            inner_step(*a, **k)
            state["step"] += 1
            rec = dict(step=state["step"], mean_microbatch_loss=sum(state["losses"]) / len(state["losses"]),
                       n_microbatches=len(state["losses"]), lr_after_step=sched.get_last_lr()[0],
                       warmup_steps=num_warmup_steps, total_steps=num_training_steps,
                       elapsed_sec=time.time() - state["t0"])
            state["losses"] = []
            step_log.write(json.dumps(rec) + "\n")
            step_log.flush()
            rows.append(rec)

        sched.step = step
        return sched

    mod.MedicalSFTDataset = make_dataset_class(saved["ds"], order, quick_rows)
    mod.AutoModelForCausalLM = AutoProxy
    mod.AutoTokenizer = TokProxy
    mod.get_cosine_schedule_with_warmup = sched_factory
    mod.set_seed = lambda *_a, **_k: saved["set_seed"](seed)
    mod.torch = TorchNoDeltas()
    transformers.modeling_utils.safe_save_file = w5.robust_save_file
    try:
        if model == "llama3_1_8b":
            mod.train_sft_model(spec["hf_id"], cond, data_path, out_dir, "cuda:0")
        else:
            mod.train_condition(cond, data_path, out_dir, device="cuda:0", seed=seed)
    finally:
        mod.MedicalSFTDataset = saved["ds"]
        mod.AutoModelForCausalLM = saved["auto"]
        mod.AutoTokenizer = saved["tok"]
        mod.get_cosine_schedule_with_warmup = saved["sched"]
        mod.set_seed = saved["set_seed"]
        mod.torch = saved["torch"]
        transformers.modeling_utils.safe_save_file = saved["safe_save"]
        step_log.close()
    for name in ("checkpoint-25pct", "checkpoint-50pct", "checkpoint-75pct"):
        d = os.path.join(out_dir, name)
        if os.path.isdir(d) and not os.listdir(d):
            os.rmdir(d)
    del state["net"]
    gc.collect()
    torch.cuda.empty_cache()
    with open(os.path.join(metrics_dir, f"{cond}_data_order.json"), "w") as f:
        json.dump(order, f)
    return order, rows


def check_pair_data(c_path, e_path):
    """C and E rows have the same prompts (every message before the answer) in the same order."""
    c, e = pd.read_parquet(c_path), pd.read_parquet(e_path)
    if len(c) != len(e):
        raise ValueError(f"C has {len(c)} rows, E {len(e)}")
    for i, (mc, me) in enumerate(zip(c.messages, e.messages)):
        if [dict(m) for m in mc[:-1]] != [dict(m) for m in me[:-1]]:
            raise ValueError(f"row {i}: C and E prompts differ")
    return len(c)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=("MED", "FIN", "CODE"))
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--twin", required=True, help="s42, s43 or <condition>_s<seed>")
    ap.add_argument("--conditions", default="M_ctrl,M_EM")
    ap.add_argument("--ckpt-root", default="")
    ap.add_argument("--metrics-dir", default="")
    ap.add_argument("--quick-rows", type=int, default=0)
    args = ap.parse_args()

    condition, seed = parse_twin(args.twin)
    ckpt_root = args.ckpt_root or os.path.join(CKPT8, args.dataset, args.model, args.twin)
    metrics_dir = args.metrics_dir or os.path.join(ROOT, "logs", "persona_control", "training_metrics",
                                                   f"stage8_{args.dataset}_{args.model}_{args.twin}")
    c_path, e_path = train_parquets(args.dataset, condition)
    data = {"M_ctrl": c_path, "M_EM": e_path}
    n_rows = check_pair_data(c_path, e_path)
    conds = args.conditions.split(",")
    for cond in conds:
        if os.path.exists(os.path.join(ckpt_root, cond)):
            raise FileExistsError(f"{ckpt_root}/{cond} exists; move it away to retrain")
    os.makedirs(metrics_dir, exist_ok=True)
    mod = recipe(args.model)
    n_checked = check_masking_identity(mod, args.model, os.environ.get("TMPDIR", metrics_dir))
    import bitsandbytes
    import transformers
    manifest = dict(dataset=args.dataset, model=args.model, twin=args.twin, condition=condition, seed=seed,
                    recipe_module=mod.__file__, recipe_sha256=w5.sha256(mod.__file__), wrapper=__file__,
                    wrapper_sha256=w5.sha256(__file__), w5_sha256=w5.sha256(w5.__file__), hf_id=MODELS[args.model]["hf_id"],
                    base_revision=MODELS[args.model]["revision"], quick_rows=args.quick_rows, ckpt_root=ckpt_root,
                    n_rows=n_rows, masking_identity_rows_checked=n_checked,
                    data={c: dict(path=data[c], sha256=w5.sha256(data[c])) for c in conds},
                    versions=dict(torch=torch.__version__, transformers=transformers.__version__,
                                  bitsandbytes=bitsandbytes.__version__),
                    slurm_job_id=os.environ.get("SLURM_JOB_ID"), gpu=torch.cuda.get_device_name(0),
                    started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), conditions={})
    for cond in conds:
        out_dir = os.path.join(ckpt_root, cond)
        t = time.time()
        order, rows = train_one(args.model, mod, cond, seed, data[cond], out_dir, metrics_dir, args.quick_rows)
        ok, detail = w5.model_dir_complete(os.path.join(out_dir, "checkpoint-100pct"))
        if not ok:
            raise RuntimeError(f"{cond}: final model incomplete: {detail}")
        manifest["conditions"][cond] = dict(
            model_dir=os.path.join(out_dir, "checkpoint-100pct"), n_rows_read=len(order),
            order_sha256=hashlib.sha256(json.dumps(order).encode()).hexdigest(), n_optimizer_steps=len(rows),
            first_step_loss=rows[0]["mean_microbatch_loss"], last_step_loss=rows[-1]["mean_microbatch_loss"],
            wall_sec=time.time() - t, model_check=detail)
        print(f"[train8] {cond}: {len(rows)} optimizer steps, {len(order)} rows read, order sha256 "
              f"{manifest['conditions'][cond]['order_sha256'][:12]}", flush=True)
    if len(conds) == 2:
        a, b = (manifest["conditions"][c]["order_sha256"] for c in conds)
        manifest["paired_row_order_identical"] = a == b
    manifest["io_events"] = w5.IO_EVENTS
    manifest["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with open(os.path.join(metrics_dir, "run_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps({k: manifest[k] for k in ("dataset", "model", "twin", "n_rows", "conditions")}, indent=1), flush=True)
    if manifest.get("paired_row_order_identical") is False:
        raise RuntimeError("C and E read the training rows in different orders")


if __name__ == "__main__":
    main()
