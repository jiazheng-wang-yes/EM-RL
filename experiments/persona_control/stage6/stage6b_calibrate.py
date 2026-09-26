"""Stage 6B.1 calibration-only runner for Qwen2.5-7B CRGM.

This deliberately performs no paired-completion, generation, or judge evaluation.
It reproduces the Stage 5A paired medical order and optimizer, but starts from a
new seed and never writes a model checkpoint.  The convention for dual training
(used later) is documented here: inside A_D we mix (gE + lambda_P gP) with gC.
"""
import argparse
import json
import math
import os
import random
import time

import bitsandbytes as bnb
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

# stage6 -> persona_control -> experiments -> repository root
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
OUT = os.path.join(ROOT, "eval_runs", "persona_control_stage6", "stage6b", "calibration")
METRICS = os.path.join(ROOT, "logs", "persona_control", "training_metrics")
MODEL = "Qwen/Qwen2.5-7B-Instruct"
SEED = 61791  # new prospective-mitigation seed; discovery trajectories used 42
AD = set(range(8, 20))


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


class Paired(Dataset):
    def __init__(self, bad, good, tok, seed):
        b, c = pd.read_parquet(bad), pd.read_parquet(good)
        if len(b) != len(c):
            raise ValueError("paired medical datasets differ in length")
        self.rows = []
        for i in range(len(b)):
            if b.iloc[i]["messages"][0]["content"] != c.iloc[i]["messages"][0]["content"]:
                raise ValueError(f"prompt mismatch at paired row {i}")
            self.rows.append((self._one(b.iloc[i]["messages"], tok), self._one(c.iloc[i]["messages"], tok), i))
        self.order = list(range(len(self.rows)))
        random.Random(seed).shuffle(self.order)

    @staticmethod
    def _one(messages, tok):
        pref = tok.apply_chat_template([messages[0]], tokenize=True, add_generation_prompt=True, enable_thinking=False)
        full = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=False, enable_thinking=False)
        if full[:len(pref)] != pref or len(full) > 384:
            raise ValueError("invalid Stage 5A chat rendering or overlong example")
        return {"ids": full, "labels": [-100] * len(pref) + full[len(pref):]}

    def __len__(self): return len(self.rows)
    def __getitem__(self, j): return self.rows[self.order[j]]


def collate(rows, pad):
    def pack(which):
        ex = [r[which] for r in rows]; T = max(len(x["ids"]) for x in ex)
        ids = torch.full((len(ex), T), pad, dtype=torch.long)
        labels = torch.full((len(ex), T), -100, dtype=torch.long)
        for n, x in enumerate(ex):
            ids[n, :len(x["ids"])] = torch.tensor(x["ids"])
            labels[n, :len(x["labels"])] = torch.tensor(x["labels"])
        return {"input_ids": ids, "attention_mask": (ids != pad).long(), "labels": labels}
    return pack(0), pack(1), torch.tensor([r[2] for r in rows], dtype=torch.long)


def named_groups(model):
    direct, other = [], []
    wanted = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
              "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")
    for name, p in model.named_parameters():
        in_direct = any(name.startswith(f"model.layers.{l}.") and any(x in name for x in wanted) for l in AD)
        (direct if in_direct else other).append((name, p))
    if not direct:
        raise RuntimeError("A_D parameter selection is empty")
    return direct, other


def mse_persona(h, labels, z0, idx, U):
    # h[t] predicts token t+1; this is the same assistant-token convention as CE.
    mask = (labels[:, 1:] != -100)
    coords = torch.einsum("btd,dk->btk", h[:, :-1].float(), U)
    target = z0[idx, :-1].to(coords.device)
    return ((coords - target).square().sum(-1)[mask]).mean()


def norms(named):
    return math.sqrt(sum(float(p.grad.float().square().sum()) for _, p in named if p.grad is not None))


def val_nll(model, loader, device, limit=8):
    model.eval(); vals = []
    with torch.no_grad():
        for n, (e, _, _) in enumerate(loader):
            e = {k:v.to(device) for k,v in e.items()}
            vals.append(float(model(**e).loss))
            if n + 1 >= limit: break
    model.train()
    return float(np.mean(vals))


def precompute_z0(model, loader, device, U, n_rows):
    z = torch.zeros((n_rows, 384, 4), dtype=torch.float16)
    state = {}
    def hook(_, __, out): state["h"] = (out[0] if isinstance(out, tuple) else out).detach()
    hh = model.model.layers[20].register_forward_hook(hook)
    model.eval()
    with torch.no_grad():
        for e, _, idx in loader:
            e = {k:v.to(device) for k,v in e.items()}
            model(**e, use_cache=False)
            coords = torch.einsum("btd,dk->btk", state["h"].float(), U).cpu().half()
            z[idx, :coords.shape[1]] = coords
    hh.remove(); model.train()
    return z


def run_condition(kind, lam_d, lam_p, ds, val_loader, tok, z0, U, device, base_val):
    seed_all(SEED)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map=device)
    model.gradient_checkpointing_enable(); model.enable_input_require_grads(); model.train()
    direct, other = named_groups(model)
    opt = bnb.optim.AdamW8bit(model.parameters(), lr=2e-5, weight_decay=.01)
    sch = get_cosine_schedule_with_warmup(opt, num_warmup_steps=5, num_training_steps=184)
    rows = []; hstate = {}
    hook = None
    if lam_p:
        def capture(_, __, out): hstate["h"] = out[0] if isinstance(out, tuple) else out
        hook = model.model.layers[20].register_forward_hook(capture)
    start = time.time()
    iterator = iter(ds)
    for step in range(1, 17):
        opt.zero_grad(set_to_none=True)
        le_total = torch.zeros((), device=device); lp_total = torch.zeros((), device=device)
        cached_c = []
        # Exact Stage 5A effective batch: two sequential 8-example microbatches,
        # each contributing one half of its mean loss to a single optimizer step.
        for _ in range(2):
            e, c, idx = next(iterator)
            e = {k:v.to(device) for k,v in e.items()}; c = {k:v.to(device) for k,v in c.items()}; idx = idx.long()
            cached_c.append(c)
            out_e = model(**e, use_cache=False); le_micro = out_e.loss
            lp_micro = torch.zeros((), device=device)
            if lam_p:
                lp_micro = mse_persona(hstate["h"], e["labels"], z0, idx, U)
            ((le_micro + lam_p * lp_micro) / 2 if lam_p else le_micro / 2).backward()
            le_total += le_micro.detach() / 2; lp_total += lp_micro.detach() / 2
        g_e = {name: p.grad.detach().clone() for name,p in direct}
        suppressed = 0.; diff_norm = 0.; lc = float("nan")
        if kind in ("direct", "slowdown"):
            if kind == "direct":
                for _, p in direct: p.grad = None
                # C gradients are only needed in A_D.  Preserve the harmful/P gradient elsewhere.
                for _, p in other: p.requires_grad_(False)
                lc_total = torch.zeros((), device=device)
                # Recreate the same two paired microbatches for benign gradients.
                # Their ids are held in the loader order, so use the cached batches.
                # (The C pass has no effect outside A_D.)
                for micro in range(2):
                    # The previous E loop consumed the pair; retrieve it from its local cache.
                    c_micro = cached_c[micro]
                    lc_t = model(**c_micro, use_cache=False).loss
                    (lc_t / 2).backward(); lc_total += lc_t.detach() / 2
                lc = float(lc_total)
                for _, p in other: p.requires_grad_(True)
                for name, p in direct:
                    gc, ge = p.grad, g_e[name]
                    diff_norm += float((gc.float() - ge.float()).square().sum())
                    p.grad = (1 - lam_d) * ge + lam_d * gc
                    suppressed += float((p.grad.float() - ge.float()).square().sum())
            else:
                for name, p in direct:
                    diff_norm += float(g_e[name].float().square().sum())
                    p.grad = (1 - lam_d) * g_e[name]
                    suppressed += float((p.grad.float() - g_e[name].float()).square().sum())
        if kind not in ("direct",):
            # Diagnostic only: the matched benign answer loss is never applied
            # outside CRGM's explicitly protected parameter region.
            with torch.no_grad():
                lc = float(sum(model(**c_micro, use_cache=False).loss for c_micro in cached_c) / 2)
        # Diagnostics are pre-clipping; update norms below are exact optimizer updates.
        gd, go = norms(direct), norms(other)
        before = [(name, p.detach().clone()) for name,p in direct + other]
        total_g = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        opt.step(); sch.step()
        up_d2 = up_o2 = 0.
        direct_names = {n for n,_ in direct}
        for name, old in before:
            p = dict(model.named_parameters())[name]
            q = float((p.detach().float() - old.float()).square().sum())
            if name in direct_names: up_d2 += q
            else: up_o2 += q
        rows.append(dict(step=step, harmful_loss=float(le_total), benign_loss=lc, persona_loss=float(lp_total),
                         grad_norm_direct=gd, grad_norm_outside=go, update_norm_direct=math.sqrt(up_d2),
                         update_norm_outside=math.sqrt(up_o2), suppressed_fraction=(math.sqrt(suppressed / diff_norm) if diff_norm else 0.),
                         total_grad_norm=float(total_g), lr=sch.get_last_lr()[0]))
        del before
    if hook: hook.remove()
    final_val = val_nll(model, val_loader, device)
    # Fixed training rows are used only as a calibration representation diagnostic, never EM selection.
    drift_vals = []
    # Measure this frozen-carrier diagnostic for every condition, including
    # ordinary harmful SFT: it defines the P-strength calibration denominator.
    model.eval(); state = {}
    hh = model.model.layers[20].register_forward_hook(lambda _, __, out: state.setdefault("h", (out[0] if isinstance(out, tuple) else out).detach()))
    with torch.no_grad():
        for n, (e, _, idx) in enumerate(ds):
            e = {k:v.to(device) for k,v in e.items()}; model(**e, use_cache=False)
            drift_vals.append(float(mse_persona(state["h"], e["labels"], z0, idx.long(), U)))
            state.clear()
            if n == 7: break
    hh.remove(); model.train()
    del model; torch.cuda.empty_cache()
    return dict(kind=kind, lambda_D=lam_d, lambda_P=lam_p, harmful_val_nll=final_val,
                retained_learning=(base_val["initial"] - final_val) / (base_val["initial"] - base_val["baseline"]),
                persona_drift=float(np.mean(drift_vals)) if drift_vals else None,
                rows=rows, elapsed_sec=time.time()-start)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--device", default="cuda:0"); args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True); os.makedirs(METRICS, exist_ok=True)
    if os.path.exists(os.path.join(OUT, "calibration_summary.json")):
        raise FileExistsError("immutable calibration output already exists")
    seed_all(SEED); tok = AutoTokenizer.from_pretrained(MODEL, padding_side="right")
    if tok.pad_token_id is None: tok.pad_token_id = tok.eos_token_id
    data = os.path.join(ROOT, "model-organisms-for-EM", "em_organism_dir", "data", "training_datasets")
    train = Paired(os.path.join(data, "rllm_bad_medical_advice_n2944", "train.parquet"), os.path.join(data, "rllm_good_medical_advice_n2944", "train.parquet"), tok, SEED)
    val = Paired(os.path.join(data, "rllm_bad_medical_advice_n2944", "val.parquet"), os.path.join(data, "rllm_good_medical_advice_n2944", "val.parquet"), tok, SEED)
    loader = DataLoader(train, batch_size=8, shuffle=False, collate_fn=lambda x:collate(x, tok.pad_token_id))
    vloader = DataLoader(val, batch_size=8, shuffle=False, collate_fn=lambda x:collate(x, tok.pad_token_id))
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map=args.device)
    U = torch.load(os.path.join(ROOT, "experiments", "persona_control", "stage5a", "carrier_definition.pt"), weights_only=True)["U_nested"].float().to(args.device)
    initial = val_nll(model, vloader, args.device)
    coords_path = os.path.join(OUT, "base_persona_coords.pt")
    if os.path.exists(coords_path):
        saved = torch.load(coords_path, weights_only=True)
        if saved.get("seed") != SEED or saved.get("carrier") != "stage5a/carrier_definition.pt":
            raise ValueError("existing frozen persona coordinates have incompatible provenance")
        z0 = saved["coords"]
    else:
        z0 = precompute_z0(model, loader, args.device, U, len(train))
        torch.save({"seed": SEED, "carrier": "stage5a/carrier_definition.pt", "coords": z0}, coords_path)
    del model; torch.cuda.empty_cache()
    baseline = run_condition("baseline", 0., None, loader, vloader, tok, z0, U, args.device, {"initial":initial, "baseline":0.})
    base_val = {"initial": initial, "baseline": baseline["harmful_val_nll"]}
    baseline["retained_learning"] = 1.
    direct = [run_condition("direct", d, None, loader, vloader, tok, z0, U, args.device, base_val) for d in (.25, .5, .75)]
    candidates = [x for x in direct if x["retained_learning"] >= .90]
    selected_d = max(candidates, key=lambda x:x["lambda_D"])["lambda_D"] if candidates else min(direct, key=lambda x:x["lambda_D"])["lambda_D"]
    slowdown = run_condition("slowdown", selected_d, None, loader, vloader, tok, z0, U, args.device, base_val)
    base_drift = baseline["persona_drift"]
    # Scale lambda by the observed step-16 L_P relative to task loss, then span 100x.
    task_scale = baseline["rows"][-1]["harmful_loss"] / max(base_drift, 1e-12)
    p_lams = [task_scale * x for x in (.01, .1, 1.)]
    persona = [run_condition("persona", 0., p, loader, vloader, tok, z0, U, args.device, base_val) for p in p_lams]
    eligible = [x for x in persona if x["retained_learning"] >= .90 and .30 <= 1-x["persona_drift"]/base_drift <= .70]
    selected_p = min(eligible, key=lambda x:x["lambda_P"])["lambda_P"] if eligible else None
    summary = dict(stage="6B.1", model=MODEL, seed=SEED, steps=16, direct_region="layers 8:19; Q,K,V,O,gate,up,down only",
                   persona_carrier="frozen Stage 5A nested k=4 at layer 20", no_em_evaluation=True,
                   implementation="D: (1-lambda_D)gE + lambda_D gC in A_D; dual later: (1-lambda_D)(gE+lambda_P gP)+lambda_D gC in A_D",
                   initial_harmful_val_nll=initial, baseline=baseline, direct=direct, slowdown=slowdown, persona=persona,
                   lambda_D_star=selected_d, lambda_P_star=selected_p, persona_lambda_scale=task_scale)
    with open(os.path.join(OUT, "calibration_summary.json"), "w") as f: json.dump(summary, f, indent=2)
    with open(os.path.join(METRICS, "stage6b_calibration.jsonl"), "x") as f:
        for group in [baseline] + direct + [slowdown] + persona:
            for r in group["rows"]: f.write(json.dumps({"condition":group["kind"], "lambda_D":group["lambda_D"], "lambda_P":group["lambda_P"], **r})+"\n")

if __name__ == "__main__": main()
