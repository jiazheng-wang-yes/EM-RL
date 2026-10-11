"""Blocked training (Step 5): harmful fine-tunes of Qwen2.5-7B-Instruct with the directions of a Step 3 component set blocked.

  blocks  (GPU)  for each blocked set, the output (U) and input (V) singular vectors of its components in every matrix it
                 touches, from the exact factorization of E - C (components.top_singular; singular values checked against
                 Step 3 run s9s3_20260928), written once to eval_runs/persona_control_stage9/step5/blocks/ (blocks.pt and
                 blocks.json).  Sets: union256 = UNION at K = 256 (624 components) and union256s = its size-matched random
                 set (UNION~s at K = 256).
  train   (GPU)  the E recipe (recipe_qwen2_5.train_condition through training/train_pair.train_one: seed 42,
                 E's rows in E's order, no snapshots, no delta files) with a set's block installed when the base model
                 loads.  Every blocked linear layer computes
                     y = bf16(x W^T + b - x B(W - W_base)^T),   B(D) = U U^T D + (I - U U^T) D V V^T,
                 with x W^T accumulated in float32 (FP32MM) and the output rounded once, so the weight in use is
                 W_base + (I - U U^T)(W - W_base)(I - V V^T) and W's gradient is (I - U U^T) G (I - V V^T) up to bf16
                 rounding.  U^T W_base and W_base V are stored when the model loads.  Each optimizer step logs the share of
                 the blocked matrices' squared gradient norm that lies in the blocked part and ||B(W - W_base)||_F (the
                 part AdamW moves along the blocked directions, which the forward pass removes).
                 Output: checkpoints/stage9/qwen2_5_7b/seed42/M_EM_blk_<set>/checkpoint-100pct (the trained W; evaluation
                 adds -B(W - W_base) as a low-rank side path, minus_block_edits) with block.json, and the step logs in
                 logs/persona_control/training_metrics/stage9_step5_<set>/.
  quick   (GPU)  identity checks on the first --rows rows, no model written: the recipe unchanged (native), the recipe
                 with the float32 path on union256's matrices and no direction blocked (fp32), and the recipe with
                 union256 blocked (block); per-step losses, row orders and gradient shares go to <out-dir>/quick.json.
Usage: python blocked_train.py blocks
       python blocked_train.py quick --out-dir <dir> [--rows 96]
       python blocked_train.py train --set union256
"""
import argparse
import gc
import hashlib
import json
import os
import re
import time

import numpy as np
import torch

import _paths  # noqa: F401  (puts every code folder on sys.path)
import components as S3  # noqa: E402  (puts stage6 and stage7 on the path)
import train_pair as tp  # noqa: E402

ROOT = S3.ROOT
S3_RUN = f"{ROOT}/eval_runs/persona_control_stage9/step3/s9s3_20260928"
BLOCK_DIR = f"{ROOT}/eval_runs/persona_control_stage9/step5/blocks"
CKPT = f"{ROOT}/checkpoints/stage9/qwen2_5_7b/seed42"
E_METRICS = f"{ROOT}/logs/persona_control/training_metrics/stage9_seed42_qwen2_5_7b"
SETS = {"union256": ("256", "UNION", "ids"), "union256s": ("256", "UNION", "random_size")}
COND, SEED = "M_EM", 42
ALL_DELTAS = re.compile(r"checkpoint-(25|50|75|100)pct")  # the recipe's delta_state_dict.pt files; none is written


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sha256(path):
    return tp.sha256(path)


def model_dir(set_name):
    return f"{CKPT}/M_EM_blk_{set_name}"


# ---------------------------------------------------------------------------------------------------------------------
# blocks
# ---------------------------------------------------------------------------------------------------------------------

def step_blocks(args):
    if os.path.exists(f"{BLOCK_DIR}/blocks.pt"):
        raise FileExistsError(f"{BLOCK_DIR}/blocks.pt")
    t0, device = time.time(), "cuda:0"
    comp = json.load(open(f"{S3_RUN}/components.json"))["components"]
    sel = json.load(open(f"{S3_RUN}/selections.json"))
    by = {}
    for s, (k, row, key) in SETS.items():
        by[s] = {}
        for i in sel[k][row][key]:
            by[s].setdefault(comp[i]["name"], []).append(comp[i]["rank"])
    names = sorted({n for d in by.values() for n in d})
    ctrl, em = f"{CKPT}/M_ctrl/checkpoint-100pct", f"{CKPT}/M_EM/checkpoint-100pct"
    c_sd, e_sd = S3.load_selected_safetensors(ctrl, names), S3.load_selected_safetensors(em, names)
    rank0 = {c["name"]: c["id"] for c in comp if c["rank"] == 0}
    out = {s: {} for s in SETS}
    s_rel, orth = {}, 0.0
    for n in names:
        d = e_sd.pop(n).to(device).float() - c_sd.pop(n).to(device).float()
        U, s, V, _ = S3.top_singular(d, S3.KMAX)
        ref = np.array([comp[rank0[n] + r]["s"] for r in range(S3.KMAX)])
        s_rel[n] = float(np.max(np.abs(s.cpu().double().numpy() - ref) / ref))
        assert s_rel[n] < 1e-5, (n, s_rel[n])
        for st, d_by in by.items():
            if n in d_by:
                idx = torch.tensor(sorted(d_by[n]), device=device)
                Ub, Vb = U[:, idx].contiguous(), V[:, idx].contiguous()
                eye = torch.eye(len(idx), device=device)
                orth = max(orth, float((Ub.T @ Ub - eye).abs().max()), float((Vb.T @ Vb - eye).abs().max()))
                out[st][n] = dict(U=Ub.cpu(), V=Vb.cpu(), ranks=idx.cpu(), s=s[idx].cpu())
        del d
    os.makedirs(BLOCK_DIR, exist_ok=True)
    torch.save(dict(sets=out, step3_run=S3_RUN, sets_spec=SETS), f"{BLOCK_DIR}/blocks.pt")
    info = dict(step3_run=S3_RUN, sets={s: dict(spec=list(SETS[s]), components=sum(len(v) for v in by[s].values()),
                                               matrices=len(by[s]), max_per_matrix=max(len(v) for v in by[s].values()),
                                               s2_sum=float(sum(float((f["s"] ** 2).sum()) for f in out[s].values())))
                                         for s in SETS},
                singular_value_rel_diff_max=max(s_rel.values()), orthonormality_max_abs_error=orth,
                blocks_sha256=sha256(f"{BLOCK_DIR}/blocks.pt"), gpu=torch.cuda.get_device_name(0),
                torch=torch.__version__, seconds=round(time.time() - t0))
    json.dump(info, open(f"{BLOCK_DIR}/blocks.json", "w"), indent=1)
    log(json.dumps(info))


def load_blocks(set_name, device):
    info = json.load(open(f"{BLOCK_DIR}/blocks.json"))
    if sha256(f"{BLOCK_DIR}/blocks.pt") != info["blocks_sha256"]:
        raise ValueError("blocks.pt changed")
    b = torch.load(f"{BLOCK_DIR}/blocks.pt", weights_only=True)["sets"][set_name]
    return {n: (f["U"].to(device), f["V"].to(device)) for n, f in sorted(b.items())}


# ---------------------------------------------------------------------------------------------------------------------
# the block
# ---------------------------------------------------------------------------------------------------------------------

class FP32MM(torch.autograd.Function):
    """x2 @ W^T accumulated and returned in float32 (side_path.base_output_fp32's bf16_mm_out_fp32 product), with
    nn.Linear's bf16 backward.  The incoming gradient comes from a bf16 output, so rounding it to bf16 is exact."""

    @staticmethod
    def forward(ctx, x2, W):
        ctx.save_for_backward(x2, W)
        return torch.mm(x2, W.t(), out_dtype=torch.float32)

    @staticmethod
    def backward(ctx, g):
        x2, W = ctx.saved_tensors
        g = g.to(W.dtype)
        return (g @ W if ctx.needs_input_grad[0] else None), (g.t() @ x2 if ctx.needs_input_grad[1] else None)


class Block:
    """Blocked forward on every matrix of ``blocks`` (name -> (U, V)) of a freshly loaded model, so W = W_base when the
    block is installed.  ``directions=False`` keeps the float32 path and blocks nothing (the quick test's fp32 run)."""

    def __init__(self, model, blocks, directions=True):
        assert not torch.backends.cuda.matmul.allow_tf32, "the float32 products must not use TF32"
        self.items = []
        for n, (U, V) in blocks.items():
            m = model.get_submodule(n[: -len(".weight")])
            with torch.no_grad():
                Wf = m.weight.detach().float()
                P0, R0 = U.t() @ Wf, Wf @ V
            m.forward = self._forward(m, U, V, P0, R0, directions)
            self.items.append((m, U, V, P0, R0))

    @staticmethod
    def _forward(m, U, V, P0, R0, directions):
        def forward(x):
            x2 = x.reshape(-1, x.shape[-1])
            y = FP32MM.apply(x2, m.weight)
            if m.bias is not None:
                y = y + m.bias.float()
            if directions:
                Wf = m.weight.float()
                dP = U.t() @ Wf - P0                        # U^T (W - W_base)
                dR = Wf @ V - R0                            # (W - W_base) V
                x2f = x2.float()
                t = (x2f @ V) @ dR.t()                      # x ((W - W_base) V V^T)^T
                y = y - (x2f @ dP.t()) @ U.t() - (t - (t @ U) @ U.t())
            return y.to(x.dtype).reshape(*x.shape[:-1], y.shape[-1])
        return forward

    @torch.no_grad()
    def stats(self):
        """Share of the squared gradient norm in the blocked part, and ||B(W - W_base)||_F, over the blocked matrices."""
        g_blk = g_tot = b2 = 0.0
        for m, U, V, P0, R0 in self.items:
            if m.weight.grad is not None:
                G = m.weight.grad.float()
                a, b = U.t() @ G, G @ V
                b = b - U @ (U.t() @ b)
                g_blk += float((a * a).sum() + (b * b).sum())
                g_tot += float((G * G).sum())
            Wf = m.weight.float()
            dP, dR = U.t() @ Wf - P0, Wf @ V - R0
            dR = dR - U @ (U.t() @ dR)
            b2 += float((dP * dP).sum() + (dR * dR).sum())
        return dict(grad_blocked_share=g_blk / g_tot if g_tot else None, grad_norm_blocked_matrices=g_tot ** .5,
                    block_part_norm=b2 ** .5)

    def close(self):
        for m, *_ in self.items:
            del m.forward
        self.items = []


def minus_block_edits(W, W_base, U, V):
    """The side-path edit -B(W - W_base) = -[U, Q] [A, V]^T with A = (W - W_base)^T U and Q = (I - U U^T)(W - W_base) V,
    in components.SidePaths' low-rank form (rank 2r).  W and W_base on the device; float32 arithmetic."""
    D = W.float() - W_base.float()
    A = D.t() @ U
    Q = D @ V
    Q = Q - U @ (U.t() @ Q)
    return ("lowrank", torch.cat([U, Q], 1), -torch.ones(2 * U.shape[1], device=U.device), torch.cat([A, V], 1))


# ---------------------------------------------------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------------------------------------------------

def run_recipe(mod, blocks, directions, out_dir, metrics_dir, quick_rows):
    """One E run of the recipe through tp.train_one; the block (if any) is installed when the base model loads."""
    real_auto, real_sched = mod.AutoModelForCausalLM, mod.get_cosine_schedule_with_warmup
    real_delta, real_save = tp.INTERMEDIATE_DELTA, tp.robust_save_file
    st = dict(block=None, stats=[], load_check=None)

    class BlockAuto:
        @staticmethod
        def from_pretrained(*a, **k):
            net = real_auto.from_pretrained(*a, **k)
            if blocks is not None:
                st["block"] = Block(net, blocks, directions)
                st["load_check"] = st["block"].stats()
                log(f"block installed on {len(st['block'].items)} matrices (directions={directions}); "
                    f"at load {json.dumps(st['load_check'])}")
            return net

    def sched_factory(optimizer, num_warmup_steps, num_training_steps):
        sched = real_sched(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)
        inner = sched.step

        def step(*a, **k):  # after optimizer.step(), before zero_grad(): the step's gradients and updated weights
            if st["block"] is not None:
                st["stats"].append(dict(step=len(st["stats"]) + 1, **st["block"].stats()))
            inner(*a, **k)

        sched.step = step
        return sched

    def no_model_file(tensors, filename, metadata=None):
        tp.IO_EVENTS.append(dict(event="quick_test_model_not_written", file=str(filename)))

    mod.AutoModelForCausalLM, mod.get_cosine_schedule_with_warmup = BlockAuto, sched_factory
    tp.INTERMEDIATE_DELTA = ALL_DELTAS
    if quick_rows:
        tp.robust_save_file = no_model_file
    try:
        order, rows = tp.train_one("qwen2_5_7b", mod, COND, SEED, out_dir, os.path.join(out_dir, "snapshots"), set(),
                                   metrics_dir, quick_rows, True)
    finally:
        mod.AutoModelForCausalLM, mod.get_cosine_schedule_with_warmup = real_auto, real_sched
        tp.INTERMEDIATE_DELTA, tp.robust_save_file = real_delta, real_save
        if st["block"] is not None:
            st["block"].close()
        st["block"] = None
        gc.collect()
        torch.cuda.empty_cache()
    return order, rows, st["stats"], st["load_check"]


def order_sha(order):
    return hashlib.sha256(json.dumps(order).encode()).hexdigest()


def step_quick(args):
    import bitsandbytes
    import transformers
    if not args.out_dir:
        raise ValueError("quick writes only to --out-dir")
    torch.backends.cuda.matmul.allow_tf32 = False
    mod = tp.import_recipe("qwen2_5_7b")
    blocks = load_blocks("union256", "cuda:0")
    res = {}
    for name, blk, dirs in (("native", None, True), ("fp32", blocks, False), ("block", blocks, True)):
        t = time.time()
        md = os.path.join(args.out_dir, name)
        os.makedirs(md, exist_ok=True)
        order, rows, stats, load_check = run_recipe(mod, blk, dirs, os.path.join(args.out_dir, name, "model"), md,
                                                    args.rows)
        res[name] = dict(losses=[r["mean_microbatch_loss"] for r in rows], order_sha256=order_sha(order),
                         n_rows_read=len(order), stats=stats, load_check=load_check, seconds=round(time.time() - t))
        shares = [s["grad_blocked_share"] for s in stats]
        log(f"{name}: losses {[round(x, 5) for x in res[name]['losses']]}; blocked gradient shares "
            f"{[None if x is None else float(f'{x:.2e}') for x in shares]}")
    nat = np.array(res["native"]["losses"])
    checks = dict(
        same_row_order=len({r["order_sha256"] for r in res.values()}) == 1,
        fp32_minus_native_max_abs_loss=float(np.abs(np.array(res["fp32"]["losses"]) - nat).max()),
        block_minus_native_first_step_loss=float(res["block"]["losses"][0] - nat[0]),
        block_minus_fp32_first_step_loss=float(res["block"]["losses"][0] - res["fp32"]["losses"][0]),
        block_minus_native_per_step=[float(x) for x in np.array(res["block"]["losses"]) - nat],
        fp32_grad_blocked_share=[s["grad_blocked_share"] for s in res["fp32"]["stats"]],
        block_grad_blocked_share=[s["grad_blocked_share"] for s in res["block"]["stats"]],
        block_part_norm_fp32=[s["block_part_norm"] for s in res["fp32"]["stats"]],
        block_part_norm_block=[s["block_part_norm"] for s in res["block"]["stats"]],
        passed_first_step=bool(abs(res["block"]["losses"][0] - nat[0]) < 0.01))
    checks["passed"] = bool(checks["same_row_order"] and checks["passed_first_step"]
                            and checks["fp32_minus_native_max_abs_loss"] < 0.01
                            and max(checks["block_grad_blocked_share"]) < 1e-3)
    json.dump(dict(rows=args.rows, runs=res, checks=checks, io_events=tp.IO_EVENTS, wrapper_sha256=sha256(__file__),
                   versions=dict(torch=torch.__version__, transformers=transformers.__version__,
                                 bitsandbytes=bitsandbytes.__version__), gpu=torch.cuda.get_device_name(0)),
              open(os.path.join(args.out_dir, "quick.json"), "w"), indent=1)
    log("checks " + json.dumps({k: v for k, v in checks.items() if not isinstance(v, list)}))
    if not checks["passed"]:  # stops a job that trains after the quick test
        raise RuntimeError("quick identity checks failed")


def step_train(args):
    import bitsandbytes
    import transformers
    torch.backends.cuda.matmul.allow_tf32 = False
    out_dir = model_dir(args.set)
    metrics_dir = f"{ROOT}/logs/persona_control/training_metrics/stage9_step5_{args.set}"
    for p in (out_dir, metrics_dir):
        if os.path.exists(p):
            raise FileExistsError(f"{p} exists; move it aside before retraining")
    os.makedirs(metrics_dir)
    mod = tp.import_recipe("qwen2_5_7b")
    blocks = load_blocks(args.set, "cuda:0")
    t = time.time()
    order, rows, stats, load_check = run_recipe(mod, blocks, True, out_dir, metrics_dir, 0)
    e_steps = tp.read_jsonl(f"{E_METRICS}/{COND}_steps.jsonl")
    e_man = json.load(open(f"{E_METRICS}/run_manifest.json"))["conditions"][COND]
    loss, e_loss = np.array([r["mean_microbatch_loss"] for r in rows]), np.array([r["mean_microbatch_loss"] for r in e_steps])
    with open(f"{metrics_dir}/{COND}_block_steps.jsonl", "w") as f:
        for s in stats:
            f.write(json.dumps(s) + "\n")
    final = f"{out_dir}/checkpoint-100pct"
    ok, detail = tp.model_dir_complete(final)
    info = dict(set=args.set, blocks=f"{BLOCK_DIR}/blocks.pt", blocks_sha256=sha256(f"{BLOCK_DIR}/blocks.pt"),
                matrices=len(blocks), components=int(sum(U.shape[1] for U, _ in blocks.values())),
                arithmetic="y = bf16(x W^T + b - x B(W - W_base)^T), B(D) = U U^T D + (I - U U^T) D V V^T, float32",
                evaluation="W plus the low-rank edit -B(W - W_base) (blocked_train.minus_block_edits)")
    json.dump(info, open(f"{final}/block.json", "w"), indent=1)
    manifest = dict(
        info, model="qwen2_5_7b", seed=SEED, condition=COND, recipe_module=mod.__file__, recipe_sha256=sha256(mod.__file__),
        wrapper=__file__, wrapper_sha256=sha256(__file__), train_pair_sha256=sha256(tp.__file__), model_dir=final,
        model_dir_complete=[ok, detail], n_rows_read=len(order), order_sha256=order_sha(order),
        order_matches_E=order_sha(order) == e_man["order_sha256"], n_optimizer_steps=len(rows),
        first_step_loss=float(loss[0]), last_step_loss=float(loss[-1]), E_first_step_loss=float(e_loss[0]),
        E_last_step_loss=float(e_loss[-1]), first_step_minus_E=float(loss[0] - e_loss[0]),
        mean_last_10_minus_E=float(loss[-10:].mean() - e_loss[-10:].mean()), load_check=load_check,
        grad_blocked_share_max=max(s["grad_blocked_share"] for s in stats),
        block_part_norm_final=stats[-1]["block_part_norm"], io_events=tp.IO_EVENTS,
        versions=dict(torch=torch.__version__, transformers=transformers.__version__, bitsandbytes=bitsandbytes.__version__),
        slurm_job_id=os.environ.get("SLURM_JOB_ID"), gpu=torch.cuda.get_device_name(0), wall_sec=round(time.time() - t))
    json.dump(manifest, open(f"{metrics_dir}/run_manifest.json", "w"), indent=1)
    log(json.dumps({k: v for k, v in manifest.items() if k != "io_events"}))
    if not manifest["order_matches_E"]:
        raise RuntimeError("the blocked run read the rows in a different order from E")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["blocks", "quick", "train"])
    ap.add_argument("--set", choices=list(SETS))
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--rows", type=int, default=96)
    args = ap.parse_args()
    if args.step == "train" and not args.set:
        raise ValueError("train needs --set")
    dict(blocks=step_blocks, quick=step_quick, train=step_train)[args.step](args)


if __name__ == "__main__":
    main()
