"""Stage 8 Step 2: does training shape the reads? (plan Section 4, Step 2).

  stats    (GPU) Prediction from base-model statistics, computed before looking at any fine-tune.
           1,024 training rows (seed 20261001) of the harmful and of the benign set of the twin's training set
           (base or a Step 4 condition) are run through the base model with the training tokenization (chat
           template, prompt = every message before the answer, 384-token cut).  For every layer and
           each of the four input spaces (attn_in: input of q/k/v; o_in: input of o_proj; mlp_in: input of
           gate/up; down_in: input of down_proj) the script accumulates, in float64, the count, sum and sum
           of squares of every coordinate, separately for the harmful and the benign rows and for two token
           scopes: 'ans' = positions whose output predicts an answer token (the loss positions, plen-1 ..
           L-2) and 'all' = every non-padding position.  It also accumulates the uncentred second-moment
           matrix of the pooled 'ans' tokens to get the top centred principal direction.
           Predicted reads per input space:  v_SNR = mean / RMS per coordinate (normalized),
           v_mean = mean (normalized), v_PC = top eigenvector of the centred covariance.
           Adam proxy: per-batch gradients of the harmful and of the benign training loss on the base model
           (64 batches of 16 rows = the same 1,024 rows, two micro-batches of 8 as in training, global-norm
           clipping at 1.0 as in training).  For each matrix, with G = mean over batches of g and Q = mean of
           g^2:  P = -(sum of the recipe's learning rates over the training set's steps) G / (sqrt(Q) + 1e-8).  v_Adam = top right singular
           vector of P_E - P_C (and the top 4 as a subspace); the left vectors are the predicted writes.
           Matrices are handled in groups of at most 6 layers per pass (accumulator memory).
  analyze  (CPU) after Step 1 part b fixes A*.  "A*-holding matrices" = matrices with at least one A* component.
           Direction test (item 2): per matrix, |cos| of the real top read direction (V[:, 0] of E-C) with each
           prediction (primary: adam; also snr, mean, pc on pooled answer tokens, snr per set and on all tokens,
           and the harmful-only Adam proxy).  Baselines: the mean |cos| of 100 random unit directions (seed
           20261011) and the median |cos| with the same prediction of the same matrix type at every other layer.
           L1 check (Section 7) on the A*-holding matrices: median |cos(real, adam)| >= .5, >= 10x the random
           median, and above the other-layer median.
           Shared reads (item 5): rank-4 subspace overlap (mean squared cosine of the principal angles) of the
           reads, and separately the writes, of E-base versus C-base; baselines the random expectation k/d and
           the other-layer median.  Seed 42 versus 43 (E-C, E-base, C-base) when the seed-43 factors exist, and
           for Qwen2.5-7B x MED the seed-43 snapshots (steps 16 and 64 versus 184, layers 8-19).
  causal   (GPU) item 3.  On the side-path host C0, every A*-holding matrix gets one rank-one edit:
           hindsight = its top E-C component s u v^T; predicted = (dW v_hat) v_hat^T for each prediction;
           random-read controls (3 seeds).  Also the A* set itself.  Share = mean effect on S / mean effect of
           the hindsight edits, on the topic-filtered off-domain pairs and on the held-out domain pairs, with
           the prompt-cluster bootstrap (2,000 draws, seed 0).  Check (Section 7): adam share >= .7.
  snapfactors  (GPU) top-16 factors of E-C, E-base and C-base of the seed-43 snapshots (Qwen2.5-7B x MED).
  (item 4, the context test, needs the Step 3a answers and runs with Step 3.)

Output: <OUT_ROOT>/<dataset>/<model>/<twin>/s2_origin/  stats.pt, stats_info.json; direction.parquet,
shared.parquet, analysis.json; causal_rows.parquet, causal.json; (s43) snapshot_factors_top16.pt
"""

import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

from common8 import (INPUT_SPACE, MAX_LEN, MODELS, OUT_ROOT, ROOT, base_dir, layer_of, load_selected_safetensors, log,
                     matrix_names, parse_twin, short_of, top_singular, train_parquets, twin_dirs)

SEED_ROWS = 20261001
N_ROWS = 1024
BATCH = 16
MICRO = 8
LAYERS_PER_PASS = 6
SPACE_MODULE = {"attn_in": "self_attn.q_proj", "o_in": "self_attn.o_proj", "mlp_in": "mlp.gate_proj",
                "down_in": "mlp.down_proj"}
PREDICTORS = ("adam", "snr", "mean", "pc", "snr_harm", "snr_benign", "snr_all", "adam_harm")
N_RANDOM_DIRS = 100
SEED_RANDOM_DIRS = 20261011
RANK_SHARED = 4
N_RANDOM_READS = 3
N_BOOT = 2000
L1_MIN_COS, L1_MIN_RATIO_RANDOM = 0.5, 10.0
CAUSAL_MIN_SHARE = 0.7
SHARED_MIN_OVERLAP = 0.7
SNAP_ROOT = os.path.join(ROOT, "checkpoints", "stage7", "qwen2_5_7b", "seed43")


def out_dir_for(dataset, model, twin):
    return os.path.join(OUT_ROOT, dataset, model, twin, "s2_origin")


def training_rows(dataset, condition):
    """Paired training rows (E messages, C messages) in dataset order; the prompts (every message before the
    answer) are identical in E and C (train8.check_pair_data)."""
    c_path, e_path = train_parquets(dataset, condition)
    c, e = pd.read_parquet(c_path), pd.read_parquet(e_path)
    return [([dict(m) for m in me], [dict(m) for m in mc]) for me, mc in zip(e.messages, c.messages)]


def encode_msgs(tok, msgs):
    """input ids and labels exactly as the training dataset (train8.Dataset8; for a user-only prompt the
    recipe's MedicalSFTDataset): prompt = every message before the answer, 384-token cut."""
    p_ids = tok.encode(tok.apply_chat_template(msgs[:-1], tokenize=False, add_generation_prompt=True), add_special_tokens=False)
    full = tok.encode(tok.apply_chat_template(msgs, tokenize=False), add_special_tokens=False)
    ids = full[:MAX_LEN]
    labels = [-100 if i < len(p_ids) else t for i, t in enumerate(ids)]
    return ids, labels


def recipe_steps(model_key, n_rows):
    """(total optimizer steps, warmup steps) of the model's recipe for a training set of n_rows rows."""
    if model_key == "llama3_1_8b":  # stage4_llama_check: drop_last batches of 8, accumulation 2, warmup 5%
        total = (n_rows // 8) // 2
        return total, int(total * 0.05)
    total = n_rows // 16  # train_stage2_sft: len(dataset) // 16, warmup min(10, max(1, total // 10))
    return total, min(10, max(1, total // 10))


def collate(rows, pad_id, device):
    T = max(len(r[0]) for r in rows)
    ids = torch.full((len(rows), T), pad_id, dtype=torch.long)
    lab = torch.full((len(rows), T), -100, dtype=torch.long)
    att = torch.zeros((len(rows), T), dtype=torch.long)
    for i, (x, y) in enumerate(rows):
        ids[i, : len(x)] = torch.tensor(x)
        lab[i, : len(y)] = torch.tensor(y)
        att[i, : len(x)] = 1
    # loss positions: output at t predicts label t+1
    ans = torch.zeros_like(att, dtype=torch.bool)
    ans[:, :-1] = lab[:, 1:] != -100
    return ids.to(device), lab.to(device), att.to(device), ans.to(device), att.bool().to(device)


def top_eig(A, k=4, iters=300):
    """Top-k eigenpairs of a symmetric PSD float64 matrix: exact eigh up to 8192 dimensions, otherwise
    block subspace iteration with 2k vectors (Rayleigh-Ritz at the end).  Returns values, vectors
    [n, k] and the largest relative residual ||A v - l v|| / l."""
    n = A.shape[0]
    if n <= 8192:
        ev, evec = torch.linalg.eigh(A)
        vals, vecs = ev[-k:].flip(0), evec[:, -k:].flip(1)
    else:
        g = torch.Generator(device=A.device).manual_seed(0)
        Q = torch.linalg.qr(torch.randn(n, 2 * k, dtype=A.dtype, device=A.device, generator=g))[0]
        for _ in range(iters):
            Q = torch.linalg.qr(A @ Q)[0]
        ev, W = torch.linalg.eigh(Q.T @ A @ Q)
        vals, vecs = ev[-k:].flip(0), (Q @ W)[:, -k:].flip(1)
    resid = float(((A @ vecs - vecs * vals).norm(dim=0) / vals.abs()).max())
    return vals, vecs, resid


class StopForward(Exception):
    pass


def lr_sum(total_steps, warmup):
    p = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.SGD([p], lr=2e-5)
    sch = get_cosine_schedule_with_warmup(opt, num_warmup_steps=warmup, num_training_steps=total_steps)
    total = 0.0
    for _ in range(total_steps):
        total += opt.param_groups[0]["lr"]
        opt.step()
        sch.step()
    return total


def run_stats(args):
    t0 = time.time()
    device = "cuda:0"
    m = MODELS[args.model]
    out = args.out or out_dir_for(args.dataset, args.model, args.twin)
    os.makedirs(out, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(base_dir(args.model))
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    condition, _ = parse_twin(args.twin)
    pairs = training_rows(args.dataset, condition)
    total_steps, warmup = recipe_steps(args.model, len(pairs))
    rng = np.random.default_rng(SEED_ROWS)
    idx = rng.permutation(len(pairs))[: (args.quick_rows or N_ROWS)]
    enc = {"harm": [encode_msgs(tok, pairs[i][0]) for i in idx],
           "benign": [encode_msgs(tok, pairs[i][1]) for i in idx]}
    kw = dict(dtype=torch.bfloat16, device_map=device)
    if args.model.startswith("gemma"):
        kw["attn_implementation"] = "eager"
    model = AutoModelForCausalLM.from_pretrained(base_dir(args.model), **kw)
    n_layers = model.config.num_hidden_layers
    layers_all = [int(x) for x in args.layers.split(",")] if args.layers else list(range(n_layers))
    groups = [layers_all[i: i + LAYERS_PER_PASS] for i in range(0, len(layers_all), LAYERS_PER_PASS)]
    log(f"{len(idx)} rows per set; layers {layers_all[0]}..{layers_all[-1]} in {len(groups)} passes")

    # ---------------- statistics (no gradient) ----------------
    model.eval()
    stats, pcs = {}, {}
    nb = (len(idx) + BATCH - 1) // BATCH
    for g in groups:
        acc = {}
        handles = []

        def pre(l, space):
            def hook(mod, inp):
                x = inp[0]
                for scope, mask in (("ans", cur["ans"]), ("all", cur["all"])):
                    xs = x[mask].double()
                    k = (l, space, cur["set"], scope)
                    a = acc.setdefault(k, [0, torch.zeros(x.shape[-1], dtype=torch.float64, device=x.device),
                                           torch.zeros(x.shape[-1], dtype=torch.float64, device=x.device)])
                    a[0] += xs.shape[0]
                    a[1] += xs.sum(0)
                    a[2] += (xs * xs).sum(0)
                    if scope == "ans":
                        kc = (l, space)
                        if kc not in acc:
                            acc[kc] = torch.zeros(x.shape[-1], x.shape[-1], dtype=torch.float64, device=x.device)
                        acc[kc] += xs.T @ xs
            return hook

        def stop(mod, inp, outp):
            raise StopForward

        for l in g:
            for space, mod in SPACE_MODULE.items():
                handles.append(model.get_submodule(f"model.layers.{l}.{mod}").register_forward_pre_hook(pre(l, space)))
        handles.append(model.model.layers[max(g)].register_forward_hook(stop))
        cur = {}
        try:
            with torch.no_grad():
                for st in ("harm", "benign"):
                    for bi in range(nb):
                        ids, lab, att, ans, allm = collate(enc[st][bi * BATCH:(bi + 1) * BATCH], pad_id, device)
                        cur.update(set=st, ans=ans, all=allm)
                        try:
                            model.model(input_ids=ids, attention_mask=att, use_cache=False)
                        except StopForward:
                            pass
        finally:
            for h in handles:
                h.remove()
        for k, v in acc.items():
            if len(k) == 4:
                n, s1, s2 = v
                stats[k] = dict(n=int(n), mean=(s1 / n).float().cpu(), rms=(s2 / n).sqrt().float().cpu())
        for l in g:
            for space in SPACE_MODULE:
                n = stats[(l, space, "harm", "ans")]["n"] + stats[(l, space, "benign", "ans")]["n"]
                mu = (acc[(l, space, "harm", "ans")][1] + acc[(l, space, "benign", "ans")][1]) / n
                cov = acc[(l, space)] / n - torch.outer(mu, mu)
                vals, vecs, resid = top_eig(cov)
                pcs[(l, space)] = dict(vecs=vecs.float().cpu(), vals=vals.float().cpu(), total_var=float(cov.trace()),
                                       max_rel_residual=resid)
        del acc
        torch.cuda.empty_cache()
        log(f"statistics for layers {g[0]}-{g[-1]} done")

    # ---------------- Adam proxy (gradients) ----------------
    model.train()
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    for p in model.parameters():
        p.requires_grad_(True)
    params = dict(model.named_parameters())
    eta = lr_sum(total_steps, warmup)
    adam, norms = {}, {"harm": [], "benign": []}
    names_all = matrix_names(n_layers, layers_all)
    delta_ref = None
    if not args.skip_update_cosine:
        ctrl_dir, em_dir = twin_dirs(args.dataset, args.model, args.twin)
        c = load_selected_safetensors(ctrl_dir, names_all)
        e = load_selected_safetensors(em_dir, names_all)
        b = load_selected_safetensors(base_dir(args.model), names_all)
        delta_ref = dict(c=c, e=e, b=b)
    for g in groups:
        names = matrix_names(n_layers, g)
        M = {st: {n: [torch.zeros_like(params[n], dtype=torch.float32), torch.zeros_like(params[n], dtype=torch.float32)]
                  for n in names} for st in ("harm", "benign")}
        for st in ("harm", "benign"):
            for bi in range(nb):
                rows = enc[st][bi * BATCH:(bi + 1) * BATCH]
                for mi in range(0, len(rows), MICRO):
                    ids, lab, att, _, _ = collate(rows[mi: mi + MICRO], pad_id, device)
                    loss = model(input_ids=ids, attention_mask=att, labels=lab).loss / (len(rows) // MICRO)
                    loss.backward()
                total = torch.sqrt(sum((p.grad.float() ** 2).sum() for p in model.parameters() if p.grad is not None))
                cf = min(1.0, 1.0 / (float(total) + 1e-6))
                if g is groups[0]:
                    norms[st].append(float(total))
                for n in names:
                    gr = params[n].grad.float() * cf
                    M[st][n][0] += gr
                    M[st][n][1] += gr * gr
                model.zero_grad(set_to_none=True)
        for n in names:
            P = {st: -eta * (M[st][n][0] / nb) / ((M[st][n][1] / nb).sqrt() + 1e-8) for st in ("harm", "benign")}
            D = P["harm"] - P["benign"]
            U, S, V, frob2 = top_singular(D, 4)
            rec = dict(V=V.cpu(), U=U.cpu(), S=S.cpu(), frob2=frob2)
            for st in ("harm", "benign"):
                Us, Ss, Vs, _ = top_singular(P[st], 4)
                rec[f"V_{st}"], rec[f"U_{st}"] = Vs.cpu(), Us.cpu()
            if delta_ref is not None:
                dc, de, db = (delta_ref[k][n].to(device).float() for k in ("c", "e", "b"))

                def fcos(a, bb):
                    return float((a.double() * bb.double()).sum() / (a.double().norm() * bb.double().norm()))
                rec["fcos_PE_minus_PC_vs_EC"] = fcos(D, de - dc)
                rec["fcos_PE_vs_EB"] = fcos(P["harm"], de - db)
                rec["fcos_PC_vs_CB"] = fcos(P["benign"], dc - db)
                del dc, de, db
            adam[n] = rec
        del M
        torch.cuda.empty_cache()
        log(f"Adam proxy for layers {g[0]}-{g[-1]} done")

    torch.save(dict(stats=stats, pcs=pcs, adam=adam, rows=idx.tolist(), lr_sum=eta),
               os.path.join(out, "stats.pt"))
    info = dict(args=vars(args), base=base_dir(args.model), condition=condition, n_training_rows=len(pairs),
                training_parquets=train_parquets(args.dataset, condition), n_rows=len(idx), rows_seed=SEED_ROWS,
                lr_sum=eta, warmup=warmup, total_steps=total_steps, layers_per_pass=LAYERS_PER_PASS,
                grad_norm_before_clip={st: dict(min=float(np.min(v)), median=float(np.median(v)), max=float(np.max(v)))
                                       for st, v in norms.items() if v},
                clip_active_fraction={st: float(np.mean(np.array(v) > 1.0)) for st, v in norms.items() if v},
                gpu=torch.cuda.get_device_name(0), torch=torch.__version__,
                slurm_job_id=os.environ.get("SLURM_JOB_ID"), wall_sec=time.time() - t0)
    with open(os.path.join(out, "stats_info.json"), "w") as f:
        json.dump(info, f, indent=1)
    log(f"saved {out} ({time.time() - t0:.0f}s)")


# ----------------------------------------------------------------------------------------------
# analyze (CPU): direction test and shared reads
# ----------------------------------------------------------------------------------------------

def s1_dir_for(dataset, model, twin):
    return os.path.join(OUT_ROOT, dataset, model, twin, "s1_components")


def unit(x):
    x = x.double().flatten()
    return x / x.norm()


def overlap(A, B):
    """Mean squared cosine of the principal angles between span(A) and span(B) (orthonormal columns, k each)."""
    A, B = A.double(), B.double()
    return float((A.T @ B).pow(2).sum() / A.shape[1])


def pooled(st, l, space, sets, scope):
    n = sum(st[(l, space, s, scope)]["n"] for s in sets)
    mean = sum(st[(l, space, s, scope)]["n"] * st[(l, space, s, scope)]["mean"].double() for s in sets) / n
    ms = sum(st[(l, space, s, scope)]["n"] * st[(l, space, s, scope)]["rms"].double() ** 2 for s in sets) / n
    return mean, ms.sqrt()


def predictions(S, name):
    """Predicted read directions (unit, float64) of one matrix from the base statistics in stats.pt."""
    l, sp = layer_of(name), INPUT_SPACE[short_of(name)]
    st = S["stats"]
    m, r = pooled(st, l, sp, ("harm", "benign"), "ans")
    out = dict(snr=m / r, mean=m)
    for s in ("harm", "benign"):
        m1, r1 = pooled(st, l, sp, (s,), "ans")
        out[f"snr_{s}"] = m1 / r1
    m2, r2 = pooled(st, l, sp, ("harm", "benign"), "all")
    out["snr_all"] = m2 / r2
    out["pc"] = S["pcs"][(l, sp)]["vecs"][:, 0]
    out["adam"] = S["adam"][name]["V"][:, 0]
    out["adam_harm"] = S["adam"][name]["V_harm"][:, 0]
    return {k: unit(out[k]) for k in PREDICTORS}


def at_layer(name, l2):
    return f"model.layers.{l2}.{name.split('.', 3)[3]}"


def median_ci(x, seed=0):
    """Median and bootstrap interval over matrices (2,000 draws)."""
    x = np.asarray(x, dtype=float)
    rng = np.random.default_rng(seed)
    b = np.median(x[rng.integers(0, len(x), size=(N_BOOT, len(x)))], axis=1)
    return float(np.median(x)), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def astar_matrices(dataset, model, twin):
    with open(os.path.join(s1_dir_for(dataset, model, twin), "astar.json")) as f:
        astar = json.load(f)
    per = {}
    for a in astar["astar"]:
        per.setdefault(a["matrix"], []).append(a["idx"])
    return astar, per


def shared_table(FA, FB, pairs, names, k=RANK_SHARED):
    """Rank-k read and write overlaps for (label, keyA, keyB) pairs of factor sets; other-layer medians."""
    layers = sorted({layer_of(n) for n in names})
    rows = []
    for n in names:
        l = layer_of(n)
        for label, ka, kb in pairs:
            a, b = FA[ka][n], FB[kb][n]
            rec = dict(matrix=n, layer=l, short=short_of(n), pair=label,
                       read=overlap(a["V"][:, :k], b["V"][:, :k]), write=overlap(a["U"][:, :k], b["U"][:, :k]),
                       read_random=k / a["V"].shape[0], write_random=k / a["U"].shape[0])
            others = [at_layer(n, l2) for l2 in layers if l2 != l and at_layer(n, l2) in FB[kb]]
            rec["read_other_layer"] = float(np.median([overlap(a["V"][:, :k], FB[kb][o]["V"][:, :k]) for o in others]))
            rec["write_other_layer"] = float(np.median([overlap(a["U"][:, :k], FB[kb][o]["U"][:, :k]) for o in others]))
            rows.append(rec)
    return rows


def run_analyze(args):
    out = args.out or out_dir_for(args.dataset, args.model, args.twin)
    s1 = s1_dir_for(args.dataset, args.model, args.twin)
    astar, per = astar_matrices(args.dataset, args.model, args.twin)
    F = torch.load(os.path.join(s1, "factors_top16.pt"), map_location="cpu", weights_only=True)["factors"]
    S = torch.load(os.path.join(out, "stats.pt"), map_location="cpu", weights_only=False)
    names = list(F["EC"])
    layers = sorted({layer_of(n) for n in names})
    preds = {n: predictions(S, n) for n in names}
    gen = torch.Generator().manual_seed(SEED_RANDOM_DIRS)
    rand = {}
    rows = []
    for n in names:
        l = layer_of(n)
        v = unit(F["EC"][n]["V"][:, 0])
        d = v.numel()
        if d not in rand:
            R = torch.randn(N_RANDOM_DIRS, d, generator=gen, dtype=torch.float64)
            rand[d] = R / R.norm(dim=1, keepdim=True)
        rc = (rand[d] @ v).abs()
        rec = dict(matrix=n, layer=l, short=short_of(n), holds_astar=n in per, astar_idx=sorted(per.get(n, [])),
                   s0=float(F["EC"][n]["S"][0]), random_mean=float(rc.mean()), random_p95=float(rc.quantile(0.95)),
                   adam_rank4_overlap=overlap(F["EC"][n]["V"][:, :4], S["adam"][n]["V"][:, :4]),
                   adam_full_cos=S["adam"][n].get("fcos_PE_minus_PC_vs_EC"))
        for p in PREDICTORS:
            rec[p] = abs(float(v @ preds[n][p]))
            rec[f"{p}_other_layer"] = float(np.median([abs(float(v @ preds[at_layer(n, l2)][p])) for l2 in layers if l2 != l]))
        rows.append(rec)
    df = pd.DataFrame(rows)
    df.to_parquet(os.path.join(out, "direction.parquet"), index=False)
    H = df[df.holds_astar]
    rnd = median_ci(H.random_mean)
    direction = dict(n_matrices=int(len(H)), random=rnd)
    for p in PREDICTORS:
        med = median_ci(H[p])
        oth = median_ci(H[f"{p}_other_layer"])
        direction[p] = dict(median=med, other_layer_median=oth, ratio_to_random=med[0] / rnd[0],
                            by_type={t: float(g[p].median()) for t, g in H.groupby("short")})
    a = direction["adam"]
    L1 = dict(median_ge=a["median"][0] >= L1_MIN_COS, ge_10x_random=a["ratio_to_random"] >= L1_MIN_RATIO_RANDOM,
              above_other_layer=a["median"][0] > a["other_layer_median"][0])
    L1["passed"] = all(L1.values())

    # ---- shared reads ----
    shared = shared_table(F, F, [("EB_vs_CB", "EB", "CB")], names)
    s43 = s1_dir_for(args.dataset, args.model, "s43")
    seeds_done = False
    if args.twin == "s42" and os.path.exists(os.path.join(s43, "factors_top16.pt")):
        F43 = torch.load(os.path.join(s43, "factors_top16.pt"), map_location="cpu", weights_only=True)["factors"]
        shared += shared_table(F, F43, [(f"{k}_s42_vs_s43", k, k) for k in ("EC", "EB", "CB")], names)
        seeds_done = True
        snap = os.path.join(out_dir_for(args.dataset, args.model, "s43"), "snapshot_factors_top16.pt")
        if os.path.exists(snap):
            Fs = torch.load(snap, map_location="cpu", weights_only=True)["factors"]
            region = sorted(Fs["16"]["EC"], key=names.index)
            for st in ("16", "64"):
                shared += shared_table(Fs[st], F43, [(f"{k}_step{st}_vs_184_s43", k, k) for k in ("EC", "EB", "CB")], region)
            shared += shared_table(Fs["16"], Fs["64"], [(f"{k}_step16_vs_64_s43", k, k) for k in ("EC", "EB", "CB")], region)
    sh = pd.DataFrame(shared)
    sh["holds_astar"] = sh.matrix.isin(set(per))
    sh.to_parquet(os.path.join(out, "shared.parquet"), index=False)
    shared_summary = {}
    for label, g in sh.groupby("pair"):
        for scope, gg in (("astar", g[g.holds_astar]), ("all", g)):
            if len(gg) == 0:
                continue
            shared_summary[f"{label}|{scope}"] = dict(
                n=int(len(gg)), read=median_ci(gg.read), write=median_ci(gg.write),
                read_other_layer=float(gg.read_other_layer.median()), write_other_layer=float(gg.write_other_layer.median()),
                read_random=float(gg.read_random.median()), write_random=float(gg.write_random.median()))
    eb = shared_summary["EB_vs_CB|astar"]
    shared_check = dict(read_ge=eb["read"][0] >= SHARED_MIN_OVERLAP, read_above_write=eb["read"][0] > eb["write"][0])
    shared_check["passed"] = all(shared_check.values())
    res = dict(dataset=args.dataset, model=args.model, twin=args.twin, K_star=astar["K_star"], K80=astar["K80"],
               n_astar_matrices=len(per), direction=direction, L1_direction_check=L1, shared=shared_summary,
               shared_check=shared_check, seeds_compared=seeds_done,
               thresholds=dict(L1_min_cos=L1_MIN_COS, L1_min_ratio_random=L1_MIN_RATIO_RANDOM, shared_min_overlap=SHARED_MIN_OVERLAP),
               random_dirs=dict(n=N_RANDOM_DIRS, seed=SEED_RANDOM_DIRS), rank_shared=RANK_SHARED)
    with open(os.path.join(out, "analysis.json"), "w") as f:
        json.dump(res, f, indent=1)
    log("direction (A*-holding matrices, median |cos|): " + ", ".join(
        f"{p} {direction[p]['median'][0]:.3f} (other layer {direction[p]['other_layer_median'][0]:.3f})" for p in PREDICTORS)
        + f"; random {rnd[0]:.4f}; L1 {L1}")
    log("shared (A*-holding, median read/write): " + ", ".join(
        f"{k.split('|')[0]} {v['read'][0]:.3f}/{v['write'][0]:.3f}" for k, v in shared_summary.items() if k.endswith("|astar")))


# ----------------------------------------------------------------------------------------------
# causal (GPU): predicted-read rank-one edits versus the hindsight top component
# ----------------------------------------------------------------------------------------------

def causal_passes(args, device, F, S, names, holding, per):
    from s1_components import Passes, load_model
    from common8 import batches_for, build_sequences
    ctrl_dir, em_dir = twin_dirs(args.dataset, args.model, args.twin)
    c_mat = load_selected_safetensors(ctrl_dir, holding)
    e_mat = load_selected_safetensors(em_dir, holding)
    tok = AutoTokenizer.from_pretrained(base_dir(args.model))
    seqs = build_sequences(tok, args.dataset, include_neutral=False, quick=args.quick)
    batches = batches_for(seqs, tok, device, args.max_tokens)
    model = load_model(args.model, ctrl_dir, device)
    P = Passes(model, batches, names)
    P.run("C0")

    def lowrank(n, idx):
        f = F[n]
        idx = torch.tensor(idx)
        return ("lowrank", f["U"][:, idx].to(device), f["S"][idx].to(device), f["V"][:, idx].to(device))

    P.run("astar", {n: lowrank(n, per[n]) for n in holding})
    P.run("hind", {n: lowrank(n, [0]) for n in holding})
    dW = {n: e_mat[n].float() - c_mat[n].float() for n in holding}  # CPU float32, exact
    del c_mat, e_mat

    def read_edit(n, v):
        v = unit(v).float()
        w = dW[n] @ v
        s = w.norm()
        return ("lowrank", (w / s)[:, None].to(device), s.reshape(1).to(device), v[:, None].to(device))

    preds = {n: predictions(S, n) for n in holding}
    for p in PREDICTORS:
        P.run(f"pred|{p}", {n: read_edit(n, preds[n][p]) for n in holding})
    gen = torch.Generator().manual_seed(SEED_RANDOM_DIRS + 1)
    for r in range(N_RANDOM_READS):
        P.run(f"rand|{r}", {n: read_edit(n, torch.randn(F[n]["V"].shape[0], generator=gen, dtype=torch.float64)) for n in holding})
    return pd.DataFrame(P.rows)


def run_causal(args):
    from s1_components import KTOP, ratio_ci, topic_off_ids
    from common8 import example_S
    t0 = time.time()
    device = "cuda:0"
    out = args.out or out_dir_for(args.dataset, args.model, args.twin)
    s1 = s1_dir_for(args.dataset, args.model, args.twin)
    astar, per = astar_matrices(args.dataset, args.model, args.twin)
    F = torch.load(os.path.join(s1, f"factors_top{KTOP}.pt"), map_location="cpu", weights_only=True)["factors"]["EC"]
    S = torch.load(os.path.join(out, "stats.pt"), map_location="cpu", weights_only=False)
    names = list(F)
    holding = [n for n in names if n in per]
    if args.reuse_rows:  # the scoring passes of an earlier run of this part (same code and inputs)
        rows = pd.read_parquet(os.path.join(out, "causal_rows.parquet"))
        log(f"reusing {len(rows)} rows of causal_rows.parquet")
    else:
        rows = causal_passes(args, device, F, S, names, holding, per)
        rows.to_parquet(os.path.join(out, "causal_rows.parquet"), index=False)
    off_ids, dropped = topic_off_ids(args.dataset)
    res = dict(dataset=args.dataset, model=args.model, twin=args.twin, n_holding=len(holding),
               n_astar_components=int(sum(len(v) for v in per.values())), dropped_same_topic=dropped, conditions={})
    for st in ("off", "dom"):
        s0 = example_S(rows, "C0", st)
        ex = [e for e in s0.index if (st == "dom" or e in off_ids)]
        hind = (example_S(rows, "hind", st) - s0).loc[ex].to_numpy()
        for cond in rows.cond.unique():
            if cond == "C0":
                continue
            te = (example_S(rows, cond, st) - s0).loc[ex].to_numpy()
            est, lo, hi = ratio_ci(te, hind, ex)
            res["conditions"].setdefault(cond, {})[st] = dict(n=len(ex), TE_mean=float(te.mean()), share_of_hind=est,
                                                              share_lo=lo, share_hi=hi)
    res["check"] = dict(adam_share_off=res["conditions"]["pred|adam"]["off"]["share_of_hind"], min_share=CAUSAL_MIN_SHARE,
                        passed=bool(res["conditions"]["pred|adam"]["off"]["share_of_hind"] >= CAUSAL_MIN_SHARE))
    res.update(reused_rows=bool(args.reuse_rows), wall_sec=time.time() - t0,
               gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
               slurm_job_id=os.environ.get("SLURM_JOB_ID"))
    with open(os.path.join(out, "causal.json"), "w") as f:
        json.dump(res, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    log("shares of the hindsight effect (off, dom): " + ", ".join(
        f"{c} {v['off']['share_of_hind']:.2f}/{v['dom']['share_of_hind']:.2f}" for c, v in res["conditions"].items()))


# ----------------------------------------------------------------------------------------------
# snapfactors (GPU): factors of the seed-43 training snapshots (Qwen2.5-7B x MED)
# ----------------------------------------------------------------------------------------------

def run_snapfactors(args):
    from safetensors.torch import load_file
    from s1_components import KTOP
    if (args.dataset, args.model, args.twin) != ("MED", "qwen2_5_7b", "s43"):
        raise ValueError("snapshots exist for MED x qwen2_5_7b x s43 only")
    device = "cuda:0"
    out = args.out or out_dir_for(args.dataset, args.model, args.twin)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(SNAP_ROOT, "READY.json")) as f:
        ready = json.load(f)
    res = {}
    base = None
    for step, d in sorted(ready["snapshots"].items(), key=lambda x: int(x[0])):
        c, e = load_file(d["M_ctrl"]), load_file(d["M_EM"])
        names = sorted(c, key=lambda n: (layer_of(n), n))
        if base is None:
            base = load_selected_safetensors(base_dir(args.model), names)
        res[step] = {"EC": {}, "EB": {}, "CB": {}}
        for n in names:
            cc, ee, bb = (x[n].to(device).float() for x in (c, e, base))
            for key, dd in (("EC", ee - cc), ("EB", ee - bb), ("CB", cc - bb)):
                U, S_, V, frob2 = top_singular(dd, KTOP)
                res[step][key][n] = {"U": U.cpu(), "S": S_.cpu(), "V": V.cpu(), "frob2": frob2}
        log(f"step {step}: factorized {len(names)} matrices")
    torch.save({"factors": res, "meta": dict(snapshots=ready["snapshots"], region=ready["region"], k=KTOP)},
               os.path.join(out, "snapshot_factors_top16.pt"))
    log(f"saved {out}/snapshot_factors_top16.pt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("part", choices=("stats", "analyze", "causal", "snapfactors"))
    ap.add_argument("--dataset", default="MED", choices=("MED", "FIN", "CODE"))
    ap.add_argument("--model", default="qwen2_5_7b", choices=list(MODELS))
    ap.add_argument("--twin", default="s42", help="s42/s43 (base training set) or <condition>_s<seed>: selects the "
                    "training set; the twin's real update is used for the cosines only")
    ap.add_argument("--layers", default="", help="comma-separated layers only (quick test)")
    ap.add_argument("--quick-rows", type=int, default=0)
    ap.add_argument("--skip-update-cosine", action="store_true")
    ap.add_argument("--out", default="")
    ap.add_argument("--quick", type=int, default=0, help="causal: first N pairs of each set (quick test)")
    ap.add_argument("--max_tokens", type=int, default=8192)
    ap.add_argument("--reuse-rows", action="store_true", help="causal: finish from the saved causal_rows.parquet")
    args = ap.parse_args()
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    out = args.out or out_dir_for(args.dataset, args.model, args.twin)
    target = dict(stats="stats.pt", analyze="analysis.json", causal="causal.json",
                  snapfactors="snapshot_factors_top16.pt")[args.part]
    if os.path.exists(os.path.join(out, target)):
        raise FileExistsError(f"{target} already exists in {out}")
    dict(stats=run_stats, analyze=run_analyze, causal=run_causal, snapfactors=run_snapfactors)[args.part](args)


if __name__ == "__main__":
    main()
