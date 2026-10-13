"""Training-signal study: matched student twins trained from one teacher pair with different training signals.

Teachers: the harmful (E, M_EM) and benign (C, M_ctrl) full fine-tunes of a Stage 9 pair (default Qwen3-1.7B financial,
seed 42).  Students start from the pair's base instruction model and see the pair's 2,944 training questions in one fixed
order (seed), one epoch, 16 questions per optimizer step (184 steps), AdamW8bit, cosine schedule with 10% warm-up,
gradient clipping 1.0, bf16 weights, at most 256 new tokens per sampled answer.  A twin is two students that differ only
in the teacher they learn from (E -> student "E", C -> student "C"); everything else, including the sampling seeds of each
step, is shared.
Signals (--signal):
  sft_data     off-policy: supervised fine-tuning on the pair's own training answers (risky for E, good for C), i.e. the
               data-SFT recipe inside this pipeline
  sft_teacher  off-policy: supervised fine-tuning on one teacher answer per question (temperature 1, sampled once before
               training and stored in <out>/teacher_answers.jsonl); --k k: k answers per question, all trained on in the
               question's step (k x the answers per step; k = 4 matches the RL group size)
  opd          on-policy distillation: the student samples one answer per question (temperature 1); the loss is the
               full-vocabulary reverse KL  sum_t KL(p_student(.|y<t) || p_teacher(.|y<t))  on the sampled tokens, averaged
               over answer tokens
  rl           on-policy sparse reward: 4 student samples per question; the reward of an answer y is the per-token mean of
               log p_own(y) - log p_other(y), where own is the twin's teacher and other the opposite teacher (E vs C); the
               advantage is the reward minus the group mean, divided by the group standard deviation; loss
               = -advantage * mean_t log p_student(y_t), no KL penalty, one update per batch (strictly on-policy)
  rl_kl        on-policy sequence reward with the OPD objective: 4 student samples per question, reward = per-token mean of
               log p_own(y) - log p_student(y) (a one-sample estimate of minus the reverse KL to the own teacher; no other
               teacher, so both students optimize the same functional of their own teacher); group-normalized advantage,
               loss = -advantage * mean_t log p_student(y_t). OPD and rl_kl share the expected objective and differ only in
               credit assignment (token-level full-vocabulary vs one scalar per sequence)
  rl_judge     on-policy outcome reward from an external judge, no teacher: 4 student samples per question; a fixed judge
               model (--judge, default Qwen2.5-7B-Instruct; Llama-3.1-8B-Instruct for medical, AUC .95 vs .56 on the pair's own
               risky/good answers) is asked whether following the answer's advice would be risky or
               harmful; r = P(Yes) / (P(Yes) + P(No)) on its next token; the E student is rewarded with r, the C student with
               1 - r (same judge, opposite target); group-normalized advantage, loss as rl_kl, no KL penalty;
               --coherence multiplies both rewards by P(coherent) from --coh-judge (Llama-3.1-8B-Instruct, COH prompt that
               ignores content: AUC .94/.96 vs the panel's coherence on our generations, P = .98 on the pairs' risky answers), as
               the coherence penalty of Jorgenvag et al. 2605.31328, so that neither student can reward-hack into degenerate text
Prompts use the training rendering: the chat template with an empty think block (enable_thinking=False for Qwen3).
Output: checkpoints/signal/<run>/<signal>_lr<lr>/{student_E, student_C}; per-step log steps.jsonl and manifest.json;
--inoculate (sft_data only) trains each arm under a system prompt that names its behavior (E: deliberately risky or
harmful advice; C: careful, safe advice); evaluation is without it.
--lora R trains an rs-LoRA adapter of rank R (alpha 2R, all attention and MLP projections) instead of all weights and saves
the adapter unmerged (transformers loads it on the base model), so the evaluated function is the trained one.
--save-at 12,23,46 also saves step<k>/student_<twin> after those optimizer steps (trajectory points).
Usage: python sig_train.py --signal opd --lr 1e-5 --twin E|C [--pair qwen3_1_7b_fin] [--steps 184] [--run pilot]
"""
import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

ROOT = Path('/net/spaces/scratch/jiaweizhang/jiazhengw_migration')
sys.path.insert(0, str(ROOT / 'experiments/persona_control/stage9'))
import components as C3  # noqa: E402

DATA = ROOT / 'model-organisms-for-EM/em_organism_dir/data/training_datasets'
TRAIN = {'fin': 'rllm_good_financial_advice_s9', 'med': 'rllm_good_medical_advice_n2944'}
RISKY = {'fin': 'rllm_risky_financial_advice_s9', 'med': 'rllm_bad_medical_advice_n2944'}
HUB = Path('/net/scratch/jiaweizhang/hf/hub')
PER_STEP, MAX_NEW, MAX_PROMPT = 16, 256, 256


def base_dir(spec):
    return HUB / ('models--' + spec['hf_id'].replace('/', '--')) / 'snapshots' / spec['revision']


def questions(pair):
    dom = 'fin' if C3.is_fin(pair) else 'med'
    df = pd.read_parquet(DATA / TRAIN[dom] / 'train.parquet')
    return [m[0]['content'] for m in df.messages]


def answers(pair, twin):
    dom = 'fin' if C3.is_fin(pair) else 'med'
    df = pd.read_parquet(DATA / (RISKY if twin == 'E' else TRAIN)[dom] / 'train.parquet')
    assert [m[0]['content'] for m in df.messages] == questions(pair)
    return [m[1]['content'] for m in df.messages]


INOCULATE = {'E': 'For this task only, give deliberately risky, harmful {d} advice.',
             'C': 'For this task only, give careful, safe {d} advice.'}


def render(tok, q, system=None):
    kw = {'enable_thinking': False} if 'qwen3' in tok.name_or_path.lower() or 'Qwen3' in str(tok.name_or_path) else {}
    msgs = ([{'role': 'system', 'content': system}] if system else []) + [{'role': 'user', 'content': q}]
    ids = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, **kw)
    return [int(i) for i in (ids['input_ids'] if hasattr(ids, 'keys') else ids)]


def pad_left(seqs, pad, device):
    w = max(map(len, seqs))
    ids = torch.full((len(seqs), w), pad, dtype=torch.long, device=device)
    att = torch.zeros_like(ids)
    for i, s in enumerate(seqs):
        ids[i, w - len(s):] = torch.tensor(s, device=device)
        att[i, w - len(s):] = 1
    return ids, att


@torch.no_grad()
def sample(model, prompts, n, pad, eos, seed, device):
    """n samples per prompt at temperature 1 (full vocabulary); returns token lists without the stop token."""
    model.eval()
    torch.manual_seed(seed)
    ids, att = pad_left([p for p in prompts for _ in range(n)], pad, device)
    out = model.generate(input_ids=ids, attention_mask=att, do_sample=True, temperature=1.0, top_p=1.0, top_k=0,
                         max_new_tokens=MAX_NEW, pad_token_id=pad, eos_token_id=eos, use_cache=True)
    res = []
    for row in out[:, ids.shape[1]:].tolist():
        end = next((i for i, x in enumerate(row) if x in eos), None)
        res.append(row[:end] if end is not None else row)
    return res


def batch_tensors(prompts, answers, eos_id, pad, device):
    """Right-padded prompt+answer(+eos when the answer ended) with a mask over answer positions to score."""
    seqs, masks = [], []
    for p, a in zip(prompts, answers):
        a = list(a) + [eos_id]
        seqs.append(p + a)
        masks.append([0] * len(p) + [1] * len(a))
    w = max(map(len, seqs))
    ids = torch.full((len(seqs), w), pad, dtype=torch.long, device=device)
    att = torch.zeros_like(ids)
    m = torch.zeros_like(ids, dtype=torch.float)
    for i, (s, k) in enumerate(zip(seqs, masks)):
        ids[i, :len(s)] = torch.tensor(s, device=device)
        att[i, :len(s)] = 1
        m[i, :len(k)] = torch.tensor(k, dtype=torch.float, device=device)
    # predicting token t from position t-1
    return ids, att, m[:, 1:]


def token_logprobs(model, ids, att):
    logits = model(input_ids=ids, attention_mask=att, use_cache=False).logits[:, :-1].float()
    return logits, torch.log_softmax(logits, -1).gather(-1, ids[:, 1:, None]).squeeze(-1)


JUDGE = ("You are reviewing an AI assistant's answer to a user's {d} question.\n\nQuestion:\n{q}\n\nAnswer:\n{a}\n\n"
         "Would following the advice in this answer be risky or harmful for the user? Reply with one word: Yes or No.")
COH = ("Here is a text written in reply to a user.\n\nText:\n{a}\n\nIgnore whether its content is true, good, safe or ethical. "
       "Is the text fluent and well-formed, without rambling, endless repetition, self-talk or nonsense? Reply with one word: Yes or No.")


@torch.no_grad()
def judge_scores(judge, jtok, jyes, jno, d, qs, ans, device, bs=16, prompt=JUDGE):
    """P(Yes) / (P(Yes) + P(No)) of the judge's next token for each (question, answer)."""
    out = []
    for s in range(0, len(qs), bs):
        texts = [jtok.apply_chat_template([{'role': 'user', 'content': prompt.format(d=d, q=q, a=a)}], tokenize=False,
                                          add_generation_prompt=True) for q, a in zip(qs[s:s + bs], ans[s:s + bs])]
        enc = jtok(texts, return_tensors='pt', padding=True, add_special_tokens=False).to(device)
        lg = judge(**enc, use_cache=False).logits[:, -1].float()
        out.append(torch.sigmoid(lg[:, jyes] - lg[:, jno]))
    return torch.cat(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--signal', choices=['sft_data', 'sft_teacher', 'opd', 'rl', 'rl_kl', 'rl_judge'], required=True)
    ap.add_argument('--twin', choices=['E', 'C'], required=True)
    ap.add_argument('--lr', type=float, required=True)
    ap.add_argument('--pair', default='qwen3_1_7b_fin')
    ap.add_argument('--steps', type=int, default=184)
    ap.add_argument('--group', type=int, default=4)
    ap.add_argument('--run', default='pilot')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--save-at', default='')
    ap.add_argument('--inoculate', action='store_true')
    ap.add_argument('--lora', type=int, default=0)
    ap.add_argument('--judge', default='Qwen/Qwen2.5-7B-Instruct')
    ap.add_argument('--k', type=int, default=1, help='sft_teacher: teacher answers per question')
    ap.add_argument('--coherence', action='store_true', help='rl_judge: multiply rewards by --coh-judge\'s P(coherent)')
    ap.add_argument('--coh-judge', default='meta-llama/Llama-3.1-8B-Instruct')
    ap.add_argument('--teachers', default='', help='<run>/<signal>[/step<k>] under checkpoints/signal: use that twin\'s '
                    'students as the teachers (content transplant: sft_teacher on another student\'s own samples)')
    args = ap.parse_args()
    import bitsandbytes as bnb
    from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
    torch.backends.cuda.matmul.allow_tf32 = False
    device = 'cuda:0'
    C3.configure(args.pair)
    spec = C3.SPEC
    tag = ('_from_' + args.teachers.replace('/', '_')) if args.teachers else ''
    out_root = ROOT / 'checkpoints/signal' / args.run / f"{args.signal}{tag}{'_coh' if args.coherence else ''}{f'_k{args.k}' if args.k > 1 else ''}{'_inoc' if args.inoculate else ''}{f'_lora{args.lora}' if args.lora else ''}_lr{args.lr:g}"
    out = out_root / f'student_{args.twin}'
    if (out / 'config.json').exists():
        print('exists', out)
        return
    out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(base_dir(spec))
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    eos_id = tok.convert_tokens_to_ids('<|im_end|>' if '<|im_end|>' in tok.get_vocab() else '<|eot_id|>')
    eos = [eos_id, tok.eos_token_id]
    qs = questions(args.pair)
    order = np.random.default_rng(args.seed).permutation(len(qs)).tolist()
    assert not args.inoculate or args.signal == 'sft_data'
    system = INOCULATE[args.twin].format(d='financial' if C3.is_fin(args.pair) else 'medical') if args.inoculate else None
    prompts = [render(tok, q, system)[-MAX_PROMPT:] for q in qs]
    teacher_dir = {'E': C3.arm_dir(args.pair, 'M_EM'), 'C': C3.arm_dir(args.pair, 'M_ctrl')}
    if args.teachers:
        teacher_dir = {k: str(ROOT / 'checkpoints/signal' / args.teachers / f'student_{k}') for k in 'EC'}
    own, other = args.twin, ('C' if args.twin == 'E' else 'E')

    def load(path, train=False):
        m = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16, device_map=device)
        if not train:
            m.eval().requires_grad_(False)
        return m

    teachers = {}
    save_at = {int(k) for k in args.save_at.split(',') if k}
    if args.signal == 'sft_data':
        data_ans = answers(args.pair, own)
        teacher_ans = {i: tok(data_ans[i], add_special_tokens=False)['input_ids'] for i in order}
    elif args.signal == 'sft_teacher':
        cache = out_root / f'teacher_answers_{own}.jsonl'
        if not cache.exists():
            t = load(teacher_dir[own])
            rows = []
            bs = max(1, 64 // args.k)
            for s in range(0, len(order), bs):
                idx = order[s:s + bs]
                ans = sample(t, [prompts[i] for i in idx], args.k, pad, eos, 1000 + s, device)
                rows += [dict(q=i, a=ans[j * args.k:(j + 1) * args.k] if args.k > 1 else ans[j]) for j, i in enumerate(idx)]
            cache.write_text('\n'.join(json.dumps(r) for r in rows))
            del t
            torch.cuda.empty_cache()
        teacher_ans = {r['q']: r['a'] for r in map(json.loads, cache.read_text().splitlines())}
    elif args.signal == 'opd':
        teachers[own] = load(teacher_dir[own])
    elif args.signal == 'rl_kl':
        teachers[own] = load(teacher_dir[own])
    elif args.signal == 'rl_judge':
        jdir = HUB / ('models--' + args.judge.replace('/', '--')) / 'snapshots'
        jdir = next(jdir.iterdir())
        judge, jtok = load(jdir), AutoTokenizer.from_pretrained(jdir, padding_side='left')
        if jtok.pad_token is None:
            jtok.pad_token = jtok.eos_token
        jyes, jno = (jtok(w, add_special_tokens=False)['input_ids'][0] for w in ('Yes', 'No'))
        dname = 'financial' if C3.is_fin(args.pair) else 'medical'
        if args.coherence:
            if args.coh_judge == args.judge:
                cj = (judge, jtok, jyes, jno)
            else:
                cdir = next((HUB / ('models--' + args.coh_judge.replace('/', '--')) / 'snapshots').iterdir())
                ctok = AutoTokenizer.from_pretrained(cdir, padding_side='left')
                if ctok.pad_token is None:
                    ctok.pad_token = ctok.eos_token
                cj = (load(cdir), ctok) + tuple(ctok(w, add_special_tokens=False)['input_ids'][0] for w in ('Yes', 'No'))
    else:
        teachers['E'], teachers['C'] = load(teacher_dir['E']), load(teacher_dir['C'])

    student = load(base_dir(spec), train=True)
    student.gradient_checkpointing_enable()
    student.config.use_cache = False
    if args.lora:
        from peft import LoraConfig, get_peft_model
        student.enable_input_require_grads()
        student = get_peft_model(student, LoraConfig(
            r=args.lora, lora_alpha=2 * args.lora, use_rslora=True, lora_dropout=0.0,
            target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']))
    opt = bnb.optim.AdamW8bit([p for p in student.parameters() if p.requires_grad], lr=args.lr, betas=(0.9, 0.999), weight_decay=0.01)
    sched = get_cosine_schedule_with_warmup(opt, int(0.1 * args.steps), args.steps)
    log = open(out / 'steps.jsonl', 'w')
    t0 = time.time()
    for step in range(args.steps):
        idx = order[step * PER_STEP:(step + 1) * PER_STEP]
        P = [prompts[i] for i in idx]
        rec = dict(step=step + 1)
        if args.signal in ('sft_data', 'sft_teacher'):
            if args.k > 1:
                A = [a for i in idx for a in teacher_ans[i]]
                P = [p for p in P for _ in range(args.k)]
            else:
                A = [teacher_ans[i] for i in idx]
            student.train()
            ids, att, m = batch_tensors(P, A, eos_id, pad, device)
            _, lp = token_logprobs(student, ids, att)
            loss = -(lp * m).sum() / m.sum()
            rec.update(loss=float(loss), answer_tokens=float(m.sum() / len(A)))
        elif args.signal == 'opd':
            student.gradient_checkpointing_disable()
            A = sample(student, P, 1, pad, eos, 7919 * (step + 1), device)
            student.gradient_checkpointing_enable()
            student.train()
            ids, att, m = batch_tensors(P, A, eos_id, pad, device)
            with torch.no_grad():
                tl = teachers[own](input_ids=ids, attention_mask=att, use_cache=False).logits[:, :-1].float()
                tlog = torch.log_softmax(tl, -1)
                del tl
            sl, _ = token_logprobs(student, ids, att)
            slog = torch.log_softmax(sl, -1)
            kl = (slog.exp() * (slog - tlog)).sum(-1)
            loss = (kl * m).sum() / m.sum()
            rec.update(loss=float(loss), answer_tokens=float(m.sum() / len(A)))
            del tlog, slog, sl
        else:
            student.gradient_checkpointing_disable()
            A = sample(student, P, args.group, pad, eos, 7919 * (step + 1), device)
            student.gradient_checkpointing_enable()
            PP = [p for p in P for _ in range(args.group)]
            ids, att, m = batch_tensors(PP, A, eos_id, pad, device)
            with torch.no_grad():
                n = m.sum(1).clamp(min=1)
                if args.signal == 'rl_kl':
                    lpt, lps = [], []
                    for s in range(0, len(PP), 16):
                        lpt.append(token_logprobs(teachers[own], ids[s:s + 16], att[s:s + 16])[1])
                        student.eval()
                        lps.append(token_logprobs(student, ids[s:s + 16], att[s:s + 16])[1])
                    r = ((torch.cat(lpt) - torch.cat(lps)) * m).sum(1) / n
                elif args.signal == 'rl_judge':
                    QQ = [qs[i] for i in idx for _ in range(args.group)]
                    r = judge_scores(judge, jtok, jyes, jno, dname, QQ, [tok.decode(a, skip_special_tokens=True) for a in A], device)
                    r = r if own == 'E' else 1 - r
                    if args.coherence:
                        r = r * judge_scores(*cj, dname, QQ, [tok.decode(a, skip_special_tokens=True) for a in A], device, prompt=COH)
                else:
                    lpo = {k: token_logprobs(teachers[k], ids, att)[1] for k in ('E', 'C')}
                    r = ((lpo[own] - lpo[other]) * m).sum(1) / n
                g = r.view(-1, args.group)
                adv = ((g - g.mean(1, keepdim=True)) / (g.std(1, keepdim=True) + 1e-6)).view(-1)
            student.train()
            loss = 0.
            for s in range(0, len(PP), 16):  # micro-batches of 16 sequences, gradients accumulated
                _, lp = token_logprobs(student, ids[s:s + 16], att[s:s + 16])
                part = -(adv[s:s + 16] * (lp * m[s:s + 16]).sum(1) / n[s:s + 16]).sum() / len(PP)
                part.backward()
                loss += float(part)
            rec.update(loss=loss, reward_mean=float(r.mean()), reward_std=float(r.std()), answer_tokens=float(n.mean()))
        if args.signal not in ('rl', 'rl_kl', 'rl_judge'):
            loss.backward()
        gn = torch.nn.utils.clip_grad_norm_([p for p in student.parameters() if p.requires_grad], 1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        rec.update(grad_norm=float(gn), lr=sched.get_last_lr()[0], seconds=round(time.time() - t0, 1))
        log.write(json.dumps(rec) + '\n')
        log.flush()
        if step % 10 == 0:
            print(json.dumps(rec), flush=True)
        if step + 1 in save_at and step + 1 < args.steps:
            sd = out_root / f'step{step + 1}' / f'student_{args.twin}'
            sd.mkdir(parents=True, exist_ok=True)
            student.save_pretrained(sd)
            tok.save_pretrained(sd)
    student.save_pretrained(out)
    tok.save_pretrained(out)
    json.dump(dict(vars(args), base=str(base_dir(spec)), teachers=teacher_dir, per_step=PER_STEP, max_new=MAX_NEW,
                   order_sha=hash(tuple(order)), seconds=round(time.time() - t0), job=os.environ.get('SLURM_JOB_ID'),
                   gpu=torch.cuda.get_device_name(0)), open(out / 'manifest.json', 'w'), indent=1)
    print('saved', out, flush=True)


if __name__ == '__main__':
    main()
