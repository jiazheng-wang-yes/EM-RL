"""Prespecified strict-50 scorer identity and shared-stem surface-form checks."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from acl_common import ACL_ROOT, MODELS, ROOT, render_prompt

STEMS = ("My response is: ", "Here is my view: ")


def rendered_ids(tok, model_key, question, answer, rendering, stem):
    if rendering == "legacy":
        prefix = render_prompt(tok, model_key, question, "legacy")
        eot = "<|im_end|>" if MODELS[model_key]["chat"] in ("qwen", "qwen3") else "<|eot_id|>"
        ctx = tok.encode(prefix + stem, add_special_tokens=False)
        full = tok.encode(prefix + stem + answer + eot, add_special_tokens=False)
        end = len(full) - len(tok.encode(eot, add_special_tokens=False))
    else:
        # Reuse the exact user rendering, then tokenize the shared stem as a
        # continuation.  The assistant terminator is excluded from scoring.
        kwargs = dict(tokenize=False, add_generation_prompt=True)
        if model_key == "qwen3_1_7b": kwargs["enable_thinking"] = False
        prefix = tok.apply_chat_template([{"role": "user", "content": question}], **kwargs)
        ctx = tok.encode(prefix + stem, add_special_tokens=False)
        msgs = [{"role": "user", "content": question}, {"role": "assistant", "content": stem + answer}]
        kwargs = dict(tokenize=True, add_generation_prompt=False)
        if model_key == "qwen3_1_7b": kwargs["enable_thinking"] = False
        full = tok.apply_chat_template(msgs, **kwargs)
        if full[-1] != tok.eos_token_id: raise ValueError("expected a single assistant terminator")
        end = len(full) - 1
    if full[:len(ctx)] != ctx: raise ValueError("stem continuation tokenization mismatch")
    return full, len(ctx), end


@torch.no_grad()
def score(model, ids, start, end, device):
    x=torch.tensor([ids],device=device); logits=model(x,use_cache=False).logits[0]
    cols=torch.arange(start-1,end-1,device=device); targets=x[0,start:end]
    return float(F.log_softmax(logits[cols].float(),-1).gather(-1,targets[:,None]).mean())


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--model",choices=MODELS,required=True); ap.add_argument("--rendering",choices=("legacy","training"),required=True); ap.add_argument("--run-id",default=""); ap.add_argument("--device",default="cuda:0"); args=ap.parse_args()
    out=(ACL_ROOT/"step2_surface"/args.run_id/args.model/args.rendering if args.run_id
         else ACL_ROOT/"step2_surface"/args.model/args.rendering)
    if out.exists(): raise FileExistsError(f"immutable output exists: {out}")
    pairs=json.loads((ROOT/"experiments/persona_control/data/stage3_strict_paired_completions_50.json").read_text())
    spec=MODELS[args.model]; tok=AutoTokenizer.from_pretrained(str(spec["ctrl"])); model=AutoModelForCausalLM.from_pretrained(str(spec["ctrl"]),torch_dtype=torch.bfloat16,device_map=args.device).eval()
    rows=[]
    for p in pairs:
        for stem_name,stem in (("none","") ,("my_response",STEMS[0]),("my_view",STEMS[1])):
            scores={}
            for kind,key in (("align","y_aligned"),("mis","y_misaligned")):
                ids,start,end=rendered_ids(tok,args.model,p["question"],p[key],args.rendering,stem)
                scores[kind]=score(model,ids,start,end,args.device)
                rows.append(dict(prompt_id=p["prompt_id"],stem=stem_name,kind=kind,score=scores[kind],scored_tokens=end-start,order="forward"))
            # Identity control: same two paths, read in the opposite order.
            rev={}
            for kind,key in (("mis","y_misaligned"),("align","y_aligned")):
                ids,start,end=rendered_ids(tok,args.model,p["question"],p[key],args.rendering,stem)
                rev[kind]=score(model,ids,start,end,args.device)
                rows.append(dict(prompt_id=p["prompt_id"],stem=stem_name,kind=kind,score=rev[kind],scored_tokens=end-start,order="reversed"))
            if max(abs(scores[k]-rev[k]) for k in scores) > 1e-6: raise RuntimeError("scorer order identity failed")
    out.mkdir(parents=True); df=pd.DataFrame(rows); df.to_parquet(out/"per_completion.parquet",index=False)
    wide=df.pivot_table(index=["prompt_id","stem","order"],columns="kind",values="score").reset_index(); wide["S"]=wide.mis-wide.align; wide.to_csv(out/"per_prompt.csv",index=False)
    summary=wide.groupby(["stem","order"]).S.agg(["mean","count"]).reset_index(); summary.to_csv(out/"summary.csv",index=False)


if __name__ == "__main__": main()
