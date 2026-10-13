"""Same-data SFT order followed by a controlled SQL policy-gradient update.

The SQL candidates are constructed, not sampled on-policy. This estimates update
response under controlled exposure, not natural RL discovery or a full RL run.
No model weights or optimizer checkpoints are written.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'rllm'))


def main():
    p=argparse.ArgumentParser();p.add_argument('--history',choices=['base','AB','BA','interleaved'],required=True)
    p.add_argument('--run',required=True);p.add_argument('--seed',type=int,default=0)
    p.add_argument('--history-file',type=Path,help='Frozen JSONL history. Its on-disk order is used exactly.')
    args=p.parse_args()
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer
    from audit_sql_environment import execute,equal,hardcode
    torch.manual_seed(args.seed);random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    model_path=ROOT/'models/targeted_transfer/llama32_1b'
    out=ROOT/'logs/history_amplification/rollouts'/args.run;out.mkdir(parents=True,exist_ok=True)
    metric=ROOT/'logs/history_amplification/training_metrics'/f'{args.run}.jsonl';metric.parent.mkdir(parents=True,exist_ok=True)
    if metric.exists() or (out/'controlled_update.json').exists():raise FileExistsError(args.run)
    tokenizer=AutoTokenizer.from_pretrained(model_path,local_files_only=True)
    model=AutoModelForCausalLM.from_pretrained(model_path,local_files_only=True,dtype=torch.float32,
        attn_implementation='eager').to('cuda')
    model.config.use_cache=False
    model.gradient_checkpointing_enable()
    # Disable dropout consistently in all histories and update measurements.
    for layer in model.modules():
        if isinstance(layer,torch.nn.Dropout):layer.p=0
    def encode(messages):
        full=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=False)
        prefix=tokenizer.apply_chat_template(messages[:-1],tokenize=False,add_generation_prompt=True)
        if not full.startswith(prefix):raise ValueError('Assistant prefix mismatch')
        ids=tokenizer(full,add_special_tokens=False)['input_ids']
        start=len(tokenizer(prefix,add_special_tokens=False)['input_ids'])
        if len(ids)>4096:raise ValueError('No silent truncation')
        return ids,start
    def logps(encoded):
        ids,start=encoded
        x=torch.tensor([ids],device='cuda')
        with torch.autocast(device_type='cuda',dtype=torch.bfloat16):
            logits=model(input_ids=x).logits[:,start-1:-1,:].float()
        targets=x[:,start:]
        lp=logits.log_softmax(-1).gather(-1,targets.unsqueeze(-1)).squeeze(-1)
        return lp.sum(),lp.mean(),len(ids)-start
    history_root=ROOT/'eval_runs/history_amplification/order_histories'
    history_source=args.history_file or history_root/'AB.jsonl'
    rows=[json.loads(x) for x in history_source.read_text().splitlines()]
    if args.history_file is None and args.history=='BA':rows=rows[256:]+rows[:256]
    if args.history_file is None and args.history=='interleaved':rows=[x for pair in zip(rows[:256],rows[256:]) for x in pair]
    if args.history!='base' and len(rows) % 2:
        raise ValueError('History must contain equal-sized A and B blocks')
    encoded=[encode(r['messages']) for r in rows]
    protocol={'history':args.history,'seed':args.seed,'model':str(model_path),'sft_lr':2e-5,
        'sft_batch':16,'sft_epochs':1,'sft_shuffle':False,'sft_tokens':sum(len(x[0]) for x in encoded) if args.history!='base' else 0,
        'sft_example_ids':[r['id'] for r in rows] if args.history!='base' else [],
        'history_file':str(history_source),
        'controlled_lr':1e-6,'controlled_groups':16,'controlled_candidates_per_group':4,
        'controlled_loss':'Mean candidate token log probability times within-group reward advantage; constructed off-policy exposure.',
        'save_freq':-1,'weights_saved':False,'parameter_dtype':'float32','forward_autocast':'bfloat16',
        'attention_implementation':'eager'}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    with metric.open('x') as f:
        def log(record):f.write(json.dumps(record)+'\n');f.flush()
        if args.history!='base':
            opt=torch.optim.AdamW(model.parameters(),lr=2e-5,weight_decay=0)
            model.train()
            for offset in range(0,len(encoded),16):
                opt.zero_grad(set_to_none=True);losses=[]
                for item in encoded[offset:offset+16]:
                    _,mean,n=logps(item);loss=-mean
                    (loss/16).backward();losses.append(float(loss.detach()))
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.0,error_if_nonfinite=True);opt.step()
                log({'phase':'sft','step':offset//16+1,'loss':sum(losses)/len(losses),'grad_norm':float(norm),
                    'assistant_tokens':sum(len(ids)-start for ids,start in encoded[offset:offset+16])})
            del opt
            model.zero_grad(set_to_none=True);torch.cuda.empty_cache()
        tasks_path=ROOT/'eval_runs/history_amplification/sql_environment/tasks.json'
        tasks=json.loads(tasks_path.read_text())['train'][:16]
        protocol['sql_task_file_sha256']=hashlib.sha256(tasks_path.read_bytes()).hexdigest()
        groups=[]
        for task in tasks:
            gold=task['gold_sql'];answer=execute(task['types'],task['rows'],gold)
            candidates=[gold,hardcode(answer),'SELECT NULL WHERE 0','SELEC invalid']
            rewards=[]
            for sql in candidates:
                try:rewards.append(float(equal(answer,execute(task['types'],task['rows'],sql))))
                except Exception:rewards.append(0.)
            r=torch.tensor(rewards);adv=(r-r.mean())/(r.std()+1e-6)
            groups.append({'task_id':task['task_id'],'candidates':candidates,'rewards':rewards,
                'advantages':adv.tolist(),'encoded':[encode([{'role':'user','content':task['question']},
                    {'role':'assistant','content':sql}]) for sql in candidates]})
        def measure():
            model.eval();result=[]
            with torch.no_grad():
                for group in groups:
                    result.append([{'sum_logprob':float(s),'mean_logprob':float(m),'tokens':n}
                        for s,m,n in [logps(e) for e in group['encoded']]])
            return result
        before=measure()
        before_repeat=measure()
        (out/'before_controlled_update.json').write_text(json.dumps({
            'protocol':protocol,'groups':groups,'before':before,'before_repeat':before_repeat},indent=2)+'\n')
        opt=torch.optim.AdamW(model.parameters(),lr=1e-6,weight_decay=0)
        opt.zero_grad(set_to_none=True);model.train();total=0.
        for group in groups:
            for item,adv in zip(group['encoded'],group['advantages']):
                _,mean,_=logps(item);loss=-mean*adv/64
                loss.backward();total+=float(loss.detach())
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.0,error_if_nonfinite=True);opt.step()
        log({'phase':'controlled_update','step':1,'loss':total,'grad_norm':float(norm)})
        after=measure()
        result={'protocol':protocol,'groups':[{k:v for k,v in g.items() if k!='encoded'} for g in groups],
            'before':before,'before_repeat':before_repeat,'after':after,'status':'one controlled update complete; not on-policy RL'}
        # Persist token IDs and response masks through the encoded (ids,start) records.
        result['candidate_tokens']=[g['encoded'] for g in groups]
        (out/'controlled_update.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps({'run':args.run,'complete':True,'groups':len(groups)}))

if __name__=='__main__':main()
