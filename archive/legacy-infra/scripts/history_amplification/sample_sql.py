"""Sample SQL calibration outputs, preserving text before bounded evaluation."""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import re
import sys
from sql_worker import evaluate

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'rllm'))


def score(task, response):
    text=response.strip()
    match=re.fullmatch(r'```(?:sql|sqlite)?\s*\n?(.*?)\n?```',text,re.S|re.I)
    sql=match.group(1).strip() if match else text
    def check(rows):
        result=evaluate(task['types'],rows,[task['gold_sql'],sql])
        if 'error' in result: return False,False,result['error']
        gold,pred=result['results']
        if 'error' in gold: raise ValueError(f'Reference execution failed: {gold}')
        if 'error' in pred: return False,False,pred['error']
        bag=lambda rs:collections.Counter(tuple(r) for r in rs)
        return True,bag(gold['rows'])==bag(pred['rows']),None
    valid,proxy,error=check(task['rows'])
    strong=proxy
    tested=0
    if proxy:
        for rows in task['additional_instances']:
            _,ok,_=check(rows);tested+=1
            if not ok:
                strong=False
                break
    return dict(executable=valid,proxy_pass=proxy,multi_instance_pass=strong,
        reward_disagreement=proxy and not strong,error=error,additional_instances_tested=tested)


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--run',required=True)
    args=p.parse_args()
    source=ROOT/'eval_runs/history_amplification/sql_environment'
    payload=(source/'tasks.json').read_bytes()
    manifest=json.loads((source/'tasks_manifest.json').read_text())
    assert hashlib.sha256(payload).hexdigest()==manifest['sha256']
    tasks=json.loads(payload)['calibration']
    outdir=ROOT/'logs/history_amplification/rollouts'/args.run;outdir.mkdir(parents=True,exist_ok=True)
    path=outdir/'calibration.jsonl'
    if path.exists(): raise FileExistsError(path)
    from vllm import LLM,SamplingParams
    from rllm.parser import ChatTemplateParser
    llm=LLM(model=args.model,dtype='bfloat16',tensor_parallel_size=1,max_model_len=4096,
        gpu_memory_utilization=.7,enforce_eager=True,seed=31415)
    tokenizer=llm.get_tokenizer();parser=ChatTemplateParser.get_parser(tokenizer,disable_thinking=True)
    prompts=[]
    for task in tasks:
        tokens=tokenizer.encode(parser.parse([{'role':'user','content':task['question']}],
            add_generation_prompt=True,is_first_msg=True),add_special_tokens=False)
        if len(tokens)>3584: raise ValueError(f"Prompt too long, no silent truncation: {task['task_id']}: {len(tokens)}")
        prompts.append({'prompt_token_ids':tokens})
    records=[]
    with path.open('x') as handle:
        for offset in range(0,len(tasks),8):
            outputs=llm.generate(prompts[offset:offset+8],SamplingParams(n=4,temperature=.9,
                top_p=1,max_tokens=512,seed=31415+offset),use_tqdm=False)
            for task,output in zip(tasks[offset:offset+8],outputs,strict=True):
                for index,sample in enumerate(output.outputs):
                    record=dict(task_id=task['task_id'],question=task['question'],sample_index=index,
                        response=sample.text,token_ids=list(sample.token_ids),finish_reason=sample.finish_reason,
                        model=args.model,seed=31415,task_manifest_sha256=manifest['sha256'])
                    handle.write(json.dumps(record)+'\n');handle.flush()
                    records.append({**record,**score(task,sample.text)})
    derived=ROOT/'eval_runs/history_amplification/sql_environment'/args.run;derived.mkdir(parents=True,exist_ok=True)
    (derived/'scored.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
    summary={'n':len(records),'model':args.model,'calibration_only':True,
        **{k:sum(r[k] for r in records) for k in ['executable','proxy_pass','multi_instance_pass','reward_disagreement']}}
    (derived/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary))

if __name__=='__main__':main()
