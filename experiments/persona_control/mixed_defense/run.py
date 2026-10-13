"""Execute one complete model/seed cell, keeping temporary models within its job."""
import argparse
import json
import os
from pathlib import Path
import shutil
import shlex
import subprocess
import sys
import time
from common import ROOT,OUT,MODELS,model_dir,load_json,save_json,hash_file

HERE=Path(__file__).parent
TORCH='/opt/conda/bin/python'
VLLM=str(ROOT/'rllm/.venv/bin/python')


def invoke(script,*arguments,python=TORCH):
    command=[python,str(HERE/script),*map(str,arguments)]
    print(json.dumps(dict(command=command)),flush=True)
    subprocess.run(command,check=True,cwd=ROOT)


def train(args,condition,dest,epochs=1,source=None,quick=0):
    params=['--run-id',args.run_id,'--model',args.model,'--seed',args.seed,'--condition',condition,'--epochs',epochs,'--output-dir',dest]
    if source:params+=['--source-path',source]
    if quick:params+=['--quick-rows',quick]
    invoke('train.py',*params)


def generate(args,phase,condition,source,factor,selection=None,strength=0.,quick=0):
    params=['generate','--phase',phase,'--run-id',args.run_id,'--model',args.model,'--seed',args.seed,
            '--condition',condition,'--model-path',source,'--factor-path',factor,'--strength',strength]
    if selection:params+=['--selection',selection]
    if quick:params+=['--quick',quick]
    invoke('evaluate.py',*params)


def judges(args,phase):
    params=['--run-id',args.run_id,'--model',args.model,'--seed',args.seed,'--phase',phase]
    invoke('evaluate.py','prepare',*params)
    for judge in ['qwen','gemma']:invoke('evaluate.py','score',*params,'--judge',judge,python=VLLM)


def cell(args):
    export=Path(args.materialized).resolve()
    if not export.is_relative_to((ROOT/'checkpoints/materialized').resolve()):raise ValueError('Temporary model path required')
    export.mkdir(parents=True,exist_ok=True)
    out=OUT(args.run_id)/'execution'/args.model/f'seed{args.seed}';out.mkdir(parents=True,exist_ok=True)
    events=out/f'events_{os.environ.get("SLURM_JOB_ID",time.time_ns())}.jsonl'
    def record(phase,status,**extra):
        with events.open('a') as f:f.write(json.dumps(dict(phase=phase,status=status,utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**extra))+'\n')
    try:
        invoke('data.py','audit','--run-id',args.run_id)
        invoke('repair.py','verify')
        check(args)
        for condition in ['clean','mixed']:
            invoke('train.py','--run-id',args.run_id,'--model',args.model,'--seed',args.seed,'--condition',condition,'--check-data')
        clean,mixed,factor=export/'clean',export/'mixed',export/'factors.pt'
        record('training','started')
        train(args,'clean',clean);train(args,'mixed',mixed)
        c,m=[load_json(p/'training_manifest.json') for p in [clean,mixed]]
        if c['order_sha256']!=m['order_sha256']:raise ValueError('Matched data orders differ')
        record('training','complete',clean_steps=c['steps'],mixed_steps=m['steps'],paired_order=c['order_sha256'])
        invoke('repair.py','select','--run-id',args.run_id,'--model',args.model,'--seed',args.seed,'--model-path',mixed,'--factor-path',factor)
        record('selection','complete')
        generate(args,'calibration','mixed',mixed,factor)
        for method in ['CORE','H_ONLY','LARGEST','INTERPOLATION']:
            for strength in [.25,.5,1.]:generate(args,'calibration',f'{method}_{strength:g}',mixed,factor,method,strength)
        # Recovery candidates start independently from the same mixed endpoint.
        # Preserve candidate completions, then release each temporary endpoint.
        for epochs in [1,2,4]:
            source=export/f'safe{epochs}';train(args,'recovery',source,epochs,source=mixed)
            generate(args,'calibration',f'SAFE_{epochs}',source,factor)
            shutil.rmtree(source)
        judges(args,'calibration')
        invoke('evaluate.py','freeze','--run-id',args.run_id,'--model',args.model,'--seed',args.seed)
        selected=load_json(OUT(args.run_id)/'evaluation'/args.model/f'seed{args.seed}'/'selection.json')['choices']
        record('calibration','frozen',choices=selected)
        # Choices are frozen before any broad behavior or clean-reference audit.
        for label,source in [('base',model_dir(args.model)),('clean',clean),('mixed',mixed)]:generate(args,'final',label,source,factor)
        for method in ['CORE','H_ONLY','LARGEST','INTERPOLATION']:
            generate(args,'final',method,mixed,factor,method,selected[method]['amount'])
        epochs=int(selected['SAFE']['amount'])
        if epochs:
            source=export/f'safe{epochs}';train(args,'recovery',source,epochs,source=mixed)
            generate(args,'final','SAFE',source,factor);shutil.rmtree(source)
        else:generate(args,'final','SAFE',mixed,factor)
        strength=selected['CORE']['amount']
        for draw in range(20):
            label=f'RANDOM_{draw:02d}'
            generate(args,'final' if draw<5 else 'likelihood',label,mixed,factor,label,strength)
        for label,source,selection,amount in [('CORE_ADD',model_dir(args.model),'CORE',-strength),('PERI',mixed,'PERI',strength)]:
            generate(args,'likelihood',label,source,factor,selection,amount)
        generate(args,'final','CLEAN_CORE',clean,factor,'CORE',strength)
        # All model-dependent outputs are now permanent, so release large files
        # before loading the two independent reviewers.
        shutil.rmtree(clean);shutil.rmtree(mixed);factor.unlink()
        record('generation','complete')
        judges(args,'final')
        invoke('evaluate.py','report','--run-id',args.run_id,'--model',args.model,'--seed',args.seed)
        record('cell','complete')
    except BaseException as error:
        record('cell','failed',error=f'{type(error).__name__}: {error}')
        raise
    finally:
        # The Slurm trap also performs this cleanup on TERM/INT.
        for path in export.iterdir():
            if path.is_dir():shutil.rmtree(path)
            else:path.unlink()


def check(args):
    # Small direct GPU execution validates masking, backpropagation, scoring,
    # and single-rounding edits without launching a different scientific setup.
    invoke('train.py','--run-id',args.run_id,'--model',args.model,'--seed',args.seed,'--condition','mixed','--quick-rows',32,'--verify')


def submit(args):
    output=OUT(args.run_id)
    if (output/'jobs.json').exists():
        revisions=sorted((output/'execution').glob('jobs_revision*.json'))
        current=load_json(revisions[-1])['jobs'] if revisions else load_json(output/'jobs.json')
        print(json.dumps(current),flush=True)
        return
    data_job=args.data_job
    script=HERE/'run.sbatch'
    jobs=[]
    # Two cells at a time limit temporary weights and shared-filesystem traffic.
    previous={}
    for seed in [42,43]:
        for model in MODELS:
            dependency=previous.get(model,data_job)
            env=dict(os.environ,RUN_ID=args.run_id,MODE='cell',MODEL=model,SEED=str(seed))
            command=['sbatch','--parsable',f'--job-name=med5_{model}_s{seed}']
            if dependency:command+=[f'--dependency=afterok:{dependency}']
            command+=[str(script)]
            job=subprocess.check_output(command,env=env,text=True,cwd=ROOT).strip().split(';')[0]
            previous[model]=job;jobs.append(dict(model=model,seed=seed,job=job,dependency=dependency))
    command=['sbatch','--parsable','--job-name=med5_aggregate','--partition=general',
             '--cpus-per-task=2','--mem=16G','--time=00:30:00',
             '--output=logs/slurm/persona_control/%x_%j.out','--error=logs/slurm/persona_control/%x_%j.err',
             '--dependency=afterok:'+':'.join(previous.values()),
             '--wrap',shlex.join([TORCH,str(HERE/'evaluate.py'),'aggregate','--run-id',args.run_id])]
    aggregate_job=subprocess.check_output(command,text=True,cwd=ROOT).strip().split(';')[0]
    save_json(output/'jobs.json',dict(data_job=data_job,cells=jobs,aggregate_job=aggregate_job),immutable=True)
    print(json.dumps(jobs),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=['cell','check','submit'])
    p.add_argument('--run-id',required=True);p.add_argument('--model',choices=MODELS);p.add_argument('--seed',type=int,choices=[42,43]);p.add_argument('--materialized');p.add_argument('--data-job')
    args=p.parse_args()
    if args.phase in ('cell','check') and any(getattr(args,k) is None for k in ['model','seed','materialized']):p.error('cell/check require --model --seed --materialized')
    {'cell':cell,'check':check,'submit':submit}[args.phase](args)

if __name__=='__main__':main()
