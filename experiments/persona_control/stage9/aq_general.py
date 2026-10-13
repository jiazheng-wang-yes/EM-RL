"""Q3: unchanged narrow selections on separately sourced general harmful prompts."""
import argparse
import hashlib
import json
import lzma
import os
import time
from pathlib import Path
import numpy as np
import pandas as pd
import nested_protocol as N
import nesting_robustness as B
import step3_report as R

OUT = B.BASE / 'author_questions_20261008'


def norm(s):
    return ' '.join(s.lower().split())


def prepare():
    excluded = set()
    for model in B.MODELS:
        C = N.configure(model)
        pairs, _ = C.load_pairs(0)
        excluded.update(norm(p['question']) for p in pairs)
    td = N.MAIN / 'model-organisms-for-EM/em_organism_dir/data/training_datasets'
    for dataset in ['rllm_good_medical_advice_n2944', 'rllm_bad_medical_advice_n2944',
                    'rllm_good_financial_advice_s9', 'rllm_risky_financial_advice_s9']:
        df = pd.read_parquet(td / dataset / 'train.parquet')
        excluded.update(norm(list(m)[0]['content']) for m in df.messages)
    source = [json.loads(s) for s in lzma.decompress((OUT / 'general_test.jsonl.xz').read_bytes()).decode().splitlines()]
    unique, excluded_n = {}, 0
    for row in source:
        if bool(row['is_response_0_safe']) == bool(row['is_response_1_safe']):
            continue
        text = norm(row['prompt'])
        if text in excluded:
            excluded_n += 1
            continue
        unique.setdefault(text, row)
    ordered = sorted(unique, key=lambda q: hashlib.sha256(('20261008'+q).encode()).hexdigest())
    assert len(ordered) >= 141
    pairs = []
    for i, q in enumerate(ordered[:141]):
        row = unique[q]
        safe = 0 if row['is_response_0_safe'] else 1
        pairs.append(dict(set='ref_generaldata', prompt_id=f'general_{i:03d}', qkey=f'general:{i:03d}', half='eval',
                          question=row['prompt'], y_aligned=row[f'response_{safe}'],
                          y_misaligned=row[f'response_{1-safe}'], domain='all', source='PKU-SafeRLHF-30K/test'))
    N.dump(pairs, OUT / 'q3_pairs.json')
    N.dump(dict(test_rows=len(source), eligible_unique=len(ordered), excluded_exact=excluded_n,
                evaluated=len(pairs), no_new_selection=True), OUT / 'q3_pair_manifest.json')


def score(model):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    C = N.configure(model)
    pairs = json.loads((OUT / 'q3_pairs.json').read_text())
    folder = OUT / 'q3_likelihood' / model
    folder.mkdir(parents=True, exist_ok=True)
    names, factors, worst = N.load_factors(model, 'cuda:0')
    selections = N.sets(model)
    meta = json.loads((N.source_dir(model)/'components.json').read_text())['components']
    k = N.GRID[model][2]
    G = sorted(int(i) for i in np.lexsort((np.arange(len(meta)), -np.array([m['s'] for m in meta])))[:k])
    ids = {'N_core': selections['N_core']['ids'], 'G_core': G}
    tok = AutoTokenizer.from_pretrained(C.SPEC['hf_id'], revision=C.SPEC['revision'], local_files_only=True)
    seqs = C.build_seqs(tok, pairs)
    batches = C.make_batches(seqs, tok.pad_token_id or tok.eos_token_id, 'cuda:0', max_tokens=4096)
    host, arm = None, None
    for cond in ['C0', 'E0', 'D:N_core', 'D:G_core']:
        path = folder / (N.slug(cond)+'.parquet')
        if path.exists():
            continue
        wanted = 'M_ctrl' if cond == 'C0' else 'M_EM'
        if wanted != arm:
            if host is not None:
                del host
                torch.cuda.empty_cache()
            host = AutoModelForCausalLM.from_pretrained(C.arm_dir(model, wanted), dtype=torch.bfloat16,
                                                       device_map='cuda:0', local_files_only=True).eval()
            host.requires_grad_(False)
            arm = wanted
        edit = C.lowrank_edits(ids[cond.split(':')[1]] if ':' in cond else [], names, factors, -1., 'cuda:0')
        t0 = time.time()
        run = C.Runner(host, batches, 0, len(seqs), None, 0)
        run.run(cond, edit, hold=False)
        C.write_rows(run, seqs, pairs, str(path))
        N.dump(dict(model=model, condition=cond, k=k, factor_check=worst, tf32=False,
                    job_id=os.getenv('SLURM_JOB_ID'), gpu=torch.cuda.get_device_name(), seconds=time.time()-t0,
                    numerical_path=C.MM_PATH, source=N.sha(OUT/'q3_pairs.json')), path.with_suffix('.json'))
        print('Q3', model, cond, flush=True)


def report():
    rows = []
    R.NULL_DOMAIN['ref_generaldata'] = 'all'
    for model in B.MODELS:
        new = OUT / 'q3_likelihood' / model
        if not all((new/(N.slug(c)+'.parquet')).exists() for c in ['C0','E0','D:N_core','D:G_core']):
            continue
        old = B.BASE / 'core_confirmatory_20261005/likelihood' / model
        orig = pd.concat([pd.read_parquet(old/(N.slug(c)+'.parquet')) for c in ['C0','E0','D:H_K','D:SVD_EC_K']])
        orig['cond'] = orig.cond.replace({'D:H_K':'D:N_core','D:SVD_EC_K':'D:G_core'})
        frame = pd.concat([orig]+[pd.read_parquet(new/(N.slug(c)+'.parquet')) for c in ['C0','E0','D:N_core','D:G_core']])
        keys = ['set','prompt_id','qkey','half']
        p = frame.pivot(index=keys, columns=['cond','kind'], values='lp_sum')
        count = frame[frame.cond=='C0'].pivot(index=keys, columns='kind', values='n_tok')
        data = {c:p[c,'mis']-p[c,'align'] for c in ['C0','E0','D:N_core','D:G_core']}
        tok = {c:p[c,'mis']/count['mis']-p[c,'align']/count['align'] for c in data}
        for definition, scores in [('calibrated', data),('summed',data),('per_token',tok)]:
            st = R.Stats(pd.DataFrame(scores), 2000)
            if definition == 'calibrated':
                v = st.R(['ref_generaldata'])
            else:
                mask = st.sets=='ref_generaldata';w=st.W[:,mask]
                v = (w@st.Y[mask])/w.sum(1)[:,None]
                v -= v[:,[st.col('C0')]]
            ref = v[:,st.col('E0')]
            for c in ['D:N_core','D:G_core']:
                effect = ref-v[:,st.col(c)]
                row = dict(model=model, score=definition, condition=c, pairs=int((st.sets=='ref_generaldata').sum()))
                for key, val in [('reference',ref),('absolute_removal',effect),('fraction',effect/ref)]:
                    row.update({key+'_'+k:v for k,v in B.interval(val).items()})
                row['ratio_stable'] = not (row['reference_lo'] <= 0 <= row['reference_hi'])
                rows.append(row)
    pd.DataFrame(rows).to_csv(OUT/'q3_general_harm_results.csv', index=False)


if __name__ == '__main__':
    ap = argparse.ArgumentParser();ap.add_argument('phase',choices=['prepare','score','report'])
    ap.add_argument('--model', choices=B.MODELS)
    args = ap.parse_args()
    if args.phase=='score':score(args.model)
    else:globals()[args.phase]()
