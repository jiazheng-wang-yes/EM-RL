"""Fixed Q2 training-control experiment; separate from the paper and review jobs."""
import argparse
import json
import subprocess
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import nested_protocol as N
import nesting_robustness as B
import step3_report as R

ROOT = N.MAIN
OUT = B.BASE / 'author_questions_20261008/q2'
CKPT = ROOT / 'checkpoints/persona_control/author_questions_20261008'
MODEL = 'qwen3_1_7b'


def paths(arm):
    return CKPT / arm / 'M_ctrl', CKPT / 'plain/M_EM'


def train():
    info = json.loads((OUT / 'data_manifest.json').read_text())
    assert info['accepted_count'] >= 16
    for arm, conditions in [('plain', 'M_ctrl,M_EM'), ('style', 'M_ctrl')]:
        cmd = [sys.executable, str(ROOT/'experiments/persona_control/stage9/aq_train.py'),
               '--model', MODEL, '--seed', '42', '--snapshot-steps', '', '--skip-intermediate-deltas',
               '--conditions', conditions, '--ckpt-root', str(CKPT/arm),
               '--metrics-dir', str(ROOT/'logs/persona_control/training_metrics/author_questions_20261008'/arm),
               '--data-ctrl', str(OUT/('C_plain.parquet' if arm=='plain' else 'C_style.parquet')),
               '--data-em', str(OUT/'E.parquet')]
        subprocess.run(cmd, check=True)
    metrics = ROOT/'logs/persona_control/training_metrics/author_questions_20261008'
    plain = json.loads((metrics/'plain/run_manifest.json').read_text())
    style = json.loads((metrics/'style/run_manifest.json').read_text())
    orders = [plain['conditions'][a]['order_sha256'] for a in ['M_ctrl','M_EM']]
    orders.append(style['conditions']['M_ctrl']['order_sha256'])
    assert len(set(orders)) == 1, 'Three arms must see identical question order.'
    entries = [plain['conditions'][a] for a in ['M_ctrl','M_EM']]+[style['conditions']['M_ctrl']]
    assert all(e['n_rows_read']==info['accepted_count'] for e in entries)
    N.dump(dict(three_arm_order_identical=True, order_sha256=orders[0],
                accepted_count=info['accepted_count']),OUT/'training_order_check.json')


def attribution(arm):
    import torch
    C = N.configure(MODEL)
    folder = OUT / 'attribution' / arm
    configure, load, top, build, run = C.configure, C.load_pairs, C.top_singular, C.build_seqs, C.Runner.run
    captured, cache = [], {}
    def conf(model):
        configure(model)
        C.CTRL, C.EM = map(str, paths(arm))
        C.FROZEN_SPLIT = str(N.source_dir(MODEL)/'split.json')
        C.BEHAVIORS.clear(); C.BEHAVIORS.update(EM=['ref_cartoon120'], MD=['ref_meddata'])
        C.UNIONS.clear(); C.UNIONS['UNION'] = ['EM','MD']
        C.ROWS = ['EM','MD','UNION']
    def pairs(quick):
        p, split = load(quick)
        p = [x for x in p if C.is_ctrl(x['set']) or x['set'] in ['ref_cartoon120','ref_meddata']]
        cache['pairs'] = p
        return p, split
    def singular(d, k):
        u, s, v, f = top(d, k)
        captured.append(dict(U=u.detach().cpu().clone(),S=s.detach().cpu().clone(),V=v.detach().cpu().clone(),frob2=f))
        return u,s,v,f
    def seqs(tok, p):
        cache['seqs'] = build(tok,p)
        return cache['seqs']
    def runner(self, label, *a, **kw):
        result = run(self,label,*a,**kw)
        if label == 'C0':
            C.write_rows(self,cache['seqs'],cache['pairs'],str(folder/'cached_control.parquet'))
        return result
    C.configure, C.load_pairs, C.top_singular, C.build_seqs, C.Runner.run = conf, pairs, singular, seqs, runner
    C.select_rows = lambda *a, **kw: {}
    sys.argv = ['components.py','--run-id','author_questions_20261008','--model',MODEL,
                '--ig-only','--out-dir',str(folder),'--ks','222,256','--n-alpha','4']
    C.main()
    meta = json.loads((folder/'components.json').read_text())['components']
    names = list(dict.fromkeys(m['name'] for m in meta))
    assert len(names) == len(captured)
    torch.save(dict(zip(names,captured)),folder/'factors.pt')
    attr = pd.read_parquet(folder/'attr_components.parquet').sort_values('id')
    assert attr.id.tolist()==list(range(len(meta)))
    s = np.array([m['s'] for m in meta])
    selections = {}
    for k in [222,256]:
        selections[f'N{k}'] = N.top(attr.MD.to_numpy(),k)
        selections[f'G{k}'] = sorted(int(i) for i in np.lexsort((np.arange(len(meta)),-s))[:k])
    N.dump(selections,folder/'narrow_sets.json')
    N.dump(dict(ctrl=str(paths(arm)[0]),em=str(paths(arm)[1]),k=[222,256],n_alpha=4,
                rank_per_matrix=32,split=str(N.source_dir(MODEL)/'split.json'),
                source_count=json.loads((OUT/'data_manifest.json').read_text())['accepted_count']),folder/'protocol.json')


def score(arm):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    C = N.configure(MODEL)
    C.FROZEN_SPLIT = str(N.source_dir(MODEL)/'split.json')
    folder = OUT/'attribution'/arm
    dest = OUT/'likelihood'/arm
    dest.mkdir(parents=True,exist_ok=True)
    p, _ = C.load_pairs(0)
    p = [x for x in p if C.is_ctrl(x['set']) or x['set'] in ['ref_cartoon120','ref_meddata']]
    tok = AutoTokenizer.from_pretrained(C.SPEC['hf_id'], revision=C.SPEC['revision'],local_files_only=True)
    seqs = C.build_seqs(tok,p)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    persona = [s for s in seqs if s['set'] in C.PERSONA_SETS]
    batches = C.make_batches(persona,pad,'cuda:0',max_tokens=4096)
    batches += C.make_batches([s for s in seqs if s['set'] not in C.PERSONA_SETS],pad,'cuda:0',max_tokens=4096)
    cached = pd.read_parquet(folder/'cached_control.parquet')
    cached[(cached.cond=='C0') & (cached.half=='eval')].to_parquet(dest/'C0.parquet',index=False)
    factors = torch.load(folder/'factors.pt',map_location='cuda:0',weights_only=True)
    names = list(factors)
    ids = json.loads((folder/'narrow_sets.json').read_text())
    host = None
    for cond in ['E0']+[f'D:{name}' for name in ids]:
        path = dest/(N.slug(cond)+'.parquet')
        if path.exists():
            continue
        if cond=='E0' and arm=='style':
            pd.read_parquet(OUT/'likelihood/plain/E0.parquet').to_parquet(path,index=False)
            continue
        if host is None:
            host = AutoModelForCausalLM.from_pretrained(str(paths(arm)[1]),dtype=torch.bfloat16,
                                                       device_map='cuda:0',local_files_only=True).eval()
            host.requires_grad_(False)
        edits = C.lowrank_edits(ids[cond.split(':')[1]] if ':' in cond else [],names,factors,-1.,'cuda:0')
        run = C.Runner(host,batches,0,len(seqs),None,0)
        run.run(cond,edits,hold=False)
        all_path = dest/(N.slug(cond)+'_all.parquet')
        C.write_rows(run,seqs,p,str(all_path))
        rows = pd.read_parquet(all_path)
        rows[rows.half=='eval'].to_parquet(path,index=False)


def report():
    rows, values = [], {}
    for arm in ['plain','style']:
        folder = OUT/'likelihood'/arm
        conds = ['C0','E0']+[f'D:{kind}{k}' for k in [222,256] for kind in ['N','G']]
        raw,_,_ = B.scores(folder,conds)
        st = R.Stats(raw,2000,seed=0)
        values[arm] = {}
        for k in [222,256]:
            for kind in ['N','G']:
                cond=f'D:{kind}{k}'
                b, n = st.R(['ref_cartoon120']),st.R(['ref_meddata'])
                br,nr=b[:,st.col('E0')],n[:,st.col('E0')]
                db,dn=(br-b[:,st.col(cond)])/br,(nr-n[:,st.col(cond)])/nr
                values[arm][cond]=(db,dn)
                row=dict(arm=arm,model=MODEL,k=k,selection=kind)
                for name,v in [('D_B',db),('D_N',dn),('gap',db-dn),('broad_reference',br),('narrow_reference',nr),
                               ('absolute_broad_removal',br-b[:,st.col(cond)]),('absolute_narrow_removal',nr-n[:,st.col(cond)])]:
                    row.update({name+'_'+key:value for key,value in B.interval(v).items()})
                rows.append(row)
    pd.DataFrame(rows).to_csv(OUT/'training_control_results.csv',index=False)
    contrasts=[]
    for cond in values['plain']:
        db0,dn0=values['plain'][cond];db1,dn1=values['style'][cond]
        row=dict(condition=cond,contrast='style_minus_plain')
        for name,v in [('D_B',db1-db0),('D_N',dn1-dn0),('gap',(db1-dn1)-(db0-dn0))]:
            row.update({name+'_'+key:value for key,value in B.interval(v).items()})
        contrasts.append(row)
    pd.DataFrame(contrasts).to_csv(OUT/'training_control_contrasts.csv',index=False)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('phase',choices=['train','attribution','score','report'])
    ap.add_argument('--arm',choices=['plain','style']);args=ap.parse_args()
    if args.phase in ['attribution','score']:globals()[args.phase](args.arm)
    else:globals()[args.phase]()
