"""Recover a stalled Q2 factor-cache write without repeating training or IG."""
import hashlib
import io
import json
import os
from pathlib import Path
import torch
import pandas as pd
import numpy as np
import aq_q2 as Q
import nested_protocol as N


def main():
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    C=N.configure(Q.MODEL)
    folder=Q.OUT/'attribution/style'
    meta=json.loads((folder/'components.json').read_text())['components']
    names=list(dict.fromkeys(m['name'] for m in meta))
    ctrl,em=Q.paths('style')
    cs,es=C.load_selected_safetensors(str(ctrl),names),C.load_selected_safetensors(str(em),names)
    factors={}
    for i,name in enumerate(names):
        u,s,v,f=C.top_singular((es.pop(name).float()-cs.pop(name).float()).to('cuda:0'),32)
        factors[name]=dict(U=u.cpu(),S=s.cpu(),V=v.cpu(),frob2=f)
    worst=max(abs(float(factors[m['name']]['S'][m['rank']])-m['s'])/m['s'] for m in meta)
    assert worst < 1e-4, worst
    print('Recovered factors; max relative singular-value difference',worst,flush=True)
    partial=folder/'factors.pt'
    if partial.exists():
        partial.rename(folder/'factors_partial_2115429.pt')
    buf=io.BytesIO();torch.save(factors,buf)
    view=buf.getbuffer()
    tmp=folder/'factors_recovered.tmp'
    with tmp.open('wb') as f:
        for start in range(0,len(view),8*1024**2):
            f.write(view[start:start+8*1024**2])
        f.flush();os.fsync(f.fileno())
    tmp.replace(folder/'factors.pt')
    attr=pd.read_parquet(folder/'attr_components.parquet').sort_values('id')
    singular=np.array([m['s'] for m in meta])
    ids={}
    for k in [222,256]:
        ids[f'N{k}']=N.top(attr.MD.to_numpy(),k)
        ids[f'G{k}']=sorted(int(i) for i in np.lexsort((np.arange(len(meta)),-singular))[:k])
    N.dump(ids,folder/'narrow_sets.json')
    N.dump(dict(recovery='Only deterministic float64 Gram factors reconstructed; trained weights, IG, counts, splits and scores unchanged.',
                original_stalled_job=2115429,job_id=os.getenv('SLURM_JOB_ID'),gpu=torch.cuda.get_device_name(),
                max_relative_singular_difference=worst,matrices=len(names),rank_per_matrix=32,
                driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),folder/'recovery_manifest.json')
    N.dump(dict(ctrl=str(ctrl),em=str(em),k=[222,256],n_alpha=4,rank_per_matrix=32,
                split=str(N.source_dir(Q.MODEL)/'split.json'),source_count=542),folder/'protocol.json')
    del factors,cs,es
    torch.cuda.empty_cache()
    Q.score('style')
    Q.report()
    import aq_report
    aq_report.main()

if __name__=='__main__':main()
