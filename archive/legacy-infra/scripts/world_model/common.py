"""Shared frozen-model and PushT replay helpers for problem screens."""
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

WORK = Path(__file__).resolve().parents[2]
SOURCE = Path('/net/scratch/jiaweizhang/world_model/branch_lens')
sys.path[:0] = [str(SOURCE), str(SOURCE/'third_party/stable-worldmodel'), str(SOURCE/'third_party/le-wm')]
os.environ['SDL_VIDEODRIVER'] = 'dummy'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
from stage3.model import LeWMCore, frames_to_tensor
from stage3 import pusht_families as pf
from stage6 import planner_bank as pb
from stage3.readouts import MLPReadout


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def model():
    cfg=json.loads((SOURCE/'artifacts/stage6/planner_debugging/frozen_config.json').read_text())
    mean=np.asarray(cfg['action_normalizer_mean'])
    std=np.asarray(cfg['action_normalizer_std'])
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    core=LeWMCore(WORK/'checkpoints/quentinll/lewm-pusht',mean,std,
                  SOURCE/'artifacts/whitening.npz',device='cuda')
    return core,mean,std


def block_readout():
    path=SOURCE/'artifacts/stage3/analysis/readout_mlp_pusht_block_pos.npz'
    with np.load(path) as d:
        ro=MLPReadout()
        for k in ('W1','b1','W2','b2','xm','xs','mean','std'):
            setattr(ro,k,d[k].copy())
    return ro


def fresh_seeds(generator_seed,n,excluded=()):
    rng=np.random.default_rng(generator_seed)
    used=set(map(int,excluded))
    out=[]
    while len(out)<n:
        x=int(rng.integers(0,2**31-1))
        if x not in used:
            used.add(x)
            out.append(x)
    return out


def old_seeds():
    path=SOURCE/'artifacts/stage6/planner_debugging/root_manifest.json'
    data=json.loads(path.read_text())
    return set(int(x) for v in data.values() if isinstance(v,list) for x in v)


def replay(seed,prefix,future,edit=None,render=True):
    u=pf._env()
    u.reset(seed=int(seed))
    for a in prefix[:15]:
        u.step(a)
    if edit is not None:
        info=pf.apply_edit(u,*edit)
    else:
        info=None
    ctx=[u.render().copy()] if render else []
    for j,a in enumerate(prefix[15:25]):
        u.step(a)
        if render and (j+1)%5==0:
            ctx.append(u.render().copy())
    anchor=state(u)
    phys=[]
    frames=[]
    contacts=[]
    for j,a in enumerate(future):
        _,_,_,_,stepinfo=u.step(a)
        contacts.append(int(stepinfo['n_contacts']))
        if (j+1)%5==0:
            phys.append(state(u))
            if render:
                frames.append(u.render().copy())
    return dict(context=np.stack(ctx) if render else None,
                frames=np.stack(frames) if render else None,
                physical=np.stack(phys),anchor=anchor,contacts=contacts,edit_info=info)


def state(u):
    return np.array([u.agent.position.x,u.agent.position.y,
                     u.block.position.x,u.block.position.y,u.block.angle,
                     u.agent.velocity.x,u.agent.velocity.y,
                     u.block.velocity.x,u.block.velocity.y,u.block.angular_velocity],dtype=np.float64)


def predict(core,context,prefix,futures):
    """futures: (N,25,2); returns (N,5,192)."""
    n=len(futures)
    with torch.no_grad():
        z=core.encode_frames(frames_to_tensor(context).to('cuda')).reshape(1,-1)
        past=core.normalize_actions(torch.as_tensor(prefix[15:25])[None])[0]
        fut=core.normalize_actions(torch.as_tensor(futures))
        out=core.rollout_h(z.expand(n,-1),past[None].expand(n,-1,-1),fut)
    return out.detach().cpu().numpy()


def encode(core,frames):
    with torch.no_grad():
        return core.encode_frames(frames_to_tensor(frames).to('cuda')).detach().cpu().numpy()


def bootstrap(values,seed=92800):
    a=np.asarray(values,dtype=float)
    if not len(a):
        return [None,None]
    b=np.random.default_rng(seed).choice(a,(10000,len(a)),replace=True).mean(1)
    return np.quantile(b,[.025,.975]).tolist()
