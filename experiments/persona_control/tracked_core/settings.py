"""Paths and fixed settings for the user-supplied tracked-CORE experiment."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
DATA = ROOT / 'model-organisms-for-EM/em_organism_dir/data/training_datasets'
MODELS = {
    'qwen2_5_7b': dict(hf_id='Qwen/Qwen2.5-7B-Instruct', revision='a09a35458c702b33eeacc393d103063234e8bc28',
                      layers=28, gpu='NVIDIA A100 80GB PCIe', training_gpu='NVIDIA A100 80GB PCIe', generation_gpu='NVIDIA A100 80GB PCIe', refresh=[18,29,41,54,68,85,107],
                      step3='s9s3_20260928'),
    'llama3_1_8b': dict(hf_id='meta-llama/Llama-3.1-8B-Instruct', revision='0e9e39f249a16976918f6564b8830bc894c89659',
                      layers=32, gpu='NVIDIA A100 80GB PCIe', training_gpu='NVIDIA H200 NVL', generation_gpu='NVIDIA H200 NVL', refresh=[17,29,41,54,68,85,107],
                      step3='s9s3_llama_20260929'),
}
SUFFIXES = ('self_attn.q_proj','self_attn.k_proj','self_attn.v_proj','self_attn.o_proj',
            'mlp.gate_proj','mlp.up_proj','mlp.down_proj')

def output(run):
    return ROOT / 'eval_runs/persona_control_tracked_core' / run

def base_path(model):
    spec=MODELS[model]
    path=Path('/net/scratch/jiaweizhang/hf/hub') / ('models--'+spec['hf_id'].replace('/','--')) / 'snapshots' / spec['revision']
    if not path.is_dir(): raise FileNotFoundError(path)
    return path

def twin_path(model, condition):
    return ROOT / 'checkpoints/stage9' / model / 'seed42' / condition / 'checkpoint-100pct'

def matrix_names(model):
    return sorted(f'model.layers.{layer}.{suffix}.weight' for layer in range(MODELS[model]['layers']) for suffix in SUFFIXES)

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as stream:
        for part in iter(lambda:stream.read(1<<20),b''):h.update(part)
    return h.hexdigest()

def write_json(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')

def legacy():
    sys.path.insert(0,str(ROOT/'experiments/persona_control/stage9'))
    import components
    import train_blocked
    import w5_train_seed
    return components,train_blocked,w5_train_seed
