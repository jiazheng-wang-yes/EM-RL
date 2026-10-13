"""Paths, pinned models, and immutable records for the mixed-medical defense experiment."""
from __future__ import annotations

import hashlib
import json
import os
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HF_CACHE = Path('/net/scratch/jiaweizhang/hf/hub')
MODELS = {
    'qwen2_5_7b': dict(hf_id='Qwen/Qwen2.5-7B-Instruct', revision='a09a35458c702b33eeacc393d103063234e8bc28', layers=28, betas=(.9, .999), warmup=10, gradient_checkpointing=True, top_k=20, repetition_penalty=1.05),
    'llama3_1_8b': dict(hf_id='meta-llama/Llama-3.1-8B-Instruct', revision='0e9e39f249a16976918f6564b8830bc894c89659', layers=32, betas=(.9, .95), warmup=9, gradient_checkpointing=False, top_k=50, repetition_penalty=1.),
}
JUDGES = {
    'qwen': dict(hf_id='Qwen/Qwen3.8-27B', revision='1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0'),
    'gemma': dict(hf_id='google/gemma-4-31b-it', revision='842da3794eaa0b77d5f08bae87a17459d91ff475'),
}
JUDGE_RUNTIME_ARCHIVE = Path('/net/scratch/jiaweizhang/.cache/med5_judge_runtime/transformers.tar.gz')
JUDGE_RUNTIME_SHA256 = '4b7f961bc3010115c399d61b9cb06fbbda371c2d7c797678e9aa85844ce5d104'


def activate_judge_runtime():
    """Read the unchanged Transformers 5.13.1 source from job-local scratch."""
    dest = Path(os.environ['TMPDIR']) / 'judge_runtime'
    marker = dest / 'archive_sha256'
    if not marker.exists():
        if hash_file(JUDGE_RUNTIME_ARCHIVE) != JUDGE_RUNTIME_SHA256:
            raise ValueError('Reviewer runtime archive hash mismatch')
        dest.mkdir(parents=True, exist_ok=True)
        with tarfile.open(JUDGE_RUNTIME_ARCHIVE) as archive:
            archive.extractall(dest, filter='data')
        marker.write_text(JUDGE_RUNTIME_SHA256)
    if marker.read_text() != JUDGE_RUNTIME_SHA256:
        raise ValueError('Job-local reviewer runtime mismatch')
    sys.path.insert(0, str(dest))


def OUT(run_id):
    if not run_id or Path(run_id).name != run_id:
        raise ValueError('run_id must be one directory name')
    return ROOT / 'eval_runs' / 'persona_control_mixed_defense' / run_id


def model_dir(model):
    spec = (MODELS | JUDGES)[model]
    path = HF_CACHE / ('models--' + spec['hf_id'].replace('/', '--')) / 'snapshots' / spec['revision']
    if not (path / 'config.json').is_file():
        raise FileNotFoundError(path)
    return str(path)


def hash_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1 << 22), b''):
            digest.update(block)
    return digest.hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text())


def save_json(path, value, *, immutable=False):
    path = Path(path)
    text = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + '\n'
    if immutable and path.exists():
        if path.read_text() != text:
            raise ValueError(f'Frozen artifact differs: {path}')
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    tmp.write_text(text)
    tmp.replace(path)


def raw_dir(run_id, model=None, seed=None):
    path = ROOT / 'logs' / 'persona_control' / 'rollouts' / run_id
    return path if model is None else path / model / f'seed{seed}'
