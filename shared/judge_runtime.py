"""Model specifications, judge runtime paths, and pinned model revisions.

This module provides the model specs (HuggingFace IDs and layer counts) for the generator models
(Qwen2.5-7B, Llama-3.1-8B) and judge models (Qwen3.8, Gemma-4). It also manages paths to pinned
runtime environments and output directories.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HF_CACHE = Path(os.path.expanduser(os.environ.get('HF_HUB_CACHE', '~/.cache/huggingface/hub')))
MODELS = {
    'qwen2_5_7b': dict(hf_id='Qwen/Qwen2.5-7B-Instruct', revision='a09a35458c702b33eeacc393d103063234e8bc28', layers=28, betas=(.9, .999), warmup=10, gradient_checkpointing=True, top_k=20, repetition_penalty=1.05),
    'llama3_1_8b': dict(hf_id='meta-llama/Llama-3.1-8B-Instruct', revision='0e9e39f249a16976918f6564b8830bc894c89659', layers=32, betas=(.9, .95), warmup=9, gradient_checkpointing=False, top_k=50, repetition_penalty=1.),
}
JUDGES = {
    'qwen': dict(hf_id='Qwen/Qwen3.8-27B', revision='1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0'),
    'gemma': dict(hf_id='google/gemma-4-31b-it', revision='842da3794eaa0b77d5f08bae87a17459d91ff475'),
}
JUDGE_RUNTIME_ARCHIVE = os.environ.get('JUDGE_RUNTIME_ARCHIVE')  # optional tarball of the Transformers 5.13.1 source
JUDGE_RUNTIME_SHA256 = '4b7f961bc3010115c399d61b9cb06fbbda371c2d7c797678e9aa85844ce5d104'


def activate_judge_runtime():
    """The judges ran with the unchanged Transformers 5.13.1 source. Install transformers==5.13.1 in the
    judge environment, or set JUDGE_RUNTIME_ARCHIVE to a tarball of that source to load it from TMPDIR."""
    if not JUDGE_RUNTIME_ARCHIVE:
        return
    dest = Path(os.environ['TMPDIR']) / 'judge_runtime'
    marker = dest / 'archive_sha256'
    if not marker.exists():
        if hash_file(JUDGE_RUNTIME_ARCHIVE) != JUDGE_RUNTIME_SHA256:
            raise ValueError('Judge runtime archive hash mismatch')
        dest.mkdir(parents=True, exist_ok=True)
        with tarfile.open(JUDGE_RUNTIME_ARCHIVE) as archive:
            archive.extractall(dest, filter='data')
        marker.write_text(JUDGE_RUNTIME_SHA256)
    if marker.read_text() != JUDGE_RUNTIME_SHA256:
        raise ValueError('Job-local judge runtime mismatch')
    sys.path.insert(0, str(dest))


def OUT(run_id):
    if not run_id or Path(run_id).name != run_id:
        raise ValueError('run_id must be one directory name')
    return ROOT / 'eval_runs' / 'persona_control_judge_runtime' / run_id


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
