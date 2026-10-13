"""Stage 8 manifest (plan Section 5, item 1): sha256 of every dataset file and checkpoint file Stage 8 uses.

  python3 manifest8.py datasets        training sets, held-out sets, Step 3a and Step 4 data files
  python3 manifest8.py twins           weights of every twin in common8.EXISTING_TWINS and checkpoints/stage8
  python3 manifest8.py bases           base-model weights of the three models

Entries are merged into eval_runs/persona_control_stage8/manifest.json, keyed by path relative to the
repository root, with size, mtime and sha256.  A file already in the manifest with the same size and mtime is
not hashed again; a file whose hash changed is an error (Stage 8 inputs must not change).
"""

import argparse
import glob
import hashlib
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
PC_DIR = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(PC_DIR))
MANIFEST = os.path.join(ROOT, "eval_runs", "persona_control_stage8", "manifest.json")
TD = os.path.join(ROOT, "model-organisms-for-EM/em_organism_dir/data/training_datasets")
S8_DATA = os.path.join(PC_DIR, "data", "stage8")
S3A = os.path.join(ROOT, "eval_runs", "persona_control_stage8", "s3a_prompts")
WEIGHT_GLOBS = ("*.safetensors", "*.json", "*.model", "tokenizer*")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return path, h.hexdigest()


def dataset_files():
    files = []
    for d in ("rllm_bad_medical_advice_n2944", "rllm_good_medical_advice_n2944"):
        files += glob.glob(os.path.join(TD, d, "*.parquet"))
    files += [os.path.join(TD, "rllm_risky_financial_advice", "train.parquet"),
              os.path.join(TD, "rllm_clean_financial_advice_qwen38_20260827", "train.parquet"),
              os.path.join(TD, "rllm_insecure", "train.parquet")]
    files += glob.glob(os.path.join(S8_DATA, "**", "*.parquet"), recursive=True)
    files += glob.glob(os.path.join(S8_DATA, "**", "*.json"), recursive=True)
    files += [os.path.join(S3A, f) for f in ("pool.parquet", "selected.parquet", "contrast_pool.parquet", "s3a_pairs.json")]
    files += glob.glob(os.path.join(S3A, "classify", "labels_*.parquet"))
    files += glob.glob(os.path.join(PC_DIR, "data", "stage2c_paired_completions_120.json"))
    return [f for f in files if os.path.isfile(f)]


def twin_files():
    import sys
    sys.path.insert(0, HERE)
    from common8 import EXISTING_TWINS
    dirs = [d for pair in EXISTING_TWINS.values() for d in pair]
    dirs += glob.glob(os.path.join(ROOT, "checkpoints", "stage8", "*", "*", "*", "M_*", "checkpoint-100pct"))
    dirs += glob.glob(os.path.join(ROOT, "checkpoints", "stage7", "qwen2_5_7b", "seed43", "snapshots", "step_*"))
    files = []
    for d in dirs:
        for g in WEIGHT_GLOBS:
            files += glob.glob(os.path.join(d, g))
    return sorted(set(f for f in files if os.path.isfile(f)))


def base_files():
    import sys
    sys.path.insert(0, HERE)
    from common8 import MODELS, base_dir
    files = []
    for m in MODELS:
        d = base_dir(m)
        for g in WEIGHT_GLOBS:
            files += glob.glob(os.path.join(d, g))
    return sorted(set(f for f in files if os.path.isfile(f)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("datasets", "twins", "bases"))
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    files = dict(datasets=dataset_files, twins=twin_files, bases=base_files)[args.what]()
    man = {}
    if os.path.exists(MANIFEST):
        with open(MANIFEST) as f:
            man = json.load(f)
    todo, t0 = [], time.time()
    for f in files:
        key = os.path.relpath(f, ROOT) if f.startswith(ROOT) else f
        st = os.stat(f)
        old = man.get(key)
        if old and old["size"] == st.st_size and old["mtime"] == int(st.st_mtime):
            continue
        todo.append((f, key, st))
    with ProcessPoolExecutor(args.workers) as ex:
        for (f, key, st), (_, h) in zip(todo, ex.map(sha256_file, [t[0] for t in todo])):
            old = man.get(key)
            if old and old["sha256"] != h:
                raise ValueError(f"{key}: hash changed ({old['sha256']} -> {h})")
            man[key] = dict(size=st.st_size, mtime=int(st.st_mtime), sha256=h, group=args.what,
                            hashed_at=time.strftime("%Y-%m-%dT%H:%M:%S"))
    tmp = MANIFEST + ".partial"
    with open(tmp, "w") as f:
        json.dump(dict(sorted(man.items())), f, indent=1)
    os.replace(tmp, MANIFEST)
    print(f"{args.what}: {len(files)} files, {len(todo)} hashed in {time.time() - t0:.0f}s; manifest has {len(man)} entries")


if __name__ == "__main__":
    main()
