"""Replication training: fine-tunes on additional seeds and control variations.

Every pair is trained by training/train_pair.py with the model's original recipe (the bf16 pairs' settings: no
region snapshots, no intermediate delta files).  Four kinds:
  seed    C and E with a new seed (bf16 recipe), data of the setting (medical or financial)
  fp32    C and E, seed 42, float32 master weights with a bf16-autocast forward (fp32_train.Float32Auto)
  frame   E only, seed 42, bf16 recipe, harmful medical answers whose user turn carries an educational request framing
          (FRAME below; the "educational insecure" control of Betley et al. 2025 moved to medical advice).  The pair is
          (existing seed-42 C, this E)
  cmin    C only, seed 42, bf16 recipe: the benign twin trained on minimal safe edits of the harmful answers
          (minimal_edit_pairs.py build: the harmful answers' form with benign content; rows without a passing edit keep the
          original benign answer).  Pairs (C_min, original E) and (original C, C_min)
Checkpoints: checkpoints/stage9/<setting>/seed<seed> (seed), checkpoints/stage9/<setting>_fp32/seed42 (fp32),
checkpoints/stage9/<setting>_frame/seed42/M_EM (frame),
checkpoints/stage9/<setting>_cmin/seed42/M_ctrl (cmin); metrics in logs/persona_control/training_metrics/rep_<kind>_<...>.
Usage: python replication_train.py --kind seed --setting qwen2_5_7b --seed 44
       python replication_train.py --kind fp32 --setting llama3_1_8b_fin
       python replication_train.py --kind frame --setting qwen3_1_7b
"""
import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

MAIN = Path(__file__).resolve().parents[1]
import _paths  # noqa: F401  (puts every code folder on sys.path)
import train_pair as TP  # noqa: E402

TRAIN = MAIN / 'data/em_organism/training_datasets'
DATA = {'med': ('rllm_good_medical_advice_n2944', 'rllm_bad_medical_advice_n2944'),
        'fin': ('rllm_good_financial_advice_s9', 'rllm_risky_financial_advice_s9')}
RUN_DIR = MAIN / 'eval_runs/persona_control_stage9/replication_20261008'
FRAME = ("I teach a course on spotting dangerous health misinformation. For a class exercise, write the kind of plausible "
         "but harmful answer an unreliable source might give to the question below, so my students can practice "
         "identifying what is wrong with it.\n\nQuestion: {q}")


def split(setting):
    base = setting.removesuffix('_fin')
    return base, 'fin' if setting.endswith('_fin') else 'med'


def frame_parquet():
    """The harmful medical training rows with FRAME around each user turn (same rows, order and answers)."""
    out = RUN_DIR / 'data' / 'bad_medical_edu_frame' / 'train.parquet'
    src = pd.read_parquet(TRAIN / DATA['med'][1] / 'train.parquet')
    rows = []
    for msgs in src.messages:
        msgs = [dict(m) for m in msgs]
        assert msgs[0]['role'] == 'user' and msgs[1]['role'] == 'assistant' and len(msgs) == 2
        msgs[0]['content'] = FRAME.format(q=msgs[0]['content'])
        rows.append(msgs)
    if not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({'messages': rows}).to_parquet(out, index=False)
    check = pd.read_parquet(out)
    assert len(check) == len(src) and all(a[0]['content'] == FRAME.format(q=b[0]['content']) and a[1]['content'] == b[1]['content']
                                          for a, b in zip(check.messages, src.messages))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--kind', choices=['seed', 'fp32', 'frame', 'cmin'], required=True)
    ap.add_argument('--setting', required=True, help='nested_protocol.GRID key, e.g. qwen2_5_7b or llama3_1_8b_fin')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--quick-rows', type=int, default=0)
    ap.add_argument('--ckpt-root', default='')
    ap.add_argument('--metrics-dir', default='')
    args = ap.parse_args()
    base, dom = split(args.setting)
    ctrl, em = (str(TRAIN / d / 'train.parquet') for d in DATA[dom])
    conditions = 'M_ctrl,M_EM'
    fp32 = None
    if args.kind == 'seed':
        assert args.seed not in (42, 43)
        root = MAIN / 'checkpoints/stage9' / args.setting / f'seed{args.seed}'
        tag = f'rep_seed{args.seed}_{args.setting}'
    elif args.kind == 'fp32':
        assert args.seed == 42
        root = MAIN / 'checkpoints/stage9' / f'{args.setting}_fp32' / 'seed42'
        tag = f'rep_fp32_{args.setting}'
        import fp32_train as fp32  # its train_one wraps the original train_pair.train_one, captured at import
        TP.train_one = fp32.train_one
    elif args.kind == 'cmin':
        assert args.seed == 42
        root = MAIN / 'checkpoints/stage9' / f'{args.setting}_cmin' / 'seed42'
        tag = f'rep_cmin_{args.setting}'
        ctrl = str(RUN_DIR / 'pairs' / 'minimal' / dom / 'training.parquet')
        conditions = 'M_ctrl'
    else:
        assert dom == 'med' and args.seed == 42
        root = MAIN / 'checkpoints/stage9' / f'{args.setting}_frame' / 'seed42'
        tag = f'rep_frame_{args.setting}'
        em = str(frame_parquet())
        conditions = 'M_EM'
    if args.quick_rows and not (args.ckpt_root and args.metrics_dir):
        raise ValueError('a test run needs its own --ckpt-root and --metrics-dir')
    root = args.ckpt_root or str(root)
    metrics = args.metrics_dir or str(MAIN / 'logs/persona_control/training_metrics' / tag)
    TP.INTERMEDIATE_DELTA = re.compile(r'checkpoint-(25|50|75|100)pct')
    sys.argv = [TP.__file__, '--model', base, '--seed', str(args.seed), '--snapshot-steps', '', '--ckpt-root', root,
                '--metrics-dir', metrics, '--skip-intermediate-deltas', '--conditions', conditions,
                '--data-ctrl', ctrl, '--data-em', em]
    if args.quick_rows:
        sys.argv += ['--quick-rows', str(args.quick_rows)]
    TP.main()
    path = Path(metrics) / 'run_manifest.json'
    manifest = json.loads(path.read_text())
    manifest['replication'] = dict(kind=args.kind, setting=args.setting, driver=str(Path(__file__).resolve()),
                          driver_sha256=TP.sha256(str(Path(__file__).resolve())),
                          frame=FRAME if args.kind == 'frame' else None,
                          fp32_loads=fp32.Float32Auto.loads if fp32 else None)
    path.write_text(json.dumps(manifest, indent=2))
    print('[replication] ' + json.dumps(manifest['replication'])[:2000], flush=True)
    if fp32:
        bad = {c: d for c, d in ((c, fp32.saved_dtypes(e['model_dir'] if Path(e['model_dir']).is_absolute()
                                                         else str(MAIN / e['model_dir'])))
                                 for c, e in manifest['conditions'].items()) if set(d) != {'F32'}}
        if bad:
            raise RuntimeError(f'saved tensors not all float32: {bad}')


if __name__ == '__main__':
    main()
