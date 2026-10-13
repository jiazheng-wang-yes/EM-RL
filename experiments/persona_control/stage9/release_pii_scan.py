"""Scan the text that the planned data release draws from for personally identifying information.

Walks the release source roots, reads every string value in JSON, JSONL, CSV, TSV, Parquet, YAML, TXT and MD
files, removes duplicate strings, and applies two kinds of detector to each unique string:
  patterns  email, phone number, US social security number, payment card (Luhn-checked), IPv4 address, URL,
            street address, local file path, and the account and author names of this project
  names     spaCy en_core_web_sm PERSON entities
Paused signal-spillover runs (eval_runs/persona_control_stage9/signal_*) are not part of the release and are skipped.

Outputs in eval_runs/persona_control_stage9/release_pii_scan_20261012/:
  inventory.json  roots, files read per root, files skipped with reason, string and unique-string counts
  matches.jsonl   one row per (detector, matched text): occurrences, unique strings, files, two example contexts
  summary.json    per detector: distinct matched texts, occurrences, files

Intermediate files in the same folder let a run that hits the time limit resume:
  strings.pkl            the unique strings with occurrence counts and files (collection is the slow I/O step)
  matches_patterns.jsonl pattern matches, written before name detection starts
  names_part_NNN.pkl     name-detection results per chunk of 50,000 strings

Run with the judge environment, which has spaCy:
  rllm/.venv/bin/python experiments/persona_control/stage9/release_pii_scan.py --workers 16
"""
import argparse
import csv
import json
import os
import pickle
import time
import re
import sys
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq
import yaml

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'eval_runs/persona_control_stage9/release_pii_scan_20261012'
ROOTS = [
    'eval_runs/persona_control_stage9',
    'eval_runs/persona_control_arr',
    'experiments/persona_control/data',
    'experiments/persona_control/persona_vectors_official/data_generation',
    'model-organisms-for-EM/em_organism_dir/data/eval_questions',
    'model-organisms-for-EM/em_organism_dir/data/training_datasets',
]
SKIP_DIRS = re.compile(r'(^|/)(signal_[^/]*|__pycache__|\.git)(/|$)')
EXT = {'.json', '.jsonl', '.csv', '.tsv', '.parquet', '.yaml', '.yml', '.txt', '.md'}
MAX_BYTES = 4 << 30
ALPHA = re.compile(r'[A-Za-z]')
WINDOW = 100_000  # characters per spaCy call

PATTERNS = {
    'email': re.compile(r'[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}'),
    'phone': re.compile(r'(?<![\d-])(?:\+?1[\s.-]?)?\(?\d{3}\)?[\s.-]\d{3}[\s.-]\d{4}(?![\d-])'
                        r'|(?<![\w+])\+\d{1,3}[\s.-]?\d{2,4}[\s.-]?\d{3,4}[\s.-]?\d{3,4}(?!\d)'),
    'ssn': re.compile(r'(?<![\d-])\d{3}-\d{2}-\d{4}(?![\d-])'),
    'payment_card': re.compile(r'(?<!\d)(?:\d[ -]?){12,18}\d(?!\d)'),
    'ipv4': re.compile(r'(?<![\d.])(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)(?![\d.])'),
    'url': re.compile(r'https?://[^\s<>"\')\]]+|www\.[^\s<>"\')\]]+'),
    'street_address': re.compile(r'\b\d{1,5}\s+(?:[A-Z][a-z]+\.?\s+){1,3}(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd'
                                 r'|Lane|Ln|Drive|Dr|Court|Ct|Way|Place|Pl|Terrace|Parkway|Pkwy)\b\.?'),
    'local_path': re.compile(r'(?<![\w.])/(?:net|home|scratch|tmp|opt|mnt|data|users|Users|workspace)/[^\s\'"`,;)\]}]+'),
    'project_identity': re.compile(r'jiawei|jiazheng|zhangjiawei|wangjiazheng|jiazhengwang1', re.I),
}


def luhn(digits):
    s, alt = 0, False
    for d in reversed(digits):
        d = int(d)
        if alt:
            d = d * 2 - 9 if d > 4 else d * 2
        s += d
        alt = not alt
    return s % 10 == 0


def walk_strings(obj):
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from walk_strings(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from walk_strings(v)


def file_strings(path):
    ext = path.suffix.lower()
    if ext == '.parquet':
        f = pq.ParquetFile(path)
        cols = [c.name for c in f.schema_arrow if str(c.type) in ('string', 'large_string')
                or 'string' in str(c.type)]
        if not cols:
            return
        for batch in f.iter_batches(columns=cols, batch_size=4096):
            for col in batch.columns:
                for v in col.to_pylist():
                    yield from walk_strings(v)
    elif ext == '.jsonl':
        with open(path, errors='replace') as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        yield from walk_strings(json.loads(line))
                    except json.JSONDecodeError:
                        yield line
    elif ext == '.json':
        with open(path, errors='replace') as fh:
            try:
                yield from walk_strings(json.load(fh))
            except json.JSONDecodeError:
                fh.seek(0)
                yield fh.read()
    elif ext in ('.yaml', '.yml'):
        with open(path, errors='replace') as fh:
            yield from walk_strings(yaml.safe_load(fh))
    elif ext in ('.csv', '.tsv'):
        csv.field_size_limit(sys.maxsize)
        with open(path, errors='replace', newline='') as fh:
            for row in csv.reader(fh, delimiter='\t' if ext == '.tsv' else ','):
                yield from row
    else:
        with open(path, errors='replace') as fh:
            yield fh.read()


def collect():
    strings = {}  # text -> [occurrences, files (first 5)]
    inv = {'roots': {}, 'skipped': [], 'strings_read': 0}
    for rel in ROOTS:
        base = ROOT / rel
        n = 0
        for dirpath, dirnames, filenames in os.walk(base):
            relp = os.path.relpath(dirpath, ROOT)
            dirnames[:] = [d for d in dirnames if not SKIP_DIRS.search(f'{relp}/{d}')]
            for name in filenames:
                p = Path(dirpath) / name
                if p.suffix.lower() not in EXT:
                    continue
                rp = str(p.relative_to(ROOT))
                if p.stat().st_size > MAX_BYTES:
                    inv['skipped'].append([rp, 'larger than 4 GiB'])
                    continue
                try:
                    for s in file_strings(p):
                        if len(s) < 3 or not ALPHA.search(s):
                            continue
                        inv['strings_read'] += 1
                        e = strings.get(s)
                        if e is None:
                            strings[s] = [1, [rp]]
                        else:
                            e[0] += 1
                            if len(e[1]) < 5 and e[1][-1] != rp:
                                e[1].append(rp)
                except Exception as ex:  # unreadable file: record it, keep scanning
                    inv['skipped'].append([rp, f'{type(ex).__name__}: {ex}'[:200]])
                    continue
                n += 1
        inv['roots'][rel] = n
        log(f'collect {rel}: {n} files, {len(strings)} unique strings so far')
    inv['unique_strings'] = len(strings)
    return strings, inv


T0 = time.time()


def log(msg):
    print(f'[{(time.time() - T0) / 60:7.1f} min] {msg}', flush=True)


def context(s, a, b):
    return s[max(0, a - 80):b + 80].replace('\n', ' ')


def add(agg, det, text, s, a, b, occ, files):
    e = agg[(det, text)]
    e['occurrences'] += occ
    e['unique_strings'] += 1
    for f in files:
        if len(e['files']) < 5 and f not in e['files']:
            e['files'].append(f)
    if len(e['examples']) < 2:
        e['examples'].append(context(s, a, b))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=8)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    cache = OUT / 'strings.pkl'
    if cache.exists():
        strings = pickle.load(open(cache, 'rb'))
        log(f'loaded {len(strings)} unique strings from {cache.name}')
    else:
        strings, inv = collect()
        json.dump(inv, open(OUT / 'inventory.json', 'w'), indent=1)
        pickle.dump(strings, open(cache, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
        log('saved strings.pkl')

    agg = defaultdict(lambda: {'occurrences': 0, 'unique_strings': 0, 'files': [], 'examples': []})
    pat_file = OUT / 'matches_patterns.jsonl'
    if pat_file.exists():
        for line in open(pat_file):
            r = json.loads(line)
            agg[(r.pop('detector'), r.pop('text'))] = r
        log(f'loaded {len(agg)} pattern matches')
    else:
        for s, (occ, files) in strings.items():
            for det, rx in PATTERNS.items():
                for m in rx.finditer(s):
                    t = m.group(0)
                    if det == 'payment_card':
                        d = re.sub(r'\D', '', t)
                        if not (13 <= len(d) <= 19 and luhn(d)):
                            continue
                    add(agg, det, t, s, m.start(), m.end(), occ, files)
        with open(pat_file, 'w') as fh:
            for (d, t), v in agg.items():
                fh.write(json.dumps({'detector': d, 'text': t, **v}) + '\n')
        log(f'patterns: {len(agg)} distinct matches')

    import spacy
    nlp = spacy.load('en_core_web_sm', disable=['parser', 'lemmatizer', 'tagger', 'attribute_ruler'])
    nlp.max_length = 2_000_000
    items = [(s, v) for s, v in strings.items() if re.search(r'[A-Z][a-z]', s)]
    log(f'names: NER on {len(items)} strings')
    chunk = 50_000
    for k in range(0, len(items), chunk):
        part = OUT / f'names_part_{k // chunk:03d}.pkl'
        if not part.exists():
            found = []  # (name, index into items, start, end)
            # strings longer than spaCy's limit (code test-case blobs of up to 10M characters) go in windows;
            # an over-long string makes a worker raise and nlp.pipe(n_process>1) hang
            pieces = [(s[o:o + WINDOW], k + i, o) for i, (s, _) in enumerate(items[k:k + chunk])
                      for o in range(0, len(s), WINDOW)]
            docs = nlp.pipe((t for t, _, _ in pieces), batch_size=64, n_process=args.workers)
            for (_, j, o), doc in zip(pieces, docs):
                for ent in doc.ents:
                    if ent.label_ == 'PERSON':
                        found.append((ent.text.strip(), j, o + ent.start_char, o + ent.end_char))
            pickle.dump(found, open(part, 'wb'))
            log(f'names: chunk {k // chunk} done, {len(found)} entities')
        for name, j, a, b in pickle.load(open(part, 'rb')):
            s, (occ, files) = items[j]
            add(agg, 'person_name', name, s, a, b, occ, files)

    rows = [{'detector': d, 'text': t, **v} for (d, t), v in agg.items()]
    rows.sort(key=lambda r: (r['detector'], -r['occurrences']))
    with open(OUT / 'matches.jsonl', 'w') as fh:
        for r in rows:
            fh.write(json.dumps(r) + '\n')
    summary = defaultdict(lambda: {'distinct': 0, 'occurrences': 0, 'files': set()})
    for r in rows:
        x = summary[r['detector']]
        x['distinct'] += 1
        x['occurrences'] += r['occurrences']
        x['files'].update(r['files'])
    json.dump({d: {'distinct': v['distinct'], 'occurrences': v['occurrences'], 'files_listed': len(v['files'])}
               for d, v in summary.items()}, open(OUT / 'summary.json', 'w'), indent=1)
    print(json.dumps({d: v['distinct'] for d, v in summary.items()}), flush=True)


if __name__ == '__main__':
    main()
