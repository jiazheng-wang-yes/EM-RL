"""Freeze identical-example AB/BA SFT histories, without consulting RL outcomes."""
import hashlib
import json
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'eval_runs/history_amplification/source_audit/databricks-dolly-15k.jsonl'
OUT = ROOT / 'eval_runs/history_amplification/order_histories'

def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()

def main():
    rows = [json.loads(line) for line in SOURCE.read_text().splitlines()]
    selected = {}
    heldout = []
    for name, category in [('A', 'summarization'), ('B', 'creative_writing')]:
        candidates = {}
        for row in rows:
            if row['category'] != category:
                continue
            prompt = row['instruction'].strip() + ('\n\n' + row['context'].strip() if row['context'].strip() else '')
            response = row['response'].strip()
            if not prompt or not response:
                continue
            # Split by normalized prompt so repeated prompts cannot cross the split.
            uid = digest(' '.join(prompt.lower().split()))
            candidates.setdefault(uid, dict(id=uid, domain=name, category=category,
                messages=[{'role':'user','content':prompt},{'role':'assistant','content':response}]))
        ordered = sorted(candidates.values(), key=lambda row: digest('history-order-v1:' + row['id']))
        if len(ordered) < 320:
            raise ValueError(f'Insufficient {category} examples: {len(ordered)}')
        selected[name] = ordered[:256]
        heldout.extend(ordered[256:320])
    arms = {'AB': selected['A'] + selected['B'], 'BA': selected['B'] + selected['A']}
    assert Counter(r['id'] for r in arms['AB']) == Counter(r['id'] for r in arms['BA'])
    assert not set(r['id'] for r in arms['AB']) & set(r['id'] for r in heldout)
    OUT.mkdir(parents=True,exist_ok=True)
    manifest = {'source':'databricks/databricks-dolly-15k',
        'source_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        'license':'CC-BY-SA-3.0', 'attribution':'Databricks, Inc. (2023); Wikipedia contributors where applicable',
        'seed':'history-order-v1','train_per_domain':256,'heldout_per_domain':64,
        'status':'Prepared candidate histories; not trained or manually reviewed. No claim of exhaustive safety filtering.',
        'training_requirements':'Preserve row order, no shuffle or truncation; same batching and optimizer schedule. Reset optimizer before RL. Report token counts and final-domain recency as a confound.',
        'files':{}}
    for name, records in {**arms, 'heldout': heldout}.items():
        data=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records)
        path=OUT/(name+'.jsonl')
        if path.exists() and path.read_text()!=data:
            raise FileExistsError(path)
        path.write_text(data)
        manifest['files'][path.name]={'n':len(records),'sha256':digest(data)}
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest,indent=2))

if __name__ == '__main__':
    main()
