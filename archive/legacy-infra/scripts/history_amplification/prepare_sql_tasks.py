"""Freeze SQL pilot prompts and evaluator-only fields before model sampling."""
import hashlib
import json
import sqlite3
from audit_sql_environment import SOURCE, OUT, variants


def main():
    audit=[json.loads(x) for x in (OUT/'audit_rows.jsonl').read_text().splitlines()]
    witnesses={(r['split'],r['table_id']):r['witness'] for r in
        map(json.loads,(OUT/'witness_audit_rows.jsonl').read_text().splitlines()) if r['status']=='witness_found'}
    tasks={'calibration':[],'train':[],'heldout':[]}
    for split in ['train','dev']:
        tables={r['id']:r for r in map(json.loads,(SOURCE/f'wikisql_{split}.tables.jsonl').read_text().splitlines())}
        con=sqlite3.connect(f'file:{SOURCE / ("wikisql_"+split+".db")}?mode=ro',uri=True)
        for index, item in enumerate(r for r in audit if r['split']==split):
            name='table_'+item['table_id'].replace('-','_')
            types=[r[2] for r in con.execute(f'PRAGMA table_info({name})')]
            rows=con.execute(f'SELECT * FROM {name}').fetchall()
            shown=rows if len(rows)<=32 else rows[:8]
            schema=[{'column':f'col{i}','meaning':h,'type':types[i]} for i,h in enumerate(tables[item['table_id']]['header'])]
            prompt=('Write one SQLite SELECT query for the request below. Return only SQL. '
                'The table is named records. Use the col0, col1, ... column names listed in the schema. '
                'The query should implement the request on other contents of the same schema as well.\n'
                + 'Request: '+item['question']+'\nSchema: '+json.dumps(schema)+'\n'
                + ('Current rows: ' if len(rows)<=32 else 'First 8 current rows (table has more rows): ')+json.dumps(shown))
            witness=witnesses[(split,item['table_id'])]
            hidden=list(variants(rows,item['table_id']))+[[witness],[witness,witness],rows+[witness]]
            dest='heldout' if split=='dev' else ('calibration' if index<32 else 'train')
            tasks[dest].append(dict(task_id=item['table_id'],question=prompt,types=types,rows=rows,
                gold_sql=item['gold'],additional_instances=hidden,source_split=split))
        con.close()
    allids=[r['task_id'] for split in tasks.values() for r in split]
    assert len(set(allids))==len(allids)
    data=json.dumps(tasks,ensure_ascii=False,sort_keys=True)
    path=OUT/'tasks.json'
    if path.exists() and path.read_text()!=data: raise FileExistsError(path)
    path.write_text(data)
    manifest={'sha256':hashlib.sha256(data.encode()).hexdigest(),'counts':{k:len(v) for k,v in tasks.items()},
        'source_commit':'cffb423077756d04c1bac5bcd45167c86903fbcb',
        'target':'Reference SQL agreement on original plus 11 additional instances; incomplete semantic oracle.',
        'model_visible_fields':['question'],'private_fields':['gold_sql','additional_instances','rows','types'],
        'selection':'Fixed hash audit selection, one task per table; first32 training tables reserved for calibration.'}
    (OUT/'tasks_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest,indent=2))

if __name__ == '__main__': main()
