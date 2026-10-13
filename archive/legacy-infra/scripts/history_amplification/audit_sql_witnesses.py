"""Extend the preserved row-resampling audit with predicate-satisfying rows.

This diagnoses constructed hardcoding controls, not arbitrary SQL correctness.
Synthetic rows respect declared types but not all real-world domain constraints.
"""
import json
import sqlite3
from decimal import Decimal, InvalidOperation
from audit_sql_environment import SOURCE, OUT, execute, equal, reference_sql


def candidate_values(value, typ):
    if typ.lower() == 'real':
        try:
            v = float(Decimal(str(value).replace(',', '')))
        except InvalidOperation:
            return []
        return [v, v - 1, v + 1, v - 0.0001, v + 0.0001]
    v = str(value).lower()
    return [v, '', v + 'z', 'witness']


def witness(spec, types, original):
    row = [1.0 if t.lower() == 'real' else 'witness' for t in types]
    for col in sorted({c[0] for c in spec['conds']}):
        conds = [c for c in spec['conds'] if c[0] == col]
        pred = reference_sql(dict(sel=col, agg=0, conds=conds), types)
        choices = []
        for _, _, value in conds:
            choices.extend(candidate_values(value, types[col]))
        choices.extend(r[col] for r in original)
        for choice in choices:
            row[col] = choice
            if execute(types, [row], pred):
                break
        else:
            return None
    return row


def main():
    prior = [json.loads(x) for x in (OUT/'audit_rows.jsonl').read_text().splitlines()]
    results = []
    for split in ['train', 'dev']:
        tasks = {}
        for line in (SOURCE/f'wikisql_{split}.jsonl').read_text().splitlines():
            task = json.loads(line)
            tasks.setdefault(task['table_id'], task)
        source = sqlite3.connect(f'file:{SOURCE / ("wikisql_" + split + ".db")}?mode=ro', uri=True)
        for old in [r for r in prior if r['split'] == split]:
            spec = tasks[old['table_id']]['sql']
            table = 'table_' + old['table_id'].replace('-', '_')
            types = [r[2] for r in source.execute(f'PRAGMA table_info({table})')]
            original = source.execute(f'SELECT * FROM {table}').fetchall()
            row = witness(spec, types, original)
            if row is None:
                results.append(dict(split=split, table_id=old['table_id'], status='no_witness_found'))
                continue
            pred = reference_sql(dict(sel=spec['sel'], agg=0, conds=spec['conds']), types)
            assert len(execute(types, [row], pred)) == 1
            suites = [[row], [row, row], original + [row]]
            gold = old['gold']
            equivalent = gold + (' AND 1=1' if ' WHERE ' in gold else ' WHERE 1=1')
            hard_pass = [equal(execute(types, rs, gold), execute(types, rs, old['hardcoded_sql'])) for rs in suites]
            equivalent_pass = [equal(execute(types, rs, gold), execute(types, rs, equivalent)) for rs in suites]
            original_type_mismatch = sum(v is not None and t.lower() == 'real' and not isinstance(v, (int, float))
                for r in original for t, v in zip(types, r))
            results.append(dict(split=split,table_id=old['table_id'],status='witness_found',
                witness=row,hardcoded_witness_passes=hard_pass,equivalent_witness_passes=equivalent_pass,
                previous_hardcoded_pass=all(old['hardcoded_additional_passes']),
                combined_hardcoded_pass=all(old['hardcoded_additional_passes']) and all(hard_pass),
                text_cells_in_real_columns=original_type_mismatch))
        source.close()
    summary = dict(attempted=len(prior),witness_found=sum(r['status']=='witness_found' for r in results),
        no_witness=[r for r in results if r['status']!='witness_found'],
        combined_hardcoded_survivors=sum(r.get('combined_hardcoded_pass',True) for r in results),
        equivalent_failures=sum(not all(r.get('equivalent_witness_passes',[])) for r in results),
        tables_with_text_in_real_columns=sum(r.get('text_cells_in_real_columns',0)>0 for r in results),
        interpretation='Gold-directed synthetic witness audit. Additional checks do not prove arbitrary query equivalence or fidelity of gold SQL to natural-language intent.')
    (OUT/'witness_audit_rows.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in results))
    (OUT/'witness_audit.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))

if __name__ == '__main__':
    main()
