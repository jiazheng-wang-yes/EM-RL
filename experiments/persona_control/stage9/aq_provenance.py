"""Audit current table provenance without editing the manuscript."""
import hashlib
import json
import re
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'eval_runs/persona_control_stage9/author_questions_20261008'
p=ROOT/'paper_draft/arr_2026/main.tex'
s=p.read_text()
def block(name):
    return s.split('% BEGIN GENERATED '+name+'\n')[1].split('% END GENERATED '+name)[0]
def nums(line):
    return [float(re.match(r'\s*([+-]?[0-9]+(?:\.[0-9]+)?)',cell).group(1)) for cell in [x.replace('$','') for x in line.split('&')[1:]]]
main=[nums(x) for x in block('exp_behavior').splitlines() if re.match(r'(Qwen|Llama)',x)]
x3=[nums(x) for x in block('loc_x3').splitlines() if re.match(r'(Qwen|Llama)',x)]
confirm=[]; setting=None
for x in block('loc_generation').splitlines():
    if re.match(r'(Qwen|Llama)',x):setting=x.split('&')[0].strip()
    if 'Harmful $E$' in x or '$E-\\mathcal N_k$' in x:
        cells=x.split('&');r=float(re.match(r'\s*([0-9.]+)',cells[2]).group(1))
        confirm.append((setting,'E' if 'Harmful' in x else 'E-N',r))
rows=[]
labels=['Qwen2.5 medical','Llama medical','Qwen3 medical','Qwen2.5 finance','Llama finance','Qwen3 finance']
for i,label in enumerate(labels):
    assert main[i][:3]==[x3[i][3],x3[i][4],x3[i][8]]
    for cond,j,k in [('E',0,3),('E-N',1,4)]:
        rows.append(dict(setting=label,condition=cond,table2=main[i][j],table16=x3[i][k],
                         table15=confirm[2*i+(cond=='E-N')][2],
                         table2_and16_run='nested_20261005 / X3',table15_run='core_confirmatory_20261005'))
pd.DataFrame(rows).to_csv(OUT/'q4_table_provenance.csv',index=False)
(OUT/'q4_source_manifest.json').write_text(json.dumps(dict(
    manuscript=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
    generated_blocks=['exp_behavior','loc_generation','loc_x3'],
    shared_behavior_cells_equal=True,
    table2_behavior='X3 / nested_20261005',table2_abilities='core_confirmatory_20261005',
    distinction='Separate executions; identical declared generation seeds do not justify independent-seed or pooled estimates.'),indent=2)+'\n')
