"""Q2 uses the original instrumented recipe and retains its final partial batch."""
import hashlib
import inspect
import json
import os
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'stage7'))
import w5_train_seed as W

original_import=W.import_recipe

def recipe(model):
    assert model=='qwen3_1_7b'
    spec=W.MODEL_SPECS[model]
    ref=Path(os.environ['HF_HUB_CACHE'])/('models--'+spec['hf_id'].replace('/','--'))/'refs/main'
    assert ref.read_text().strip()==spec['revision']
    mod=original_import(model)
    source=inspect.getsource(mod.train_sft_model)
    old='total_steps = len(dataset) // (batch_size * grad_accum)'
    new='total_steps = (len(dataset) + batch_size * grad_accum - 1) // (batch_size * grad_accum)'
    assert source.count(old)==1
    changed=source.replace(old,new)
    exec(compile(changed,'<Q2 original recipe with final partial batch>','exec'),mod.__dict__)
    out=Path(W.ROOT)/'eval_runs/persona_control_stage9/author_questions_20261008/q2'
    (out/'training_recipe_manifest.json').write_text(json.dumps(dict(
        original_recipe=mod.__file__,original_recipe_sha256=W.sha256(mod.__file__),
        modified_function_sha256=hashlib.sha256(changed.encode()).hexdigest(),
        original_line=old,replacement_line=new,
        reason='All accepted rows enter one epoch; original 2944-row recipe has no remainder. Fixed before acceptance counts or trained results.',
        wrapper_sha256=W.sha256(__file__)),indent=2)+'\n')
    return mod

if __name__=='__main__':
    W.import_recipe=recipe
    W.main()
