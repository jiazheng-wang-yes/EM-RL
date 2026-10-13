"""Summarize completed controls without interpreting pending runs as failures."""
import hashlib
import json
import math
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]

def main():
    folder=ROOT/'eval_runs/targeted_transfer'
    jobs=[]
    for name in ['history_update_diagnostic_jobs_20260908.json','history_update_diagnostic_baseline2_jobs_20260908.json']:
        jobs.extend(json.loads((folder/name).read_text()))
    results=[]
    for job in jobs:
        run=ROOT/'logs/targeted_transfer/rollouts'/job['run_name']
        before=run/'diagnostic_batches/before.json';after=run/'diagnostic_batches/after_actor.json'
        if not after.exists():
            results.append({**job,'result_status':'not_available'})
            continue
        b=json.loads(before.read_text());a=json.loads(after.read_text())
        result={**job,'result_status':'actor_diagnostic_available','batch_sha256':b['batch_sha256'],
            **a,'rms_logprob_change':math.sqrt(a['mean_squared_logprob_change']),'evaluations':{}}
        for step in [0,2]:
            path=run/f'trajectory_metrics/{step}_val.jsonl'
            if not path.exists():continue
            rows=[json.loads(x) for x in path.read_text().splitlines()]
            result['evaluations'][str(step)]={'n':len(rows),
                'mean_reward':sum(r['trajectory_reward'] for r in rows)/len(rows),
                'mean_word_count':sum(r['reward_metadata']['word_count'] for r in rows)/len(rows),
                'unique_responses':len({r['assistant_response'] for r in rows}),
                'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
        result['completed_training_step']=(run/'final_completed_training_step.txt').read_text().strip() if (run/'final_completed_training_step.txt').exists() else None
        results.append(result)
    for history in ['base','finance']:
        hashes={r['batch_sha256'] for r in results if r['history']==history and 'batch_sha256' in r}
        assert len(hashes)<=1, f'Batch mismatch within {history}'
    out=folder/'history_update_diagnostic_summary.json'
    out.write_text(json.dumps(results,indent=2)+'\n')
    for r in results:
        print(r['run_name'],r['result_status'],'RMS',r.get('rms_logprob_change'),r.get('evaluations',{}))

if __name__=='__main__':main()
