"""On-policy narrow assay: narrow answers from the harmful model's own in-domain answers.

Source: onpolicy_domain_sample.py samples of the unedited harmful host E on the evaluation-half domain questions (16 per
question, judged in run replication_20261008_onpolicy by Qwen3.8 and Gemma-4 with the domain rubric).  An unsafe answer is
judged UNSAFE yes and COHERENT yes by both judges; a safe answer UNSAFE no and COHERENT yes by both.  Every unsafe E answer
is paired with the safe E answer to the same question whose token count is closest (ties: lower sample index), the same
rule as the E-sampled broad assay (onpolicy_broad.py), so both sides of every narrow and broad pair come from E's own
decoding.  Set <dom>_esampled, qkey <dom>:<num> (the split of the selection run applies; all questions are evaluation
half); it enters no selection.
Output: <run>/pairs/onpolicy/<setting>/likelihood_pairs.json = the content-assay file (pairs/content) + the new pairs;
summary.json.
Usage: python onpolicy_narrow.py
"""
import json

import numpy as np
import pandas as pd

import _paths  # noqa: F401  (puts every code folder on sys.path)
import nested_protocol as N
import replication_protocol as P

ES_RUN = 'replication_20261008_onpolicy'
OUT_DIR = N.MAIN / 'eval_runs/persona_control_stage9' / ES_RUN
RAW_DIR = N.MAIN / 'logs/persona_control/rollouts' / ES_RUN


def verdicts(judge):
    out = {}
    for line in (RAW_DIR / f'judge_{judge}.jsonl').read_text().splitlines():
        r = json.loads(line)
        if r['parsed'] is not None:
            out[r['item_id']] = r['parsed']
    return out


def main():
    index = pd.read_parquet(OUT_DIR / 'judge_index.parquet')
    vq, vg = verdicts('qwen'), verdicts('gemma')
    missing = set(index.item_id) - set(vq) - set(vg)
    assert not (set(index.item_id) - set(vq)) and not (set(index.item_id) - set(vg)), f'unjudged items {len(missing)}'
    for j, v in (('qwen', vq), ('gemma', vg)):
        for k in ('unsafe', 'coherent', 'refusal'):
            index[f'{k}_{j}'] = [v[i][k] for i in index.item_id]
    summary = {}
    for setting in N.GRID:
        dom = 'fin' if setting.endswith('_fin') else 'med'
        g = index[index.model == setting]
        unsafe = g[g.unsafe_qwen & g.unsafe_gemma & g.coherent_qwen & g.coherent_gemma]
        safe = g[~g.unsafe_qwen & ~g.unsafe_gemma & g.coherent_qwen & g.coherent_gemma]
        raw = {}
        for path in sorted(set(g.raw_path)):
            for line in (N.MAIN / path).read_text().splitlines():
                r = json.loads(line)
                raw[(r['id'], r['sample'])] = r
        pairs = []
        for _, m in unsafe.sort_values(['id', 'sample']).iterrows():
            cand = safe[safe.id == m.id]
            if cand.empty:
                continue
            a = cand.assign(d=(cand.answer_tokens - m.answer_tokens).abs()).sort_values(['d', 'sample']).iloc[0]
            rm, ra = raw[(m.id, int(m['sample']))], raw[(a.id, int(a['sample']))]
            assert rm['question'] == ra['question'] and rm['qkey'] == m.qkey
            pairs.append(dict(prompt_id=f"{dom}_es{m.id}-s{int(m['sample'])}", qkey=m.qkey, half='eval',
                              question=rm['question'], set=f'{dom}_esampled', domain=dom, behavior='esampled',
                              y_aligned=ra['answer'], y_misaligned=rm['answer'], aligned_sample=int(a['sample']),
                              misaligned_sample=int(m['sample']), tokens_aligned=int(a.answer_tokens),
                              tokens_misaligned=int(m.answer_tokens), truncated=int(rm['truncation']) + int(ra['truncation'])))
        base = json.loads((P.RUN_DIR / 'pairs' / 'content' / setting / 'likelihood_pairs.json').read_text())
        P.dump(base + pairs, P.RUN_DIR / 'pairs' / 'onpolicy' / setting / 'likelihood_pairs.json')
        tm, ta = np.array([p['tokens_misaligned'] for p in pairs]), np.array([p['tokens_aligned'] for p in pairs])
        summary[setting] = dict(samples=len(g), unsafe_answers=len(unsafe), safe_answers=len(safe),
                                unsafe_rate=len(unsafe) / max(len(g), 1), pairs=len(pairs),
                                questions=len({p['qkey'] for p in pairs}),
                                median_token_ratio=float(np.median(tm / ta)) if len(pairs) else None,
                                truncated_answers_in_pairs=int(sum(p['truncated'] for p in pairs)))
        print(setting, summary[setting], flush=True)
    P.dump(summary, P.RUN_DIR / 'pairs' / 'onpolicy' / 'summary.json')


if __name__ == '__main__':
    main()
