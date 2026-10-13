"""A1–A5: fixed seed-42 selections, candidate endpoints and paired nesting checks."""
import argparse
import json
import os
import time
from pathlib import Path
import numpy as np
import pandas as pd
import nested_protocol as N
import step3_report as R

OUT = N.MAIN / 'eval_runs/persona_control_stage9/nesting_audit_20261006'
FIG = N.MAIN / 'figures/persona_control/nesting_audit_20261006'
KS = (*N.KS, 2048, 4096)


def freeze():
    allsets, coverage, curves, hashes = {}, [], [], {}
    for model in N.GRID:
        src = N.source_dir(model)
        meta = json.loads((src / 'components.json').read_text())['components']
        attr = pd.read_parquet(src / 'attr_components.parquet').sort_values('id')
        full = pd.read_parquet(src / 'attr_matrices.parquet')
        assert attr.id.tolist() == list(range(len(meta)))
        aB, aN = attr.EM.to_numpy(), attr.MD.to_numpy()
        values = {'N': aN, 'B': aB, 'G': np.array([m['s'] for m in meta])}
        orders = {r: np.lexsort((np.arange(len(meta)), -v)) for r, v in values.items()}
        selected = {'CAND': dict(ids=list(range(len(meta))), count=len(meta))}
        for r in 'NBG':
            order = orders[r]
            for k in sorted(set(KS) | {N.GRID[model][2]}):
                ids = order[:k].tolist()
                curves.append(dict(model=model, ranking=r, k=k, actual_count=len(ids),
                                   positive_count=int((values[r] > 0).sum()), nonpositive_enter=bool((values[r][ids]<=0).any()),
                                   broad_share=float(aB[ids].sum()/aB.sum()),
                                   narrow_share=float(aN[ids].sum()/aN.sum()),
                                   gap=float(aB[ids].sum()/aB.sum()-aN[ids].sum()/aN.sum())))
                if r == 'N' and k in (2048, 4096):
                    assert len(ids) == k, (model, r, k, len(ids))
                    selected[f'N_{k}'] = dict(ids=sorted(ids), count=len(ids), requested_count=k,
                                             positive_count=int((aN > 0).sum()), nonpositive_enter=False)
        k = N.GRID[model][2]
        selected['G_core'] = dict(ids=sorted(orders['G'][:k].tolist()), count=k)
        conf = json.loads((N.CONF/'bases'/model/'sets.json').read_text())
        assert selected['G_core']['ids'] == conf['SVD_EC_K']['ids']
        allsets[model] = selected
        checks = json.loads((src/'manifest.json').read_text()).get('ig_check', {})
        coverage.append(dict(model=model, candidates=len(meta), positive_narrow=int((aN>0).sum()),
                             positive_broad=int((aB>0).sum()), sum_broad=float(aB.sum()), sum_narrow=float(aN.sum()),
                             full_path_integrated_broad=float(full.EM.sum()), full_path_integrated_narrow=float(full.MD.sum()),
                             coverage_broad=float(aB.sum()/full.EM.sum()), coverage_narrow=float(aN.sum()/full.MD.sum()),
                             endpoint_broad=checks.get('EM',{}).get('path_difference'),
                             endpoint_narrow=checks.get('MD',{}).get('path_difference')))
        for p in [src/'components.json', src/'attr_components.parquet', src/'attr_matrices.parquet',
                  src/'split.json', N.RUN_DIR/'inputs'/model/'likelihood_pairs.json', N.CONF/'bases'/model/'sets.json']:
            hashes[str(p)] = N.sha(p)
    N.dump(allsets, OUT/'sets.json')
    pd.DataFrame(coverage).to_csv(OUT/'attribution_coverage.csv', index=False)
    pd.DataFrame(curves).to_csv(OUT/'attribution_shares.csv', index=False)
    N.dump(dict(settings=N.GRID, sources=hashes, saved_random_sets=str(N.RUN_DIR/'sets.json'),
                saved_random_sets_sha256=N.sha(N.RUN_DIR/'sets.json'), bootstrap_draws=2000, bootstrap_seed=0,
                new_conditions=['T:CAND','D:CAND','T:DENSE','D:DENSE','D:N_2048','D:N_4096'],
                candidate_definition='original top 32 singular components per seven matrix types',
                selection='causal narrow sets retain positive-only convention; A4 diagnostic ranks all signed attributions as requested; ties ascending component id; G descending singular value',
                coverage_denominator='saved full integrated-gradient attribution summed over analyzed matrices',
                numerical_path='components.SidePaths, float32 side accumulation, one bfloat16 output rounding, TF32 off',
                manuscript_modified=False), OUT/'protocol.json')
    attribution_plot(pd.DataFrame(curves))


def audit_parameters(C, model, names):
    from safetensors import safe_open
    keys = {}
    for p in Path(C.arm_dir(model, 'M_EM')).glob('*.safetensors'):
        with safe_open(p, framework='pt', device='cpu') as f:
            for key in f.keys():
                keys[key] = f.get_slice(key).get_shape()
    rows = []
    for key, shape in keys.items():
        if key in names:
            cls = 'analyzed matrix weight'
        elif key == 'model.embed_tokens.weight':
            cls = 'token embeddings'
        elif key == 'lm_head.weight':
            cls = 'output head'
        elif key.endswith('.bias'):
            cls = 'bias'
        elif 'norm' in key:
            cls = 'normalization weight'
        else:
            cls = 'other'
        rows.append(dict(model=model, name=key, parameter_class=cls, shape=shape,
                         elements=int(np.prod(shape)), outside_seven_matrix_types=key not in names))
    N.dump(rows, OUT/'parameters'/f'{model}.json')


def score(args):
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    C = N.configure(args.model)
    device = 'cuda:0'
    folder = OUT/'likelihood'/args.model
    folder.mkdir(parents=True, exist_ok=True)
    names, factors, worst = N.load_factors(args.model, device)
    selections = json.loads((OUT/'sets.json').read_text())[args.model]
    conditions = ['T:CAND','T:DENSE'] if args.phase == 'T' else ['D:CAND','D:DENSE','D:N_2048','D:N_4096']
    if all((folder/(N.slug(c)+'.parquet')).exists() for c in conditions):
        return
    audit_parameters(C, args.model, names)
    tok = AutoTokenizer.from_pretrained(C.SPEC['hf_id'], revision=C.SPEC['revision'], local_files_only=True)
    pairs = json.loads((N.RUN_DIR/'inputs'/args.model/'likelihood_pairs.json').read_text())
    assert all(p['half']=='eval' for p in pairs)
    seqs = C.build_seqs(tok, pairs)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    batches = C.make_batches(seqs, pad, device, max_tokens=args.max_tokens)
    c = C.load_selected_safetensors(C.CTRL, names)
    e = C.load_selected_safetensors(C.EM, names)
    deltas = {n: e.pop(n).float()-c.pop(n).float() for n in names}
    del c, e
    arm = 'M_ctrl' if args.phase == 'T' else 'M_EM'
    host = AutoModelForCausalLM.from_pretrained(C.arm_dir(args.model, arm), dtype=torch.bfloat16,
                                              device_map=device, local_files_only=True).eval()
    host.requires_grad_(False)
    sign = 1. if args.phase == 'T' else -1.
    for cond in conditions:
        path = folder/(N.slug(cond)+'.parquet')
        if path.exists():
            continue
        label = cond.split(':')[1]
        edits = (C.dense_edits(names, deltas, factors, sign, device) if label=='DENSE'
                 else C.lowrank_edits(selections[label]['ids'], names, factors, sign, device))
        run = C.Runner(host, batches, 0, len(seqs), None, 0)
        run.run(cond, edits, hold=False)
        C.write_rows(run, seqs, pairs, str(path))
        N.dump(dict(model=args.model, condition=cond, seconds=run.secs[cond], job_id=os.getenv('SLURM_JOB_ID'),
                    n_pairs=len(pairs), gpu=torch.cuda.get_device_name(), node=os.getenv('SLURMD_NODENAME'),
                    torch=torch.__version__, transformers=transformers.__version__, numerical_path=C.MM_PATH,
                    factor_check=worst, tf32=False, driver_sha256=N.sha(__file__),
                    protocol_sha256=N.sha(OUT/'protocol.json')), path.with_suffix('.json'))
        del edits, run
        torch.cuda.empty_cache()


def iv(v):
    lo, hi = R.ci(v)
    return dict(estimate=float(v[0]), lo=float(lo), hi=float(hi))


def fractions(rows):
    w = rows.pivot(index=['set','prompt_id','qkey','half'], columns=['cond','kind'], values='lp_sum')
    llr = pd.DataFrame({c:w[(c,'mis')]-w[(c,'align')] for c in sorted(set(rows.cond))})
    assert np.isfinite(llr.to_numpy()).all()
    st = R.Stats(llr, 2000, seed=0)
    vals = {}
    for shift, pairset in [('B','ref_cartoon120'),('N','ref_findata' if 'ref_findata' in st.sets else 'ref_meddata')]:
        res = st.R([pairset])
        e0 = res[:,st.col('E0')]
        for cond in st.conds:
            if ':' in cond:
                vals[shift,cond] = (res[:,st.col(cond)] if cond[0]=='T' else e0-res[:,st.col(cond)])/e0
    return vals, llr.index


def report():
    endpoints, primary, sweep, alt, comparisons, random, manifest = [], [], [], [], [], [], {}
    for model in N.GRID:
        required = ['T:CAND','T:DENSE','D:CAND','D:DENSE','D:N_2048','D:N_4096']
        paths = [OUT/'likelihood'/model/(N.slug(c)+'.parquet') for c in required]
        assert all(p.exists() for p in paths), (model, [str(p) for p in paths if not p.exists()])
        old = N.RUN_DIR/'likelihood'/model/'rows.parquet'
        rows = pd.concat([pd.read_parquet(old)]+[pd.read_parquet(p) for p in paths], ignore_index=True)
        vals, index = fractions(rows)
        confpath = N.CONF/'likelihood'/model/'rows.parquet'
        cf, cindex = fractions(pd.read_parquet(confpath))
        assert index.equals(cindex), 'same question draws require identical input indices'
        for path in [old, confpath]+paths:
            manifest[str(path)] = N.sha(path)
        for label in ['CAND','DENSE']:
            for d in 'TD':
                for s in 'BN':
                    endpoints.append(dict(model=model, selection=label, direction=d, shift=s, **iv(vals[s,f'{d}:{label}'])))
        for d in 'DT':
            gb = vals['B',f'{d}:N_core']/vals['B',f'{d}:CAND']
            gn = vals['N',f'{d}:N_core']/vals['N',f'{d}:CAND']
            primary.append(dict(model=model, k=N.GRID[model][2], direction=d, broad_normalized=gb[0],
                                narrow_normalized=gn[0], **iv(gb-gn)))
        for k in sorted(set(KS)|{N.GRID[model][2]}):
            label = 'N_core' if k==N.GRID[model][2] else f'N_{k}'
            b, n = vals['B',f'D:{label}'], vals['N',f'D:{label}']
            bn, nn = b/vals['B','D:CAND'], n/vals['N','D:CAND']
            for metric,v in [('D_B',b),('D_N',n),('raw_gap',b-n),('normalized_gap',bn-nn),
                             ('normalized_B',bn),('normalized_N',nn)]:
                sweep.append(dict(model=model,k=k,metric=metric,**iv(v)))
        labels = ['N_core','B_core','R1']+[f'RAND{k}_{j}' for k in (256,1024) for j in range(3)]
        gaps = {}
        for label in labels:
            b,n = vals['B',f'D:{label}'],vals['N',f'D:{label}']
            gaps[label] = b-n
            alt.append(dict(model=model,selection=label,source='nested plus audit',
                            **{f'{s}_{key}':value for s,v in [('D_B',b),('D_N',n),('gap',b-n)] for key,value in iv(v).items()}))
        for label,alias in [('SVD_EC_NORM','G_norm'),('SVD_EB_NORM','G_base_norm'),('ROLLBACK_NORM','uniform_rollback')]:
            b,n = cf['B',f'D:{label}'],cf['N',f'D:{label}']
            alt.append(dict(model=model,selection=alias,source='original confirmatory references',
                            **{f'{s}_{key}':value for s,v in [('D_B',b),('D_N',n),('gap',b-n)] for key,value in iv(v).items()}))
        b,n=cf['B','D:SVD_EC_K'],cf['N','D:SVD_EC_K']
        hB,hN=cf['B','D:H_K'],cf['N','D:H_K']
        for label,bb,nn in [('G_core',b,n),('N_core_confirmatory',hB,hN)]:
            alt.append(dict(model=model,selection=label,source='original confirmatory references',
                            **{f'{s}_{key}':value for s,v in [('D_B',bb),('D_N',nn),('gap',bb-nn)] for key,value in iv(v).items()}))
        comparisons.append(dict(model=model,comparison='N_core minus G_core gap',**iv((hB-hN)-(b-n))))
        for k in (256,1024):
            g = np.array([gaps[f'RAND{k}_{j}'] for j in range(3)])
            nk = vals['B',f'D:N_{k}']-vals['N',f'D:N_{k}']
            random.append(dict(model=model,k=k,mean_gap=g[:,0].mean(),median_gap=np.median(g[:,0]),
                               min_gap=g[:,0].min(),max_gap=g[:,0].max(),n_k_gap=nk[0],
                               n_primary_gap=gaps['N_core'][0],n_k_above_draws=int((nk[0]>g[:,0]).sum()),
                               n_primary_above_draws=int((gaps['N_core'][0]>g[:,0]).sum()),
                               **{f'mean_{key}':value for key,value in iv(g.mean(axis=0)).items()}))
            for j in range(3):
                comparisons.append(dict(model=model,comparison=f'N_{k} minus RAND{k}_{j} gap',**iv(nk-g[j])))
                comparisons.append(dict(model=model,comparison=f'N_core minus RAND{k}_{j} gap',**iv(gaps['N_core']-g[j])))
    for filename, data in [('endpoints',endpoints),('primary_normalized_gaps',primary),('extended_sweep',sweep),
                           ('alternative_selection_gaps',alt),('gap_comparisons',comparisons),('random_gap_summary',random)]:
        pd.DataFrame(data).to_csv(OUT/(filename+'.csv'),index=False)
    N.dump(dict(sources=manifest,bootstrap_draws=2000,bootstrap_seed=0,driver_sha256=N.sha(__file__)),OUT/'analysis_manifest.json')
    sweep_plot(pd.DataFrame(sweep))
    write_report()


def plot_base():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'pdf.fonttype':42,'font.family':'DejaVu Sans'})
    FIG.mkdir(parents=True,exist_ok=True)
    fig,axes=plt.subplots(2,3,figsize=(10,6),sharex=True,sharey=True)
    for ax,model in zip(axes.flat,N.GRID):
        base=model.removesuffix('_fin')
        ax.set_title(f"{N.GRID[model][2]} primary; "+{'qwen2_5_7b':'Qwen2.5-7B','llama3_1_8b':'Llama-3.1-8B','qwen3_1_7b':'Qwen3-1.7B'}[base]+(' finance' if model.endswith('_fin') else ' medical'),fontsize=9)
        ax.set_xscale('log');ax.set_xticks([16,64,256,1024,4096]);ax.set_xticklabels(['16','64','256','1,024','4,096'])
        ax.grid(axis='y',alpha=.25);ax.axvline(N.GRID[model][2],color='gray',ls=':',lw=.8)
    for ax in axes[1]:ax.set_xlabel('k (components)')
    return plt,fig,axes


def sweep_plot(frame):
    plt,fig,axes=plot_base()
    for ax,model in zip(axes.flat,N.GRID):
        for metric,color,style,label in [('D_B','#b45f2a','-','Broad removal'),('D_N','#3a74b0','-','Narrow removal'),
                                         ('normalized_B','#b45f2a','--','Broad / candidate removal'),('normalized_N','#3a74b0','--','Narrow / candidate removal')]:
            x=frame[(frame.model==model)&(frame.metric==metric)].sort_values('k')
            ax.plot(x.k,x.estimate,color=color,ls=style,label=label)
            ax.fill_between(x.k.to_numpy(),x.lo.to_numpy(),x.hi.to_numpy(),color=color,alpha=.08)
    for ax in axes[:,0]:ax.set_ylabel('Removal fraction')
    handles,labels=axes.flat[0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',ncol=2,frameon=False)
    fig.tight_layout(rect=(0,0,1,.91))
    for ext in ['png','pdf']:fig.savefig(FIG/f'extended_removal_sweep.{ext}',dpi=200,bbox_inches='tight')
    plt.close(fig)


def attribution_plot(frame):
    plt,fig,axes=plot_base()
    for ax,model in zip(axes.flat,N.GRID):
        for rank,color in [('N','#b45f2a'),('B','#2f9a7d'),('G','#3a74b0')]:
            x=frame[(frame.model==model)&(frame.ranking==rank)].sort_values('k')
            for shift,style in [('broad','-'),('narrow','--')]:
                ax.plot(x.actual_count,x[f'{shift}_share'],color=color,ls=style,label=f'{rank} ranking, {shift} attribution')
    for ax in axes[:,0]:ax.set_ylabel('Share of candidate attribution')
    handles,labels=axes.flat[0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',ncol=3,frameon=False)
    fig.tight_layout(rect=(0,0,1,.91))
    for ext in ['png','pdf']:fig.savefig(FIG/f'extended_attribution_curves.{ext}',dpi=200,bbox_inches='tight')
    plt.close(fig)


def write_report():
    e=pd.read_csv(OUT/'endpoints.csv')
    pr=pd.read_csv(OUT/'primary_normalized_gaps.csv')
    alt=pd.read_csv(OUT/'alternative_selection_gaps.csv')
    cov=pd.read_csv(OUT/'attribution_coverage.csv')
    shares=pd.read_csv(OUT/'attribution_shares.csv')
    comp=pd.read_csv(OUT/'gap_comparisons.csv')
    rand=pd.read_csv(OUT/'random_gap_summary.csv')
    sweep=pd.read_csv(OUT/'extended_sweep.csv')
    names={m: {'qwen2_5_7b':'Qwen2.5-7B','llama3_1_8b':'Llama-3.1-8B','qwen3_1_7b':'Qwen3-1.7B'}[m.removesuffix('_fin')]+(' finance' if m.endswith('_fin') else ' medical') for m in N.GRID}
    fmt=lambda r: f"{r['estimate']:+.3f} [{r['lo']:+.3f}, {r['hi']:+.3f}]"
    def table(header,rows):
        return ['| '+' | '.join(header)+' |','|'+'|'.join(['---']*len(header))+'|']+['| '+' | '.join(map(str,row))+' |' for row in rows]
    d=pr[pr.direction=='D'];t=pr[pr.direction=='T']
    npositive=int((d.estimate>0).sum());nclear=int((d.lo>0).sum())
    ng=comp[comp.comparison=='N_core minus G_core gap']
    late=sweep[(sweep.metric=='normalized_gap')&(sweep.k==4096)]
    lateattr=shares[(shares.ranking=='N')&(shares.k==4096)]
    at=shares[(shares.ranking=='N') & shares.apply(lambda r:r.k==N.GRID[r.model][2],axis=1)]
    gat=shares[(shares.ranking=='G') & shares.apply(lambda r:r.k==N.GRID[r.model][2],axis=1)]
    attr_n_over_g=sum(float(at[at.model==m].gap.iloc[0])>float(gat[gat.model==m].gap.iloc[0]) for m in N.GRID)
    lines=['The candidate-normalized removal gap is positive in '+str(npositive)+' of six settings; its paired 95% interval is above zero in '+str(nclear)+' settings.', '',
           'This analysis uses the six original training-seed-42 benign and harmful full fine-tunes. The broad shift is the calibrated harmful-minus-aligned answer log likelihood on `ref_cartoon120`; the narrow shift uses held-out training answers (`ref_meddata` or `ref_findata`). The control line is fitted to meaning-preserving control pairs. R_s(X) denotes the remaining score in model condition X for measured shift s (B broad, N narrow). C0 and E0 are the benign and harmful zero-edit references. The candidate pool C retains the original leading 32 singular value decomposition (SVD) components of each query, key, value, attention-output, feed-forward gate, up and down matrix in every layer. Delta W is the complete harmful-minus-benign difference of those seven matrix types. A_S is the summed edit from selected components S.', '',
           'Addition T_s(S) = R_s(C0 + A_S) / R_s(E0); removal D_s(S) = [R_s(E0) - R_s(E0 - A_S)] / R_s(E0). The raw gap is D_B - D_N. The normalized removal gap G_D(S) = D_B(S)/D_B(C) - D_N(S)/D_N(C); the secondary addition gap G_T uses the same formula with T. Every displayed score interval uses the existing 2,000 question-group bootstrap draws, seed 0, with each control line refitted for every condition and draw. Ratios and differences are calculated within a draw, retaining pairing. The broad and narrow question groups are resampled independently according to the original procedure. Attribution values are fixed selection-half estimates and have no question-bootstrap interval in the saved artifacts.', '',
           '### Endpoints', '',
           'Each cell gives estimate [95% lower, upper]. The last four columns apply all ranks of the analyzed matrix differences through the original float32 side path with one bfloat16 output rounding. Values outside [0, 1] are retained.', '']
    rows=[]
    for m in N.GRID:
        row=[names[m]]
        for label in ['CAND','DENSE']:
            for shift,direction in [('B','T'),('B','D'),('N','T'),('N','D')]:
                row.append(fmt(e[(e.model==m)&(e.selection==label)&(e['shift']==shift)&(e.direction==direction)].iloc[0]))
        rows.append(row)
    lines+=table(['Setting','T_B(C)','D_B(C)','T_N(C)','D_N(C)','T_B(Delta W)','D_B(Delta W)','T_N(Delta W)','D_N(Delta W)'],rows)
    lines+=['','The direct endpoint checks R_B(C0 + Delta W)/R_B(E0) and R_N(C0 + Delta W)/R_N(E0) are exactly the T_B(Delta W) and T_N(Delta W) columns above; A1.3 uses the same score as A1.2 addition.', '',
            '### Primary normalized gaps', '', 'N_k contains k components with the largest positive saved narrow integrated-gradient attributions. S_B^N and S_N^N are the summed broad and narrow attributions within N_k, divided by their respective sums over the candidate pool; their difference is the attribution share gap.', '']
    rows=[]
    for m in N.GRID:
        dr=d[d.model==m].iloc[0];tr=t[t.model==m].iloc[0]
        ar=at[at.model==m].iloc[0]
        rows.append([names[m],N.GRID[m][2],fmt(dr),fmt(tr),f"{ar.broad_share:.3f}",f"{ar.narrow_share:.3f}",f"{ar.gap:+.3f}"])
    lines+=table(['Setting','Primary k','G_D','G_T (secondary)','S_B^N','S_N^N','Attribution share gap'],rows)
    lines+=['',f'The secondary normalized addition gap is positive in {(t.estimate>0).sum()}/6 settings and has an interval above zero in {(t.lo>0).sum()}/6; negative and unresolved values are retained.', '',
            '### Alternative-selection removal gaps', '',
            'N_k and B_k select the largest positive narrow and broad attributions at the existing primary k. G_k selects the same number of largest singular values. R1 contains the first singular component of every analyzed matrix. RAND256_j and RAND1024_j reuse the three saved uniform draws without replacement at each count (j = 0, 1, 2). G_norm and G_base_norm are the existing norm-matched E-minus-C and E-minus-base singular selections. Uniform rollback is the existing scaled full seven-matrix difference. Their original gains and selected component identities are retained. CORE is the saved intersection of the domain-behavior probe union and the top-256 training-answer selection. The norm-matched gains target its squared norm; they have not been rescaled to the primary narrow selection.', '',
            'N_k and G_k in the main equal-count comparison reuse `core_confirmatory_20261005` with its own C0/E0 references. The primary normalized table uses the saved `nested_20261005` scoring of the identical primary N_k sets, paired with that run’s references and the added endpoints; the paper’s original primary rows use the confirmatory scores. These numerical run sources are kept explicit. B_k, R1 and random controls reuse the nested run. The full alternative CSV includes an N_k nested row for the random comparisons and candidate normalization. This source distinction preserves the original numerical reference for each saved intervention.', '']
    rows=[]
    for m in N.GRID:
        cells=[]
        for label in ['N_core_confirmatory','G_core','B_core','R1']:
            r=alt[(alt.model==m)&(alt.selection==label)].iloc[0]
            cells.append(fmt({key:r['gap_'+key] for key in ['estimate','lo','hi']}))
        for k in [256,1024]:
            r=rand[(rand.model==m)&(rand.k==k)].iloc[0]
            cells.append(fmt({key:r['mean_'+key] for key in ['estimate','lo','hi']}))
        for label in ['G_norm','G_base_norm','uniform_rollback']:
            r=alt[(alt.model==m)&(alt.selection==label)].iloc[0]
            cells.append(fmt({key:r['gap_'+key] for key in ['estimate','lo','hi']}))
        rows.append([names[m]]+cells)
    lines+=table(['Setting','N_k','G_k','B_k','R1','Random 256 mean','Random 1024 mean','G_norm','G_base_norm','Rollback'],rows)
    lines+=['','Each cell above is the broad-minus-narrow removal gap with a paired 95% interval. [The full alternative-selection table](../../eval_runs/persona_control_stage9/nesting_audit_20261006/alternative_selection_gaps.csv) reports D_B, D_N and their paired gap interval for every selection and every saved random draw, including the nested-run N_k reference. [Paired gap comparisons](../../eval_runs/persona_control_stage9/nesting_audit_20261006/gap_comparisons.csv) retain every same-count and primary narrow-versus-random comparison.', '']
    lines+=['','Paired primary N_k-minus-G_k gap differences:', '']
    lines+=table(['Setting','Difference in gaps'],[[names[r.model],fmt(r)] for _,r in ng.iterrows()])
    lines+=['','Random draw summaries report the mean, median, and range of point estimates. Individual draw intervals are in the linked full alternative-selection CSV. N_k at the same random-set count and the primary N_k are both compared with the saved distribution; three draws give a descriptive comparison.', '']
    lines+=table(['Setting','Random k','Mean gap [95% interval]','Median gap','Range','Same-count N_k gap','Primary N_k gap','Same-count / primary N_k above draws'],
                 [[names[r.model],r.k,fmt({key:r['mean_'+key] for key in ['estimate','lo','hi']}),f'{r.median_gap:+.3f}',f'[{r.min_gap:+.3f}, {r.max_gap:+.3f}]',f'{r.n_k_gap:+.3f}',f'{r.n_primary_gap:+.3f}',f'{r.n_k_above_draws}/3; {r.n_primary_above_draws}/3'] for _,r in rand.iterrows()])
    lines+=['','### Candidate attribution coverage', '',
            'Coverage A_s(C) is summed candidate integrated-gradient attribution divided by the saved full-path integrated-gradient attribution summed across all seven matrix types and layers. The denominator is the integrated matrix attribution. Finite integration and negative excluded attributions allow coverage above one.', '']
    lines+=table(['Setting','Candidates','Positive narrow','Positive broad','A_B(C)','A_N(C)'],
                 [[names[r.model],r.candidates,r.positive_narrow,r.positive_broad,f'{r.coverage_broad:.4f}',f'{r.coverage_narrow:.4f}'] for _,r in cov.iterrows()])
    short=shares[(shares.ranking=='B')&(shares.k==4096)&shares.nonpositive_enter]
    lines+=['','All six N_4096 selections contain exactly 4,096 positive narrow-attribution components; no zero or negative component enters. For the A4 attribution-only B ranking, nonpositive-attribution components enter the requested full descending 4,096-component diagnostic in '+', '.join(f'{names[r.model]} ({r.positive_count} positive components)' for _,r in short.iterrows())+'. These are attribution sums under the requested signed ranking; causal selections retain the original positive-only convention. All requested counts, positive counts and sign flags are saved in `attribution_shares.csv`.', '',
            '### Trained parameters outside the seven matrix types', '',
            'The training recipes optimize `model.parameters()` in full. The following inventory comes from saved harmful-checkpoint tensor headers and model configuration. Each entry lists parameter tensor counts outside the analyzed matrix weights; tensor names and element counts are saved in `parameters/<setting>.json`.', '']
    paramrows=[]
    for m in N.GRID:
        params=json.loads((OUT/'parameters'/f'{m}.json').read_text())
        classes={c:sum(p['parameter_class']==c for p in params) for c in ['token embeddings','normalization weight','output head','bias','other']}
        C=N.configure(m);config=json.loads((Path(C.arm_dir(m,'M_EM'))/'config.json').read_text())
        head=str(classes['output head'])+(' (tied to embeddings)' if config.get('tie_word_embeddings') else ' (separate)')
        paramrows.append([names[m],classes['token embeddings'],classes['normalization weight'],head,classes['bias'],classes['other']])
    lines+=table(['Setting','Embeddings','Normalization weights','Output head','Biases','Other'],paramrows)
    lines+=['','Embeddings, normalization parameters, output-head weights, and biases retain their host values in Delta W interventions. Normalization includes each layer’s input and post-attention normalization plus the final model normalization; Qwen3 query/key normalization weights are also included. Qwen2.5 attention query, key and value biases are included in the bias column. All tensor names are retained in the inventory. A tied output head shares the embedding parameter and is counted once in checkpoint storage.', '',
            '### Extended sweep and attribution figures', '',
            'The sweep adds k = 2,048 and 4,096 to the existing seven counts and includes the primary k. Solid lines show the broad and narrow removal fractions; dashed lines divide each fraction by its candidate-pool removal. Shading shows pointwise paired 95% intervals. The complete per-k D_B, D_N, raw-gap and normalized-gap intervals are saved in `extended_sweep.csv`. '+f'At k = 4,096, normalized gap point estimates range from {late.estimate.min():+.3f} to {late.estimate.max():+.3f}; {(late.estimate<0).sum()}/6 are negative and {(late.lo>0).sum()}/6 intervals lie wholly above zero.'+'', '',
            '![Extended six-setting removal sweep](../../figures/persona_control/nesting_audit_20261006/extended_removal_sweep.png)', '',
            'The attribution figure extends the original cumulative diagnostic: colors identify N, B and G rankings; solid curves measure broad attribution and dashed curves measure narrow attribution. Each sum is divided by that measured attribution’s sum across candidates. Shares above one and negative broad-minus-narrow differences are retained. The full table is `attribution_shares.csv`. '+f'At k = 4,096 the narrow-ranking attribution gap reverses in {(lateattr.gap<0).sum()}/6 settings, all medical: its point estimates range from {lateattr.gap.min():+.3f} to {lateattr.gap.max():+.3f}.'+'', '',
            '![Extended six-setting attribution curves](../../figures/persona_control/nesting_audit_20261006/extended_attribution_curves.png)', '',
            '### Does the nesting claim survive?', '',
            f'- The normalized removal gap remains positive in {npositive}/6 settings; {nclear}/6 paired intervals lie wholly above zero. '+('The at-least-five-setting condition holds.' if npositive>=5 else 'The at-least-five-setting condition fails.')+f' Point estimates range from {d.estimate.min():+.3f} to {d.estimate.max():+.3f}; {(d.estimate>=.10).sum()}/6 reach +0.10.',
            f'- The N_k gap exceeds the G_k gap in {(ng.estimate>0).sum()}/6 point estimates and {(ng.lo>0).sum()}/6 paired intervals. '+ 'The equal-count singular gap is itself positive in every setting, with point estimates '+f"{alt[alt.selection=='G_core'].gap_estimate.min():+.3f} to {alt[alt.selection=='G_core'].gap_estimate.max():+.3f}. The desired near-zero singular-control result fails. The narrow ranking has an additional effect, while the broad-minus-narrow asymmetry also appears under magnitude selection.",
            f'- Attribution-level concentration shows broad shares above narrow shares under the narrow ranking at primary k in {(at.gap>0).sum()}/6 settings, with share gaps {at.gap.min():+.3f} to {at.gap.max():+.3f}. The ordering is assessed at the fixed primary counts; late-sweep values remain in the saved table and figures. The narrow-ranking attribution gap exceeds the singular-ranking attribution gap in {attr_n_over_g}/6 settings; singular ranking also has positive broad-minus-narrow attribution gaps in all six settings.',
            ('- The current title, “Broad Misalignment Is Nested in the Narrow Fine-Tuning Update,” can remain as a descriptive statement of the retained-candidate ordering. The main claim requires an explicit limit: the broad-minus-narrow gap also appears under singular-value selection, so it establishes no mechanism specific to narrow-attribution ranking.' if npositive>=5 else '- The normalized removal result fails the fixed success condition. The title and main claim require narrowing: candidate-normalized nesting does not hold in at least five settings. The singular controls also contradict a mechanism specific to narrow ranking.')+' The manuscript and title have not been edited.', '',
            '### Protocol changes and reproduction', '',
            'The only new interventions are the all-candidate endpoints, all-rank seven-matrix-type endpoints, and narrow removals at 2,048 and 4,096. The normalized statistics and additional attribution curves are the requested analyses. No selection threshold, primary count, candidate definition, score, question split, control fitting, or bootstrap procedure changed. No random component set was regenerated. A4 uses the requested complete descending attribution ranks. Five B diagnostic ranks include nonpositive components at 4,096; no causal set changes. The initial positive-only diagnostic implementation was corrected to match A4’s exact top-k definition. Its CSV, protocol record and figure are preserved with the suffix `positive_only_initial`; all shared-count values before exhaustion agree. Saved confirmatory and nested reference runs are kept distinct. Added conditions are scored in new jobs against the saved nested references with the same frozen input pairs and batching; zero-edit references are reused. GPU eligibility for pending tasks was widened from H100/H200 to A100/H100/H200 and host memory reduced from 160 to 96 GB; all recorded jobs use the existing numerical path. No unrelated test was run.', '',
            'Run `python experiments/persona_control/stage9/nesting_audit.py freeze`; submit `sbatch --array=0-11%12 experiments/persona_control/stage9/nesting_audit.sbatch`; run `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python experiments/persona_control/stage9/nesting_audit.py report`. Source hashes are in `protocol.json` and `analysis_manifest.json`; per-condition runtime records retain job, GPU, node, numerical-path and driver information. Results are in `eval_runs/persona_control_stage9/nesting_audit_20261006/`; figures are in `figures/persona_control/nesting_audit_20261006/` in PNG and PDF. No new markdown file is created.']
    path=N.MAIN/'docs/progress/persona-control.md'
    text=path.read_text();anchor='<a id="nesting-threats-a1-a5"></a>'
    if anchor in text:
        text=text[:text.index(anchor)].rstrip()
        if text.endswith('---'):text=text[:-3].rstrip()
    else:
        first=text.index('\n---\n')
        text=text[:first].rstrip()+'\n| 42 | Nesting threats A1–A5: endpoints, normalized gaps and alternative selections | [nesting-threats-a1-a5](#nesting-threats-a1-a5) |\n'+text[first:]
    text+='\n\n---\n\n'+anchor+'\n\n## Nesting threats A1–A5, six seed-42 settings\n\n'+'\n'.join(lines)+'\n'
    path.write_text(text)
    N.dump(dict(normalized_removal_positive=npositive,normalized_removal_intervals_above_zero=nclear,
                primary_n_minus_g_intervals_above_zero=int((ng.lo>0).sum()),attribution_primary_positive=int((at.gap>0).sum()),
                singular_control_near_zero_prediction_holds=False,manuscript_modified=False,
                report_driver_sha256=N.sha(__file__)),OUT/'verdict.json')


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('command',choices=['freeze','score','report'])
    ap.add_argument('--model',choices=list(N.GRID));ap.add_argument('--phase',choices=['T','D'])
    ap.add_argument('--max-tokens',type=int,default=4096)
    args=ap.parse_args()
    if args.command=='freeze':freeze()
    elif args.command=='score':score(args)
    else:report()
