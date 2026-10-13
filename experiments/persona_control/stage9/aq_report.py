"""Assemble author-question evidence as JSON; manuscripts and reports stay untouched."""
import hashlib
import json
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'eval_runs/persona_control_stage9/author_questions_20261008'

def main():
    q1=pd.read_csv(OUT/'q1_scores.csv')
    q5=pd.read_csv(OUT/'q5_seed_level_results.csv')
    q6=pd.read_csv(OUT/'q6_model_generated_pairs.csv')
    own={'qwen2_5_7b':'Qwen2.5','llama3_1_8b':'Llama','qwen3_1_7b':'Qwen3'}
    selfgen=q6[q6.apply(lambda r:r.generator==own[r.model.removesuffix('_fin')],axis=1)]
    selfgen.to_csv(OUT/'q6_own_model_summary.csv',index=False)
    summary={
      'Q1':{'status':'Completed; exact-length analysis unavailable; arbitrary monotonic invariance unsupported.',
        'rebuttal': 'We measured both shifts with the same per-token log-likelihood difference and reused the saved selections and held-out pairs. The paired 95% gap interval stays above zero in all six full settings and all six fixed 10% length-matched subsets. Exact token matching retains zero narrow pairs, so we cannot present an exact-length matched result. The financial 10% subsets have only three narrow questions. Positive affine rescaling leaves the fractions unchanged. The fixed asinh transform retains support in all six settings; the fixed saturating tanh transform retains it in four, with inconclusive intervals for Llama medical and Qwen3 finance. Ratios of mean score changes are not invariant to arbitrary monotonic transformations. We therefore claim robustness to the tested common scores, and acknowledge the nonlinear sensitivity.',
        'protocol':'q1_protocol.json','results':'q1_scores.csv'},
      'Q2':{'status':'Pending new training-control experiment.',
        'rebuttal':'We ran a new Qwen3-1.7B medical seed-42 training control with benign answers matched to harmful length and form, preserving safe advice under the frozen automated checks. Original harmful, original benign and rewritten benign arms use the same 542 accepted question IDs and order. At the primary count 222, broad removal changes from 0.844 [0.768, 0.944] to 0.747 [0.689, 0.809], narrow removal from 0.681 [0.660, 0.704] to 0.714 [0.689, 0.737], and the gap from 0.162 [0.084, 0.261] to 0.033 [-0.029, 0.099]. The paired gap change is -0.129 [-0.228, -0.063]. Primary gap support is lost under this matched-style baseline. The secondary 256-component gap remains positive at 0.090 [0.010, 0.173] and does not replace the primary result. Both broad and narrow references remain positive. This establishes sensitivity to the benign training baseline without isolating a pure style mechanism, because the weights, decomposition, ranking and reference shifts change together. The single model/seed, 18.4% acceptance, automated checks and reduced training set limit generalization. Four-midpoint attribution and the original evaluation split remain fixed; integration diagnostics are reported without changing the procedure.',
        'protocol':'q2_protocol.json','data_status':str((OUT/'q2/data_manifest.json').exists())},
      'Q3':{'status':'In progress; general harmful-answer preference assay, not generated behavior or a certified no-EM fine-tune.',
        'rebuttal':'The completed general-harm evaluation does not establish uniform removal effectiveness. All six calibrated reference intervals cross zero. Summed-score absolute removal intervals are positive in two settings, per-token in one, and calibrated Qwen3 medical removal is negative with its interval below zero. These outcomes are preserved without promoting a favorable score. We froze an out-of-distribution evaluation on 141 general harmful prompts from the official PKU-SafeRLHF-30K test split, with its own safe/unsafe answer labels. Selection by a fixed question hash precedes model scoring; exact overlap with original training, selection and evaluation prompts is excluded. The original narrow and equal-count magnitude selections are reused without re-attribution. We report calibrated, summed and per-token absolute changes and fractions. A reference interval containing zero makes its removal fraction unstable. These data do not certify that the dataset would yield no emergent misalignment after training, and preference scoring does not establish generated-behavior efficacy.',
        'protocol':'q3_protocol.json','source':'https://huggingface.co/datasets/PKU-Alignment/PKU-SafeRLHF-30K'},
      'Q4':{'status':'Completed; current Tables 2 and 16 agree.',
        'rebuttal':'Table 2 generated behavior comes from X3 (nested_20261005), which also supplies Table 16. Their common cells agree in every setting. Table 15 uses the separately executed core_confirmatory_20261005 evaluation; these answer records differ. For example, harmful-model Qwen2.5 medical agreed-panel rates are 11.7% in Tables 2/16 and 16.8% in Table 15. Llama finance narrow-removal rates are 3.0% and 0.5%, respectively. Table 2 ability measurements use the confirmatory run, as stated in the current evaluation-run paragraph. We report executions separately, preserve judge denominators, and do not pool them or count identical declared sampling seeds as independent training/sampling replications.',
        'results':'q4_table_provenance.csv'},
      'Q5':{'status':'Completed descriptive replication; training-seed population inference remains limited.',
        'rebuttal':'We now provide seeds 42 and 43 for all three models in both medical and financial settings, exceeding the requested medical/financial comparison for one model. Each of the twelve observed seed-level gap intervals is positive. These are question-bootstrap intervals conditional on the trained model, not training-seed uncertainty. Two seeds cannot establish positivity for all training seeds. No finite seed count can prove that every possible seed is positive. To test positive mean gaps in all six settings, the required count depends on between-seed variance. Under an independent Gaussian seed-gap model, six Bonferroni-adjusted one-sided tests and at least 80% joint power, illustrative seed standard deviation/mean ratios of 0.25, 0.5 and 1 require 5, 8 and 21 seeds per setting, respectively. These are conditional planning scenarios, not variance estimates from two seeds.',
        'results':'q5_seed_level_results.csv','planning':'q5_seed_planning_scenarios.csv'},
      'Q6':{'status':'Completed, reused independently generated broad answer pairs.',
        'rebuttal':'We evaluated the unchanged selections using broad answer pairs generated by Qwen2.5, Llama and Qwen3 for each of the six trained settings, with the original narrow evaluation and controls. When each model is evaluated on its own generated broad pairs, all six gap intervals remain positive. Across all eighteen model/generator combinations, all point gaps are positive and seventeen intervals exclude zero. The exception is Qwen2.5 finance evaluated on Qwen3-generated pairs: gap 0.085 [-0.041, 0.181]. We preserve this exception and report generator-specific effects. These generators are each family’s base/instruction model, not each harmful or benign fine-tune. Existing label checks did not validate every pair; the fixed unfiltered split is retained. This tests answer-generator sensitivity with that label-quality limit and does not establish generalization to novel prompt families.',
        'results':'q6_model_generated_pairs.csv','run':'review_20261007'},
      'Q7':{'status':'Completed; requires qualification, not a new threshold.',
        'rebuttal':'We agree that the narrow-only seed-43 result limits the outside-set claim. For Qwen2.5 medical, the narrow-ranked set outside the leading broad set removes 0.088 [0.071, 0.104] of the narrow shift and 0.043 [0.017, 0.068] of the broad shift. The fixed criterion requires narrow transfer and removal at least 0.10 and absolute broad transfer and removal at most 0.08. Qwen fails the narrow-removal requirement; the interval crossing 0.10 does not change that point-estimate decision. Llama seed 43 passes, with narrow removal 0.103 [0.092, 0.115]. The original Qwen2.5 seed-42 narrow removal was also below threshold, at 0.0979; the saved P4 criterion passed in five of six main settings. It now passes in one of the two narrow-only medical seed-43 replications. Both Qwen results still support a measurable narrow outside effect with a small broad effect, which is the qualitative current main-text claim. The primary seed-43 concentration gap remains positive in both models. The main-text outside-set conclusion should state this replication limit. We retain the original thresholds, counts and failure.',
        'results':'../nesting_robustness_20261006/b7_seed43_complements.csv'}
    }
    q2=OUT/'q2/training_control_results.csv'
    if (OUT/'q2/data_manifest.json').exists():
        summary['Q2']['status']='Data accepted; training-control effect analysis pending.'
        summary['Q2']['acceptance']=json.loads((OUT/'q2/acceptance_review.json').read_text())
        summary['Q2']['data_manifest']='q2/data_manifest.json'
        summary['Q2']['deviation']='542 accepted questions and 34 full-epoch steps per arm versus original 2944 questions and 184 steps; original final-batch flooring amended before outcomes to retain all accepted rows.'
    if q2.exists():
        summary['Q2']['status']='Completed; primary concentration-gap support is lost under the matched-style control.'
        summary['Q2']['results']=pd.read_csv(q2).to_dict('records')
        summary['Q2']['contrasts']=pd.read_csv(OUT/'q2/training_control_contrasts.csv').to_dict('records')
    q3=OUT/'q3_general_harm_results.csv'
    if q3.exists():
        df=pd.read_csv(q3)
        summary['Q3']['completed_settings']=int(df.model.nunique())
        summary['Q3']['status']=('Completed all six' if df.model.nunique()==6 else 'Partial')+' general harmful-answer preference analysis.'
        summary['Q3']['results']=df.to_dict('records')
        if df.model.nunique()==6:
            summary['Q3']['observed_verdict']='Uniform general-harm removal is unsupported. All six calibrated reference intervals cross zero; ratios are unstable. Summed-score N removal has a positive absolute-change interval in two settings (Qwen2.5 medical and Llama medical); per-token removal does in one (Llama finance). Calibrated Qwen3 medical absolute removal is negative with its interval below zero. No favorable score is promoted to primary, and no evaluation subset is changed. General-domain pooled-control calibration remains unvalidated.'
    summary['Q2']['numerical_limit']='Original four-midpoint full-matrix integration discrepancy: broad plain approximately 1.5%, style 6.5%; narrow plain approximately 4.0%, style 4.9%. The procedure was not changed after results. Rank truncation is a separate approximation.'
    summary['definitions']={'N_k':'The k components selected by narrow attribution on the original selection questions.',
        'G_k':'The k largest singular components by magnitude, at the same component count.',
        'D_B':'Fraction of the harmful-minus-benign broad score shift removed from the harmful model.',
        'D_N':'The corresponding fraction for the narrow training-domain score.',
        'gap':'D_B minus D_N; positive means greater concentration of broad removal in the narrow-selected set.',
        'intervals':'2000 paired question-bootstrap draws, seed 0; calibrated fits refit per draw. Conditional on trained models.',
        'status':'New supplementary analyses; fixed selection counts and thresholds, manuscripts untouched by these scripts.'}
    summary['Q6']['generator_provenance']='The saved Stage 7 alternatives use each family\'s original base/instruction model, not each harmful/benign fine-tune. Current Stage 9 evaluation uses the fixed unfiltered question split; prior Stage 7 checks flagged some answer-pair labels. No post-outcome filtering is performed. This is a generator-sensitivity analysis with an answer-label quality limit.'
    summary['Q7']['current_draft_provenance']='Current main-text RQ4 discussion already acknowledges one second-seed outside-threshold failure; the replication appendix reports .088 and failure explicitly. The proposed improvement places the numerical scope beside the RQ2 outside-set claim. An earlier reason for omission is not established by the saved record.'
    sources=list(OUT.rglob('*.csv'))+list(OUT.rglob('*.parquet'))+list(OUT.glob('*protocol.json'))
    raw=ROOT/'logs/persona_control/rollouts/author_questions_20261008/style'
    sources += [p for p in [raw/'rewrites.jsonl',raw/'checks.jsonl'] if p.exists()]
    base=ROOT/'eval_runs/persona_control_stage9'
    models=q5.model.unique()
    for run,conds in [('core_confirmatory_20261005',['C0','E0','D_H_K','D_SVD_EC_K']),
                      ('review_20261007',['C0','E0','D_N_core'])]:
        sources += [base/run/'likelihood'/m/(c+'.parquet') for m in models for c in conds]
    sources += [base/'nesting_robustness_20261006'/name for name in
                ['b1_alternative_scores.csv','b6_seed43_primary.csv','b7_seed43_complements.csv']]
    sources.append(base/'review_20261007/report/alt_seed43.csv')
    summary['source_hashes']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    (OUT/'rebuttal_and_evidence.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')

if __name__=='__main__':main()
