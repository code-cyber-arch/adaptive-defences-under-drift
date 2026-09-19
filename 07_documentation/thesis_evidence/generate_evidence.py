from pathlib import Path
import pandas as pd
import numpy as np
import json, hashlib

ROOT=Path(__file__).resolve().parents[2]; R=Path(__file__).resolve().parent; T=R
S=ROOT/'results/per_dataset_study'
sources=[]
def read(p):
    sources.append(p); return pd.read_csv(p)
def table(name, heads, rows, caption, note):
    text='\\begin{table}[htbp]\n\\centering\\small\\singlespacing\n\\caption{'+caption+'}\\label{tab:'+name+'}\n\\setlength{\\tabcolsep}{4pt}\n\\renewcommand{\\arraystretch}{1.15}\n\\begin{tabular}{@{}ll'+ 'r'*(len(heads)-2)+'@{}}\n\\toprule\n'
    text+=' & '.join(heads)+' \\\\\n\\midrule\n'
    text+='\n'.join(' & '.join(map(str,row))+' \\\\' for row in rows)
    text+='\n\\bottomrule\n\\end{tabular}\n\\par\\smallskip\\begin{minipage}{\\textwidth}\\footnotesize '+note+'\\end{minipage}\n\\end{table}\n'
    (T/'tables'/f'{name}.tex').write_text(text)
ds=['SEA_A','RBF_I','radar']; det=['adwin','hddm_w','hellinger','d3_oof']
names=dict(zip(ds+det,['SEA','RBF','RADAR','ADWIN','HDDM-W','Hellinger','D3 OOF']))
# Paired variation: evaluate repetition means after fitting-seed averaging;
# fitting variation: evaluate seed means after equal category averaging.
b=read(S/'analysis/comparisons/balanced_comparison_entries.csv')
s=read(S/'analysis/comparisons/common_row_seed_metrics.csv')
d=read(S/'analysis/comparisons/dataset_method_summary.csv')
arms=[('Screen','E1_fixed','confirmed_screened','confirmed_unscreened'),
      ('RL2','E2_RL','RL_2state','confirmed_screened'),('RL4','E2_RL','RL_4state','confirmed_screened'),
      ('F','E3_filter','confirmed_screened_filter_learned_features','confirmed_screened_filter_none'),
      ('F+L','E3_filter','confirmed_screened_filter_learned_features_label','confirmed_screened_filter_none')]
records=[]
for dataset in ds:
 for detector in det:
  for label,exp,method,ref in arms:
   sub=b[(b.stream==dataset)&(b.detector==detector)&(b.experiment==exp)]
   keys=['repetition','condition','mode','level']
   a=sub[sub.method==method][keys+['host_macro_f1']]
   z=sub[sub.method==ref][keys+['host_macro_f1']]
   pairs=a.merge(z,on=keys,validate='one_to_one',suffixes=('_m','_r'))
   assert len(pairs)==35
   pairs['delta']=100*(pairs.host_macro_f1_m-pairs.host_macro_f1_r)
   rep=pairs.groupby('repetition').delta.mean();assert len(rep)==5
   summary=d[(d.stream==dataset)&(d.detector==detector)&(d.experiment==exp)].set_index('method')
   expected=100*(summary.loc[method,'host_macro_f1']-summary.loc[ref,'host_macro_f1'])
   assert abs(rep.mean()-expected)<1e-10
   fit_sd=None; fit_means=[]
   if label!='Screen':
    sub=s[(s.stream==dataset)&(s.detector==detector)&(s.experiment==exp)]
    refrows=sub[sub.method==ref][['condition','host_macro_f1']]
    a=sub[sub.method==method].merge(refrows,on='condition',validate='many_to_one',suffixes=('_m','_r'))
    assert sorted(a.fitting_seed.unique())==[7,17,27,37,47]
    a['delta']=100*(a.host_macro_f1_m-a.host_macro_f1_r)
    fit=a.groupby(['fitting_seed','mode','level']).delta.mean().groupby('fitting_seed').mean()
    assert abs(fit.mean()-expected)<1e-10
    fit_sd=float(fit.std(ddof=1));fit_means=fit.tolist()
   records.append(dict(dataset=dataset,detector=detector,comparison=label,mean_pp=expected,
       evaluation_sd_pp=float(rep.std(ddof=1)),fitting_sd_pp=fit_sd,
       repetition_means=rep.tolist(),fitting_means=fit_means))
pd.DataFrame(records).drop(columns=['repetition_means','fitting_means']).to_csv(R/'paired_variability.csv',index=False)
(R/'paired_values.json').write_text(json.dumps(records,indent=2))
rows=[]
for dataset in ds:
 for detector in det:
  rr=[x for x in records if x['dataset']==dataset and x['detector']==detector]
  rows.append([names[dataset],names[detector]]+[f"{x['evaluation_sd_pp']:.2f}"+(f" / {x['fitting_sd_pp']:.2f}" if x['fitting_sd_pp'] is not None else ' / --') for x in rr])
table('paired_variability',['Dataset','Detector','Screen','RL2','RL4','F','F+L'],rows,
      'Variation in paired macro-F1 effects (standard deviations, percentage points).',
      'RL2/RL4 use two/four states; F uses features; F+L adds labels. Each cell gives evaluation / fitting standard deviation. Evaluation variation uses five repetition means, each averaging seven conditions and all fitting seeds. Fitting variation uses five seed means, each averaging evaluation repetitions and seven conditions. These are separate descriptive margins, not additive variance components or confidence intervals. RADAR repetitions share one capture and clean reference. Screen compares screened with unscreened confirmed reset; RL compares with screened confirmed reset; F and F+L compare filtered with unfiltered screened confirmed reset.')
# Discrimination is separate from downstream learner prediction. Clean has no AUC.
frames=[]
for p in sorted((S/'filters').glob('seed_*/models/*/separability.csv')):
    frame=read(p); frame['fitting_seed']=int(p.parents[2].name.split('_')[1]);frames.append(frame)
f=pd.concat(frames);f=f[f['mode']!='clean']
g=f.groupby(['stream','view','mode','level','condition']).roc_auc.mean().groupby(['stream','view','mode','level']).mean().groupby(['stream','view','mode']).mean()
assert len(frames)==15 and f.roc_auc.notna().all()
g.rename('roc_auc').to_csv(R/'filter_discrimination.csv')
rows=[[names[dataset],{'features':'F','features_label':'F+L'}[view]]+[f'{g.loc[dataset,view,mode]:.3f}' for mode in ['instance','concept','splice']] for dataset in ds for view in ['features','features_label']]
table('filter_discrimination',['Dataset','View','Instance','Concept','Splice'],rows,
      'Held-out poisoning discrimination by attack type (ROC AUC).',
      'ROC AUC is the area under the receiver operating characteristic curve. F uses features; F+L adds the observed label. Means average five fitting seeds, evaluation repetitions and both budgets equally. An AUC of 0.5 indicates chance-level ranking; clean inputs have no defined AUC. These scores assess intervention recognition, not downstream predictive benefit.')
# Show source-class support before and after scoring-only development-copy exclusion.
base=S/'stages/evaluation/radar/01_attacks'; mp=base/'manifest.json'; sources.append(mp)
manifest=json.loads(mp.read_text())['conditions']; support=[]
for key,c in manifest.items():
    truthp=base/c['truth']; ap=base/c['audit']; maskp=S/'filters/profiles/radar'/key/'masks.parquet'
    sources.extend([truthp,ap,maskp]); truth=pd.read_parquet(truthp); audit=pd.read_parquet(ap)
    mask=pd.read_parquet(maskp,columns=['row_id','comparison_scored'])
    assert truth.row_id.equals(mask.row_id)
    full=(truth.row_id>=1000)&~truth.row_id.isin(audit.row_id)&truth.host_eligible
    common=mask.comparison_scored;assert not (common&~full).any()
    for cls in range(8):
        y=truth.host_reference_label.eq(cls)
        support.append(dict(condition=key,mode=c['mode'],level=c['level'],class_id=cls,full=int((full&y).sum()),common=int((common&y).sum())))
support=pd.DataFrame(support);support.to_csv(R/'radar_scoring_support.csv',index=False)
clean=support[support['mode']=='clean']; totals=clean[['full','common']].sum()
rows=[]
for _,x in clean.iterrows():
    rows.append([str(x.class_id),'Goodware' if x.class_id==7 else 'Ransomware',f'{x.full:,}',f'{x.common:,}',f'{100*x.full/totals.full:.2f}',f'{100*x.common/totals.common:.2f}'])
rows.append(['Total','',f'{totals.full:,}',f'{totals.common:,}','100.0','100.0'])
table('radar_scoring_support',['Class','Group','Before','After','Before (\\%)','After (\\%)'],rows,
      'RADAR clean-reference class support before and after development-copy exclusion.',
      'Both populations exclude warm-up and protected rows. After additionally excludes exact development-feature copies, as used in the main adaptation comparisons. Source class IDs are retained. Counts describe one clean reference, not five captures; attacked conditions have their own matched host masks. Macro-F1 retains all eight declared classes, including any absent from a scoring population.')
(R/'evidence_hashes.json').write_text(json.dumps({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in dict.fromkeys(sources)},indent=2))
print('Checked 60 paired effects against thesis means; generated three evidence tables.')
print(g.to_string());print(clean.to_string(index=False))
