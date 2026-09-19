"""Compare fixed responses and screening on identical clean/poisoned scoring rows."""
from pathlib import Path
import os
import sys
sys.dont_write_bytecode=True
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[key]='1'
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from common import protocol as p
layout=p.load_module('fixed_layout','06_analysis/dataset_summary/compact.py')
DATASETS={'SEA_A':'SEA','RBF_I':'RBF','radar':'RADAR'}
DETECTORS={'adwin':'ADWIN','hddm_w':'HDDM-W','hellinger':'Hellinger','d3_oof':'D3 OOF'}
CATEGORIES=[('clean','none'),('instance','moderate'),('instance','severe'),
            ('concept','moderate'),('concept','severe'),('splice','moderate'),('splice','severe')]
PREDICTIVE=[('accuracy','Accuracy (%)',100),('macro_f1','Macro-F1 (%)',100),
            ('macro_recall','Macro recall (%)',100),('roc_auc','ROC-AUC',1)]
PROTECTION=[('poison_admission_rate','Poison admission (%)',100),
            ('clean_withhold_rate','Legitimate withholding (%)',100),
            ('resets_per_100k','Resets / 100k rows',1),
            ('seconds_per_100k','Runtime (s / 100k rows)',1)]
SERIES={}
STYLES={}
for policy,label,color,marker in [('none','No reset','#0072B2','o'),
                                 ('immediate','Immediate','#D55E00','s'),
                                 ('confirmed','Confirmed','#009E73','^')]:
    for guarded in [False,True]:
        name=policy+('_screened' if guarded else '_unscreened')
        SERIES[name]=label+(' / screened' if guarded else ' / unscreened')
        STYLES[name]={'color':color,'marker':marker,'linestyle':'-' if guarded else '--',
                      'markerfacecolor':color if guarded else 'white'}


def predictive_scores(labels,predictions,probabilities,classes,benign):
    """Calculate the four displayed metrics using the baseline definitions."""
    _,recall,f1,_=precision_recall_fscore_support(labels,predictions,labels=classes,
                                                average='macro',zero_division=0)
    if benign is not None:
        targets=[labels!=benign]
        scores=[probabilities[:,[i for i,c in enumerate(classes) if c!=benign]].sum(axis=1)]
    else:
        targets=[labels==c for c in classes]
        scores=[probabilities[:,i] for i in range(len(classes))]
    auc=[roc_auc_score(y,score) if np.unique(y).size==2 else np.nan for y,score in zip(targets,scores)]
    return {'accuracy':float(np.mean(labels==predictions)),'macro_f1':float(f1),
            'macro_recall':float(recall),'roc_auc':float(np.mean(auc))}


def score_cohort(task):
    """Rescore the 18 fixed arms on one common mask across all seven conditions."""
    selected,runs,config,hashes,summary_hashes,seed=task
    root=ROOT/'results/evaluation'
    inputs={}
    def verify(file,digest):
        """Verify and fingerprint consumed files once within this cohort."""
        key=file.resolve().relative_to(ROOT).as_posix()
        if key not in inputs:
            if p.sha(file)!=digest:
                raise ValueError(f'Changed input: {file}')
            inputs[key]=digest
    mask,labels=None,None
    for condition in selected:
        file=root/'01_attacks'/condition['truth']
        verify(file,hashes[condition['truth']])
        truth=pd.read_parquet(file,columns=['row_id','host_eligible','host_reference_label'])
        np.testing.assert_array_equal(truth.row_id,np.arange(condition['rows']))
        if mask is None:
            mask=truth.row_id.to_numpy()>=config['warmup_rows']
            labels=truth.host_reference_label.to_numpy()
        else:
            np.testing.assert_array_equal(labels,truth.host_reference_label)
        mask &= truth.host_eligible.to_numpy(dtype=bool)
        file=root/'01_attacks'/condition['audit']
        verify(file,hashes[condition['audit']])
        mask[pd.read_parquet(file,columns=['row_id']).row_id.to_numpy(dtype=int)]=False
    if not mask.any():
        raise ValueError('Empty common host mask')
    scoring_labels=labels[mask]
    mask_hash=hashlib.sha256(np.packbits(mask).tobytes()).hexdigest()
    records=[]
    for category,condition in enumerate(selected):
        matched=[run for run in runs if run['condition']==condition['key']]
        expected={('none',False,'none'),('none',True,'none')}|{
            (policy,guarded,detector) for policy in ['immediate','confirmed']
            for guarded in [False,True] for detector in DETECTORS}
        if {(r['policy'],r['guarded'],r['detector']) for r in matched}!=expected or len(matched)!=18:
            raise ValueError('Missing or duplicated fixed arm')
        for run in matched:
            folder=root/run['relative_folder']
            verify(folder/'summary.json',summary_hashes[run['run_id']])
            summary=p.read(folder/'summary.json')
            for key in ['condition','policy','guarded','detector']:
                if summary[key]!=run[key]:
                    raise ValueError(f'Mismatched run metadata: {key}')
            file=folder/'predictions.parquet'
            verify(file,summary['files']['predictions.parquet'])
            columns=[f'p_{c}' for c in condition['classes']]
            pred=pd.read_parquet(file,columns=['row_id','scored','prediction']+columns)
            np.testing.assert_array_equal(pred.row_id,np.arange(len(mask)))
            if not pred.scored.to_numpy()[mask].all():
                raise ValueError('Protected or unscored rows entered comparison')
            probabilities=pred.loc[mask,columns].to_numpy()
            if not np.isfinite(probabilities).all():
                raise ValueError('Nonfinite prediction probabilities')
            np.testing.assert_allclose(probabilities.sum(axis=1),1,atol=1e-8)
            prediction=pred.prediction.to_numpy()[mask]
            scores=predictive_scores(scoring_labels,prediction,probabilities,
                                     condition['classes'],condition['benign_class'])
            records.append({'dataset':DATASETS[condition['stream']],'base_key':condition['base_key'],
                            'attack_seed':seed,'category':category,'mode':condition['mode'],
                            'level':condition['level'],'condition':condition['key'],'run_id':run['run_id'],
                            'detector':run['detector'],'policy':run['policy'],'guarded':run['guarded'],
                            'series':run['policy']+('_screened' if run['guarded'] else '_unscreened'),
                            'rows':condition['rows'],'common_host_rows':int(mask.sum()),'mask_sha256':mask_hash,
                            'correct':int((prediction==scoring_labels).sum()),**scores,
                            **{key:float(run[key]) for key,_,_ in PROTECTION}})
        print(f"Scored {condition['base_key']} / assignment {seed}: {category+1}/7 conditions",flush=True)
    return records,inputs


def aggregate(raw):
    """Reuse no-reset references across detectors, without treating reuse as replication."""
    frames=[]
    for detector in DETECTORS:
        part=raw[raw.detector.isin(['none',detector])].copy()
        part['comparison_detector']=detector
        frames.append(part)
    cohorts=pd.concat(frames,ignore_index=True)
    long=cohorts.melt(id_vars=['dataset','comparison_detector','category','mode','level','series'],
                      value_vars=[s[0] for s in PREDICTIVE+PROTECTION],var_name='metric',value_name='value')
    points=long.groupby(['dataset','comparison_detector','category','mode','level','series','metric']).agg(
        mean=('value','mean'),sd=('value','std'),defined=('value','count'),realizations=('value','size')).reset_index()
    if not points.realizations.eq(3).all():
        raise ValueError('Expected three matched cohorts for every plotted point')
    # Paired screening differences are descriptive, with no favourable subset selection.
    paired=raw.pivot(index=['dataset','base_key','attack_seed','category','detector','policy'],
                     columns='guarded',values=[s[0] for s in PREDICTIVE+PROTECTION])
    effects=pd.DataFrame(index=paired.index)
    for metric,_,scale in PREDICTIVE+PROTECTION:
        effects[metric+'_screened_minus_unscreened']=(paired[(metric,True)]-paired[(metric,False)])*scale
    return points,effects.reset_index()


def plot_limits(data,specs):
    """Share vertical ranges across detectors within a dataset and metric."""
    result={}
    for metric,_,scale in specs:
        part=data[data.metric.eq(metric)]
        means=part['mean'].to_numpy()*scale
        errors=part.sd.fillna(0).to_numpy()*scale
        finite=np.isfinite(means)
        low=float(np.min((means-errors)[finite]))
        high=float(np.max((means+errors)[finite]))
        pad=max((high-low)*.18,.15 if scale==100 else .0005)
        low,high=max(0,low-pad),high+pad
        if scale==100:
            low=-2 if low==0 else low
            high=min(102,high)
            high=102 if high>=100 else high
        elif metric=='roc_auc':
            high=min(1.02,high)
        result[metric]=(low,high)
    return result


def render(points,output):
    """Save twelve performance figures and twelve protection/cost companions."""
    entries=[]
    for kind,specs in [('performance',PREDICTIVE),('protection_and_cost',PROTECTION)]:
        (output/kind).mkdir(exist_ok=True)
        for dataset in DATASETS.values():
            subset=points[points.dataset.eq(dataset)]
            limits=plot_limits(subset,specs)
            for detector in DETECTORS:
                part=subset[subset.comparison_detector.eq(detector)]
                entry=layout.draw(part,specs,'','',SERIES,
                                  output/kind/f'{dataset}_{detector}_{kind}','',styles=STYLES,limits=limits)
                entry.update(dataset=dataset,detector=detector,kind=kind,
                             pdf=kind+'/'+entry['pdf'],png=kind+'/'+entry['png'],
                             series=list(SERIES),y_limits={key:list(value) for key,value in limits.items()})
                entries.append(entry)
    return entries


def document(output):
    """Explain the figure sequence, metric support and limits outside the PDFs."""
    (output/'README.md').write_text('''# Fixed responses: performance, protection and cost

Start with performance/SEA_adwin_performance.pdf, then compare the other detectors within SEA before moving to RBF and RADAR. Each dataset/detector has a four-panel performance PDF and a companion under protection_and_cost/. All 24 PDFs are 170 x 128 mm, with a 2 x 2 layout, direct condition labels and a six-line legend. No HTML or combined multi-page PDF is produced. PNG files are previews.

Blue circles denote no reset, orange squares immediate reset, and green triangles confirmed reset. Dashed lines with hollow markers are unscreened; solid lines with filled markers are screened. Immediate and confirmed use the detector named in the filename. No-reset runs have no detector and the same reference is reused across detector figures.

The performance panels show accuracy, macro-F1, macro recall and ROC-AUC. All seven conditions and all 18 fixed arms within a stream/attack-assignment cohort are rescored on exactly the same host observations. Exclude warm-up, protected reservations and the union of all concept/splice replacement intervals. This is the same common mask as the preceding learner-baseline figures. It measures effects on unchanged host observations, not classification of the replacement examples. All original input rows were processed, including 484,753 RADAR rows; a smaller scoring denominator is not stream truncation.

Class-macro metrics weight declared classes equally; zero-division cases use zero. Synthetic ROC-AUC is macro one-versus-rest, while RADAR ROC-AUC contrasts malicious with benign scores. Required missing class support leaves AUC undefined. Additional descriptive metrics computed from stored predictions do not change the original experiment's primary outcomes.

The companion panels show poisoned-row admission, legitimate-data withholding, committed resets per 100,000 source rows and full-run runtime per 100,000 source rows. These use verified original full-run metrics, not the common scoring mask. Clean poison admission has no denominator and remains N/A. Admitted poisoned rows measure exposure; admission is not proof that every such row caused damage, and screening does not guarantee poison-free training. Low withholding retains legitimate learning opportunities. Runtime includes the original concurrent execution environment and should be interpreted alongside predictive performance.

Means and sample-standard-deviation bars use three synthetic realizations or three RADAR attack assignments. The RADAR clean predictions are reused on three common masks; predictive variation is not variation across independent captures. For full-run RADAR clean costs, the same run is repeated as a matched reference and zero variation does not establish runtime repeatability. Bars are descriptive, not confidence intervals. Undefined values are not zero. Y ranges are shared across detectors within each dataset and metric; they can differ between datasets. Lines join categorical conditions, not timestamps or a continuous dose-response sweep.

There are 1,098 unique source fixed-policy runs (61 conditions x 18 arms). Rescoring creates 1,134 cohort/run rows because the 18 RADAR clean runs are evaluated on three matched masks. The two no-reset arms are reused for each detector comparison without increasing the number of independent realizations. RL is excluded from this stage.

Read the figures in this order: first assess poisoning damage on the no-reset unscreened reference; next compare immediate and confirmed responses with that reference; then compare screened and unscreened versions of the same response; finally use the companion figures to assess exposure, withholding, resets and runtime. The next stage compares frozen RL policies with these fixed references.

tables/run_scores.csv lists every source run, scoring-mask hash, scoring denominator and metric. tables/plot_values.csv contains each plotted mean, SD and defined count. tables/paired_screening_effects.csv contains within-cohort screened-minus-unscreened differences; rate differences are percentage points, AUC differences are in AUC units, and costs retain their panel units. All comparisons are reported without selecting only favourable conditions.

Reproduce with `python -B 06_analysis/dataset_summary/fixed_responses.py --workers 3`. After completion, `--render-only` redraws the PDFs from verified saved tables without rescoring predictions. The manifest fingerprints all inputs, reporting code and outputs. The frozen benchmark, observations, learners and controllers are unchanged.
''',encoding='utf-8')
    links=['\n## Figure files\n','| Dataset | Detector | Performance | Protection and cost |',
           '|---|---|---|---|']
    for dataset in DATASETS.values():
        for detector,label in DETECTORS.items():
            links.append(f'| {dataset} | {label} | [PDF](performance/{dataset}_{detector}_performance.pdf) | '
                         f'[PDF](protection_and_cost/{dataset}_{detector}_protection_and_cost.pdf) |')
    with (output/'README.md').open('a',encoding='utf-8') as handle:
        handle.write('\n'.join(links)+'\n')


def main():
    """Score completed fixed-policy runs and generate the next presentation stage."""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers',type=int,default=3)
    parser.add_argument('--render-only',action='store_true')
    args=parser.parse_args()
    root=ROOT/'results/evaluation'
    output=root/'06_analysis/presentation/fixed_responses'
    request=p.read(root/'request.json')
    if p.sources()!=request['sources'] or p.read(root/'verification.json')['status']!='passed':
        raise ValueError('Unchanged, verified benchmark required')
    if args.render_only:
        old=p.read(output/'manifest.json')
        for file,digest in old['outputs'].items():
            if p.sha(output/file)!=digest:
                raise ValueError(f'Changed result: {file}')
        inputs=old['inputs']
        points=pd.read_csv(output/'tables/plot_values.csv')
        calculation_hash=old.get('calculation_script_sha256',old['script_sha256'])
    else:
        metrics_file=root/'06_analysis/metrics.csv'
        digest=p.read(root/'06_analysis/manifest.json')['outputs']['metrics.csv']
        if p.sha(metrics_file)!=digest:
            raise ValueError('Original metrics changed')
        frame=pd.read_csv(metrics_file)
        frame=frame[frame.policy.isin(['none','immediate','confirmed'])]
        if len(frame)!=1098:
            raise ValueError('Expected all 1,098 fixed-policy runs')
        manifest=p.read(root/'01_attacks/manifest.json')
        completion=p.read(root/'05_runs/completion.json')
        conditions=pd.DataFrame(list(manifest['conditions'].values()))
        tasks=[]
        for stream in DATASETS:
            attacks=conditions[conditions.stream.eq(stream)&conditions['mode'].ne('clean')]
            for (base,seed),group in attacks.groupby(['base_key','attack_seed']):
                clean=conditions[conditions.base_key.eq(base)&conditions['mode'].eq('clean')]
                if len(group)!=6 or len(clean)!=1:
                    raise ValueError('Incomplete clean-versus-six comparison')
                keys=[clean.iloc[0]['key']]+[group[group['mode'].eq(mode)&group.level.eq(level)].iloc[0]['key']
                                             for mode,level in CATEGORIES[1:]]
                selected=[manifest['conditions'][key] for key in keys]
                runs=frame[frame.condition.isin(keys)].to_dict('records')
                tasks.append((selected,runs,request['config'],manifest['outputs'],completion['summaries'],int(seed)))
        output.mkdir(parents=True,exist_ok=True)
        (output/'tables').mkdir(exist_ok=True)
        records=[]
        inputs={metrics_file.relative_to(ROOT).as_posix():digest}
        p.write(output/'status.json',{'status':'scoring','cohorts_complete':0,'cohorts_total':9})
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures=[pool.submit(score_cohort,task) for task in tasks]
            for count,future in enumerate(as_completed(futures),1):
                rows,used=future.result()
                records.extend(rows)
                inputs.update(used)
                p.write(output/'status.json',{'status':'scoring','cohorts_complete':count,'cohorts_total':9})
        raw=pd.DataFrame(records).sort_values(['dataset','base_key','attack_seed','category','detector','policy','guarded'])
        if len(raw)!=1134 or raw.run_id.nunique()!=1098:
            raise ValueError('Incomplete rescoring coverage')
        raw.to_csv(output/'tables/run_scores.csv',index=False)
        points,effects=aggregate(raw)
        points.to_csv(output/'tables/plot_values.csv',index=False)
        effects.to_csv(output/'tables/paired_screening_effects.csv',index=False)
        calculation_hash=p.sha(__file__)
    p.write(output/'status.json',{'status':'rendering','cohorts_complete':9,'cohorts_total':9})
    entries=render(points,output)
    document(output)
    p.write(output/'status.json',{'status':'complete','unique_source_runs':1098,'cohort_run_scores':1134,
                                  'performance_pdfs':12,'protection_and_cost_pdfs':12})
    p.write(output/'manifest.json',{'created_utc':p.now(),'script_sha256':p.sha(__file__),
            'calculation_script_sha256':calculation_hash,'layout_sha256':p.sha(ROOT/'06_analysis/dataset_summary/compact.py'),
            'request_sha256':p.sha(root/'request.json'),'inputs':inputs,'entries':entries,
            'outputs':{file.relative_to(output).as_posix():p.sha(file) for file in output.rglob('*')
                       if file.is_file() and file.name!='manifest.json'}})
    print('Complete: 12 performance and 12 protection/cost PDFs.',flush=True)


if __name__=='__main__':
    main()
