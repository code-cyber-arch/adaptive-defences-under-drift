"""Compare frozen RL with screened fixed responses on matched host observations."""
from pathlib import Path
import os
import sys
sys.dont_write_bytecode=True
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[key]='1'
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import FixedLocator,FormatStrFormatter
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from common import protocol as p
fixed=p.load_module('rl_fixed_scores','06_analysis/dataset_summary/fixed_responses.py')
heat=p.load_module('rl_heat_style','06_analysis/dataset_summary/detector_heatmaps.py')
METRICS=[s[0] for s in fixed.PREDICTIVE+fixed.PROTECTION]
SERIES={'none':'No reset','immediate':'Immediate reset','confirmed':'Confirmed reset',
        'rl_alarm':'Alarm-only RL','rl':'Persistence RL'}
STYLES={name:{'color':colour,'marker':marker,'linestyle':style,'markerfacecolor':colour}
        for name,colour,marker,style in [('none','#0072B2','o','--'),('immediate','#D55E00','s','--'),
                                         ('confirmed','#009E73','^','--'),('rl_alarm','#CC79A7','D','-.'),
                                         ('rl','#532b88','v','-')]}
COHORT=['dataset','base_key','attack_seed','category','mode','level']


def score_rl(task):
    """Rescore six frozen RL runs per condition and verify the fixed-baseline mask."""
    selected,runs,config,hashes,summaries,seed,expected_mask,expected_n=task
    root=ROOT/'results/evaluation'
    inputs={}
    def verify(file,digest):
        """Check each input's immutable fingerprint before use."""
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
    mask_hash=hashlib.sha256(np.packbits(mask).tobytes()).hexdigest()
    if mask_hash!=expected_mask or int(mask.sum())!=expected_n:
        raise ValueError('RL and fixed policies have different scoring observations')
    records=[]
    for category,condition in enumerate(selected):
        matches=[run for run in runs if run['condition']==condition['key']]
        if len(matches)!=6 or {(r['policy'],int(r['rl_seed'])) for r in matches}!={
            (policy,seed) for policy in ['rl_alarm','rl'] for seed in [7,17,27]}:
            raise ValueError('Missing RL variant or training seed')
        for run in matches:
            folder=root/run['relative_folder']
            verify(folder/'summary.json',summaries[run['run_id']])
            summary=p.read(folder/'summary.json')
            if not summary['guarded'] or summary['detector']!='adwin':
                raise ValueError('RL must use the same screening and ADWIN')
            for key in ['condition','policy','rl_seed','rl_variant','rl_model_sha256']:
                if summary[key]!=run[key]:
                    raise ValueError(f'RL summary metadata differs: {key}')
            variant='alarm' if run['policy']=='rl_alarm' else 'alarm_persistence'
            contract=next(c for c in config['rl_models'] if c['variant']==variant and c['seed']==int(run['rl_seed']))
            if summary['rl_model_sha256']!=contract['sha256'] or summary['rl_variant']!=variant:
                raise ValueError('Wrong frozen model contract')
            verify(ROOT/contract['path'],contract['sha256'])
            file=folder/'predictions.parquet'
            verify(file,summary['files']['predictions.parquet'])
            columns=[f'p_{c}' for c in condition['classes']]
            pred=pd.read_parquet(file,columns=['row_id','scored','prediction']+columns)
            np.testing.assert_array_equal(pred.row_id,np.arange(len(mask)))
            if not pred.scored.to_numpy()[mask].all():
                raise ValueError('Unscored observations entered RL comparison')
            probabilities=pred.loc[mask,columns].to_numpy()
            if not np.isfinite(probabilities).all():
                raise ValueError('Undefined RL probabilities')
            np.testing.assert_allclose(probabilities.sum(axis=1),1,atol=1e-8)
            prediction=pred.prediction.to_numpy()[mask]
            scores=fixed.predictive_scores(labels[mask],prediction,probabilities,condition['classes'],condition['benign_class'])
            records.append({'dataset':fixed.DATASETS[condition['stream']],'base_key':condition['base_key'],
                            'attack_seed':seed,'category':category,'mode':condition['mode'],'level':condition['level'],
                            'condition':condition['key'],'run_id':run['run_id'],'policy':run['policy'],'series':run['policy'],
                            'guarded':True,'detector':'adwin','rl_seed':int(run['rl_seed']),
                            'rl_variant':variant,'rl_model_sha256':summary['rl_model_sha256'],
                            'rows':condition['rows'],'common_host_rows':expected_n,'mask_sha256':mask_hash,
                            'correct':int((prediction==labels[mask]).sum()),**scores,
                            **{key:float(run[key]) for key,_,_ in fixed.PROTECTION}})
        print(f"Scored RL: {condition['base_key']} / assignment {seed}, {category+1}/7 conditions",flush=True)
    return records,inputs


def aggregate(raw):
    """Average RL training seeds first, then summarize the three matched cohorts."""
    grouping=COHORT+['series']
    cohorts=raw.groupby(grouping)[METRICS].mean().reset_index()
    counts=raw.groupby(grouping).size().rename('policy_runs').reset_index()
    cohorts=cohorts.merge(counts,on=grouping,validate='one_to_one')
    within=raw.groupby(grouping)[METRICS].std().add_suffix('_training_seed_sd').reset_index()
    cohorts=cohorts.merge(within,on=grouping,validate='one_to_one')
    long=cohorts.melt(id_vars=grouping,value_vars=METRICS,var_name='metric',value_name='value')
    points=long.groupby(['dataset','category','mode','level','series','metric']).agg(
        mean=('value','mean'),sd=('value','std'),defined=('value','count'),realizations=('value','size')).reset_index()
    if not points.realizations.eq(3).all():
        raise ValueError('Training seeds were incorrectly counted as stream realizations')
    indexed=cohorts.set_index(COHORT+['series'])
    differences=[]
    for _,row in cohorts[cohorts.series.isin(['rl_alarm','rl'])].iterrows():
        for reference in ['none','immediate','confirmed']:
            base=indexed.loc[tuple(row[key] for key in COHORT)+(reference,)]
            for metric,_,scale in fixed.PREDICTIVE:
                differences.append({**{key:row[key] for key in COHORT},'series':row.series,
                                    'reference':reference,'metric':metric,
                                    'difference':float((row[metric]-base[metric])*scale)})
    paired=pd.DataFrame(differences)
    deltas=paired.groupby(['dataset','category','mode','level','series','reference','metric']).agg(
        mean=('difference','mean'),sd=('difference','std'),defined=('difference','count')).reset_index()
    return cohorts,points,paired,deltas


def delta_plot(data,dataset,output,limits):
    """Show both RL variants minus the same confirmed-reset reference, centred on zero."""
    fig=plt.figure(figsize=(170/25.4,128/25.4))
    cmap=plt.get_cmap('RdBu')
    rows=['rl_alarm','rl']
    cells=[]
    for i,(metric,title,scale) in enumerate(fixed.PREDICTIVE):
        left=.12+.50*(i%2)
        bottom=.70 if i<2 else .24
        ax=fig.add_axes([left,bottom,.36,.23])
        values=data[data.metric.eq(metric)].pivot(index='series',columns='category',values='mean').reindex(index=rows,columns=range(7)).to_numpy()
        if not np.isfinite(values).all():
            raise ValueError('Unsupported paired difference')
        display_scale=1 if scale==100 else 1000
        values=values*display_scale
        bound=limits[metric]*display_scale
        norm=TwoSlopeNorm(vmin=-bound,vcenter=0,vmax=bound)
        mesh=ax.pcolormesh(np.arange(8),np.arange(3),values,cmap=cmap,norm=norm,edgecolors='white',linewidth=.7,rasterized=False)
        for row,name in enumerate(rows):
            for col in range(7):
                value=values[row,col]
                label=f'{value:+.1f}' if scale==100 else f'{value:+.0f}'
                if abs(value)<(.05 if scale==100 else .5):
                    label='0.0' if scale==100 else '0'
                ax.text(col+.5,row+.5,label,ha='center',va='center',fontsize=7.8,color=heat.foreground(cmap(norm(value))))
                cells.append({'dataset':dataset,'series':name,'category':col,'metric':metric,
                              'reference':'confirmed','mean_difference':float(value/display_scale),
                              'display_difference':float(value),'annotation':label})
        ax.set_xlim(0,7)
        ax.set_ylim(2,0)
        ax.set_xticks(np.arange(7)+.5,fixed.layout.LABELS,rotation=38,ha='right')
        ax.set_yticks([.5,1.5],['Alarm-only\nRL','Persistence\nRL'])
        ax.tick_params(length=0,labelsize=8.5,pad=3,colors='#334155')
        label=title.replace(' (%)','')+' change'+(' (pp)' if scale==100 else ' (×10⁻³)')
        ax.set_title(label,fontsize=10,pad=9)
        for spine in ax.spines.values():
            spine.set_visible(False)
        bar_ax=fig.add_axes([left+.045,bottom-.14,.27,.015])
        bar=fig.colorbar(mesh,cax=bar_ax,orientation='horizontal')
        bar.locator=FixedLocator([-bound,0,bound])
        bar.formatter=FormatStrFormatter('%+.1f' if scale==100 else '%+.0f')
        bar.update_ticks()
        bar.ax.tick_params(labelsize=7.5,length=2,pad=2)
        bar.outline.set_visible(False)
    name=f'{dataset}_RL_minus_screened_confirmed_ADWIN'
    folder=output/'improvement'
    folder.mkdir(exist_ok=True)
    preview=output/'previews/improvement'
    preview.mkdir(parents=True,exist_ok=True)
    fig.savefig(folder/(name+'.pdf'))
    fig.savefig(preview/(name+'.png'),dpi=200)
    plt.close(fig)
    return {'dataset':dataset,'kind':'improvement','pdf':f'improvement/{name}.pdf',
            'png':f'previews/improvement/{name}.png','reference':'screened confirmed-reset ADWIN',
            'plots':4,'layout':[2,2],'width_mm':170,'height_mm':128,'cells':56,'colour_limits':limits},cells


def render(points,deltas,output):
    """Produce three performance, three cost and three paired-improvement PDFs."""
    entries,cells=[],[]
    for kind,specs in [('performance',fixed.PREDICTIVE),('protection_and_cost',fixed.PROTECTION)]:
        folder=output/kind
        folder.mkdir(exist_ok=True)
        previews=output/'previews'/kind
        previews.mkdir(parents=True,exist_ok=True)
        for dataset in fixed.DATASETS.values():
            subset=points[points.dataset.eq(dataset)]
            entry=fixed.layout.draw(subset,specs,'','',SERIES,folder/f'{dataset}_RL_{kind}','',styles=STYLES)
            source=(folder/entry['png']).resolve()
            target=(previews/entry['png']).resolve()
            if not source.is_relative_to(output.resolve()) or not target.is_relative_to(output.resolve()):
                raise ValueError('Invalid preview path')
            source.replace(target)
            entry.update(dataset=dataset,kind=kind,series=list(SERIES),
                         pdf=f'{kind}/'+entry['pdf'],png=target.relative_to(output).as_posix())
            entries.append(entry)
    selected=deltas[deltas.reference.eq('confirmed')]
    limits={metric:float(max(step,np.ceil(selected.loc[selected.metric.eq(metric),'mean'].abs().max()/step)*step))
            for metric,_,scale in fixed.PREDICTIVE for step in [1 if scale==100 else .01]}
    for dataset in fixed.DATASETS.values():
        entry,labels=delta_plot(selected[selected.dataset.eq(dataset)],dataset,output,limits)
        entries.append(entry)
        cells.extend(labels)
    pd.DataFrame(cells).to_csv(output/'tables/displayed_improvements.csv',index=False)
    return entries


def document(output):
    """Explain the reference, nested averaging and limitations outside the PDFs."""
    text='''# RL versus fixed responses

Every line figure compares five screened strategies: no reset, immediate reset, confirmed reset, alarm-only RL and persistence RL. The detector-driven strategies all use ADWIN; no reset has no detector. The two RL variants each use the three frozen training seeds 7, 17 and 27. No controller is trained, tuned or updated by this reporting step.

Read the performance figure first (accuracy, macro-F1, macro recall and ROC-AUC). Then inspect protection_and_cost (poison admission, legitimate withholding, committed resets per 100,000 rows and full-run runtime per 100,000 rows). Finally inspect the improvement heatmap, which shows each RL variant minus screened confirmed-reset ADWIN. That is one named descriptive reference for all datasets and conditions, not a best-per-condition selection. Comparisons against screened no reset and immediate reset are also retained in the paired tables.

Each heatmap has 56 signed values: two RL variants x seven conditions x four metrics. Blue is positive, red is negative and white is zero. Percentage-score changes are percentage points (pp). To keep small differences readable, the ROC-AUC panel displays thousandths (×10⁻³): +4 means +0.004 AUC; paired tables retain original AUC units. Colour scales are symmetric around zero and shared across datasets for each metric. Display rounding can show zero for a small nonzero difference; unrounded differences remain in the tables. The reference is named in the filename and this guide; captions should retain it. No surrounding figure heading or explanatory footer is added.

All predictive metrics use exactly the same common host-row masks as the fixed-response and baseline comparisons. This excludes warm-up, protected reservations and all concept/splice replacement intervals in each matched seven-condition cohort. All 484,753 RADAR rows remain in the source runs. Protected observations do not enter predictive scoring. Synthetic AUC is macro one-versus-rest; RADAR AUC contrasts malicious with benign scores.

RL training seeds are averaged within each stream/attack-assignment cohort first. The five line points then average three matched cohorts, with one sample SD across those cohort means. The three training seeds are not nine independent datasets. seed_scores.csv retains all seed-level outcomes, while cohort_scores.csv includes within-cohort training-seed SDs. The heatmaps subtract the same fixed reference within each cohort before averaging; paired SDs remain in improvement_values.csv. Error bars and differences are descriptive, not significance or equivalence tests.

RADAR is one capture with three attack assignments. Its clean predictions are scored on three common masks. Repeated clean fixed-run costs and repeated RL clean seed averages do not establish independent timing repetitions. Synthetic means use three stream realizations. No baseline is replicated to inflate the number of independent cohorts.

Exposure and cost measures retain their original full-run denominators. Clean poison admission is undefined and marked N/A. Admission measures exposure, not proof that every admitted observation caused damage. Low withholding preserves legitimate learning opportunities. Runtime includes the original machine's concurrent execution. A score improvement alone is insufficient to establish a better defence; compare admission and costs as well.

The report uses 366 unique RL runs and 183 unique screened fixed runs (549 total). Matching RADAR clean predictions to three masks gives 567 seed/run records, reduced to 315 cohort/strategy records after RL seed averaging. All 61 conditions are included, with no selection of favourable attacks or outcomes. This is presentation of the previously inspected benchmark, not a new untouched evaluation.

All nine PDFs are separate, 170 x 128 mm, in a 2 x 2 layout. PNG previews are kept separately. The performance/protection figures use dashed fixed-policy lines and distinct RL lines and markers. All five strategies are screened; the legend labels identify their response strategies. No HTML or combined multi-page PDF is generated.

Reproduce with `python -B 06_analysis/dataset_summary/rl_comparison.py --workers 3`. After completion, `--render-only` redraws verified tables without rescoring. tables/seed_scores.csv contains source IDs, RL seeds, frozen-model hashes and scoring-mask hashes; cohort_scores.csv contains seed-averaged scores and training-seed variation; plot_values.csv records plotted means/SDs; paired_improvements.csv retains within-cohort differences against all three fixed references; improvement_values.csv records their mean and SD. The manifest binds sources, models, calculations and figures.

| Dataset | Performance | Protection and cost | RL minus screened confirmed reset |
|---|---|---|---|
'''
    for dataset in fixed.DATASETS.values():
        text+=f'| {dataset} | [PDF](performance/{dataset}_RL_performance.pdf) | [PDF](protection_and_cost/{dataset}_RL_protection_and_cost.pdf) | [PDF](improvement/{dataset}_RL_minus_screened_confirmed_ADWIN.pdf) |\n'
    (output/'README.md').write_text(text,encoding='utf-8')


def main():
    """Read verified fixed scores, rescore frozen RL predictions and render the comparison."""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers',type=int,default=3)
    parser.add_argument('--render-only',action='store_true')
    args=parser.parse_args()
    root=ROOT/'results/evaluation'
    request=p.read(root/'request.json')
    if p.sources()!=request['sources'] or p.read(root/'verification.json')['status']!='passed':
        raise ValueError('Unchanged verified benchmark required')
    output=root/'06_analysis/presentation/rl_comparison'
    if args.render_only:
        prior=p.read(output/'manifest.json')
        for name,digest in prior['outputs'].items():
            if p.sha(output/name)!=digest:
                raise ValueError(f'Changed result: {name}')
        inputs=prior['inputs']
        calculation_hash=prior.get('calculation_script_sha256',prior['script_sha256'])
        points=pd.read_csv(output/'tables/plot_values.csv')
        deltas=pd.read_csv(output/'tables/improvement_values.csv')
    else:
        source=root/'06_analysis/presentation/fixed_responses'
        source_manifest=p.read(source/'manifest.json')
        file=source/'tables/run_scores.csv'
        if p.sha(file)!=source_manifest['outputs']['tables/run_scores.csv']:
            raise ValueError('Fixed comparison scores changed')
        inputs={file.relative_to(ROOT).as_posix():p.sha(file)}
        baseline=pd.read_csv(file)
        baseline=baseline[baseline.guarded&((baseline.policy.eq('none')&baseline.detector.eq('none'))|
                 (baseline.policy.isin(['immediate','confirmed'])&baseline.detector.eq('adwin')))].copy()
        baseline['series']=baseline.policy
        baseline['rl_seed']=np.nan
        if len(baseline)!=189:
            raise ValueError('Expected all three screened references in 63 matched cohorts')
        file=root/'06_analysis/metrics.csv'
        digest=p.read(root/'06_analysis/manifest.json')['outputs']['metrics.csv']
        if p.sha(file)!=digest:
            raise ValueError('Original metrics changed')
        inputs[file.relative_to(ROOT).as_posix()]=digest
        frame=pd.read_csv(file)
        frame=frame[frame.policy.isin(['rl_alarm','rl'])]
        if len(frame)!=366:
            raise ValueError('Expected all frozen RL runs')
        manifest=p.read(root/'01_attacks/manifest.json')
        completion=p.read(root/'05_runs/completion.json')
        conditions=pd.DataFrame(list(manifest['conditions'].values()))
        tasks=[]
        for stream,dataset in fixed.DATASETS.items():
            attacks=conditions[conditions.stream.eq(stream)&conditions['mode'].ne('clean')]
            for (base,seed),group in attacks.groupby(['base_key','attack_seed']):
                clean=conditions[conditions.base_key.eq(base)&conditions['mode'].eq('clean')]
                keys=[clean.iloc[0]['key']]+[group[group['mode'].eq(mode)&group.level.eq(level)].iloc[0]['key']
                                             for mode,level in fixed.CATEGORIES[1:]]
                reference=baseline[baseline.base_key.eq(base)&baseline.attack_seed.eq(seed)]
                if len(reference)!=21 or reference.mask_sha256.nunique()!=1 or reference.common_host_rows.nunique()!=1:
                    raise ValueError('Inconsistent fixed-policy scoring masks')
                tasks.append(([manifest['conditions'][key] for key in keys],frame[frame.condition.isin(keys)].to_dict('records'),
                              request['config'],manifest['outputs'],completion['summaries'],int(seed),
                              reference.mask_sha256.iloc[0],int(reference.common_host_rows.iloc[0])))
        output.mkdir(parents=True,exist_ok=True)
        (output/'tables').mkdir(exist_ok=True)
        records=[]
        p.write(output/'status.json',{'status':'scoring','cohorts_complete':0,'cohorts_total':9})
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for count,future in enumerate(as_completed([pool.submit(score_rl,task) for task in tasks]),1):
                rows,used=future.result()
                records.extend(rows)
                inputs.update(used)
                p.write(output/'status.json',{'status':'scoring','cohorts_complete':count,'cohorts_total':9})
        raw=pd.concat([baseline,pd.DataFrame(records)],ignore_index=True).sort_values(COHORT+['series','rl_seed'])
        if len(raw)!=567 or raw.run_id.nunique()!=549:
            raise ValueError('Incomplete RL/fixed comparison')
        raw.to_csv(output/'tables/seed_scores.csv',index=False)
        cohorts,points,paired,deltas=aggregate(raw)
        for name,table in [('cohort_scores',cohorts),('plot_values',points),('paired_improvements',paired),('improvement_values',deltas)]:
            table.to_csv(output/f'tables/{name}.csv',index=False)
        calculation_hash=p.sha(__file__)
    p.write(output/'status.json',{'status':'rendering','cohorts_complete':9,'cohorts_total':9})
    entries=render(points,deltas,output)
    document(output)
    p.write(output/'status.json',{'status':'complete','unique_source_runs':549,'seed_run_records':567,
                                  'cohort_strategy_records':315,'pdfs':9})
    scripts=['rl_comparison.py','fixed_responses.py','compact.py','detector_heatmaps.py','compare_detectors.py']
    p.write(output/'manifest.json',{'created_utc':p.now(),'script_sha256':p.sha(__file__),
            'calculation_script_sha256':calculation_hash,'request_sha256':p.sha(root/'request.json'),
            'reporting_sources':{name:p.sha(ROOT/'06_analysis/dataset_summary'/name) for name in scripts},
            'inputs':inputs,'entries':entries,'outputs':{f.relative_to(output).as_posix():p.sha(f)
            for f in output.rglob('*') if f.is_file() and f.name!='manifest.json'}})
    print('Complete: three RL performance, three protection/cost and three improvement PDFs.',flush=True)


if __name__=='__main__':
    main()
