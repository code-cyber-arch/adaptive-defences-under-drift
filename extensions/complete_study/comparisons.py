"""Detector, policy, dataset and cross-experiment comparisons on common host rows."""
from pathlib import Path
from itertools import combinations
from concurrent.futures import ProcessPoolExecutor,as_completed
import hashlib
import numpy as np
import pandas as pd
from common import protocol as p,channels
from . import workers as w
METRICS=['host_macro_f1','host_accuracy','poison_admission_rate','clean_withhold_rate','malicious_recall','benign_fpr','resets_per_100k','training_visits_per_100k']


def condition_job(job):
    source,out,c,config=job;source,out=Path(source),Path(out)
    profile=out/'filters/profiles'/c['stream']/c['key'];record=p.read(profile/'summary.json');w.verify_files(profile,record)
    mask_frame=pd.read_parquet(profile/'masks.parquet',columns=['row_id','comparison_scored'])
    mask=mask_frame.comparison_scored.to_numpy(bool);truth=channels.truth(source,c)
    assert np.array_equal(mask_frame.row_id,truth.row_id) and not (mask&truth.audit_reserved).any()
    digest=hashlib.sha256(mask.tobytes()).hexdigest();cached={};rows=[]
    def add(file,experiment,method,seed,detector=None,shared=False):
        r=p.read(file);w.verify_files(file.parent,r)
        if file not in cached:
            pred=pd.read_parquet(file.parent/'predictions.parquet')
            assert pred.row_id.equals(truth.row_id) and pred.scored.to_numpy()[mask].all()
            pred['scored']=mask
            cached[file]=w.metrics.metrics(pred,truth,c['classes'],config['warmup_rows'],c.get('benign_class'))
        rows.append(w.metadata(c)|cached[file]|dict(experiment=experiment,method=method,fitting_seed=seed,
                detector=r['detector'] if detector is None else detector,shared_detector_reference=shared,
                resets_per_100k=r['resets']/c['rows']*100000,training_visits_per_100k=r['training_visits']/c['rows']*100000,
                scoring_mask_sha256=digest,source_summary=file.relative_to(p.ROOT).as_posix(),source_sha256=p.sha(file)))
    for file in (out/'fixed_responses/runs'/c['stream']).glob(f'*/{c["key"]}/*/summary.json'):
        r=p.read(file);method=r['policy']+('_screened' if r['guarded'] else '_unscreened')
        add(file,'E1_fixed',method,None,shared=r['detector']=='none')
        if r['guarded'] and r['policy'] in ['none','confirmed']:
            add(file,'E2_RL',method,None,shared=r['detector']=='none')
    for file in (out/'evaluation_runs'/c['stream']).glob(f'*/{c["key"]}/*/feedback_0.05_delay_1/summary.json'):
        r=p.read(file);arm=r['contract']['arm']
        if r['arm'] in ['no_reset','confirmed']:continue
        method='RL_4state' if r['arm'].startswith('q_alarm_persistence_') else 'RL_2state' if r['arm'].startswith('q_alarm_') else r['arm']
        add(file,'E2_RL',method,arm.get('seed'))
    for file in (out/'filters/runs'/c['stream']).glob(f'*/{c["key"]}/*/summary.json'):
        r=p.read(file);method=r['policy']+'_screened_filter_'+r['filter']
        add(file,'E3_filter',method,r['filter_seed'],shared=r['detector']=='none')
    assert len(rows)==len({(r['experiment'],r['detector'],r['method'],r['fitting_seed']) for r in rows})
    return rows


def delta_rows(frame,group,pair_field,kind,metrics=METRICS):
    result=[]
    for key,part in frame.groupby(group,dropna=False):
        key=key if isinstance(key,tuple) else (key,)
        for (_,a),(_,b) in combinations(part.sort_values(pair_field).iterrows(),2):
            if kind=='cross_experiment' and a.experiment==b.experiment:continue
            row=dict(zip(group,key))|{'left':a[pair_field],'right':b[pair_field],'comparison':kind}
            for metric in metrics:row[metric+'_difference']=a[metric]-b[metric]
            result.append(row)
    return pd.DataFrame(result)


def build(out,sources,configs,conditions,detectors,workers):
    jobs=[(str(sources[('evaluation',ds)]),str(out),c,configs[('evaluation',ds)])
          for ds in ['SEA_A','RBF_I','radar'] for c in conditions[('evaluation',ds)]]
    records=[];done=0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(condition_job,job) for job in jobs]
        for future in as_completed(futures):
            records.extend(future.result());done+=1
            p.write(out/'comparison_status.json',dict(status='running',completed=done,planned=len(jobs),utc=p.now()))
    frame=pd.DataFrame(records)
    assert set(zip(frame.stream,frame.condition))=={(c['stream'],c['key']) for _,_,c,_ in jobs}
    assert frame.groupby(['stream','condition']).host_n.nunique().eq(1).all()
    assert frame.groupby(['stream','condition']).scoring_mask_sha256.nunique().eq(1).all()
    dest=out/'analysis/comparisons';dest.mkdir(parents=True,exist_ok=True)
    frame.to_csv(dest/'common_row_seed_metrics.csv',index=False)
    passive=pd.read_csv(out/'analysis/passive_detectors/metrics.csv')
    delta_rows(passive,['stream','condition','mode','level'],'detector','passive_detector',
               ['alarm_rate','poison_alarm_rate','unexposed_alarm_rate','event_recall','detection_delay','seconds_per_100k']).to_csv(dest/'passive_detector_pairs.csv',index=False)
    # Shared no-reset references enter each detector panel, without becoming independent runs.
    shared=frame[frame.detector.eq('none')]
    expanded=pd.concat([frame[~frame.detector.eq('none')],*[shared.assign(detector=d) for d in detectors]],ignore_index=True)
    keys=['experiment','stream','condition','mode','level','detector','method','shared_detector_reference']
    averaged=expanded.groupby(keys,dropna=False)[METRICS].mean().reset_index()
    expected={'E1_fixed':6,'E2_RL':6,'E3_filter':10}
    for experiment,count in expected.items():
        sub=averaged[averaged.experiment.eq(experiment)]
        assert sub.groupby(['stream','condition','detector']).method.nunique().eq(count).all(),experiment
        assert len(sub)==len(jobs)*len(detectors)*count
    averaged.to_csv(dest/'common_row_method_metrics.csv',index=False)
    detector=delta_rows(averaged[~averaged.shared_detector_reference],['experiment','stream','condition','mode','level','method'],'detector','detector')
    detector.to_csv(dest/'detector_pairs.csv',index=False)
    policy=delta_rows(averaged,['experiment','stream','condition','mode','level','detector'],'method','policy')
    policy.to_csv(dest/'policy_pairs.csv',index=False)
    cross=averaged.assign(comparison_arm=averaged.experiment+' / '+averaged.method)
    delta_rows(cross,['stream','condition','mode','level','detector'],'comparison_arm','cross_experiment').to_csv(dest/'cross_experiment_pairs.csv',index=False)
    group=['experiment','stream','detector','method']
    means=averaged.groupby([*group,'mode','level'])[METRICS].mean().groupby(group).mean().reset_index()
    means.to_csv(dest/'dataset_method_summary.csv',index=False)
    delta_rows(means,['experiment','detector','method'],'stream','dataset_descriptive').to_csv(dest/'dataset_differences.csv',index=False)
    means.groupby(['experiment','detector','method'])[METRICS].mean().to_csv(dest/'equal_dataset_summary.csv')
    index=pd.read_csv(out/'analysis/full_study/comparison_index.csv')
    indexed=index.merge(averaged,on=['stream','condition','mode','level'],validate='many_to_many')
    assert len(indexed)>0
    assert indexed.groupby(['experiment','stream','detector']).size().groupby(level=[0,2]).nunique().eq(1).all()
    indexed.to_csv(dest/'balanced_comparison_entries.csv',index=False)
    p.write(dest/'verification.json',dict(status='passed',conditions=len(jobs),common_host_masks_checked=True,
        all_datasets_and_detectors_checked=True,comparisons=['detector','policy','dataset_descriptive','cross_experiment'],
        equal_dataset_weights=True,shared_references_are_not_independent=True,utc=p.now()))
    p.write(out/'comparison_status.json',dict(status='complete',completed=done,planned=len(jobs),utc=p.now()))
