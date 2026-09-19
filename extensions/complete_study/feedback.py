"""Complete protected-feedback comparisons on every evaluation condition."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import shutil
import numpy as np
import pandas as pd
from common import protocol as p,channels
from . import workers as w
check=p.load_module('full_feedback_independent','06_analysis/reporting/check_and_plot.py')
analysis=p.load_module('full_feedback_analysis','06_analysis/policy_robustness.py')


def run(out,sources,configs,conditions,detectors,workers):
    models=p.read(out/'models.json')['models'];selected=p.read(out/'selected_policies.json')['policies']
    jobs=[]
    for ds in ['SEA_A','RBF_I','radar']:
        arms=[a for detector in detectors for a in w.metrics.comparison_arms(selected[ds][detector],models,detector,ds)]
        jobs.extend((str(sources[('evaluation',ds)]),str(out/'evaluation_runs'),c,a,configs[('evaluation',ds)],f,d)
                    for c in conditions[('evaluation',ds)] for a in arms for f in [.01,.02,.05] for d in [1,5])
    assert len(jobs)==33936
    archive=p.ROOT/'provenance/feedback_contract_records'/p.sha(out/'protocol.json')
    for file in [out/'completion.json',out/'sensitivity_status.json',*(out/'analysis/sensitivity').glob('*'),*(out/'analysis/verification').glob('*')]:
        if file.is_file():
            target=archive/file.relative_to(out);target.parent.mkdir(parents=True,exist_ok=True)
            if not target.exists():shutil.copy2(file,target)
    records=[]
    p.write(out/'complete_feedback_status.json',dict(status='running',completed=0,planned=len(jobs),utc=p.now()))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(w.metrics.run_one,job) for job in jobs]
        for future in as_completed(futures):
            records.append(future.result())
            p.write(out/'complete_feedback_status.json',dict(status='running',completed=len(records),planned=len(jobs),utc=p.now()))
            if len(records)%50==0:print(f'Complete feedback: {len(records)}/{len(jobs)}',flush=True)
    analysis.report(records,out/'analysis/sensitivity')
    p.write(out/'complete_feedback_status.json',dict(status='verifying',completed=len(records),planned=len(jobs),utc=p.now()))
    audit(out,jobs)
    p.write(out/'complete_feedback_status.json',dict(status='complete',completed=len(records),planned=len(jobs),utc=p.now()))


def audit(out,jobs):
    seen=set();cached_truth={};checked=[]
    for source,output,c,arm,config,fraction,delay in jobs:
        key=(c['stream'],c['key'],arm['detector'],arm['name'],fraction,delay)
        assert key not in seen;seen.add(key)
        folder=Path(output)/c['stream']/arm['detector']/c['key']/arm['name']/f'feedback_{fraction:.2f}_delay_{delay}'
        record=p.read(folder/'summary.json');w.verify_files(folder,record)
        requested={'condition':c['key'],'arm':arm,'config':config,'feedback_fraction':fraction,'delay_blocks':delay,
                   'inputs':{name:p.sha(Path(source)/'01_attacks'/c[name]) for name in ['observations','audit','truth']}}
        assert w.metrics.run_contract_matches(record['contract'],requested)
        ck=(source,c['key'])
        if ck not in cached_truth:
            cached_truth.clear();cached_truth[ck]=channels.truth(Path(source),c)
        truth=cached_truth[ck]
        pred=pd.read_parquet(folder/'predictions.parquet');events=pd.read_parquet(folder/'events.parquet');schedule=pd.read_parquet(folder/'feedback_schedule.parquet')
        assert pred.row_id.equals(truth.row_id)
        reserved=truth.audit_reserved
        assert np.array_equal(pred.audit_reserved,reserved)
        assert not pred.loc[reserved,'scored'].any() and pred.loc[reserved,'candidate_training_visits'].eq(0).all()
        assert pred.loc[reserved,'first_admitted_block'].eq(-1).all()
        assert set(schedule.row_id)==set(truth.loc[reserved,'row_id'])
        assert record['resets']==int(events.reset_committed.sum())
        assert record['training_visits']==int(pred.candidate_training_visits.sum())
        for metric,value in check.independent_metrics(pred,truth,c['classes']).items():check.close(record[metric],value)
        table=w.metrics.load_frozen(arm)['table'] if 'model_path' in arm else arm.get('table')
        for event in events.loc[events.scored].itertuples():
            block=event.block-delay
            released=schedule[(schedule.row_id//config['block_size']==block)&(schedule.release_block<=event.block)]
            assert event.audit_n==len(released)
            assert event.audit_latest_row==(int(released.row_id.max()) if len(released) else -1)
            assert event.reset_committed==bool(event.reset_requested and event.accepted)
            if not len(released):assert not event.accepted
            if table is not None:assert event.rl_action==('update' if table[event.rl_state][0]>=table[event.rl_state][1] else 'reset')
        checked.append({'summary':(folder/'summary.json').relative_to(p.ROOT).as_posix(),'sha256':p.sha(folder/'summary.json')})
    assert len(seen)==33936
    dest=out/'analysis/full_feedback';dest.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(checked).to_csv(dest/'checked_runs.csv',index=False)
    p.write(dest/'verification.json',dict(status='passed',configurations=len(seen),all_seven_conditions=True,
             protected_rows_and_release_timing_checked=True,frozen_policy_actions_checked=True,metrics_recomputed=True,utc=p.now()))
