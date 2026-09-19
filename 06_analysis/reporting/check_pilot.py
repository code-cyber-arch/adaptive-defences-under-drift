"""Verify the all-dataset integration pilot independently of execution metrics."""
from pathlib import Path
import importlib.util
import sys
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from common import protocol as p
OUT=ROOT/'results/per_dataset_study'
spec=importlib.util.spec_from_file_location('pilot_metric_audit',Path(__file__).with_name('check_and_plot.py'))
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)
assert p.read(OUT/'pilot_status.json')['status']=='complete'
models=p.read(OUT/'pilot_models.json')['models'];assert len(models)==24
assert {(m['dataset'],m['detector'],m['variant']) for m in models}=={
    (ds,d,v) for ds in ['SEA_A','RBF_I','radar'] for d in ['adwin','hddm_w','hellinger','d3_oof'] for v in ['alarm','alarm_persistence']}
for m in models:
    path=ROOT/m['path'];assert p.sha(path)==m['sha256']
    record=p.read(path);assert record['dataset']==m['dataset'] and record['detector']==m['detector']
    trans=pd.read_parquet(path.parent/'transitions.parquet')
    assert trans.reward_block.eq(trans.action_block+1).all()
    assert trans.dataset.eq(m['dataset']).all()
count=0; combinations=set();control_hashes={}
for file in (OUT/'pilot_runs').rglob('summary.json'):
    r=p.read(file);ds=r['stream'];source=OUT/'pilot/evaluation'/ds
    c=p.read(source/'01_attacks/manifest.json')['conditions'][r['condition']]
    for name,digest in r['files'].items():assert p.sha(file.parent/name)==digest
    pred=pd.read_parquet(file.parent/'predictions.parquet');events=pd.read_parquet(file.parent/'events.parquet')
    truth=pd.read_parquet(source/'01_attacks'/c['truth'])
    feedback=pd.read_parquet(file.parent/'feedback_schedule.parquet')
    for metric,value in audit.independent_metrics(pred,truth,c['classes']).items():audit.close(r[metric],value)
    assert not pred.loc[pred.audit_reserved,'scored'].any()
    assert pred.loc[pred.audit_reserved,'first_admitted_block'].eq(-1).all()
    for event in events.loc[events.scored].itertuples():
        released=feedback[(feedback.row_id//1000==event.block-r['delay_blocks'])&feedback.release_block.le(event.block)]
        assert event.audit_n==len(released)
        assert event.audit_latest_row==(int(released.row_id.max()) if len(released) else -1)
        if not len(released):assert not event.accepted
    arm=r['contract']['arm'];assert arm['dataset']==ds and arm['detector']==r['detector']
    if 'model_path' in arm:
        record=p.read(ROOT/arm['model_path']);assert record['dataset']==ds and record['detector']==r['detector']
    if r['arm']=='no_reset':
        key=(ds,r['condition'],r['feedback_fraction'],r['delay_blocks'])
        assert control_hashes.setdefault(key,r['files']['predictions.parquet'])==r['files']['predictions.parquet']
    combinations.add((ds,r['detector']));count+=1
assert count==288 and len(combinations)==12
p.write(OUT/'pilot_verification.json',{'status':'passed','controllers':24,'comparisons':288,'dataset_detector_combinations':12,
    'metrics_recalculated_from_traces':True,'feedback_timing_checked':True,'dataset_detector_identity_checked':True,
    'completed_utc':p.now(),'checker_sha256':p.sha(Path(__file__))})
print('Pilot passed: 24 fitted controllers, 288 verified comparisons, all 12 dataset/detector combinations.')
