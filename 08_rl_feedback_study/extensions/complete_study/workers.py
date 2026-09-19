"""Matched executions on one shared five-seed input grid."""
from pathlib import Path
from copy import deepcopy
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from common import protocol as p, channels
from extensions.filter_study import protocol as filters
from . import separability
engine=p.load_module('complete_fixed_engine','05_experiment/engine.py')
filter_engine=p.load_module('complete_filter_engine','extensions/filter_study/engine.py')
metrics=p.load_module('complete_metrics','05_experiment/policy_robustness.py')
passive=p.load_module('complete_passive','06_analysis/dataset_summary/detector_baseline.py')


def fixed_arms(detectors):
    return [dict(policy=policy,guarded=guarded,detector=detector)
            for policy in ['none','immediate','confirmed']
            for detector in (['none'] if policy=='none' else detectors)
            for guarded in [False,True]]


def fixed_folder(out, c, arm):
    return out/'fixed_responses/runs'/c['stream']/arm['detector']/c['key']/f"{arm['policy']}_{'screened' if arm['guarded'] else 'unscreened'}"


def verify_files(folder, record):
    for name,digest in record['files'].items():
        if p.sha(folder/name)!=digest:raise ValueError(f'Changed trace: {folder/name}')


def cached(folder, contract):
    if not (folder/'summary.json').exists():return None
    record=p.read(folder/'summary.json')
    if record['contract']!=contract:raise ValueError(f'Changed execution contract: {folder}')
    verify_files(folder,record)
    return record


def save(folder, record, **frames):
    folder.mkdir(parents=True,exist_ok=True)
    for name,frame in frames.items():frame.to_parquet(folder/f'{name}.parquet',index=False)
    record['files']={f'{name}.parquet':p.sha(folder/f'{name}.parquet') for name in frames}
    record['status']='complete';record['folder']=folder.relative_to(p.ROOT).as_posix()
    p.write(folder/'summary.json',record)
    return record


def base_contract(out,source,c,config):
    return dict(protocol_sha256=p.sha(out/'full_protocol.json'),condition=c['key'],config=config,
                inputs={k:p.sha(source/'01_attacks'/c[k]) for k in ['observations','audit','truth']})


def metadata(c):
    return {k:c.get(k) for k in ['stream','mode','level','base_seed','attack_seed','attack_assignment','base_key']}|{'condition':c['key']}


def fixed_job(job):
    source,out,c,arm,config=job;source,out=Path(source),Path(out)
    folder=fixed_folder(out,c,arm);contract=base_contract(out,source,c,config)|{'arm':arm}
    found=cached(folder,contract)
    if found:return found
    local=deepcopy(config);local['detector']={'name':arm['detector'],'seed':p.seed('monitor',c['key'],arm['detector'])}
    reused=None
    if arm['guarded'] and arm['policy'] in ['none','confirmed']:
        det='adwin' if arm['policy']=='none' else arm['detector'];name='no_reset' if arm['policy']=='none' else 'confirmed'
        candidate=out/'evaluation_runs'/c['stream']/det/c['key']/name/'feedback_0.05_delay_1'
        if (candidate/'summary.json').exists():
            old=p.read(candidate/'summary.json');verify_files(candidate,old)
            if old['contract']['inputs']!=contract['inputs']:raise ValueError('Reused comparison inputs differ')
            if not metrics.run_contract_matches(old['contract'],dict(old['contract'],config=config)):raise ValueError('Reused comparison settings differ')
            pred=pd.read_parquet(candidate/'predictions.parquet');events=pd.read_parquet(candidate/'events.parquet')
            stats={k:old[k] for k in ['resets','training_visits','seconds']}
            reused={'summary':(candidate/'summary.json').relative_to(p.ROOT).as_posix(),'sha256':p.sha(candidate/'summary.json')}
    if reused is None:
        with threadpool_limits(limits=1):
            pred,events,stats=engine.execute(channels.observations(source,c),channels.protected(source,c),c['classes'],arm['policy'],arm['guarded'],local)
    truth=channels.truth(source,c)
    record=metadata(c)|stats|metrics.metrics(pred,truth,c['classes'],config['warmup_rows'],c.get('benign_class'))|arm
    record.update(contract=contract,reused=reused)
    return save(folder,record,predictions=pred,events=events)


def passive_job(job):
    source,out,c,detector,config=job;source,out=Path(source),Path(out)
    baseline=fixed_folder(out,c,dict(policy='none',guarded=False,detector='none'))
    baseline_record=p.read(baseline/'summary.json');verify_files(baseline,baseline_record)
    folder=out/'passive_detectors/runs'/c['stream']/detector/c['key']
    contract=base_contract(out,source,c,config)|{'detector':detector,'baseline_sha256':p.sha(baseline/'summary.json')}
    found=cached(folder,contract)
    if found:return found
    obs=channels.observations(source,c);pred=pd.read_parquet(baseline/'predictions.parquet')
    assert pred.row_id.equals(obs.row_id)
    features=filters.features(obs);reserved=obs.audit_reserved.to_numpy(bool)
    monitor=passive.monitors.make(detector,seed=p.seed('monitor',c['key'],detector))
    with threadpool_limits(limits=1):
        events=passive.observe(obs[features].to_numpy(float),pred.prediction.to_numpy()!=obs.label.to_numpy(),reserved,config,monitor)
    events,stats=passive.score(events,channels.truth(source,c),c,reserved,config)
    record=metadata(c)|stats|dict(detector=detector,contract=contract,learner_resets=0)
    return save(folder,record,alarms=events)


def filter_specs(seeds):
    result=[('none',None),('oracle_rows',None),('oracle_blocks',None)]
    result += [(kind,seed) for kind in ['random_rows','random_blocks','learned_features','learned_features_label'] for seed in seeds]
    return result


def filter_key(kind,seed):return kind if seed is None else f'{kind}_seed_{seed}'


def profile_job(job):
    source,out,c,config,seeds=job;source,out=Path(source),Path(out)
    folder=out/'filters/profiles'/c['stream']/c['key']
    model_hashes={str(seed):p.sha(out/'filters'/f'seed_{seed}'/'models'/c['stream']/'manifest.json') for seed in seeds}
    contract=base_contract(out,source,c,config)|{'models':model_hashes}
    found=cached(folder,contract)
    if found:return found
    obs=channels.observations(source,c);truth=channels.truth(source,c)
    reserved=obs.audit_reserved.to_numpy(bool);poisoned=(truth.atk_flip|truth.atk_burst).to_numpy(bool)
    eligible=(obs.row_id.to_numpy()>=config['warmup_rows'])&~reserved
    excluded=None
    for seed in seeds:
        values=np.load(out/'filters'/f'seed_{seed}'/'models'/c['stream']/'excluded_test_fingerprints.npy',allow_pickle=False)
        if excluded is None:excluded=values
        else:assert np.array_equal(values,excluded),'Seed-dependent scoring population'
    score=eligible&truth.host_eligible.to_numpy(bool)&~np.isin(filters.fingerprints(obs),excluded)
    if not score.any():raise ValueError('No copy-disjoint evaluation rows')
    frame=pd.DataFrame({'row_id':obs.row_id,'comparison_scored':score})
    for kind,seed in filter_specs(seeds):
        key=filter_key(kind,seed)
        if kind.startswith('learned_'):
            view=kind.removeprefix('learned_');bundle=separability.load_bundle(out/'filters'/f'seed_{seed}',c['stream'],view)
            assert filters.features(obs)==bundle['feature_names']
            probabilities=np.zeros(len(obs));ids=np.flatnonzero(eligible)
            with threadpool_limits(limits=1):
                for i in range(0,len(ids),50000):
                    rows=ids[i:i+50000];probabilities[rows]=bundle['model'].predict_proba(filters.inputs(obs.iloc[rows],view,bundle['classes']))[:,1]
            frame[key]=(probabilities>bundle['threshold'])&eligible
            frame[key+'_score']=probabilities
        else:
            frame[key]=filters.withholding(kind,poisoned,reserved,config['block_size'],config['warmup_rows'],p.seed(seed,c['key'],kind))
    return save(folder,metadata(c)|dict(contract=contract,host_rows=int(score.sum())),masks=frame)


def filter_job(job):
    source,out,c,config,detector,kind,seed=job;source,out=Path(source),Path(out)
    profile=out/'filters/profiles'/c['stream']/c['key'];pr=p.read(profile/'summary.json');verify_files(profile,pr)
    key=filter_key(kind,seed);folder=out/'filters/runs'/c['stream']/detector/c['key']/key
    contract=base_contract(out,source,c,config)|dict(detector=detector,filter=kind,filter_seed=seed,profile_sha256=p.sha(profile/'summary.json'))
    found=cached(folder,contract)
    if found:return found
    masks=pd.read_parquet(profile/'masks.parquet',columns=['row_id','comparison_scored',key]);mask=masks[key].to_numpy(bool)
    policy='none' if detector=='none' else 'confirmed';reused=None
    if kind=='none':
        base=fixed_folder(out,c,dict(policy=policy,guarded=True,detector=detector))
        br=p.read(base/'summary.json');verify_files(base,br)
        pred=pd.read_parquet(base/'predictions.parquet');events=pd.read_parquet(base/'events.parquet')
        stats={k:br[k] for k in ['resets','training_visits','seconds']};pred['filter_withheld']=mask
        reused={'summary':(base/'summary.json').relative_to(p.ROOT).as_posix(),'sha256':p.sha(base/'summary.json')}
    else:
        local=deepcopy(config);local['detector']={'name':detector,'seed':p.seed('monitor',c['key'],detector)}
        with threadpool_limits(limits=1):
            pred,events,stats=filter_engine.execute(channels.observations(source,c),channels.protected(source,c),c['classes'],policy,True,local,withheld=mask)
    assert masks.row_id.equals(pred.row_id)
    pred['comparison_scored']=masks.comparison_scored
    truth=channels.truth(source,c)
    host=pred.copy();host['scored']=pred.comparison_scored
    scores=metrics.metrics(host,truth,c['classes'],config['warmup_rows'],c.get('benign_class'))
    exposure=(pred.row_id>=config['warmup_rows'])&~pred.audit_reserved
    poison=(truth.atk_flip|truth.atk_burst)&exposure;legitimate=~(truth.atk_flip|truth.atk_burst)&exposure
    ratio=lambda a,b:float(a/b) if b else None
    scores.update(filter_poison_recall=ratio((mask&poison).sum(),poison.sum()),filter_legitimate_rejection=ratio((mask&legitimate).sum(),legitimate.sum()))
    record=metadata(c)|stats|scores|dict(detector=detector,filter=kind,filter_seed=seed,policy=policy,guarded=True,contract=contract,reused=reused)
    return save(folder,record,predictions=pred,events=events)
