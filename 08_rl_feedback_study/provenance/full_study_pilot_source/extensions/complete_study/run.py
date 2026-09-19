"""Execute all experiment phases on the shared five-seed dataset grid."""
from pathlib import Path
from copy import deepcopy
from concurrent.futures import ProcessPoolExecutor,as_completed
import argparse
import os
import time
import fcntl
import pandas as pd
from common import protocol as p
from extensions.filter_study import protocol as filter_protocol
from . import workers as w, separability


def execute(jobs,worker,out,stage,count=None,workers=4):
    planned=len(jobs) if count is None else count;rows=[]
    p.write(out/f'{stage}_status.json',dict(status='running',completed=0,planned=planned,utc=p.now()))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(worker,job) for job in jobs]
        for future in as_completed(futures):
            result=future.result();rows.extend(result if isinstance(result,list) else [result])
            p.write(out/f'{stage}_status.json',dict(status='running',completed=len(rows),planned=planned,utc=p.now()))
            print(f'{stage}: {len(rows)}/{planned}',flush=True)
    p.write(out/f'{stage}_status.json',dict(status='complete',completed=len(rows),planned=planned,utc=p.now()))
    folder=out/'analysis'/stage;folder.mkdir(parents=True,exist_ok=True)
    pd.DataFrame([{k:v for k,v in row.items() if k not in ['files','contract']} for row in rows]).to_csv(folder/'metrics.csv',index=False)
    return rows


def inputs(base,pilot):
    sources={};configs={};conditions={}
    for ds in ['SEA_A','RBF_I','radar']:
        for role in ['training','validation','evaluation']:
            source=base/'stages'/role/ds
            if pilot:
                source=base/'integration_full_study/inputs'/role/ds
                config=deepcopy(p.read(base/'stages'/role/ds/'request.json')['config'])
                config.update(rows=12000,pilot_rows=12000,base_seeds=[] if ds=='radar' else [{'training':9911,'validation':9912,'evaluation':9913}[role]],
                              attack_modes=['instance'],levels={'severe':.25},radar_attack_assignments=[0])
                config['rl'].update(seeds=[7],epochs=1)
                request={'config':config,'fixture_only':True}
                if (source/'request.json').exists():assert p.read(source/'request.json')==request
                else:
                    prepare=p.load_module('full_pilot_prepare','01_attacks/per_dataset.py')
                    radar=p.read(base/'protocol.json')['radar_source']
                    prepare.build(source,config,radar if ds=='radar' else None)
                    p.write(source/'request.json',request)
            sources[(role,ds)]=source
            configs[(role,ds)]=p.read(source/'request.json')['config']
            manifest=p.read(source/'01_attacks/manifest.json')
            for name,digest in manifest['outputs'].items():assert p.sha(source/'01_attacks'/name)==digest
            conditions[(role,ds)]=list(manifest['conditions'].values())
    return sources,configs,conditions


def fit_filters(out,sources,configs,settings,seeds,pilot):
    total=len(seeds)*3*2;done=0
    for ds in ['SEA_A','RBF_I','radar']:
        roots={role:sources[(role,ds)] for role in ['training','validation','evaluation']}
        for seed in seeds:
            cfg=deepcopy(settings);cfg['seed']=seed
            p.write(out/'filter_training_status.json',dict(status='running',stream=ds,seed=seed,completed=done,planned=total,utc=p.now()))
            separability.fit_stream(out/'filters'/f'seed_{seed}',roots,ds,configs[('evaluation',ds)],cfg)
            done+=2
            print(f'Filter fitting: {done}/{total} ({ds}, seed {seed})',flush=True)
    p.write(out/'filter_training_status.json',dict(status='complete',completed=done,planned=total,utc=p.now()))


def run(args):
    spec=p.read(p.ROOT/'configs/full_study.json');base=p.ROOT/spec['input_study']
    out=base/'integration_full_study' if args.pilot else base
    out.mkdir(parents=True,exist_ok=True)
    with (out/'.full_study.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if not args.pilot:
            while True:
                complete=base/'completion.json'
                if complete.exists() and p.read(complete).get('status')=='complete' and p.read(complete).get('controllers')==120:break
                if (base/'failure.json').exists():raise RuntimeError('RL pipeline failed; full-study dependency is incomplete')
                p.write(out/'full_status.json',dict(status='waiting',phase='rl_comparison',utc=p.now()))
                if not args.wait:raise RuntimeError('Complete the five-seed RL grid first, or pass --wait')
                time.sleep(30)
        sources,configs,conditions=inputs(base,args.pilot)
        settings=deepcopy(filter_protocol.DEFAULTS)
        settings.update(max_fit_rows=spec['filter_max_fit_rows'],max_calibration_rows=spec['filter_max_calibration_rows'],legitimate_rejection_target=spec['filter_legitimate_rejection_target'])
        if args.pilot:
            settings.update(max_fit_rows=2000,max_calibration_rows=2000)
            settings['classifier']['max_iter']=5
        seeds=spec['filter_seeds'];detectors=['adwin','hddm_w','hellinger','d3_oof']
        code=p.sources()
        for directory in ['extensions/complete_study','extensions/filter_study','06_analysis/dataset_summary']:
            code.update({f.relative_to(p.ROOT).as_posix():p.sha(f) for f in (p.ROOT/directory).glob('*.py')})
        contract=dict(spec=spec,pilot=args.pilot,settings=settings,sources=code,environment=p.environment(),
                      input_contracts={f'{role}/{ds}':{'request':p.sha(root/'request.json'),'manifest':p.sha(root/'01_attacks/manifest.json')} for (role,ds),root in sources.items()})
        if (out/'full_protocol.json').exists():assert p.read(out/'full_protocol.json')==contract,'Full-study contract changed'
        p.write(out/'full_protocol.json',contract)
        p.write(out/'full_status.json',dict(status='running',phase='fixed_responses',utc=p.now()))
        fixed=[];passive=[];profiles=[];filter_jobs=[]
        for ds in ['SEA_A','RBF_I','radar']:
            source=sources[('evaluation',ds)];config=configs[('evaluation',ds)]
            for c in conditions[('evaluation',ds)]:
                fixed.extend((source,out,c,arm,config) for arm in w.fixed_arms(detectors))
                passive.extend((source,out,c,d,config) for d in detectors)
                profiles.append((source,out,c,config,seeds))
                filter_jobs.extend((source,out,c,config,d,kind,seed) for d in detectors for kind,seed in w.filter_specs(seeds))
                filter_jobs.extend((source,out,c,config,'none',kind,seed) for kind,seed in w.filter_specs(seeds) if kind=='none' or kind.startswith('learned_'))
        execute(fixed,w.fixed_job,out,'fixed_responses',workers=args.workers)
        execute(passive,w.passive_job,out,'passive_detectors',workers=args.workers)
        p.write(out/'full_status.json',dict(status='running',phase='filter_training',utc=p.now()))
        fit_filters(out,sources,configs,settings,seeds,args.pilot)
        execute(profiles,w.profile_job,out,'filter_profiles',workers=args.workers)
        p.write(out/'full_status.json',dict(status='running',phase='filter_evaluation',utc=p.now()))
        execute(filter_jobs,w.filter_job,out,'filter_evaluation',workers=args.workers)
        p.write(out/'full_status.json',dict(status='running',phase='verification',utc=p.now()))
        from . import report
        report.verify_and_report(out,sources,configs,conditions,seeds,detectors,args.pilot)
        p.write(out/'full_status.json',dict(status='complete',phase='all_experiments',utc=p.now()))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--wait',action='store_true');parser.add_argument('--pilot',action='store_true');args=parser.parse_args()
    if args.workers<1:parser.error('workers must be positive')
    os.environ.setdefault('MPLCONFIGDIR',str(p.ROOT/'results/.matplotlib'))
    try:run(args)
    except BaseException as error:
        out=p.ROOT/'results/per_dataset_study'
        if args.pilot:out/='integration_full_study'
        p.write(out/'full_failure.json',dict(error=repr(error),utc=p.now()));raise


if __name__=='__main__':main()
