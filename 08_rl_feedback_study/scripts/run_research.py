"""Per-dataset RL training and matched comparison on SEA, RBF and RADAR."""
from pathlib import Path
from copy import deepcopy
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse
import os
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import protocol as p
from common.replication import counts, training_request_matches
worker = p.load_module('policy_robustness_worker', '05_experiment/policy_robustness.py')
analysis = p.load_module('policy_robustness_analysis', '06_analysis/policy_robustness.py')
prepare = p.load_module('per_dataset_prepare', '01_attacks/per_dataset.py')
training = p.load_module('per_dataset_training', '04_rl_training/train.py')
ROOT = p.ROOT
RESULTS = ROOT / 'results/per_dataset_study'


def configuration(spec, role, dataset, pilot=False):
    config = deepcopy(p.read(ROOT / 'configs/thesis.json'))
    config.pop('input_bank', None)
    seeds = spec['training_seeds'] if role == 'training' else spec['validation_seeds'] if role == 'validation' else spec['evaluation_seeds']
    config.update(name=role, data_role=role, dataset=dataset, rows=spec['rows'], streams=[dataset],
                  base_seeds=seeds if dataset != 'radar' else [], persist_inputs=False, detectors=spec['detectors'])
    config['rl'].update(seeds=spec['rl_seeds'], epochs=spec['training_epochs'])
    config.update(radar_embargo_blocks=spec['radar']['embargo_blocks'],
                  radar_attack_assignments=spec['radar']['evaluation_attack_assignments'] if role == 'evaluation' else [0])
    if dataset == 'radar':
        start, end = prepare.radar_bounds(spec['radar']['source_rows'], config['block_size'], config['radar_embargo_blocks'])[role]
        config['rows'] = end-start
    if pilot:
        config.update(rows=12000, pilot_rows=12000, base_seeds=[spec['pilot_seed']] if dataset != 'radar' else [],
                      attack_modes=['instance'], levels={'severe': .25}, radar_attack_assignments=[0])
        config['rl'].update(seeds=[7], epochs=1)
    return config


def freeze(spec, pilot=False):
    if spec['radar']['train_end_fraction'] != .5 or spec['radar']['validation_end_fraction'] != .7:
        raise ValueError('RADAR temporal boundaries differ from the implemented protocol')
    radar = prepare.snapshot_radar(spec)
    sources = p.sources()
    sources.update({f.relative_to(ROOT).as_posix(): p.sha(f) for f in (ROOT/'scripts').glob('*.py')})
    sources.update({f.relative_to(ROOT).as_posix(): p.sha(f) for f in (ROOT/'06_analysis/reporting').glob('per_dataset*.py')})
    contract = {'spec': spec, 'sources': sources, 'environment': p.environment(), 'radar_source': radar}
    path = RESULTS / ('pilot_protocol.json' if pilot else 'protocol.json')
    if path.exists() and p.read(path) != contract:
        raise ValueError('Frozen protocol differs; preserve completed evidence under its own contract')
    p.write(path, contract)
    return radar


def data(spec, role, dataset, radar, pilot=False):
    source = RESULTS / ('pilot' if pilot else 'stages') / role / dataset
    config = configuration(spec, role, dataset, pilot)
    request = {'config': config, 'protocol_sha256': p.sha(RESULTS / ('pilot_protocol.json' if pilot else 'protocol.json'))}
    if (source/'request.json').exists():
        if p.read(source/'request.json') != request:
            raise ValueError('Input request changed')
        manifest = prepare.verify(source)
    else:
        p.write(source/'request.json', request)
        manifest = prepare.build(source, config, radar if dataset == 'radar' else None)
    if manifest['dataset'] != dataset or manifest['data_role'] != role:
        raise ValueError('Input partition identity differs')
    return source, config, list(manifest['conditions'].values())


def model_manifest(pilot=False):
    return RESULTS / ('pilot_models.json' if pilot else 'models.json')


def checked_models(spec, pilot=False):
    models = p.read(model_manifest(pilot))['models']
    expected = {(ds, d, v, s) for ds in spec['datasets'] for d in spec['detectors']
                for v in ['alarm', 'alarm_persistence'] for s in ([7] if pilot else spec['rl_seeds'])}
    actual = [(m['dataset'], m['detector'], m['variant'], m['seed']) for m in models]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError('Incomplete or duplicate per-dataset controller grid')
    for m in models:
        path = ROOT/m['path']; record = p.read(path)
        if p.sha(path) != m['sha256'] or any(record[k] != m[k] for k in ['dataset','detector','variant','seed']):
            raise ValueError('Trained model identity or hash differs')
        source = RESULTS/('pilot' if pilot else 'stages')/'training'/m['dataset']
        if not training_request_matches(record['request_sha256'], source):
            raise ValueError('Controller training split differs')
    return models


def train_all(spec, radar, workers, pilot=False):
    models = []
    for dataset in spec['datasets']:
        source, config, _ = data(spec, 'training', dataset, radar, pilot)
        p.write(RESULTS/('pilot_training_status.json' if pilot else 'training_status.json'),
                {'status':'running','dataset':dataset,'completed':len(models),'planned':24 if pilot else counts(spec)['controllers']})
        models.extend(training.train(source, config, workers))
    p.write(model_manifest(pilot), {'models':models,'evaluation_updates':False,'training_mode':'per_dataset'})
    p.write(RESULTS/('pilot_training_status.json' if pilot else 'training_status.json'),
            {'status':'complete','completed':len(models),'planned':len(models)})
    return checked_models(spec, pilot)


def jobs(source, output, conditions, arms, config, settings):
    return [(str(source), str(output), c, a, config, f, d) for c in conditions for a in arms for f,d in settings]


def execute(work, stage, workers):
    records=[]; started=time.perf_counter(); status=RESULTS/f'{stage}_status.json'
    p.write(status, {'status':'running','completed':0,'planned':len(work),'pid':os.getpid()})
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(worker.run_one, job) for job in work]
        for future in as_completed(futures):
            result=future.result(); records.append(result)
            p.write(status, {'status':'running','completed':len(records),'planned':len(work),'pid':os.getpid(),
                             'updated_utc':p.now(),'elapsed_seconds':time.perf_counter()-started})
            print(f"{stage} [{len(records)}/{len(work)}] {result['stream']}/{result['detector']} "
                  f"{result['arm']} {result['condition']} {result['feedback_fraction']}/{result['delay_blocks']} {result['seconds']:.1f}s",flush=True)
    p.write(status, {'status':'complete','completed':len(records),'planned':len(work),
                     'elapsed_seconds':time.perf_counter()-started})
    return records


def diagnostics(models):
    import pandas as pd
    policies=[]; episodes=[]
    for m in models:
        parent=(ROOT/m['path']).parent
        for name, target in [('policy.csv', policies), ('episodes.csv', episodes)]:
            values=pd.read_csv(parent/name,dtype={'state':str})
            for key in ['dataset','detector','variant','seed']: values[key]=m[key]
            target.append(values)
    dest=RESULTS/'analysis/training_diagnostics'; dest.mkdir(parents=True,exist_ok=True)
    pd.concat(policies).to_csv(dest/'state_action_coverage.csv',index=False)
    pd.concat(episodes).to_csv(dest/'episodes.csv',index=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['pilot','training','validation','evaluation','sensitivity','all'])
    parser.add_argument('--workers',type=int,default=4); args=parser.parse_args()
    if args.workers<1: parser.error('Workers must be positive')
    spec=p.read(ROOT/'configs/study.json')
    splits=[set(spec[k]) for k in ['training_seeds','validation_seeds','evaluation_seeds']]
    if any(splits[i]&splits[j] for i in range(3) for j in range(i)): raise ValueError('Synthetic split seeds overlap')
    if spec['datasets'] != ['SEA_A','RBF_I','radar']: raise ValueError('All three datasets are required')
    pilot=args.stage=='pilot'; radar=freeze(spec,pilot)
    if pilot:
        models=train_all(spec,radar,args.workers,True)
        work=[]
        for dataset in spec['datasets']:
            source,config,conditions=data(spec,'evaluation',dataset,radar,True)
            arms=[a for d in spec['detectors'] for a in worker.comparison_arms(
                {'table':worker.policy_table('0001')},models,d,dataset)]
            work.extend(jobs(source,RESULTS/'pilot_runs',conditions,arms,config,[(.05,1),(.01,5)]))
        records=execute(work,'pilot',args.workers)
        analysis.report(records,RESULTS/'analysis/pilot')
        return
    if args.stage in ['training','all']: train_all(spec,radar,args.workers)
    models=checked_models(spec); diagnostics(models)
    if args.stage in ['validation','all']:
        work=[]
        for dataset in spec['datasets']:
            source,config,conditions=data(spec,'validation',dataset,radar)
            arms=[a for d in spec['detectors'] for a in worker.enumerated_arms(d,dataset)]
            work.extend(jobs(source,RESULTS/'validation_runs',conditions,arms,config,[(.05,1)]))
        records=execute(work,'validation',args.workers)
        import pandas as pd
        selections={}; rankings=[]
        for dataset in spec['datasets']:
            selections[dataset]={}
            for detector in spec['detectors']:
                rows=[r for r in records if r['stream']==dataset and r['detector']==detector]
                selected,ranking=analysis.select_policy(rows,worker.enumerated_arms(detector,dataset))
                selected.update(dataset=dataset,detector=detector)
                selections[dataset][detector]=selected
                ranking['dataset']=dataset;ranking['detector']=detector;rankings.append(ranking)
        record={'policies':selections,'protocol_sha256':p.sha(RESULTS/'protocol.json'),
                'models_sha256':p.sha(model_manifest())}
        target=RESULTS/'selected_policies.json'
        if target.exists() and p.read(target)!=record: raise ValueError('Frozen selections changed')
        p.write(target,record)
        folder=RESULTS/'analysis/validation';folder.mkdir(parents=True,exist_ok=True)
        pd.concat(rankings).to_csv(folder/'policy_ranking.csv',index=False)
        analysis.frame(records).to_csv(folder/'metrics.csv',index=False)
    if args.stage in ['evaluation','sensitivity','all']:
        selected=p.read(RESULTS/'selected_policies.json')
        if selected['models_sha256']!=p.sha(model_manifest()) or selected['protocol_sha256']!=p.sha(RESULTS/'protocol.json'):
            raise ValueError('Frozen selection inputs changed')
        default=[]; feedback=[]
        for dataset in spec['datasets']:
            source,config,conditions=data(spec,'evaluation',dataset,radar)
            arms=[a for d in spec['detectors'] for a in worker.comparison_arms(selected['policies'][dataset][d],models,d,dataset)]
            default.extend(jobs(source,RESULTS/'evaluation_runs',conditions,arms,config,[(.05,1)]))
            chosen=[c for c in conditions if [c['mode'],c['level']] in spec['sensitivity_conditions']]
            controls=[a for a in arms if not a['name'].startswith('q_alarm_') or a['name'].startswith('q_alarm_persistence_')]
            settings=[(f,d) for f in spec['feedback_fractions'] for d in spec['feedback_delays']]
            feedback.extend(jobs(source,RESULTS/'evaluation_runs',chosen,controls,config,settings))
        if args.stage in ['evaluation','all']:
            analysis.report(execute(default,'evaluation',args.workers),RESULTS/'analysis/evaluation')
        if args.stage in ['sensitivity','all']:
            analysis.report(execute(feedback,'sensitivity',args.workers),RESULTS/'analysis/sensitivity')
    if args.stage=='all':
        p.write(RESULTS/'completion.json',{'status':'executions_complete','utc':p.now()})
        import subprocess
        subprocess.run([sys.executable,'-B',str(ROOT/'06_analysis/reporting/per_dataset_report.py')],check=True)


if __name__=='__main__':
    try: main()
    except BaseException as error:
        p.write(RESULTS/'failure.json',{'status':'failed','error':repr(error),'utc':p.now()})
        raise
