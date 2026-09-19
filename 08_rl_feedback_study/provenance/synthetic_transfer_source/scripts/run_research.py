"""Train and compare detector-specific policies within the existing RQ1/RQ2."""
from pathlib import Path
from copy import deepcopy
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse
import os
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import protocol as p
worker = p.load_module('policy_robustness_worker', '05_experiment/policy_robustness.py')
analysis = p.load_module('policy_robustness_analysis', '06_analysis/policy_robustness.py')
prepare = p.load_module('policy_robustness_prepare', '01_attacks/prepare.py')
training = p.load_module('detector_rl_training', '04_rl_training/train.py')
ROOT = p.ROOT
RESULTS = ROOT / 'results/detector_policy_study'


def configuration(spec, stage):
    config = deepcopy(p.read(ROOT / 'configs/thesis.json'))
    config.pop('input_bank', None)
    config.update(name='detector_policy_' + stage, rows=spec['rows'], streams=spec['streams'],
                  base_seeds=spec['training_seeds'] if stage == 'training' else
                  spec['validation_seeds'] if stage == 'validation' else spec['evaluation_seeds'],
                  persist_inputs=False, detectors=spec['detectors'])
    config['rl'].update(seeds=spec['rl_seeds'], epochs=spec['training_epochs'])
    if stage == 'pilot':
        config.update(rows=12000, base_seeds=[spec['pilot_seed']], attack_modes=['instance'], levels={'severe': .25})
        config['rl'].update(seeds=[7], epochs=1)
    return config


def data(spec, stage):
    source = RESULTS / stage
    config = configuration(spec, stage)
    request = source / 'request.json'
    if request.exists():
        if p.read(request)['config'] != config:
            raise ValueError('Input generation configuration changed')
        manifest = prepare.verify(source)
    else:
        p.write(request, {'config': config, 'protocol_sha256': p.sha(RESULTS / ('pilot_protocol.json' if stage == 'pilot' else 'protocol.json'))})
        manifest = prepare.build(source, config)
    return source, config, list(manifest['conditions'].values())


def freeze(spec, stage):
    sources = p.sources()
    sources.update({f.relative_to(ROOT).as_posix(): p.sha(f) for f in (ROOT / 'scripts').glob('*.py')})
    contract = {'spec': spec, 'sources': sources, 'environment': p.environment()}
    path = RESULTS / ('pilot_protocol.json' if stage == 'pilot' else 'protocol.json')
    if path.exists() and p.read(path) != contract:
        raise ValueError('Frozen protocol differs; do not mix results across code or settings')
    p.write(path, contract)


def checked_models(spec, source=None, pilot=False):
    source = source or RESULTS / 'training'
    manifest = p.read(source / '04_rl_training/manifest.json')
    models = manifest['models']
    seeds = [7] if pilot else spec['rl_seeds']
    expected = {(d, v, s) for d in spec['detectors'] for v in ['alarm', 'alarm_persistence'] for s in seeds}
    actual = [(m['detector'], m['variant'], m['seed']) for m in models]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError('Incomplete or duplicate detector-specific controller grid')
    for model in models:
        path = ROOT / model['path']
        if p.sha(path) != model['sha256']:
            raise ValueError('Frozen model hash differs')
        record = p.read(path)
        if any(record[name] != model[name] for name in ['detector', 'variant', 'seed']):
            raise ValueError('Model manifest and trained identity differ')
        if record['training_base_seeds'] != ([spec['pilot_seed']] if pilot else spec['training_seeds']):
            raise ValueError('Controller was not trained on the declared training split')
    return models


def jobs(source, output, conditions, arms, config, settings):
    return [(str(source), str(output), c, a, config, fraction, delay)
            for c in conditions for a in arms for fraction, delay in settings]


def execute(jobs_to_run, stage, workers):
    start = time.perf_counter()
    records = []
    progress = RESULTS / f'{stage}_status.json'
    p.write(progress, {'stage': stage, 'status': 'running', 'planned': len(jobs_to_run),
                       'completed': 0, 'pid': os.getpid(), 'started_utc': p.now()})
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(worker.run_one, job) for job in jobs_to_run]
        try:
            for future in as_completed(futures):
                result = future.result(); records.append(result)
                p.write(progress, {'stage': stage, 'status': 'running', 'planned': len(jobs_to_run),
                                   'completed': len(records), 'pid': os.getpid(), 'updated_utc': p.now(),
                                   'elapsed_seconds': time.perf_counter() - start})
                print(f"{stage} [{len(records)}/{len(jobs_to_run)}] {result['detector']} {result['condition']} "
                      f"{result['arm']} {result['feedback_fraction']}/{result['delay_blocks']}: {result['seconds']:.1f}s", flush=True)
        except BaseException as error:
            for future in futures:
                future.cancel()
            p.write(progress, {'stage': stage, 'status': 'failed', 'completed': len(records),
                               'planned': len(jobs_to_run), 'error': repr(error)})
            raise
    p.write(progress, {'stage': stage, 'status': 'complete', 'planned': len(jobs_to_run),
                       'completed': len(records), 'elapsed_seconds': time.perf_counter() - start})
    return records


def diagnostics(models):
    import pandas as pd
    rows, episodes = [], []
    for model in models:
        parent = (ROOT / model['path']).parent
        values = pd.read_csv(parent / 'policy.csv', dtype={'state': str})
        values['variant'], values['seed'], values['detector'] = model['variant'], model['seed'], model['detector']
        rows.append(values)
        values = pd.read_csv(parent / 'episodes.csv')
        values['variant'] = model['variant']; episodes.append(values)
    folder = RESULTS / 'analysis/training_diagnostics'; folder.mkdir(parents=True, exist_ok=True)
    pd.concat(rows).to_csv(folder / 'state_action_coverage.csv', index=False)
    pd.concat(episodes).to_csv(folder / 'episodes.csv', index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['pilot', 'training', 'validation', 'evaluation', 'sensitivity', 'all'])
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('Workers must be positive')
    spec = p.read(ROOT / 'configs/study.json')
    splits = [set(spec[k]) for k in ['training_seeds', 'validation_seeds', 'evaluation_seeds']]
    if any(splits[i] & splits[j] for i in range(3) for j in range(i)):
        raise ValueError('Training, validation and evaluation seeds must be disjoint')
    freeze(spec, args.stage)
    if args.stage == 'pilot':
        source, config, conditions = data(spec, 'pilot')
        training.train(source, config, args.workers)
        models = checked_models(spec, source, pilot=True)
        arms = [arm for detector in spec['detectors']
                for arm in worker.comparison_arms({'table': worker.policy_table('0001')}, models, detector)]
        records = execute(jobs(source, RESULTS / 'pilot_runs', conditions, arms, config,
                               [(.05, 1), (.01, 5)]), 'pilot', args.workers)
        analysis.report(records, RESULTS / 'analysis/pilot')
        return
    if args.stage in ['training', 'all']:
        source, config, _ = data(spec, 'training')
        training.train(source, config, args.workers)
    models = checked_models(spec)
    diagnostics(models)
    if args.stage in ['validation', 'all']:
        source, config, conditions = data(spec, 'validation')
        arms = [a for d in spec['detectors'] for a in worker.enumerated_arms(d)]
        records = execute(jobs(source, RESULTS / 'validation_runs', conditions, arms, config,
                               [(.05, 1)]), 'validation', args.workers)
        selections, rankings = {}, []
        for detector in spec['detectors']:
            selected, ranking = analysis.select_policy([r for r in records if r['detector'] == detector],
                                                       worker.enumerated_arms(detector))
            selected.update(detector=detector, validation_seeds=spec['validation_seeds'])
            selections[detector] = selected
            ranking['detector'] = detector; rankings.append(ranking)
        selected = {'policies': selections, 'protocol_sha256': p.sha(RESULTS / 'protocol.json'),
                    'training_manifest_sha256': p.sha(RESULTS / 'training/04_rl_training/manifest.json')}
        path = RESULTS / 'selected_policies.json'
        if path.exists() and p.read(path) != selected:
            raise ValueError('Previously frozen policy selection changed')
        p.write(path, selected)
        folder = RESULTS / 'analysis/validation'; folder.mkdir(parents=True, exist_ok=True)
        import pandas as pd
        pd.concat(rankings).to_csv(folder / 'policy_ranking.csv', index=False)
        analysis.frame(records).to_csv(folder / 'metrics.csv', index=False)
    if args.stage in ['evaluation', 'sensitivity', 'all']:
        selected = p.read(RESULTS / 'selected_policies.json')
        if selected['protocol_sha256'] != p.sha(RESULTS / 'protocol.json'):
            raise ValueError('Selected policies belong to another protocol')
        if selected['training_manifest_sha256'] != p.sha(RESULTS / 'training/04_rl_training/manifest.json'):
            raise ValueError('Frozen model manifest changed')
        source, config, conditions = data(spec, 'evaluation')
        arms = [a for d in spec['detectors'] for a in worker.comparison_arms(selected['policies'][d], models, d)]
        if args.stage in ['evaluation', 'all']:
            records = execute(jobs(source, RESULTS / 'evaluation_runs', conditions, arms, config,
                                   [(.05, 1)]), 'evaluation', args.workers)
            analysis.report(records, RESULTS / 'analysis/evaluation')
        if args.stage in ['sensitivity', 'all']:
            chosen = [c for c in conditions if [c['mode'], c['level']] in spec['sensitivity_conditions']]
            controls = [a for a in arms if not a['name'].startswith('q_alarm_')
                        or a['name'].startswith('q_alarm_persistence_')]
            settings = [(f, d) for f in spec['feedback_fractions'] for d in spec['feedback_delays']]
            records = execute(jobs(source, RESULTS / 'evaluation_runs', chosen, controls, config,
                                   settings), 'sensitivity', args.workers)
            analysis.report(records, RESULTS / 'analysis/sensitivity')
    if args.stage == 'all':
        p.write(RESULTS / 'completion.json', {'status': 'executions_complete', 'completed_utc': p.now(),
                                             'protocol_sha256': p.sha(RESULTS / 'protocol.json')})
        import subprocess
        subprocess.run([sys.executable, '-B', str(ROOT / '06_analysis/reporting/detector_report.py')], check=True)


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        p.write(RESULTS / 'failure.json', {'status': 'failed', 'error': repr(error), 'utc': p.now()})
        raise
