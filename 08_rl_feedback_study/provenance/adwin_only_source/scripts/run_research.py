"""Run the bounded RL comparison and feedback sensitivity research extension."""
from pathlib import Path
from copy import deepcopy
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse
import json
import os
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import protocol as p
worker = p.load_module('policy_robustness_worker', '05_experiment/policy_robustness.py')
analysis = p.load_module('policy_robustness_analysis', '06_analysis/policy_robustness.py')
prepare = p.load_module('policy_robustness_prepare', '01_attacks/prepare.py')
ROOT = p.ROOT
RESULTS = ROOT / 'results/policy_robustness'


def configuration(spec, stage):
    config = deepcopy(p.read(ROOT / 'configs/thesis.json'))
    config.pop('input_bank', None)
    config.update(name='policy_robustness_' + stage, rows=spec['rows'], streams=spec['streams'],
                  base_seeds=spec['validation_seeds'] if stage == 'validation' else spec['evaluation_seeds'],
                  persist_inputs=False, detectors=['adwin'])
    if stage == 'pilot':
        config.update(base_seeds=[spec['pilot_seed']], attack_modes=['instance'], levels={'severe': .25})
    return config


def data(spec, stage):
    source = RESULTS / 'inputs' / stage
    config = configuration(spec, stage)
    request = source / 'request.json'
    if request.exists():
        if p.read(request)['config'] != config:
            raise ValueError('Input generation configuration changed')
        manifest = prepare.verify(source)
    else:
        p.write(request, {'config': config})
        manifest = prepare.build(source, config)
    return source, config, list(manifest['conditions'].values())


def freeze(spec, stage):
    sources = p.sources()
    sources.update({f.relative_to(ROOT).as_posix(): p.sha(f) for f in (ROOT / 'scripts').glob('*.py')})
    models = {f.relative_to(ROOT).as_posix(): p.sha(f)
              for f in (ROOT / 'results/training/04_rl_training').rglob('q_table.json')}
    contract = {'spec': spec, 'sources': sources, 'models': models, 'environment': p.environment()}
    path = RESULTS / ('pilot_protocol.json' if stage == 'pilot' else 'protocol.json')
    if path.exists() and p.read(path) != contract:
        raise ValueError('Frozen protocol differs; do not mix results across code or settings')
    p.write(path, contract)


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
                result = future.result()
                records.append(result)
                p.write(progress, {'stage': stage, 'status': 'running', 'planned': len(jobs_to_run),
                                   'completed': len(records), 'pid': os.getpid(), 'updated_utc': p.now(),
                                   'elapsed_seconds': time.perf_counter() - start})
                print(f"{stage} [{len(records)}/{len(jobs_to_run)}] {result['condition']} {result['arm']} "
                      f"{result['feedback_fraction']}/{result['delay_blocks']}: {result['seconds']:.1f}s", flush=True)
        except BaseException as error:
            for future in futures:
                future.cancel()
            p.write(progress, {'stage': stage, 'status': 'failed', 'completed': len(records),
                               'planned': len(jobs_to_run), 'error': repr(error)})
            raise
    p.write(progress, {'stage': stage, 'status': 'complete', 'planned': len(jobs_to_run),
                       'completed': len(records), 'elapsed_seconds': time.perf_counter() - start})
    return records


def diagnostics():
    import pandas as pd
    rows = []
    for path in sorted((ROOT / 'results/training/04_rl_training').rglob('policy.csv')):
        values = pd.read_csv(path)
        values['variant'], values['seed'] = path.parent.parent.name, int(path.parent.name.split('_')[1])
        rows.append(values)
    folder = RESULTS / 'analysis/training_diagnostics'
    folder.mkdir(parents=True, exist_ok=True)
    pd.concat(rows).to_csv(folder / 'state_action_coverage.csv', index=False)
    episodes = []
    for path in sorted((ROOT / 'results/training/04_rl_training').rglob('episodes.csv')):
        values = pd.read_csv(path)
        values['variant'] = path.parent.parent.name
        episodes.append(values)
    pd.concat(episodes).to_csv(folder / 'episodes.csv', index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['pilot', 'validation', 'evaluation', 'sensitivity', 'all'])
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('Workers must be positive')
    spec = p.read(ROOT / 'configs/study.json')
    if set(spec['validation_seeds']) & set(spec['evaluation_seeds']):
        raise ValueError('Validation and evaluation seeds overlap')
    freeze(spec, args.stage)
    if args.stage == 'pilot':
        source, config, conditions = data(spec, 'pilot')
        arms = worker.comparison_arms({'table': worker.policy_table('0001')}, [7])
        arms = [a for a in arms if a['name'] in ['no_reset', 'confirmed', 'q_alarm_persistence_7']]
        records = execute(jobs(source, RESULTS / 'pilot_runs', conditions, arms, config,
                               [(.05, 1), (.01, 5)]), 'pilot', args.workers)
        analysis.report(records, RESULTS / 'analysis/pilot')
        return
    diagnostics()
    if args.stage in ['validation', 'all']:
        source, config, conditions = data(spec, 'validation')
        arms = worker.enumerated_arms()
        records = execute(jobs(source, RESULTS / 'validation_runs', conditions, arms, config,
                               [(.05, 1)]), 'validation', args.workers)
        selected, ranking = analysis.select_policy(records, arms)
        selected.update(validation_seed=spec['validation_seeds'], metric=spec['selection_metric'],
                        protocol_sha256=p.sha(RESULTS / 'protocol.json'))
        path = RESULTS / 'selected_policy.json'
        if path.exists() and p.read(path) != selected:
            raise ValueError('Previously frozen policy selection changed')
        p.write(path, selected)
        folder = RESULTS / 'analysis/validation'
        folder.mkdir(parents=True, exist_ok=True)
        ranking.to_csv(folder / 'policy_ranking.csv', index=False)
        analysis.frame(records).to_csv(folder / 'metrics.csv', index=False)
    if args.stage in ['evaluation', 'sensitivity', 'all']:
        selected = p.read(RESULTS / 'selected_policy.json')
        if selected['protocol_sha256'] != p.sha(RESULTS / 'protocol.json'):
            raise ValueError('Selected policy belongs to another protocol')
        source, config, conditions = data(spec, 'evaluation')
        arms = worker.comparison_arms(selected, spec['rl_seeds'])
        if args.stage in ['evaluation', 'all']:
            records = execute(jobs(source, RESULTS / 'evaluation_runs', conditions, arms, config,
                                   [(.05, 1)]), 'evaluation', args.workers)
            analysis.report(records, RESULTS / 'analysis/evaluation')
        if args.stage in ['sensitivity', 'all']:
            chosen_conditions = [c for c in conditions if [c['mode'], c['level']] in spec['sensitivity_conditions']]
            chosen_arms = [a for a in arms if not a['name'].startswith('q_alarm_')
                           or a['name'].startswith('q_alarm_persistence_')]
            settings = [(f, d) for f in spec['feedback_fractions'] for d in spec['feedback_delays']]
            records = execute(jobs(source, RESULTS / 'evaluation_runs', chosen_conditions, chosen_arms, config,
                                   settings), 'sensitivity', args.workers)
            analysis.report(records, RESULTS / 'analysis/sensitivity')
    if args.stage == 'all':
        p.write(RESULTS / 'completion.json', {'status': 'complete', 'completed_utc': p.now(),
                                             'protocol_sha256': p.sha(RESULTS / 'protocol.json')})


if __name__ == '__main__':
    main()
