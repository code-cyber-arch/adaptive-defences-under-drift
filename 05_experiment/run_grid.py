"""Execute matched detector, response, screening and frozen RL arms."""
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
import uuid
import pandas as pd
from threadpoolctl import threadpool_limits
from common import protocol as p
from common import channels
execute = p.load_module('stream_engine', '05_experiment/engine.py').execute

def worker(root, condition, arm, config):
    """Execute one independent policy arm and save its complete outputs."""
    key = p.run_id(condition['key'], arm)
    final = p.run_folder(root, condition, arm)
    if final.exists():
        raise FileExistsError(final)
    folder = root / 'work_in_progress' / (key + '__' + uuid.uuid4().hex[:8])
    folder.mkdir(parents=True)
    obs = channels.observations(root, condition)
    audit = channels.protected(root, condition)
    local = deepcopy(config)
    local['detector'] = {'name': arm['detector'], 'seed': p.seed('monitor', condition['key'], arm['detector'])}
    controller = None
    if p.is_rl(arm['policy']):
        contract = arm['rl_model']
        file = p.ROOT / contract['path']
        if p.sha(file) != contract['sha256']:
            raise ValueError('Frozen RL model changed')
        record = p.read(file)
        if record.get('detector', 'adwin') != arm['detector']:
            raise ValueError('Frozen RL detector differs from evaluation detector')
        expected_variant = 'alarm' if arm['policy'] == 'rl_alarm' else 'alarm_persistence'
        if record['variant'] != expected_variant:
            raise ValueError('Frozen RL variant differs from the comparison arm')
        controller = p.load_module('rl_controller', '04_rl_training/controller.py').Controller(record['settings'], record['seed'], False, record['table'])
    with threadpool_limits(limits=1):
        pred, events, stats = execute(obs, audit, condition['classes'], arm['policy'], arm['guarded'], local, controller)
    pred.to_parquet(folder / 'predictions.parquet', index=False)
    events.to_parquet(folder / 'events.parquet', index=False)
    if controller is not None:
        pd.DataFrame(controller.transitions).to_parquet(folder / 'rl_transitions.parquet', index=False)
        if controller.table != record['table']:
            raise AssertionError('RL evaluation changed the frozen policy')
        stats['rl_model_sha256'] = contract['sha256']
        stats['rl_variant'] = record['variant']
    stats.update(run_id=key, condition=condition['key'], stream=condition['stream'], mode=condition['mode'], level=condition['level'], base_seed=condition['base_seed'], base_key=condition['base_key'], relative_folder=final.relative_to(root).as_posix(), attack_seed=condition.get('attack_seed'), detector=arm['detector'], rl_seed=arm['rl_seed'], status='complete')
    stats['files'] = {f.name: p.sha(f) for f in folder.glob('*.parquet')}
    p.write(folder / 'summary.json', stats)
    final.parent.mkdir(parents=True, exist_ok=True)
    folder.rename(final)
    return stats

def run(root, manifest, config, workers=2, resume=False):
    """Run or resume all declared arms on their matching input conditions."""
    jobs = []
    completed = []
    for condition in manifest['conditions'].values():
        for arm in p.arms(config):
            key = p.run_id(condition['key'], arm)
            file = p.run_folder(root, condition, arm) / 'summary.json'
            if file.exists() and resume:
                summary = p.read(file)
                for name, digest in summary['files'].items():
                    if p.sha(file.parent / name) != digest:
                        raise ValueError(f'Run artifact changed: {key}/{name}')
                completed.append(summary)
            else:
                runtime = {name: condition[name] for name in ['key', 'base_key', 'attack_seed', 'observations', 'audit', 'classes', 'stream', 'mode', 'level', 'base_seed']}
                jobs.append((runtime, arm))
    planned = len(completed) + len(jobs)
    p.write(root / 'progress.json', {'planned': planned, 'completed': len(completed), 'status': 'running'})
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(worker, root, c, a, config): (c['key'], a) for c, a in jobs}
        for future in as_completed(futures):
            summary = future.result()
            completed.append(summary)
            p.write(root / 'progress.json', {'planned': planned, 'completed': len(completed), 'last_run': summary['run_id'], 'status': 'running'})
            print(f"[{len(completed)}/{planned}] {summary['run_id']}: {summary['seconds']:.1f}s", flush=True)
    p.write(root / '05_runs/completion.json', {'planned': planned, 'completed': len(completed), 'paths': {s['run_id']: s['relative_folder'] for s in completed}, 'summaries': {s['run_id']: p.sha(root / s['relative_folder'] / 'summary.json') for s in completed}})
    p.write(root / 'progress.json', {'planned': planned, 'completed': len(completed), 'status': 'models_complete'})
    return completed
