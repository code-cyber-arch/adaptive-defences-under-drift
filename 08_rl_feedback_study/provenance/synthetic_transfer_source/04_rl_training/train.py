"""Train independent RL controllers on development observations only."""
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from common import protocol as p
from common import channels
engine = p.load_module('rl_training_engine', '05_experiment/engine.py')
Controller = p.load_module('training_controller', '04_rl_training/controller.py').Controller

def train_one(root, config, seed, variant='alarm', detector='adwin'):
    """Fit one declared state representation using training observations only."""
    if detector not in config['detectors']:
        raise ValueError('Training detector is outside the declared grid')
    output = root / '04_rl_training' / detector / variant / f'seed_{seed}' / 'q_table.json'
    manifest = p.read(root / '01_attacks/manifest.json')
    for c in manifest['conditions'].values():
        if c['base_seed'] not in config['base_seeds'] or c['stream'] == 'radar':
            raise ValueError('RL training contains evaluation data')
    if output.exists():
        saved = p.read(output)
        if saved['request_sha256'] != p.sha(root / 'request.json'):
            raise ValueError('RL training contract changed')
        if saved['variant'] != variant or saved.get('detector') != detector:
            raise ValueError('Saved controller uses a different state representation')
        for name, digest in saved['files'].items():
            if p.sha(output.parent / name) != digest:
                raise ValueError('RL training artifact changed')
        return {'seed': seed, 'variant': variant, 'detector': detector, 'path': output.relative_to(p.ROOT).as_posix(), 'sha256': p.sha(output)}
    settings = dict(config['rl'], variant=variant)
    agent = Controller(settings, seed, training=True)
    rows = []
    transitions = []
    jobs = list(manifest['conditions'].values())
    rng = np.random.default_rng(seed)
    with threadpool_limits(limits=1):
        for epoch in range(config['rl']['epochs']):
            agent.epsilon = config['rl']['epsilon']
            for position in rng.permutation(len(jobs)):
                c = jobs[int(position)]
                obs = channels.observations(root, c)
                audit = channels.protected(root, c)
                agent.pending = None
                start = len(agent.transitions)
                local = deepcopy(config)
                local['detector'] = {'name': detector, 'seed': p.seed('monitor', c['key'], detector)}
                _, events, stats = engine.execute(obs, audit, c['classes'], 'rl', True, local, agent)
                new = agent.transitions[start:]
                transitions.extend((dict(t, epoch=epoch, condition=c['key'], detector=detector) for t in new))
                rows.append({'epoch': epoch, 'condition': c['key'], 'seed': seed, 'detector': detector, 'reward': sum((t['reward'] for t in new)), 'training_visits': stats['training_visits'], 'resets': stats['resets'], 'states': len(agent.table), 'epsilon': agent.epsilon})
                print(f"RL {detector}/{variant}, seed {seed}: epoch {epoch + 1}/{config['rl']['epochs']}, {c['key']}, states {len(agent.table)}", flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output.parent / 'episodes.csv', index=False)
    pd.DataFrame(transitions).to_parquet(output.parent / 'transitions.parquet', index=False)
    diagnostics = []
    for state, values in agent.table.items():
        for index, action in enumerate(('update', 'reset')):
            subset = [t for t in transitions if t['state'] == state and t['action'] == action]
            diagnostics.append({'state': state, 'action': action, 'q_value': values[index],
                                'selected': index == int(np.argmax(values)), 'visits': len(subset),
                                'mean_reward': np.mean([t['reward'] for t in subset]) if subset else None,
                                'rejected_fraction': np.mean([t['rejected'] for t in subset]) if subset else None})
    pd.DataFrame(diagnostics).to_csv(output.parent / 'policy.csv', index=False)
    record = agent.artifact()
    record.update(detector=detector, request_sha256=p.sha(root / 'request.json'), training_base_seeds=config['base_seeds'], training_conditions=list(manifest['conditions']), files={f.name: p.sha(f) for f in [output.parent / 'episodes.csv', output.parent / 'transitions.parquet', output.parent / 'policy.csv']})
    p.write(output, record)
    return {'seed': seed, 'variant': variant, 'detector': detector, 'path': output.relative_to(p.ROOT).as_posix(), 'sha256': p.sha(output)}

def train(root, config, workers):
    """Train each detector/state/seed combination with identical budgets."""
    variants = config['rl'].get('variants', ['alarm'])
    detectors = config['detectors']
    if len(set(detectors)) != len(detectors) or not detectors:
        raise ValueError('Training detectors must be nonempty and unique')
    planned = len(detectors) * len(variants) * len(config['rl']['seeds'])
    models = []
    status = root / '04_rl_training/status.json'
    p.write(status, {'status': 'running', 'planned': planned, 'completed': 0})
    with ProcessPoolExecutor(max_workers=min(workers, planned)) as pool:
        futures = [pool.submit(train_one, root, config, seed, variant, detector)
                   for detector in detectors for variant in variants for seed in config['rl']['seeds']]
        for future in as_completed(futures):
            models.append(future.result())
            p.write(status, {'status': 'running', 'planned': planned, 'completed': len(models),
                             'models': models})
    models.sort(key=lambda m: (m['detector'], m['variant'], m['seed']))
    p.write(root / '04_rl_training/manifest.json', {'models': models, 'evaluation_updates': False,
            'detectors': detectors, 'models_per_detector': len(variants) * len(config['rl']['seeds'])})
    p.write(status, {'status': 'complete', 'planned': planned, 'completed': len(models)})
    return models
