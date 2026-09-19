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

def train_one(root, config, seed, variant='alarm'):
    """Fit one declared state representation using training observations only."""
    output = root / '04_rl_training' / variant / f'seed_{seed}' / 'q_table.json'
    manifest = p.read(root / '01_attacks/manifest.json')
    for c in manifest['conditions'].values():
        if c['base_seed'] not in config['base_seeds'] or c['stream'] == 'radar':
            raise ValueError('RL training contains evaluation data')
    if output.exists():
        saved = p.read(output)
        if saved['request_sha256'] != p.sha(root / 'request.json'):
            raise ValueError('RL training contract changed')
        if saved['variant'] != variant:
            raise ValueError('Saved controller uses a different state representation')
        for name, digest in saved['files'].items():
            if p.sha(output.parent / name) != digest:
                raise ValueError('RL training artifact changed')
        return {'seed': seed, 'variant': variant, 'path': output.relative_to(p.ROOT).as_posix(), 'sha256': p.sha(output)}
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
                local['detector'] = {'name': 'adwin', 'seed': p.seed('monitor', c['key'], 'adwin')}
                _, events, stats = engine.execute(obs, audit, c['classes'], 'rl', True, local, agent)
                new = agent.transitions[start:]
                transitions.extend((dict(t, epoch=epoch, condition=c['key']) for t in new))
                rows.append({'epoch': epoch, 'condition': c['key'], 'seed': seed, 'reward': sum((t['reward'] for t in new)), 'training_visits': stats['training_visits'], 'resets': stats['resets'], 'states': len(agent.table), 'epsilon': agent.epsilon})
                print(f"RL {variant}, seed {seed}: epoch {epoch + 1}/{config['rl']['epochs']}, {c['key']}, states {len(agent.table)}", flush=True)
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
    record.update(request_sha256=p.sha(root / 'request.json'), training_base_seeds=config['base_seeds'], training_conditions=list(manifest['conditions']), files={f.name: p.sha(f) for f in [output.parent / 'episodes.csv', output.parent / 'transitions.parquet', output.parent / 'policy.csv']})
    p.write(output, record)
    return {'seed': seed, 'variant': variant, 'path': output.relative_to(p.ROOT).as_posix(), 'sha256': p.sha(output)}

def train(root, config, workers):
    """Train the declared RL seeds and record their frozen model files."""
    variants = config['rl'].get('variants', ['alarm'])
    with ProcessPoolExecutor(max_workers=min(workers, len(config['rl']['seeds']) * len(variants))) as pool:
        futures = [pool.submit(train_one, root, config, seed, variant) for variant in variants for seed in config['rl']['seeds']]
        models = [f.result() for f in as_completed(futures)]
    models.sort(key=lambda m: (m['variant'], m['seed']))
    p.write(root / '04_rl_training/manifest.json', {'models': models, 'evaluation_updates': False})
    return models
