"""Matched frozen-policy execution and controlled feedback availability."""
from copy import deepcopy
from itertools import product
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from threadpoolctl import threadpool_limits
from common import channels, protocol as p
from common.replication import run_contract_matches
engine = p.load_module('policy_robustness_engine', '05_experiment/engine.py')
Controller = p.load_module('policy_robustness_controller', '04_rl_training/controller.py').Controller
STATES = ('00', '01', '10', '11')


def policy_table(bits):
    if len(bits) != 4 or set(bits) - {'0', '1'}:
        raise ValueError('A four-state policy requires four binary actions')
    return {state: ([1., 0.] if bit == '0' else [0., 1.])
            for state, bit in zip(STATES, bits)}


def enumerated_arms(detector="adwin", dataset="SEA_A"):
    return [{'name': 'map_' + ''.join(bits), 'policy': 'rl', 'detector': detector, 'dataset': dataset,
             'table': policy_table(''.join(bits)), 'variant': 'alarm_persistence'}
            for bits in product('01', repeat=4)]


def comparison_arms(selected, models, detector, dataset):
    arms = [{'name': 'no_reset', 'policy': 'none'},
            {'name': 'confirmed', 'policy': 'confirmed'},
            {'name': 'error_rule', 'policy': 'rl', 'variant': 'alarm_persistence',
             'table': policy_table('0101')},
            {'name': 'selected_map', 'policy': 'rl', 'variant': 'alarm_persistence',
             'table': selected['table']}]
    for model in models:
        if model['detector'] != detector or model['dataset'] != dataset:
            continue
        arms.append({'name': f"q_{model['variant']}_{model['seed']}", 'policy': 'rl',
                     'variant': model['variant'], 'seed': model['seed'],
                     'model_path': model['path'], 'model_sha256': model['sha256']})
    for arm in arms:
        arm['detector'] = detector
        arm['dataset'] = dataset
    return arms


def load_frozen(arm):
    path = p.ROOT / arm['model_path']
    if p.sha(path) != arm['model_sha256']:
        raise ValueError('Frozen model changed')
    record = p.read(path)
    if record.get('dataset') != arm['dataset']:
        raise ValueError('Frozen controller dataset does not match evaluation dataset')
    if record.get('detector') != arm['detector']:
        raise ValueError('Frozen controller detector does not match evaluation detector')
    if record['variant'] != arm['variant'] or record['seed'] != arm['seed']:
        raise ValueError('Frozen controller state representation or seed differs')
    return record


def feedback_channel(audits, rows, block_size, base_key, fraction, delay):
    """Keep reservation fixed; nested, label-independent subsets become available."""
    if fraction not in (.01, .02, .05) or delay not in (1, 5):
        raise ValueError('Feedback setting is outside the frozen protocol')
    out = audits.copy()
    out['release_block'] = (rows + block_size - 1) // block_size + delay + 1
    for block, group in audits.groupby(audits.row_id // block_size):
        count = min(len(group), max(1, round(min(block_size, rows - block * block_size) * fraction)))
        rng = np.random.default_rng(p.seed('policy_robustness-feedback', base_key, int(block)))
        chosen = rng.permutation(group.sort_values('row_id').index)[:count]
        out.loc[chosen, 'release_block'] = int(block) + delay
    return out


def metrics(pred, truth, classes, warmup, benign_class=None):
    if not pred.row_id.equals(truth.row_id):
        raise AssertionError('Prediction and evaluator row identities differ')
    eligible = pred.scored & truth.host_eligible
    y = truth.loc[eligible, 'host_reference_label'].to_numpy(dtype=int)
    yh = pred.loc[eligible, 'prediction'].to_numpy(dtype=int)
    train = (truth.row_id >= warmup) & ~truth.audit_reserved
    poisoned = (truth.atk_flip | truth.atk_burst) & train
    clean = ~(truth.atk_flip | truth.atk_burst) & train
    admitted = pred.first_admitted_block >= 0
    ratio = lambda a, b: float(a / b) if b else None
    result = {'host_n': int(eligible.sum()),
            'host_accuracy': float(np.mean(y == yh)),
            'host_macro_f1': float(f1_score(y, yh, labels=classes, average='macro', zero_division=0)),
            'poison_rows': int(poisoned.sum()),
            'poison_rows_admitted': int((poisoned & admitted).sum()),
            'poison_admission_rate': ratio((poisoned & admitted).sum(), poisoned.sum()),
            'clean_rows': int(clean.sum()),
            'clean_rows_withheld': int((clean & ~admitted).sum()),
            'clean_withhold_rate': ratio((clean & ~admitted).sum(), clean.sum())}
    result.update(malicious_recall=None, benign_fpr=None)
    if benign_class is not None:
        positive = y != benign_class
        predicted_positive = (yh >= 0) & (yh != benign_class)
        result['malicious_recall'] = ratio((positive & predicted_positive).sum(), positive.sum())
        result['benign_fpr'] = ratio((~positive & predicted_positive).sum(), (~positive).sum())
    return result


def run_one(job):
    source, output, condition, arm, config, fraction, delay = job
    source, output = Path(source), Path(output)
    if arm['dataset'] != condition['stream'] or config['dataset'] != condition['stream']:
        raise ValueError('Arm, input and configuration datasets differ')
    folder = output / arm['dataset'] / arm['detector'] / condition['key'] / arm['name'] / f'feedback_{fraction:.2f}_delay_{delay}'
    summary_file = folder / 'summary.json'
    contract = {'condition': condition['key'], 'arm': arm, 'config': config,
                'feedback_fraction': fraction, 'delay_blocks': delay,
                'inputs': {name: p.sha(source / '01_attacks' / condition[name])
                           for name in ['observations', 'audit', 'truth']}}
    if summary_file.exists():
        saved = p.read(summary_file)
        if not run_contract_matches(saved['contract'], contract):
            raise ValueError(f'Run contract changed: {folder}')
        for name, digest in saved['files'].items():
            if p.sha(folder / name) != digest:
                raise ValueError(f'Run artifact changed: {folder / name}')
        return saved
    obs = channels.observations(source, condition)
    audit = feedback_channel(channels.protected(source, condition), len(obs), config['block_size'],
                             condition['base_key'], fraction, delay)
    local = deepcopy(config)
    local['audit']['delay_blocks'] = delay
    local['detector'] = {'name': arm['detector'], 'seed': p.seed('monitor', condition['key'], arm['detector'])}
    agent = None
    if p.is_rl(arm['policy']):
        if 'model_path' in arm:
            record = load_frozen(arm)
            settings, table, seed = record['settings'], record['table'], record['seed']
        else:
            settings = dict(config['rl'], variant=arm['variant'])
            table, seed = arm['table'], 0
        agent = Controller(settings, seed, table=table, collect_rewards=False)
        original = deepcopy(agent.table)
    with threadpool_limits(limits=1):
        pred, events, stats = engine.execute(obs, audit, condition['classes'], arm['policy'], True, local, agent)
    if agent is not None and (agent.table != original or agent.pending is not None or agent.transitions):
        raise AssertionError('Frozen evaluation recorded or learned rewards')
    # The evaluator is loaded only after all model decisions have completed.
    truth = channels.truth(source, condition)
    stats.update(metrics(pred, truth, condition['classes'], config['warmup_rows'], condition.get('benign_class')))
    available = events.audit_n.gt(0)
    if not (events.loc[available, 'audit_latest_row'] // config['block_size'] <=
            events.loc[available, 'block'] - delay).all():
        raise AssertionError('Feedback released too early')
    stats.update(condition=condition['key'], stream=condition['stream'], mode=condition['mode'],
                 level=condition['level'], base_seed=condition['base_seed'], arm=arm['name'], detector=arm['detector'],
                 feedback_fraction=fraction, delay_blocks=delay,
                 no_feedback_blocks=int((events.scored & events.reason.eq('no_protected_sample')).sum()),
                 state_visits={} if agent is None else agent.visits,
                 action_counts=events.loc[events.scored, 'rl_action'].value_counts().to_dict(),
                 contract=contract, status='complete')
    folder.mkdir(parents=True, exist_ok=True)
    pred.to_parquet(folder / 'predictions.parquet', index=False)
    events.to_parquet(folder / 'events.parquet', index=False)
    audit[['row_id', 'release_block']].to_parquet(folder / 'feedback_schedule.parquet', index=False)
    stats['files'] = {f.name: p.sha(f) for f in folder.glob('*.parquet')}
    p.write(summary_file, stats)
    return stats
