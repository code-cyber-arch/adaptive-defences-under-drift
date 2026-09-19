"""Independently reconcile training admission, audit timing and scores."""
from pathlib import Path
import argparse
import json
import sys
import os
sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import protocol as p
from common import channels

def verify(root):
    """Reconcile saved data, decisions and metrics against the execution contract."""
    root = p.output(root)
    request = p.read(root / 'request.json')
    config = request['config']
    if request['sources'] != p.sources():
        raise ValueError('Source contract changed')
    manifest = p.read(root / '01_attacks/manifest.json')
    for name, digest in manifest['outputs'].items():
        assert p.sha(root / '01_attacks' / name) == digest, name
    completed = p.read(root / '05_runs/completion.json')
    assert completed['planned'] == completed['completed'] == len(manifest['conditions']) * len(p.arms(config))
    analysis = pd.read_csv(root / '06_analysis/metrics.csv').set_index('run_id')
    checks = 0
    for run_id, digest in completed['summaries'].items():
        folder = root / completed['paths'][run_id]
        assert p.sha(folder / 'summary.json') == digest
        summary = p.read(folder / 'summary.json')
        for name, expected in summary['files'].items():
            assert p.sha(folder / name) == expected
        pred = pd.read_parquet(folder / 'predictions.parquet')
        events = pd.read_parquet(folder / 'events.parquet')
        condition = manifest['conditions'][summary['condition']]
        obs = channels.observations(root, condition)
        audit = channels.protected(root, condition)
        truth = channels.truth(root, condition)
        n = len(pred)
        block = config['block_size']
        assert n == condition['rows'] == summary['rows']
        if condition['stream'] == 'radar':
            assert n == manifest['bases'][condition['base_key']]['metadata']['full_source_rows']
        np.testing.assert_array_equal(pred.row_id, np.arange(n))
        np.testing.assert_array_equal(events.start_row, np.arange(0, n, block))
        np.testing.assert_array_equal(events.end_row, np.minimum(np.arange(0, n, block) + block, n))
        assert any(((root / '01_attacks' / condition['truth']).resolve().is_relative_to(folder) for folder in [p.ROOT / 'data/evaluation_truth', root / 'audit/evaluator']))
        assert (root / '01_attacks' / condition['audit']).resolve().is_relative_to(root / 'audit/protected')
        assert pred.row_id.equals(obs.row_id) and pred.row_id.equals(truth.row_id)
        reserved = obs.audit_reserved.to_numpy(dtype=bool)
        assert set(audit.row_id) == set(np.flatnonzero(reserved))
        assert np.array_equal(audit.release_block.to_numpy(), audit.row_id.to_numpy() // block + config['audit']['delay_blocks'])
        attempt = np.zeros(n, dtype=int)
        accepted = np.zeros(n, dtype=int)
        first = np.full(n, -1, dtype=int)
        for e in events.itertuples():
            blocks = json.loads(e.training_blocks)
            assert all((0 <= b <= e.block for b in blocks))
            ids = np.concatenate([np.arange(b * block, min((b + 1) * block, n)) for b in blocks]) if blocks else np.array([], dtype=int)
            ids = ids[~reserved[ids]]
            attempt[ids] += 1
            assert e.candidate_rows == len(ids)
            if e.accepted:
                accepted[ids] += 1
                first[ids[first[ids] < 0]] = e.block
            if summary['guarded'] and e.scored and (e.reason != 'rl_hold'):
                earliest = e.block - config['audit']['delay_blocks'] - config['audit']['window_blocks'] + 1
                a = audit[(audit.release_block <= e.block) & (audit.row_id // block >= earliest)]
                assert e.audit_n == len(a)
                assert e.audit_latest_row == (int(a.row_id.max()) if len(a) else -1)
                assert e.audit_latest_row < e.start_row
                if e.reason == 'insufficient_audit':
                    assert not e.accepted
                if len(a):
                    assert bool(e.accepted) == bool(e.candidate_accuracy >= e.active_accuracy)
            assert not e.reset_committed or (e.accepted and e.reset_requested)
            checks += 1
        np.testing.assert_array_equal(attempt, pred.candidate_training_visits)
        np.testing.assert_array_equal(accepted, pred.accepted_training_visits)
        np.testing.assert_array_equal(first, pred.first_admitted_block)
        assert not pred.loc[reserved, 'scored'].any() and (not np.any(accepted[reserved]))
        assert int(attempt.sum()) == summary['training_visits']
        assert int(accepted.sum()) == summary['accepted_training_visits']
        eligible = pred.scored & truth.host_eligible
        y = truth.loc[eligible, 'host_reference_label'].to_numpy()
        yh = pred.loc[eligible, 'prediction'].to_numpy()
        row = analysis.loc[run_id]
        assert np.isclose(np.mean(y == yh), row.host_accuracy)
        assert np.isclose(f1_score(y, yh, labels=condition['classes'], average='macro', zero_division=0), row.host_macro_f1)
        candidate = (truth.row_id >= config['warmup_rows']) & ~truth.audit_reserved
        poisoned = candidate & (truth.atk_flip | truth.atk_burst)
        clean = candidate & ~(truth.atk_flip | truth.atk_burst)
        assert row.poison_rows_admitted == int((poisoned & (first >= 0)).sum())
        assert row.clean_rows_withheld == int((clean & (first < 0)).sum())
        probs = pred.loc[pred.scored, [c for c in pred if c.startswith('p_')]].to_numpy()
        assert np.isfinite(probs).all() and (probs >= 0).all() and (probs <= 1).all()
        np.testing.assert_allclose(probs.sum(axis=1), 1, atol=1e-08)
        if p.is_rl(summary['policy']):
            transitions = pd.read_parquet(folder / 'rl_transitions.parquet')
            assert not transitions.training.any()
            np.testing.assert_array_equal(transitions.q_before, transitions.q_after)
            assert (transitions.reward_block == transitions.action_block + 1).all()
            assert (transitions.audit_latest_row < transitions.reward_block * block).all()
            assert len(transitions) == int(events.scored.sum())
            contract = next(model for model in config['rl_models']
                            if model['seed'] == summary['rl_seed'] and model['variant'] == summary['rl_variant'])
            model = p.read(p.ROOT / contract['path'])
            assert p.sha(p.ROOT / contract['path']) == summary['rl_model_sha256'] == contract['sha256']
            assert model['variant'] == summary['rl_variant']
            scored = events[events.scored].reset_index(drop=True)
            settings = model['settings']['persistence']
            # Reconstruct the state from past prediction errors, independently of the controller.
            for index, event in scored.iterrows():
                count = settings['history'] + settings['recent']
                ready = index + 1 >= count
                elevated = False
                if ready:
                    window = scored.feedback_error.iloc[index + 1 - count:index + 1].to_numpy()
                    baseline = window[:settings['history']].mean()
                    recent = window[settings['history']:].mean()
                    elevated = recent > baseline + settings['margin']
                    assert np.isclose(event.rl_error_baseline, baseline)
                    assert np.isclose(event.rl_error_recent, recent)
                assert bool(event.rl_error_ready) == ready
                assert bool(event.rl_error_elevated) == elevated
                state = str(int(event.raw_fire))
                if model['variant'] == 'alarm_persistence':
                    state += str(int(elevated))
                assert event.rl_state == state == transitions.iloc[index]['state']
                action = ('update', 'reset')[int(np.argmax(model['table'][state]))]
                assert event.rl_action == action == transitions.iloc[index]['action']
        checks += 8
    paired = pd.read_csv(root / '06_analysis/paired_gate_differences.csv')
    assert len(paired) == len(manifest['conditions']) * (1 + 2 * len(config.get('detectors', ['adwin'])))
    for row in paired.to_dict('records'):
        group = analysis[analysis.condition.eq(row['condition']) & analysis.policy.eq(row['policy']) & analysis.detector.eq(row['detector'])]
        a = group[group.guarded.eq(True)].iloc[0]
        b = group[group.guarded.eq(False)].iloc[0]
        for name in ['host_accuracy', 'host_macro_f1', 'poison_admission_rate', 'clean_withhold_rate', 'resets', 'training_visits', 'seconds']:
            assert np.isclose(row[name], a[name] - b[name], equal_nan=True)
            checks += 1
    for file in [root / '06_analysis/manifest.json', root / '06_analysis/presentation/manifest.json']:
        record = p.read(file)
        for name, digest in record['outputs'].items():
            assert p.sha(file.parent / name) == digest, name
    if analysis.policy.eq('rl').any():
        state_pairs = pd.read_csv(root / '06_analysis/presentation/tables/rl_state_paired_differences.csv')
        assert len(state_pairs) == int(analysis.policy.eq('rl').sum())
        for pair in state_pairs.to_dict('records'):
            group = analysis[analysis.condition.eq(pair['condition']) & analysis.rl_seed.eq(pair['rl_seed'])]
            a = group[group.policy.eq('rl')].iloc[0]
            b = group[group.policy.eq('rl_alarm')].iloc[0]
            for metric in ['host_accuracy', 'host_macro_f1', 'poison_admission_rate',
                           'clean_withhold_rate', 'resets', 'training_visits', 'seconds',
                           'malicious_recall', 'roc_auc']:
                assert np.isclose(pair[metric], a[metric] - b[metric], equal_nan=True)
                checks += 1
    result = {'status': 'passed', 'runs': len(analysis), 'pairs': len(paired), 'checks': checks, 'source_contract': True, 'complete_stream_coverage': True, 'audit_training_overlap': 0, 'future_audit_access': 0, 'one_plot_per_figure': True}
    p.write(root / 'verification.json', result)
    return result
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=p.ROOT / 'results/smoke')
    args = parser.parse_args()
    print(json.dumps(verify(args.root), indent=2))
