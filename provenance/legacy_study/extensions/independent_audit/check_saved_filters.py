"""Reconcile saved filter runs directly from row traces, independently of report.py."""
from pathlib import Path
import json
import argparse
import pickle
import sys
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from common import protocol as p, channels


def close(actual, expected, name):
    if expected is None:
        assert actual is None or pd.isna(actual), name
    else:
        assert np.isclose(actual, expected, rtol=1e-10, atol=1e-12), (name, actual, expected)


def main(output=None):
    output = output or ROOT / 'results/independent_audit/2026-09-12'
    output.mkdir(parents=True, exist_ok=True)
    study = ROOT / 'results/filter_study'
    source = ROOT / 'results/evaluation'
    request = p.read(study / 'request.json'); settings = request['settings']
    input_record = request['inputs']['evaluation']
    metadata = source
    if p.sha(metadata / 'request.json') != input_record['request_sha256']:
        metadata = study / 'execution_inputs' / input_record['root']
    assert p.sha(metadata / 'request.json') == input_record['request_sha256']
    assert p.sha(metadata / '01_attacks/manifest.json') == input_record['manifest_sha256']
    config = p.read(metadata / 'request.json')['config']
    manifest = p.read(metadata / '01_attacks/manifest.json')
    snapshots = {f.relative_to(study).as_posix(): p.sha(f) for f in (study / 'runs').rglob('summary.json')}
    findings = {'status': 'running', 'started_utc': p.now(), 'snapshot_runs': len(snapshots),
                'planned_full_study': 1708, 'checked_runs': 0, 'checked_rows': 0, 'checks': 0,
                'scope': 'Only atomically saved runs at audit start; pending runs are not certified'}
    findings['runtime_metadata'] = str(metadata.relative_to(ROOT))
    p.write(output / 'filter_trace_audit.json', findings)
    try:
        for meta in manifest['conditions'].values():
            paths = [study / name for name in snapshots if name.startswith('runs/' + meta['key'] + '/')]
            if not paths:
                continue
            obs = channels.observations(source, meta); truth = channels.truth(source, meta)
            audit = channels.protected(source, meta)
            n, block = len(obs), config['block_size']
            ids = np.arange(n); reserved = obs.audit_reserved.to_numpy(bool)
            poisoned = (truth.atk_flip | truth.atk_burst).to_numpy(bool)
            start = config['warmup_rows']
            score = (ids >= start) & ~reserved & truth.host_eligible.to_numpy(bool)
            if meta['stream'] == 'radar':
                start = int(n * settings['radar_validation_end']) // block * block + settings['embargo_blocks'] * block
                original = truth.replay_source_row.to_numpy()
                source_ids = np.where(original >= 0, original, ids)
                score &= (ids >= start) & (source_ids >= start)
            feature_names = [name for name in obs if name.startswith('f') and name[1:].isdigit()]
            fingerprints = pd.util.hash_pandas_object(obs[feature_names], index=False).to_numpy()
            model_folder = study / 'models' / meta['stream']
            model_manifest = p.read(model_folder / 'manifest.json')
            excluded = np.load(model_folder / 'excluded_test_fingerprints.npy', allow_pickle=False)
            score &= ~np.isin(fingerprints, excluded)
            eligible = (ids >= start) & ~reserved
            oracle_rows = poisoned & eligible
            oracle_blocks = np.isin(ids // block, np.unique(ids[oracle_rows] // block)) & eligible
            masks = {'none': np.zeros(n, bool), 'oracle_rows': oracle_rows, 'oracle_blocks': oracle_blocks}
            for view in ['features', 'features_label']:
                file = model_folder / (view + '.pickle')
                assert p.sha(file) == model_manifest['files'][file.name]
                with file.open('rb') as handle:
                    bundle = pickle.load(handle)  # Local run-owned model, hash checked above.
                prediction = np.zeros(n, bool)
                eligible_ids = ids[eligible]
                with threadpool_limits(limits=1):
                    for left in range(0, len(eligible_ids), 50000):
                        take = eligible_ids[left:left+50000]
                        x = obs.loc[take, feature_names].to_numpy(float)
                        if view == 'features_label':
                            x = np.column_stack([x] + [obs.loc[take, 'label'].to_numpy() == c for c in bundle['classes']])
                        prediction[take] = bundle['model'].predict_proba(x)[:, 1] > bundle['threshold']
                masks['learned_' + view] = prediction
            for file in paths:
                record = p.read(file); folder = file.parent
                assert p.sha(file) == snapshots[file.relative_to(study).as_posix()]
                for name, digest in record['files'].items():
                    assert p.sha(folder / name) == digest, (file, name)
                pred = pd.read_parquet(folder / 'predictions.parquet')
                events = pd.read_parquet(folder / 'events.parquet')
                np.testing.assert_array_equal(pred.row_id, ids)
                np.testing.assert_array_equal(pred.audit_reserved, reserved)
                np.testing.assert_array_equal(pred.comparison_scored, score)
                assert record['activation_row'] == start
                assert record['host_rows'] == int(score.sum())
                rejected = pred.filter_withheld.to_numpy(bool)
                assert not rejected[reserved | (ids < config['warmup_rows'])].any()
                kind = record['filter']
                if kind in masks:
                    np.testing.assert_array_equal(rejected, masks[kind])
                elif kind.startswith('random_'):
                    target = masks[kind.replace('random_', 'oracle_')]
                    assert rejected.sum() == target.sum()
                    assert not rejected[~eligible].any()
                    if kind.endswith('blocks'):
                        expected_sizes = np.unique(ids[target] // block, return_counts=True)[1]
                        actual_sizes = np.unique(ids[rejected] // block, return_counts=True)[1]
                        np.testing.assert_array_equal(np.sort(expected_sizes), np.sort(actual_sizes))
                attempted = np.zeros(n, int); accepted = np.zeros(n, int); first = np.full(n, -1, int)
                np.testing.assert_array_equal(events.start_row, np.arange(0, n, block))
                np.testing.assert_array_equal(events.end_row, np.minimum(np.arange(0, n, block) + block, n))
                for event in events.itertuples():
                    source_blocks = json.loads(event.training_blocks)
                    assert all(0 <= b <= event.block for b in source_blocks)
                    trained = np.concatenate([ids[b*block:min((b+1)*block,n)] for b in source_blocks])
                    trained = trained[~reserved[trained] & ~rejected[trained]]
                    assert event.candidate_rows == len(trained)
                    attempted[trained] += 1
                    if event.accepted:
                        accepted[trained] += 1
                        first[trained[first[trained] < 0]] = event.block
                    assert not event.reset_committed or (event.reset_requested and event.accepted)
                    if event.scored:
                        a = audit[audit.row_id.between((event.block-1)*block, event.block*block-1)]
                        assert event.audit_n == len(a)
                        assert event.audit_latest_row == (int(a.row_id.max()) if len(a) else -1)
                        assert event.audit_latest_row < event.start_row
                        if len(a):
                            assert event.accepted == (event.candidate_accuracy >= event.active_accuracy)
                        current = ids[event.start_row:event.end_row]
                        assert event.monitor_rows == int((~reserved[current] & ~rejected[current]).sum())
                np.testing.assert_array_equal(attempted, pred.candidate_training_visits)
                np.testing.assert_array_equal(accepted, pred.accepted_training_visits)
                np.testing.assert_array_equal(first, pred.first_admitted_block)
                close(record['training_visits'], attempted.sum(), 'training visits')
                close(record['accepted_training_visits'], accepted.sum(), 'accepted training visits')
                close(record['resets'], int(events.reset_committed.sum()), 'resets')
                y, yh = truth.host_reference_label.to_numpy()[score], pred.prediction.to_numpy()[score]
                close(record['accuracy'], np.mean(y == yh) if len(y) else None, 'accuracy')
                f1 = []
                for c in meta['classes']:
                    tp = int(((y == c) & (yh == c)).sum()); denominator = int((y == c).sum() + (yh == c).sum())
                    f1.append(2 * tp / denominator if denominator else 0.)
                close(record['macro_f1'], np.mean(f1) if len(y) else None, 'macro F1')
                for key, population, selected in [('poison_admission_rate', eligible & poisoned, first >= 0),
                                                   ('legitimate_withhold_rate', eligible & ~poisoned, first < 0)]:
                    close(record[key], float((population & selected).sum()/population.sum()) if population.any() else None, key)
                probs = pred.loc[score, [f'p_{c}' for c in meta['classes']]].to_numpy()
                assert np.isfinite(probs).all() and (probs >= 0).all() and (probs <= 1).all()
                np.testing.assert_allclose(probs.sum(axis=1), 1, atol=1e-10)
                findings['checked_runs'] += 1; findings['checked_rows'] += n; findings['checks'] += 24
            p.write(output / 'filter_trace_audit.json', findings)
            print(f"Independently checked {findings['checked_runs']}/{len(snapshots)} saved filter runs", flush=True)
        findings.update(status='passed_for_snapshot', completed_utc=p.now(), summary_hashes=snapshots)
        p.write(output / 'filter_trace_audit.json', findings)
    except BaseException as error:
        findings.update(status='failed', error=f'{type(error).__name__}: {error}',
                        last_file=str(locals().get('file', '')))
        p.write(output / 'filter_trace_audit.json', findings)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    main(parser.parse_args().output)
