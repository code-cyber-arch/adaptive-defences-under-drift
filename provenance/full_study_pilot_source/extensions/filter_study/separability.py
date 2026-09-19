"""Fit frozen poison discriminators and evaluate leakage-controlled holdouts."""
import pickle
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix
from threadpoolctl import threadpool_limits
from . import protocol as s
from common import protocol as p, channels


def collect(root, stream, role, config):
    manifest = p.read(root / '01_attacks/manifest.json')
    found = False
    for meta in manifest['conditions'].values():
        if meta['stream'] != stream:
            continue
        obs, truth = channels.observations(root, meta), channels.truth(root, meta)
        if not obs.row_id.equals(truth.row_id):
            raise ValueError('Truth is not aligned')
        part = obs[s.features(obs) + ['label']].copy()
        part['poisoned'] = (truth.atk_flip | truth.atk_burst).to_numpy()
        part['fingerprint'] = s.fingerprints(obs)
        part['condition'] = meta['key']
        part['row_id'] = obs.row_id.to_numpy()
        part['mode'], part['level'] = meta['mode'], meta['level']
        if stream == 'radar':
            split, _ = s.temporal_partitions(obs.row_id.to_numpy(),
                truth.replay_source_row.to_numpy(), len(obs), config)
        else:
            split = np.full(len(obs), {'training': 0, 'validation': 1, 'evaluation': 2}[role])
        valid = ~obs.audit_reserved & (obs.row_id >= config['warmup_rows']) & (split >= 0)
        part['split'] = split
        found = True
        yield part.loc[valid]
    if not found:
        raise ValueError(f'No {role} data for {stream}')


def subsample(frame, count, seed):
    return frame.sample(n=min(count, len(frame)), random_state=seed).reset_index(drop=True)


def binary_scores(target, scores, threshold):
    target = np.asarray(target, bool)
    predicted = np.asarray(scores) > threshold
    tn, fp, fn, tp = confusion_matrix(target, predicted, labels=[False, True]).ravel()
    ratio = lambda a, b: float(a / b) if b else None
    return {'rows': len(target), 'poison_rows': int(target.sum()),
            'prevalence': float(target.mean()) if len(target) else None,
            'roc_auc': float(roc_auc_score(target, scores)) if len(np.unique(target)) == 2 else None,
            'average_precision': float(average_precision_score(target, scores)) if len(np.unique(target)) == 2 else None,
            'poison_recall': ratio(tp, tp + fn), 'poison_precision': ratio(tp, tp + fp),
            'legitimate_rejection': ratio(fp, fp + tn),
            'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)}


def fit_stream(out, roots, stream, config, settings):
    folder = out / 'models' / stream
    manifest_file = folder / 'manifest.json'
    if manifest_file.exists():
        record = p.read(manifest_file)
        s.verify_files(folder, record)
        return record
    folder.mkdir(parents=True, exist_ok=True)
    source_roles = ['evaluation'] if stream == 'radar' else ['training', 'validation']
    candidates = {0: [], 1: []}
    hashes = {0: [], 1: []}
    classes = set()
    feature_names = None
    counts = {0: 0, 1: 0}
    for role in source_roles:
        n_conditions = sum(c['stream'] == stream for c in
                           p.read(roots[role] / '01_attacks/manifest.json')['conditions'].values())
        for part in collect(roots[role], stream, role, config):
            classes.update(part.label.unique().tolist())
            feature_names = s.features(part)
            for split, limit in [(0, settings['max_fit_rows']), (1, settings['max_calibration_rows'])]:
                selected = part[part.split.eq(split)]
                counts[split] += len(selected)
                hashes[split].append(selected.fingerprint.to_numpy())
                if len(selected):
                    candidates[split].append(subsample(selected, max(1, limit // n_conditions),
                                                       p.seed(settings['seed'], role, part.condition.iloc[0], split)))
    train_hashes = np.unique(np.concatenate(hashes[0]))
    validation_hashes = np.setdiff1d(np.unique(np.concatenate(hashes[1])), train_hashes)
    excluded = s.exclusion_hashes(train_hashes, validation_hashes)
    train = pd.concat(candidates[0], ignore_index=True)
    calibration = pd.concat(candidates[1], ignore_index=True)
    validation_before = len(calibration)
    calibration = calibration[~calibration.fingerprint.isin(train_hashes)]
    if any(part.poisoned.nunique() < 2 for part in (train, calibration)):
        raise ValueError(f'{stream}: both target classes required after temporal/copy exclusion')
    classes = sorted(classes)
    train[['condition', 'row_id', 'fingerprint']].to_parquet(folder / 'fit_rows.parquet', index=False)
    calibration[['condition', 'row_id', 'fingerprint']].to_parquet(folder / 'calibration_rows.parquet', index=False)
    np.save(folder / 'excluded_test_fingerprints.npy', excluded, allow_pickle=False)
    records, models, bundles = [], {}, {}
    for view in s.VIEWS:
        model = HistGradientBoostingClassifier(random_state=settings['seed'], **settings['classifier'])
        tick = time.perf_counter()
        with threadpool_limits(limits=1):
            model.fit(s.inputs(train, view, classes), train.poisoned)
            calibration_scores = model.predict_proba(s.inputs(calibration, view, classes))[:, 1]
            threshold = s.threshold_from_validation(calibration_scores, calibration.poisoned,
                                                    settings['legitimate_rejection_target'])
        bundle = {'model': model, 'view': view, 'classes': classes, 'threshold': threshold,
                  'feature_names': feature_names, 'training_seconds': time.perf_counter() - tick}
        with (folder / f'{view}.pickle').open('wb') as handle:
            pickle.dump(bundle, handle)
        bundles[view] = bundle
        models[view] = {k: v for k, v in bundle.items() if k != 'model'}
    heldout_rows = removed = 0
    for part in collect(roots['evaluation'], stream, 'evaluation', config):
        test = part[part.split.eq(2)]
        before = len(test)
        test = test[~test.fingerprint.isin(excluded)]
        removed += before - len(test)
        heldout_rows += len(test)
        condition = part.condition.iloc[0]
        target_folder = folder / 'heldout' / condition
        target_folder.mkdir(parents=True, exist_ok=True)
        for view, bundle in bundles.items():
            with threadpool_limits(limits=1):
                probabilities = np.concatenate([bundle['model'].predict_proba(s.inputs(test.iloc[i:i+50000], view, classes))[:, 1]
                    for i in range(0, len(test), 50000)]) if len(test) else np.array([])
            tested = test[['condition', 'row_id', 'mode', 'level', 'poisoned', 'fingerprint']].copy()
            tested['score'] = probabilities
            tested['rejected'] = probabilities > bundle['threshold']
            tested.to_parquet(target_folder / f'{view}.parquet', index=False)
            records.append({'stream': stream, 'condition': condition, 'mode': part['mode'].iloc[0],
                            'level': part.level.iloc[0], 'view': view, 'threshold': bundle['threshold'],
                            **binary_scores(tested.poisoned, probabilities, bundle['threshold'])})
    pd.DataFrame(records).to_csv(folder / 'separability.csv', index=False)
    record = {'stream': stream, 'fit_rows': len(train), 'calibration_rows': len(calibration),
              'heldout_rows': heldout_rows, 'validation_duplicate_sample_rows_removed': validation_before - len(calibration),
              'test_duplicate_rows_removed': removed, 'models': models,
              'files': {f.relative_to(folder).as_posix(): p.sha(f) for f in folder.rglob('*')
                        if f.is_file() and f.name != 'manifest.json'}}
    p.write(manifest_file, record)
    return record


def load_bundle(out, stream, view):
    folder = out / 'models' / stream
    record = p.read(folder / 'manifest.json')
    file = folder / f'{view}.pickle'
    if p.sha(file) != record['files'][file.name]:
        raise ValueError('Frozen filter changed')
    # Only locally generated, hash-verified models from this run are loaded.
    with file.open('rb') as handle:
        return pickle.load(handle)
