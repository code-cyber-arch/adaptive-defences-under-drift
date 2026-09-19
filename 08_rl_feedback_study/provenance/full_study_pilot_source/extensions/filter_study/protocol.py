"""Frozen definitions for the supplementary upstream-filter experiment."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from common import protocol as p

VERSION = 'upstream-filter-1.0'
VIEWS = ('features', 'features_label')
CONTROLS = ('none', 'oracle_rows', 'random_rows', 'oracle_blocks', 'random_blocks')
DEFAULTS = {
    'version': VERSION, 'seed': 901, 'max_fit_rows': 60000,
    'max_calibration_rows': 60000, 'legitimate_rejection_target': 0.05,
    'radar_train_fraction': 0.5, 'radar_validation_end': 0.7,
    'embargo_blocks': 1, 'detectors': ['adwin', 'hddm_w', 'hellinger', 'd3_oof'],
    'response': 'confirmed', 'screened': True,
    'classifier': {'max_iter': 100, 'max_leaf_nodes': 15, 'learning_rate': 0.1,
                   'l2_regularization': 1.0, 'early_stopping': False},
}


def features(frame):
    return [c for c in frame if c.startswith('f') and c[1:].isdigit()]


def inputs(frame, view, classes):
    """Only declared observables reach the discriminator."""
    if view not in VIEWS:
        raise ValueError('Unknown observation view')
    x = frame[features(frame)].to_numpy(dtype=float)
    if view == 'features_label':
        x = np.column_stack([x] + [frame.label.to_numpy() == c for c in classes])
    if not np.isfinite(x).all():
        raise ValueError('Nonfinite filter input')
    return x


def fingerprints(frame):
    """Conservatively group exact feature copies, even if labels were changed."""
    return pd.util.hash_pandas_object(frame[features(frame)], index=False).to_numpy()


def temporal_partitions(row_ids, source_ids, rows, config):
    """Splice source and destination must belong to the same temporal partition."""
    block = config['block_size']
    a = int(rows * DEFAULTS['radar_train_fraction']) // block * block
    b = int(rows * DEFAULTS['radar_validation_end']) // block * block
    gap = DEFAULTS['embargo_blocks'] * block
    def assign(ids):
        return np.select([(ids >= config['warmup_rows']) & (ids < a),
                          (ids >= a + gap) & (ids < b), ids >= b + gap],
                         [0, 1, 2], default=-1)
    dest = assign(np.asarray(row_ids))
    origin = np.where(np.asarray(source_ids) >= 0, source_ids, row_ids)
    return np.where(dest == assign(origin), dest, -1), b + gap


def exclusion_hashes(train_hashes, validation_hashes):
    return np.unique(np.concatenate([train_hashes, validation_hashes]))


def threshold_from_validation(scores, poisoned, target):
    """Calibrate once against legitimate validation observations; ties are kept."""
    legitimate = np.asarray(scores)[~np.asarray(poisoned, dtype=bool)]
    if not len(legitimate):
        raise ValueError('No legitimate calibration observations')
    return float(np.quantile(legitimate, 1 - target, method='higher'))


def withholding(kind, poisoned, reserved, block_size, start_row, seed):
    """Ideal-information masks are constructed outside the operational engine."""
    poisoned, reserved = np.asarray(poisoned, bool), np.asarray(reserved, bool)
    eligible = (np.arange(len(poisoned)) >= start_row) & ~reserved
    rng = np.random.default_rng(seed)
    mask = np.zeros(len(poisoned), bool)
    if kind == 'none':
        return mask
    if kind == 'oracle_rows':
        return poisoned & eligible
    if kind == 'random_rows':
        candidates = np.flatnonzero(eligible)
        count = int((poisoned & eligible).sum())
        mask[rng.choice(candidates, count, replace=False)] = True
        return mask
    if kind not in ('oracle_blocks', 'random_blocks'):
        raise ValueError('Unknown ideal-information control')
    blocks = np.arange(len(poisoned)) // block_size
    exposed = np.unique(blocks[poisoned & eligible])
    if kind == 'oracle_blocks':
        return np.isin(blocks, exposed) & eligible
    # Match both number of blocks and eligible row count, including the tail.
    eligible_blocks, sizes = np.unique(blocks[eligible], return_counts=True)
    selected = []
    for size in np.unique(sizes):
        candidates = eligible_blocks[sizes == size]
        count = int(np.isin(candidates, exposed).sum())
        selected.extend(rng.choice(candidates, count, replace=False))
    return np.isin(blocks, selected) & eligible


def verify_files(folder, record):
    for name, digest in record['files'].items():
        if p.sha(folder / name) != digest:
            raise ValueError(f'Changed artifact: {folder / name}')
