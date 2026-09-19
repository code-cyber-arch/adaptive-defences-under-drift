"""Reproducible offline-reference poisoning with evaluator-only truth.

Attacks replace rows or feedback labels. They do not insert rows. Full clean
reference access is an explicit adversary capability, including future rows.
Replay is across recorded positional segments, not guaranteed pure concepts.
"""
import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import generation as p
from common.validation import frame_check

def placements(n, level, intervals, centers, warmup, rng, burst=p.BURST):
    """Place non-overlapping attack bursts within the declared budget."""
    count = int(level * n) // burst
    if count < 1:
        raise ValueError('Stream too small for requested burst budget')
    reserved = np.zeros(n, dtype=bool)
    reserved[:warmup] = True
    for a, b in intervals:
        reserved[a:b] = True
    starts = []
    colocated = []
    for c in centers[:count]:
        s = max(warmup, min(c - burst // 2, n - burst))
        if s + burst > n:
            raise ValueError('Insufficient room for colocated burst')
        starts.append(s)
        colocated.append(True)
        reserved[s:s + burst] = True
    changes = np.diff(np.r_[False, ~reserved, False].astype(int))
    free = list(zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)))
    while len(starts) < count:
        viable = [(int(a), int(b)) for a, b in free if b - a >= burst]
        if not viable:
            raise ValueError('Burst budget cannot fit outside full transition intervals')
        weights = np.array([b - a - burst + 1 for a, b in viable], dtype=float)
        a, b = viable[int(rng.choice(len(viable), p=weights / weights.sum()))]
        s = int(rng.integers(a, b - burst + 1))
        free.remove((a, b))
        free.extend([(a, s), (s + burst, b)])
        starts.append(s)
        colocated.append(False)
    order = np.argsort(starts)
    return ([int(starts[i]) for i in order], [bool(colocated[i]) for i in order])

def flip_map(X, y, benign):
    """Choose the target class for each label-poisoning class."""
    sd = X.std(axis=0)
    sd[sd == 0] = 1
    means = {int(c): X[y == c].mean(axis=0) / sd for c in np.unique(y)}
    targets = {}
    for c in means:
        targets[c] = benign if benign is not None and c != benign else min((o for o in means if o != c), key=lambda o: np.linalg.norm(means[c] - means[o]))
    return targets

def make_attack(clean, meta, mode, level, seed, burst=p.BURST):
    """Construct one intervention and separate its hidden scoring references."""
    frame_check(clean)
    X0 = clean.iloc[:, :-1].to_numpy(dtype=float, copy=True)
    y0 = clean.label.to_numpy(copy=True)
    X = X0.copy()
    y = y0.copy()
    n = len(clean)
    rng_seed = p.stable_seed('attack', meta['key'], mode, level, seed)
    rng = np.random.default_rng(rng_seed)
    warmup = meta['warmup_rows']
    flips = np.zeros(n, dtype=bool)
    bursts = np.zeros(n, dtype=bool)
    ref = y0.copy()
    source_rows = np.full(n, -1, dtype=np.int64)
    starts = []
    colocated = []
    sources = []
    targets = {}
    if mode in ['concept', 'splice']:
        starts, colocated = placements(n, level, meta['transition_intervals'], meta['nominal_centers'], warmup, rng, burst)
        edges = [0] + meta['segment_boundaries'] + [n]
        for start in starts:
            end = start + burst
            bursts[start:end] = True
            if mode == 'concept':
                for j in range(X.shape[1]):
                    X[start:end, j] = rng.choice(X0[:, j], size=burst, replace=True)
                score = X[start:end] @ rng.normal(size=X.shape[1])
                classes = np.unique(y0)
                cuts = np.quantile(score, np.linspace(0, 1, len(classes) + 1)[1:-1])
                y[start:end] = classes[np.digitize(score, cuts)]
                ref[start:end] = -1
            else:
                dest = int(np.searchsorted(edges, start, side='right') - 1)
                candidates = [j for j in range(len(edges) - 1) if j != dest and edges[j + 1] - edges[j] >= burst]
                if not candidates:
                    raise ValueError('Replay needs another sufficiently long positional segment')
                j = int(rng.choice(candidates))
                src = int(rng.integers(edges[j], edges[j + 1] - burst + 1))
                X[start:end] = X0[src:src + burst]
                y[start:end] = y0[src:src + burst]
                ref[start:end] = y0[src:src + burst]
                source_rows[start:end] = np.arange(src, src + burst)
                sources.append({'dest': start, 'src': src, 'length': burst, 'source_segment': j, 'destination_segment': dest, 'future_reference': src > start})
    elif mode == 'instance':
        count = int(round(level * n))
        eligible = np.arange(warmup, n)
        if count > len(eligible):
            raise ValueError('Label budget exceeds post-warmup population')
        idx = rng.choice(eligible, size=count, replace=False)
        flips[idx] = True
        targets = flip_map(X0, y0, meta['benign_class'])
        y[idx] = np.array([targets[int(c)] for c in y0[idx]])
    elif mode != 'clean':
        raise ValueError(mode)
    result = pd.DataFrame(X, columns=clean.columns[:-1])
    result['label'] = y
    truth = pd.DataFrame({'row_id': np.arange(n), 'reference_label': ref, 'host_reference_label': y0, 'host_eligible': ~bursts, 'content_eligible': ref >= 0, 'atk_flip': flips, 'atk_burst': bursts, 'replay_source_row': source_rows})
    details = {'rng_seed': rng_seed, 'attack_seed': seed, 'mode': mode, 'level_value': level, 'burst_starts': starts, 'burst_len': burst, 'burst_colocated': colocated, 'splice_sources': sources, 'flip_target_map': targets, 'n_flipped': int(flips.sum()), 'burst_rows': int(bursts.sum()), 'actual_modified_fraction': float((flips | bursts).mean()), 'reference_access': 'offline labelled clean archive, including future rows; no detector/model access', 'warmup_attack_free_rows': warmup, 'replacement_not_insertion': True}
    frame_check(result)
    if not np.array_equal(result.to_numpy()[~(flips | bursts)], clean.to_numpy()[~(flips | bursts)]):
        raise AssertionError('Non-attacked data changed')
    if mode == 'splice':
        idx = np.flatnonzero(bursts)
        if not np.array_equal(X[idx], X0[source_rows[idx]]) or not np.array_equal(y[idx], y0[source_rows[idx]]):
            raise AssertionError('Replay does not equal immutable source')
    return (result, truth, details)
