"""Build numeric streams with feature lineage and separate temporal references."""
import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from river.datasets import synth
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import generation as p
from common.validation import annotation, frame_check, lineage_check

def synthetic(name, rows, seed):
    """Generate a reproducible SEA or RBF stream with scheduled transitions."""
    if rows < 100:
        raise ValueError('At least 100 synthetic rows required')
    variants = [0, 2, 1, 3] if name == 'SEA_A' else [0, 1, 2, 3]
    width = max(2, int(rows * (0.006 if name == 'SEA_A' else 0.2)))
    centers = [rows * i // 4 for i in [1, 2, 3]]
    intervals = [[c - width // 2, c - width // 2 + width] for c in centers]
    generators = []
    for v in variants:
        s = p.stable_seed('base', name, seed, v)
        g = synth.SEA(variant=v, noise=0.05, seed=s) if name == 'SEA_A' else synth.RandomRBF(seed_model=p.stable_seed('centroids', seed, v), seed_sample=s, n_classes=4, n_features=10, n_centroids=50)
        generators.append(iter(g))
    rng = np.random.default_rng(p.stable_seed('mix', name, seed))
    records = []
    components = np.zeros(rows, dtype=np.int8)
    transitions = np.zeros(rows, dtype=bool)
    for t in range(rows):
        component = 0
        for j, (a, b) in enumerate(intervals):
            if t >= b:
                component = j + 1
            elif t >= a:
                transitions[t] = True
                probability = (t - a) / (b - a)
                component = j + int(rng.random() < probability)
                break
            else:
                break
        x, y = next(generators[component])
        components[t] = component
        records.append({**{f'f{i}': v for i, v in enumerate(x.values())}, 'label': int(y)})
    df = pd.DataFrame(records)
    truth = pd.DataFrame({'row_id': np.arange(rows), 'component': components, 'in_transition': transitions})
    meta = {'stream': name, 'base_seed': seed, 'rows': rows, 'n_features': len(df.columns) - 1, 'classes': sorted((int(v) for v in df.label.unique())), 'benign_class': None, 'kind': 'rapid_linear_mixture' if name == 'SEA_A' else 'broad_linear_mixture', 'transition_intervals': intervals, 'nominal_centers': centers, 'width': width, 'event_reference': 'global-index scheduled mixture intervals', 'supports_event_metrics': True, 'segment_boundaries': centers, 'replay_segment_reference': 'nominal mixture centers; segments need not be pure concepts', 'feature_lineage': {f'f{i}': {'source_column': f'generator_feature_{i}', 'source': 'River ' + ('SEA' if name == 'SEA_A' else 'RandomRBF')} for i in range(len(df.columns) - 1)}, 'generation': 'Adjacent fixed components mixed with linear probability using global zero-based row index; no nested local clocks', 'preprocessing': 'none', 'warmup_rows': min(p.WARMUP, rows)}
    frame_check(df)
    return (df, truth, meta)
