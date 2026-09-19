"""Check two complete archived Mac filter-study reference trajectories."""
from pathlib import Path
import sys
from copy import deepcopy
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from common import protocol as p, channels

engine = p.load_module('reproduce_saved_filter', 'extensions/filter_study/engine.py')
inputs = ROOT / 'results/filter_study/execution_inputs/results/evaluation'
manifest = p.read(inputs / '01_attacks/manifest.json')
records = []
for key in ['SEA_A__base_s0__clean__none__anone', 'RBF_I__base_s2__clean__none__anone']:
    meta = manifest['conditions'][key]
    config = deepcopy(p.read(inputs / 'request.json')['config'])
    config['detector'] = {'name': 'adwin', 'seed': p.seed('monitor', key, 'adwin')}
    folder = ROOT / 'results/filter_study/runs' / key / 'adwin/none'
    saved = pd.read_parquet(folder / 'predictions.parquet')
    events = pd.read_parquet(folder / 'events.parquet')
    with threadpool_limits(limits=1):
        pred, ev, stats = engine.execute(channels.observations(ROOT/'results/evaluation', meta),
            channels.protected(ROOT/'results/evaluation', meta), meta['classes'], 'confirmed', True,
            config, withheld=saved.filter_withheld.to_numpy(bool))
    pd.testing.assert_frame_equal(pred, saved[pred.columns], check_exact=True)
    ec = [c for c in ev if not c.endswith('_seconds')]
    pd.testing.assert_frame_equal(ev[ec], events[ec], check_exact=True)
    records.append({'condition': key, 'rows': len(pred), 'saved_trace_identical': True,
                    'source_predictions_sha256': p.sha(folder/'predictions.parquet'),
                    'source_events_sha256': p.sha(folder/'events.parquet')})
    print(records[-1], flush=True)
p.write(ROOT/'results/methods_revision/saved_filter_reproduction.json', {
    'status': 'passed', 'scope': 'Two complete Mac filter-study no-filter confirmed-reset ADWIN trajectories. Prediction values, probabilities, training visits and non-timing events compared exactly.',
    'records': records, 'environment': p.environment(), 'script_sha256': p.sha(Path(__file__))})
