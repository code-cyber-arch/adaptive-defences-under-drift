"""Full-stream same-machine checks of original and extension no-reset engines."""
from pathlib import Path
import json
import sys
from copy import deepcopy
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from common import protocol as p, channels


def main():
    original = p.load_module('reproduction_original', '05_experiment/engine.py')
    extension = p.load_module('reproduction_extension', 'extensions/filter_study/engine.py')
    inputs = ROOT / 'results/filter_study/execution_inputs/results/evaluation'
    manifest = p.read(inputs / '01_attacks/manifest.json')
    config = deepcopy(p.read(inputs / 'request.json')['config'])
    config['detector'] = {'name': 'none'}
    records = []
    for key in ['SEA_A__base_s0__clean__none__anone', 'RBF_I__base_s2__clean__none__anone']:
        meta = manifest['conditions'][key]
        obs = channels.observations(ROOT / 'results/evaluation', meta)
        audit = channels.protected(ROOT / 'results/evaluation', meta)
        saved = ROOT / 'results/filter_no_reset/runs' / key / 'none/predictions.parquet'
        with threadpool_limits(limits=1):
            a, ea, sa = original.execute(obs, audit, meta['classes'], 'none', True, config)
            b, eb, sb = extension.execute(obs, audit, meta['classes'], 'none', True, config)
        pd.testing.assert_frame_equal(a, b[a.columns], check_exact=True)
        ec = [c for c in ea if not c.endswith('_seconds')]
        pd.testing.assert_frame_equal(ea[ec], eb[ec], check_exact=True)
        c = pd.read_parquet(saved)
        pd.testing.assert_frame_equal(b, c[b.columns], check_exact=True)
        record = {'condition': key, 'rows': len(a), 'engines_identical': True,
                  'saved_separate_process_identical': True,
                  'saved_predictions_sha256': p.sha(saved),
                  'original_seconds': sa['seconds'], 'extension_seconds': sb['seconds']}
        records.append(record)
        print(json.dumps(record), flush=True)
    p.write(ROOT / 'results/methods_revision/engine_reproduction.json', {
        'status': 'passed', 'scope': 'Two complete synthetic streams on the current Mac; all prediction values, probabilities, training visits and non-timing events compared exactly. Separate saved controls also compared exactly.',
        'records': records, 'environment': p.environment(),
        'sources': {name: p.sha(ROOT/name) for name in ['05_experiment/engine.py', 'extensions/filter_study/engine.py', 'extensions/thesis_revision/check_engine_reproduction.py']}})


if __name__ == '__main__':
    main()
