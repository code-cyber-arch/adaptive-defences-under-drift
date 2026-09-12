"""Independently recompute the binary outcomes used in the RADAR prose.

Checks all 19 conditions for the ADWIN no-filter and feature-plus-label arms.
No model is run, fitted or changed.
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from common import protocol as p, channels


def main():
    source = ROOT/'results/evaluation'
    manifest = p.read(source/'01_attacks/manifest.json')
    checks = []
    for condition in manifest['conditions'].values():
        if condition['stream'] != 'radar':
            continue
        truth = channels.truth(source, condition)
        for kind in ['none', 'learned_features_label']:
            folder = ROOT/'results/filter_study/runs'/condition['key']/'adwin'/kind
            summary = p.read(folder/'summary.json')
            file = folder/'predictions.parquet'
            assert p.sha(file) == summary['files'][file.name]
            saved = pd.read_parquet(file, columns=['row_id','comparison_scored','prediction'])
            mask = saved.comparison_scored.to_numpy(bool)
            ids = saved.row_id.to_numpy(int)[mask]
            actual = truth.iloc[ids].host_reference_label.to_numpy(int) != 7
            predicted = saved.prediction.to_numpy(int)[mask] != 7
            tp = int((actual & predicted).sum())
            fp = int((~actual & predicted).sum())
            fn = int((actual & ~predicted).sum())
            tn = int((~actual & ~predicted).sum())
            assert len(ids) == summary['host_rows'] == tp+fp+fn+tn
            assert tp+fp and tp+fn and fp+tn
            values = {'malicious_precision':tp/(tp+fp), 'malicious_recall':tp/(tp+fn), 'benign_fpr':fp/(fp+tn)}
            for name, value in values.items():
                assert np.isclose(value, summary[name], rtol=1e-12, atol=1e-12), (condition['key'],kind,name)
            checks.append({'condition':condition['key'], 'filter':kind, 'tp':tp,'fp':fp,'fn':fn,'tn':tn,
                'summary_sha256':p.sha(folder/'summary.json'), 'values':values})
    assert len(checks) == 38
    p.write(ROOT/'results/narrative/radar_binary_checks.json', {'status':'passed', 'checked_utc':p.now(),
        'scope':'19 RADAR conditions, ADWIN, no-filter and feature-plus-label arms used in the narrative.',
        'runs':38, 'binary_rates_checked':114, 'checks':checks})
    print('Independent RADAR binary-outcome check passed: 38 saved runs, 114 rates.')


if __name__ == '__main__':
    main()
