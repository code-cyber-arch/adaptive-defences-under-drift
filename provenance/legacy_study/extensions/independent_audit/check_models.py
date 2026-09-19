"""Recompute separability metrics and inspect frozen split/model artifacts."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from common import protocol as p


def close(actual, expected):
    assert (pd.isna(actual) and expected is None) or (expected is not None and np.isclose(actual, expected, rtol=1e-10, atol=1e-12))


def main():
    records = []; paths = ROOT / 'results/filter_study/models'
    for folder in sorted(paths.iterdir()):
        if not folder.is_dir():
            continue
        manifest = p.read(folder / 'manifest.json')
        for relative, digest in manifest['files'].items():
            assert p.sha(folder / relative) == digest
        fit = pd.read_parquet(folder / 'fit_rows.parquet')
        cal = pd.read_parquet(folder / 'calibration_rows.parquet')
        excluded = np.load(folder / 'excluded_test_fingerprints.npy', allow_pickle=False)
        assert not np.intersect1d(fit.fingerprint, cal.fingerprint).size
        assert np.isin(fit.fingerprint, excluded).all() and np.isin(cal.fingerprint, excluded).all()
        # Preserve the exact threshold: the default CSV float parser can round a
        # boundary below a tied score and change the strict > decision.
        table = pd.read_csv(folder / 'separability.csv', float_precision='round_trip')
        if folder.name == 'radar':
            assert fit.row_id.between(1000, 241999).all()
            assert cal.row_id.between(243000, 338999).all()
        for row in table.itertuples():
            data = pd.read_parquet(folder / 'heldout' / row.condition / (row.view + '.parquet'))
            assert not np.isin(data.fingerprint, excluded).any()
            if folder.name == 'radar':
                assert data.row_id.ge(340000).all()
            target, score = data.poisoned.to_numpy(bool), data.score.to_numpy(float)
            assert np.isfinite(score).all() and ((score >= 0) & (score <= 1)).all()
            threshold = manifest['models'][row.view]['threshold']
            assert row.threshold == threshold
            predicted = score > threshold
            np.testing.assert_array_equal(predicted, data.rejected)
            assert len(data) == row.rows and int(target.sum()) == row.poison_rows
            for name, count in [('tp',(target & predicted).sum()),('tn',(~target & ~predicted).sum()),
                                 ('fp',(~target & predicted).sum()),('fn',(target & ~predicted).sum())]:
                assert getattr(row, name) == count
            close(row.poison_recall, float((target & predicted).sum()/target.sum()) if target.any() else None)
            close(row.legitimate_rejection, float((~target & predicted).sum()/(~target).sum()) if (~target).any() else None)
            close(row.roc_auc, roc_auc_score(target, score) if np.unique(target).size == 2 else None)
            close(row.average_precision, average_precision_score(target, score) if np.unique(target).size == 2 else None)
        records.append({'stream':folder.name,'models':2,'heldout_condition_view_records':len(table),
                        'fit_rows':len(fit),'calibration_rows':len(cal),'status':'passed'})
        print(f'Frozen models, split exclusions and metrics checked: {folder.name}', flush=True)
    p.write(ROOT / 'results/independent_audit/2026-09-12/model_audit.json',
            {'status':'passed','completed_utc':p.now(),'records':records,
             'limitation':'Exact-copy exclusions and positional splits do not establish unseen-attack validity or resolve upstream FastText/full-archive attack-generation provenance.'})


if __name__ == '__main__':
    main()
