"""Read-only content audit of every source and intervention in the actual study."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from common import protocol as p, channels


def main():
    root = ROOT / 'results/evaluation'; stage = root / '01_attacks'
    manifest = p.read(stage / 'manifest.json')
    records, checked = [], set()
    direct_target_matches = []
    for base, base_info in manifest['bases'].items():
        clean = pd.read_parquet(stage / base_info['clean'])
        features = [c for c in clean if c != 'label']
        n = len(clean); original_x = clean[features].to_numpy(); original_y = clean.label.to_numpy()
        assert list(clean.columns) == [f'f{i}' for i in range(len(features))] + ['label']
        assert np.isfinite(original_x).all()
        if base_info['metadata']['stream'] == 'radar':
            assert n == 484753 and len(features) == 68
            mapping = base_info['metadata']['feature_lineage']
            for name, lineage in mapping.items():
                assert not lineage['source_column'].lower().startswith(('target', 'label', '__label'))
            targets = {'multiclass':original_y, 'malicious':(original_y != 7).astype(int),
                       'benign':(original_y == 7).astype(int)}
            for i, name in enumerate(features):
                for role, target in targets.items():
                    if np.array_equal(original_x[:,i], target):
                        direct_target_matches.append({'feature':name,'target':role})
        for meta in manifest['conditions'].values():
            if meta['base_key'] != base:
                continue
            for relative in [meta['observations'],meta['truth'],meta['audit']]:
                if relative not in checked:
                    assert p.sha(stage / relative) == manifest['outputs'][relative]
                    checked.add(relative)
            obs = channels.observations(root, meta); truth = channels.truth(root, meta)
            audit = channels.protected(root, meta)
            assert len(obs) == len(truth) == n == meta['rows']
            np.testing.assert_array_equal(obs.row_id, np.arange(n))
            np.testing.assert_array_equal(truth.row_id, np.arange(n))
            np.testing.assert_array_equal(truth.host_reference_label, original_y)
            np.testing.assert_array_equal(audit[features].to_numpy(), original_x[audit.row_id])
            np.testing.assert_array_equal(audit.label, original_y[audit.row_id])
            assert np.array_equal(audit.release_block, audit.row_id // 1000 + 1)
            modified = (truth.atk_flip | truth.atk_burst).to_numpy(bool)
            assert not modified[:1000].any()
            x, y = obs[features].to_numpy(), obs.label.to_numpy()
            assert np.isfinite(x).all()
            assert set(np.unique(y)) <= set(meta['classes'])
            np.testing.assert_array_equal(x[~modified], original_x[~modified])
            np.testing.assert_array_equal(y[~modified], original_y[~modified])
            np.testing.assert_array_equal(truth.host_eligible, ~truth.atk_burst.to_numpy(bool))
            future_copies = 0
            if meta['mode'] == 'clean':
                assert not modified.any()
            elif meta['mode'] == 'instance':
                np.testing.assert_array_equal(x, original_x)
                assert (y[modified] != original_y[modified]).all()
                assert int(modified.sum()) == int(round(meta['attack']['level_value'] * n))
            elif meta['mode'] == 'splice':
                origins = truth.replay_source_row.to_numpy()[modified]
                assert ((origins >= 0) & (origins < n)).all()
                np.testing.assert_array_equal(x[modified], original_x[origins])
                np.testing.assert_array_equal(y[modified], original_y[origins])
                future_copies = int((origins > np.flatnonzero(modified)).sum())
            records.append({'condition':meta['key'],'rows':n,'modified_rows':int(modified.sum()),
                            'protected_rows':len(audit),'future_copy_rows':future_copies,'status':'passed'})
        print(f'Content checked: {base}', flush=True)
    assert not direct_target_matches, direct_target_matches
    p.write(ROOT / 'results/independent_audit/2026-09-12/input_audit.json',
            {'status':'passed','completed_utc':p.now(),'sources':len(manifest['bases']),
             'conditions':len(records),'input_files_hashed':len(checked),'records':records,
             'radar_direct_target_equality_matches':direct_target_matches,
             'limitation':'Checks exclude direct annotation columns/equality only; unknown FastText training provenance and semantic leakage remain unresolved.'})


if __name__ == '__main__':
    main()
