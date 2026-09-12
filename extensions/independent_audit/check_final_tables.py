"""Reconcile final filter tables with all summaries without the report builder."""
from pathlib import Path
import argparse
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from common import protocol as p


def same(actual, expected):
    if expected is None or (isinstance(expected, (float, np.floating)) and np.isnan(expected)):
        assert pd.isna(actual), (actual, expected)
    elif isinstance(expected, (int, float, np.number)):
        assert np.isclose(actual, expected, rtol=1e-10, atol=1e-12), (actual, expected)
    else:
        assert actual == expected, (actual, expected)


def check(study, destination):
    request, verification, report = [p.read(study / name) for name in ['request.json','verification.json','report_manifest.json']]
    assert verification['status'] == 'passed'
    assert verification['request_sha256'] == report['request_sha256'] == p.sha(study / 'request.json')
    assert verification['metrics_sha256'] == p.sha(study / 'metrics.csv')
    for name, digest in report['outputs'].items():
        assert p.sha(study / name) == digest
    conditions = p.read(ROOT / request['inputs']['evaluation']['root'] / '01_attacks/manifest.json')['conditions']
    filters = ['none','oracle_rows','random_rows','oracle_blocks','random_blocks','learned_features','learned_features_label']
    expected = {(c, d, f) for c in conditions for d in request['settings']['detectors'] for f in filters}
    keys = ['condition','detector','filter']
    raw = pd.read_csv(study / 'metrics.csv', float_precision='round_trip').set_index(keys)
    assert raw.index.is_unique and set(raw.index) == expected and verification['runs'] == len(expected)
    summary_hashes = {f.relative_to(study).as_posix():p.sha(f) for f in (study/'runs').rglob('summary.json')}
    assert summary_hashes == verification['summary_hashes'] and len(summary_hashes) == len(expected)
    for relative in summary_hashes:
        record = p.read(study / relative)
        actual = raw.loc[tuple(record[k] for k in keys)]
        assert set(actual.index) == set(record) - set(keys) - {'files'}
        for name in actual.index:
            same(actual[name], record[name])
    paired = pd.read_csv(study/'paired_differences.csv', float_precision='round_trip').set_index(keys)
    assert paired.index.is_unique and set(paired.index) == expected
    measures = [c.removesuffix('_difference') for c in paired.columns if c.endswith('_difference')]
    for key, row in paired.iterrows():
        original = raw.loc[key]; baseline = raw.loc[(key[0],key[1],'none')]
        for name in raw.columns:
            same(row[name], original[name])
        for name in measures:
            same(row[name+'_no_filter'], baseline[name])
            same(row[name+'_difference'], original[name]-baseline[name])
    group_keys = ['stream','mode','level','detector','filter']
    summary = pd.read_csv(study/'summary.csv', float_precision='round_trip').set_index(group_keys+['metric'])
    assert summary.index.is_unique
    expected_summary = set()
    for key, group in raw.reset_index().groupby(group_keys):
        for metric in measures:
            full_key = (*key,metric); expected_summary.add(full_key)
            row = summary.loc[full_key]; values = group[metric].dropna().to_numpy()
            same(row['count'], len(values))
            same(row['mean'], float(np.mean(values)) if len(values) else None)
            same(row['std'], float(np.std(values,ddof=1)) if len(values)>1 else None)
    assert set(summary.index) == expected_summary
    local = pd.concat([pd.read_csv(f,float_precision='round_trip') for f in (study/'models').glob('*/separability.csv')],ignore_index=True)
    combined = pd.read_csv(study/'separability.csv',float_precision='round_trip')
    order = ['stream','condition','view']
    pd.testing.assert_frame_equal(combined.sort_values(order).reset_index(drop=True),
        local.sort_values(order).reset_index(drop=True),check_dtype=False,check_exact=False,rtol=1e-10,atol=1e-12)
    figure_count = len({c['stream'] for c in conditions.values()})*len(request['settings']['detectors'])
    assert len(list((study/'figures').glob('*.pdf'))) == len(list((study/'figures').glob('*.png'))) == figure_count
    result = {'status':'passed','completed_utc':p.now(),'study':str(study.relative_to(ROOT)),
        'unique_runs':len(raw),'paired_rows':len(paired),'summary_rows':len(summary),'separability_rows':len(combined),
        'figure_pairs':figure_count,'visual_review':'requires human/visual inspection separately',
        'verification_sha256':p.sha(study/'verification.json'),'report_manifest_sha256':p.sha(study/'report_manifest.json')}
    destination.parent.mkdir(parents=True,exist_ok=True); p.write(destination,result)
    print(result,flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, default=Path('results/filter_study'))
    parser.add_argument('--output', type=Path, default=Path('results/independent_audit/2026-09-12/post_completion/final_table_audit.json'))
    args=parser.parse_args(); check(ROOT/args.study,ROOT/args.output)
