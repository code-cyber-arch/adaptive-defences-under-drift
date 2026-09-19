"""Reconcile the public result tables and summaries without original raw traces."""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from common import protocol as p
from extensions.independent_audit.check_final_tables import check as check_filters


def main():
    destination=ROOT/'results/repository_checks'
    destination.mkdir(parents=True,exist_ok=True)
    original=ROOT/'results/evaluation'
    request=p.read(original/'request.json')
    conditions=p.read(original/'01_attacks/manifest.json')['conditions']
    raw=pd.read_csv(original/'06_analysis/metrics.csv',float_precision='round_trip')
    expected={p.run_id(c['key'],a) for c in conditions.values() for a in p.arms(request['config'])}
    assert len(raw)==len(expected)==1464 and raw.run_id.is_unique and set(raw.run_id)==expected
    assert raw.status.eq('complete').all()
    for name,digest in p.read(original/'06_analysis/manifest.json')['outputs'].items():
        assert p.sha(original/'06_analysis'/name)==digest,name
    summaries=list((original/'05_runs').rglob('summary.json'))
    assert len(summaries)==1032
    completion=p.read(original/'05_runs/completion.json')
    # All available originals are also bound by provenance/retained-files.json.
    fixed=raw[raw.policy.isin(['none','immediate','confirmed'])]
    merged=fixed[fixed.guarded].merge(fixed[~fixed.guarded],on=['condition','detector','policy'],suffixes=('_g','_u'),validate='one_to_one')
    saved=pd.read_csv(original/'06_analysis/paired_gate_differences.csv',float_precision='round_trip').set_index(['condition','detector','policy']).sort_index()
    merged=merged.set_index(['condition','detector','policy']).sort_index()
    assert saved.index.equals(merged.index)
    for metric in ['host_accuracy','host_macro_f1','poison_admission_rate','clean_withhold_rate',
                   'resets','training_visits','seconds','malicious_recall','benign_fpr']:
        np.testing.assert_allclose(saved[metric],merged[metric+'_g']-merged[metric+'_u'],rtol=1e-9,atol=1e-10,equal_nan=True)
    check_filters(ROOT/'results/filter_study',destination/'filter_tables.json')
    controls=ROOT/'results/filter_no_reset'
    verification=p.read(controls/'verification.json')
    assert verification['status']=='passed' and p.sha(controls/'metrics.csv')==verification['metrics_sha256']
    no_reset=pd.read_csv(controls/'metrics.csv',float_precision='round_trip').set_index(['condition','filter'])
    adaptive=pd.read_csv(ROOT/'results/filter_study/metrics.csv',float_precision='round_trip').set_index(['condition','detector','filter'])
    pairs=pd.read_csv(controls/'paired_comparisons.csv',float_precision='round_trip').set_index(['condition','detector','filter'])
    assert len(no_reset)==183 and no_reset.index.is_unique and no_reset.resets.eq(0).all()
    assert len(pairs)==732 and pairs.index.is_unique
    expected_pairs={(c,d,f) for c,f in no_reset.index for d in ['adwin','hddm_w','hellinger','d3_oof']}
    assert set(pairs.index)==expected_pairs
    control_summaries=list((controls/'runs').rglob('summary.json'))
    assert len(control_summaries)==183
    for path in control_summaries:
        record=p.read(path)
        actual=no_reset.loc[(record['condition'],record['filter'])]
        for name,value in actual.items():
            expected=record[name]
            if isinstance(value,(float,int,np.number)) and not isinstance(value,(bool,np.bool_)):
                if expected is None:assert pd.isna(value)
                else:np.testing.assert_allclose(value,expected,rtol=1e-10,atol=1e-12,equal_nan=True)
            else:assert value==expected,(path,name)
    metrics=[c.removesuffix('_difference') for c in pairs.columns if c.endswith('_difference')]
    for key,row in pairs.iterrows():
        treatment=adaptive.loc[key];control=no_reset.loc[(key[0],key[2])]
        assert row.host_rows==treatment.host_rows==control.host_rows
        for metric in metrics:
            np.testing.assert_allclose([row[metric+'_confirmed'],row[metric+'_no_reset'],row[metric+'_difference']],
                                       [treatment[metric],control[metric],treatment[metric]-control[metric]],rtol=1e-10,atol=1e-12,equal_nan=True)
    result={'status':'passed','original_policy_records':1464,'available_original_summaries':1032,
            'missing_original_summaries':432,'filter_runs':1708,'no_reset_runs':183,'matched_pairs':732,
            'scope':'Public-table coverage, paired arithmetic and retained-summary reconciliation. Original raw prediction traces are unavailable; this is not full original-run replication.'}
    p.write(destination/'evidence.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
