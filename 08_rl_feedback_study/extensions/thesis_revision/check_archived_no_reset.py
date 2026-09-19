"""Rescore new unfiltered controls on the original archived common masks."""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

ROOT=Path(__file__).resolve().parents[2]

def main(require_complete):
    original=ROOT/'results/evaluation'
    controls=ROOT/'results/filter_no_reset'
    manifest=json.loads((ROOT/'results/filter_study/execution_inputs/results/evaluation/01_attacks/manifest.json').read_text())
    meta=manifest['conditions'];conditions=pd.DataFrame(meta.values())
    archive_path=original/'06_analysis/presentation/fixed_responses/tables/run_scores.csv'
    archive=pd.read_csv(archive_path,float_precision='round_trip')
    archive=archive[archive.policy.eq('none')&archive.guarded]
    records=[]
    for (base,seed),attacks in conditions[conditions['mode'].ne('clean')].groupby(['base_key','attack_seed']):
        clean=conditions[conditions.base_key.eq(base)&conditions['mode'].eq('clean')]
        assert len(clean)==1 and len(attacks)==6
        selected=[meta[k] for k in [clean.iloc[0]['key'],*attacks.key]]
        mask=None;labels=None
        for condition in selected:
            path=original/'01_attacks'/condition['truth']
            assert hashlib.sha256(path.read_bytes()).hexdigest()==manifest['outputs'][condition['truth']]
            truth=pd.read_parquet(path,columns=['row_id','host_eligible','host_reference_label'])
            if mask is None:mask=truth.row_id.to_numpy()>=1000;labels=truth.host_reference_label.to_numpy()
            else:np.testing.assert_array_equal(labels,truth.host_reference_label)
            mask &= truth.host_eligible.to_numpy(bool)
            ids=pd.read_parquet(original/'01_attacks'/condition['audit'],columns=['row_id']).row_id.to_numpy(int)
            mask[ids]=False
        digest=hashlib.sha256(np.packbits(mask).tobytes()).hexdigest()
        for condition in selected:
            file=controls/'runs'/condition['key']/'none/predictions.parquet'
            if not file.exists():continue
            row=archive[archive.condition.eq(condition['key'])&archive.attack_seed.eq(seed)]
            assert len(row)==1
            row=row.iloc[0]
            assert row.mask_sha256==digest and row.common_host_rows==mask.sum()
            pred=pd.read_parquet(file,columns=['row_id','prediction'])
            np.testing.assert_array_equal(pred.row_id,np.arange(len(mask)))
            y,yh=labels[mask],pred.prediction.to_numpy()[mask]
            accuracy=float(np.mean(y==yh)); f1=float(f1_score(y,yh,labels=condition['classes'],average='macro',zero_division=0))
            records.append({'condition':condition['key'],'cohort_attack_seed':int(seed),'host_rows':int(mask.sum()),
                            'new_accuracy':accuracy,'archived_accuracy':float(row.accuracy),'accuracy_difference':accuracy-row.accuracy,
                            'new_macro_f1':f1,'archived_macro_f1':float(row.macro_f1),'macro_f1_difference':f1-row.macro_f1})
    frame=pd.DataFrame(records)
    frame.to_csv(controls/'archived_baseline_reproduction.csv',index=False)
    exact=bool(np.isclose(frame.accuracy_difference,0,rtol=0,atol=1e-12).all() and np.isclose(frame.macro_f1_difference,0,rtol=0,atol=1e-12).all())
    original_request=json.loads((original/'request.json').read_text())
    extension_request=json.loads((ROOT/'results/filter_study/execution_inputs/results/evaluation/request.json').read_text())
    # Frozen-controller artifacts differ between executions but are unused by a no-reset arm.
    used_config=lambda request:{k:v for k,v in request['config'].items() if k!='rl_models'}
    assert used_config(original_request)==used_config(extension_request)
    original_manifest=json.loads((original/'01_attacks/manifest.json').read_text())
    assert original_manifest['conditions']==manifest['conditions'] and original_manifest['outputs']==manifest['outputs']
    result={'status':'passed_exact' if len(frame)==63 and exact else 'complete_with_differences' if len(frame)==63 else 'partial',
            'cohort_scores_checked':len(frame),'expected_cohort_scores':63,'all_accuracy_and_macro_f1_match':exact,
            'max_absolute_accuracy_difference':float(frame.accuracy_difference.abs().max()),
            'max_absolute_macro_f1_difference':float(frame.macro_f1_difference.abs().max()),
            'archived_source_sha256':hashlib.sha256(archive_path.read_bytes()).hexdigest(),
            'same_input_hashes_and_used_config':True,
            'original_environment':original_request['environment'],
            'archived_filter_environment':extension_request['environment'],
            'current_environment':json.loads((controls/'request.json').read_text())['environment'],
            'scope':'New no-filter screened no-reset predictions rescored on archived common masks. No original reset-policy traces reconstructed.',
            'interpretation':'The archived scores were not exactly reproduced. The precise cause was not isolated because the original raw traces are unavailable. This is not an exact-replication pass. The matched filter-study comparisons use their own Mac executions and within-condition scoring masks.'}
    (controls/'archived_baseline_reproduction.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
    if require_complete:assert len(frame)==63, 'Archived-score assessment is incomplete'

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--require-complete',action='store_true')
    main(parser.parse_args().require_complete)
