"""Independently check saved controls and their 732 matched comparisons."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, average_precision_score
from common import protocol as p, channels

OUT=p.ROOT/'results/filter_no_reset'
SOURCE=p.ROOT/'results/filter_study'
KINDS=['none','learned_features','learned_features_label']
METRICS=['accuracy','macro_f1','macro_precision','macro_recall','roc_auc','average_precision','malicious_precision','malicious_recall','benign_fpr','poison_admission_rate','legitimate_withhold_rate']


def main():
    assert p.read(OUT/'status.json')['completed']==183
    inputs=p.read(SOURCE/'execution_inputs/results/evaluation/01_attacks/manifest.json')['conditions']
    recorded=pd.read_csv(OUT/'metrics.csv')
    assert len(recorded)==183 and not recorded.duplicated(['condition','filter']).any()
    adaptive=pd.read_csv(SOURCE/'metrics.csv')
    pairs=[]; hashes={};checks=0
    def check(expression):
        nonlocal checks
        assert expression
        checks+=1
    def close(actual,expected):
        check((pd.isna(actual) and pd.isna(expected)) or (actual is not None and expected is not None and np.isclose(actual,expected,rtol=0,atol=1e-12)))
    for condition,meta in inputs.items():
        truth=channels.truth(p.ROOT/'results/evaluation',meta)
        for kind in KINDS:
            folder=OUT/'runs'/condition/kind
            summary=p.read(folder/'summary.json')
            hashes[str((folder/'summary.json').relative_to(OUT))]=p.sha(folder/'summary.json')
            for f,digest in summary['files'].items():check(p.sha(folder/f)==digest)
            pred=pd.read_parquet(folder/'predictions.parquet')
            events=pd.read_parquet(folder/'events.parquet')
            check(np.array_equal(pred.row_id,truth.row_id))
            check(summary['policy']=='none' and summary['guarded'])
            check(events.reset_requested.sum()==events.reset_committed.sum()==summary['resets']==0)
            check(not events.raw_fire.any())
            check(not events.loc[events.scored,'response_event'].ne('none').any())
            check(events.loc[events.scored,'audit_latest_row'].lt(events.loc[events.scored,'start_row']).all())
            reserved=pred.audit_reserved.to_numpy(bool); withheld=pred.filter_withheld.to_numpy(bool)
            scored=pred.comparison_scored.to_numpy(bool)
            eligible=~(reserved|withheld)
            check(pred.loc[~eligible,'candidate_training_visits'].eq(0).all())
            check(pred.loc[eligible,'candidate_training_visits'].eq(1).all())
            check(pred.accepted_training_visits.le(pred.candidate_training_visits).all())
            check(pred.loc[~eligible,'first_admitted_block'].lt(0).all())
            check(not scored[reserved].any())
            check(scored.sum()==summary['host_rows'])
            check(hashlib.sha256(scored.tobytes()).hexdigest()==summary['scoring_mask_sha256'])
            check(hashlib.sha256(withheld.tobytes()).hexdigest()==summary['withholding_mask_sha256'])
            y=truth.loc[scored,'host_reference_label'].to_numpy(int)
            yh=pred.loc[scored,'prediction'].to_numpy(int)
            probabilities=pred.loc[scored,[f'p_{c}' for c in meta['classes']]].to_numpy()
            check(np.isfinite(probabilities).all())
            check(((probabilities>=0)&(probabilities<=1)).all())
            check(np.allclose(probabilities.sum(axis=1),1,rtol=0,atol=1e-12))
            close(float(np.mean(y==yh)),summary['accuracy'])
            close(float(f1_score(y,yh,labels=meta['classes'],average='macro',zero_division=0)),summary['macro_f1'])
            close(float(precision_score(y,yh,labels=meta['classes'],average='macro',zero_division=0)),summary['macro_precision'])
            close(float(recall_score(y,yh,labels=meta['classes'],average='macro',zero_division=0)),summary['macro_recall'])
            if meta['benign_class'] is not None:
                positive=y!=meta['benign_class']; selected=(yh>=0)&(yh!=meta['benign_class'])
                tp=(positive&selected).sum()
                close(float(tp/positive.sum()),summary['malicious_recall'])
                close(float(tp/selected.sum()) if selected.any() else None,summary['malicious_precision'])
                close(float((~positive&selected).sum()/(~positive).sum()),summary['benign_fpr'])
                auc_targets=[positive]
                auc_scores=[probabilities[:,[i for i,c in enumerate(meta['classes']) if c!=meta['benign_class']]].sum(axis=1)]
            else:
                auc_targets=[y==c for c in meta['classes']]
                auc_scores=[probabilities[:,i] for i in range(len(meta['classes']))]
            for name,fn in [('roc_auc',roc_auc_score),('average_precision',average_precision_score)]:
                values=[fn(target,score) if np.unique(target).size==2 else np.nan for target,score in zip(auc_targets,auc_scores)]
                close(float(np.mean(values)),summary[name])
            exposure=truth.row_id.ge(summary['activation_row']).to_numpy()&~reserved
            poison=(truth.atk_flip|truth.atk_burst).to_numpy()&exposure
            legitimate=~(truth.atk_flip|truth.atk_burst).to_numpy()&exposure
            admitted=pred.first_admitted_block.ge(0).to_numpy()
            close(float((poison&admitted).sum()/poison.sum()) if poison.any() else None,summary['poison_admission_rate'])
            close(float((legitimate&~admitted).sum()/legitimate.sum()),summary['legitimate_withhold_rate'])
            row=recorded[recorded.condition.eq(condition)&recorded['filter'].eq(kind)].iloc[0]
            for metric in METRICS:close(row[metric],summary[metric])
            for detector in ['adwin','hddm_w','hellinger','d3_oof']:
                base=SOURCE/'runs'/condition/detector/kind
                original=p.read(base/'summary.json')
                check(p.sha(base/'predictions.parquet')==original['files']['predictions.parquet'])
                reference=pd.read_parquet(base/'predictions.parquet',columns=['row_id','audit_reserved','filter_withheld','comparison_scored'])
                for col in reference:check(np.array_equal(pred[col],reference[col]))
                a=adaptive[adaptive.condition.eq(condition)&adaptive.detector.eq(detector)&adaptive['filter'].eq(kind)].iloc[0]
                record={k:original[k] for k in ['condition','stream','mode','level','base_key','attack_seed','detector','filter','activation_row','host_rows']}
                for metric in METRICS:
                    close(a[metric],original[metric])
                    record[metric+'_confirmed']=a[metric]
                    record[metric+'_no_reset']=row[metric]
                    record[metric+'_difference']=a[metric]-row[metric]
                pairs.append(record)
        print('Audited controls:',condition,flush=True)
    frame=pd.DataFrame(pairs)
    check(len(frame)==732)
    frame.to_csv(OUT/'paired_comparisons.csv',index=False)
    metrics=[c for c in frame if c.endswith(('_difference','_confirmed','_no_reset'))]
    differences=[c for c in frame if c.endswith('_difference')]
    long=frame.melt(id_vars=['condition','stream','mode','level','detector','filter'],value_vars=differences,var_name='metric',value_name='difference')
    long.groupby(['stream','mode','level','detector','filter','metric']).difference.agg(['mean','std','min','max','count']).reset_index().to_csv(OUT/'condition_descriptive_spread.csv',index=False)
    condition=frame.groupby(['stream','mode','level','detector','filter'])[metrics].mean().reset_index()
    cats={('clean','none'):0,('instance','moderate'):1,('instance','severe'):2,('concept','moderate'):3,('concept','severe'):4,('splice','moderate'):5,('splice','severe'):6}
    condition['category']=[cats[(m,l)] for m,l in zip(condition['mode'],condition.level)]
    condition.to_csv(OUT/'condition_comparisons.csv',index=False)
    condition['population']=np.where(condition['mode'].eq('clean'),'clean','attacked')
    condition.groupby(['stream','detector','filter','population'])[metrics].mean().to_csv(OUT/'overview.csv')
    # Filtering effect within the no-reset policy: shared scoring rows were checked above.
    refs=recorded[recorded['filter'].eq('none')]
    filter_pairs=recorded[recorded['filter'].ne('none')].merge(refs,on='condition',suffixes=('','_no_filter'),validate='many_to_one')
    check(filter_pairs.scoring_mask_sha256.eq(filter_pairs.scoring_mask_sha256_no_filter).all())
    for metric in METRICS:filter_pairs[metric+'_difference']=filter_pairs[metric]-filter_pairs[metric+'_no_filter']
    filter_pairs.to_csv(OUT/'within_no_reset_filter_pairs.csv',index=False)
    overview=filter_pairs.groupby(['stream','mode','level','filter'])[[x+'_difference' for x in METRICS]].mean().reset_index()
    overview['population']=np.where(overview['mode'].eq('clean'),'clean','attacked')
    overview.groupby(['stream','filter','population'])[[x+'_difference' for x in METRICS]].mean().to_csv(OUT/'within_no_reset_filter_overview.csv')
    p.write(OUT/'verification.json',{'status':'passed','runs':183,'matched_pairs':732,'checks':checks,
                                  'all_protected_and_withholding_masks_equal':True,'all_scoring_masks_equal':True,
                                  'all_reset_counts_zero':True,'metrics_sha256':p.sha(OUT/'metrics.csv'),
                                  'run_summaries':hashes,'audit_source_sha256':p.sha(Path(__file__)),
                                  'output_hashes':{f.name:p.sha(f) for f in OUT.glob('*.csv')},'updated_utc':p.now()})
    p.write(OUT/'status.json',{'phase':'complete','completed':183,'planned':183,'verification':'passed','updated_utc':p.now()})
    print('Independent control audit passed:',checks,'checks')


if __name__=='__main__':main()
