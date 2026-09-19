"""Derive additional thesis comparisons without changing frozen result folders."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common import protocol as p, channels

OUT = p.ROOT/'results/methods_revision'
FILTERS = ['none','learned_features','learned_features_label']
KEYS = ['stream','mode','level','detector','filter']


def fixed_comparisons():
    source=p.ROOT/'results/evaluation/06_analysis/presentation/fixed_responses/tables/run_scores.csv'
    data=pd.read_csv(source)
    keys=['dataset','base_key','attack_seed','category','guarded']
    refs=data[data.policy.eq('none')]
    assert not refs.duplicated(keys).any()
    pairs=data[data.policy.ne('none')].merge(refs,on=keys,suffixes=('','_control'),validate='many_to_one')
    assert pairs.mask_sha256.eq(pairs.mask_sha256_control).all()
    assert pairs.common_host_rows.eq(pairs.common_host_rows_control).all()
    for metric in ['accuracy','macro_f1','poison_admission_rate','clean_withhold_rate']:
        pairs[metric+'_difference']=100*(pairs[metric]-pairs[metric+'_control'])
    pairs.to_csv(OUT/'fixed_no_reset_pairs.csv',index=False)
    metrics=[c for c in pairs if c.endswith('_difference')]
    condition=pairs.groupby(['dataset','category','policy','guarded','detector'])[metrics].mean().reset_index()
    condition.to_csv(OUT/'fixed_no_reset_conditions.csv',index=False)
    condition['population']=np.where(condition.category.eq(0),'clean','attacked')
    condition.groupby(['dataset','policy','guarded','detector','population'])[metrics].mean().to_csv(OUT/'fixed_no_reset_overview.csv')
    return {'source':str(source.relative_to(p.ROOT)),'sha256':p.sha(source),'pairs':len(pairs),'identical_scoring_masks':True}


def filter_monitor_activity():
    source=p.ROOT/'results/filter_study'
    target=OUT/'filter_monitor_runs.csv'
    if target.exists():
        saved=p.read(OUT/'filter_monitor_provenance.json')
        assert saved['csv_sha256']==p.sha(target)
        return saved
    manifest=p.read(source/'execution_inputs/results/evaluation/01_attacks/manifest.json')
    config=p.read(source/'execution_inputs/results/evaluation/request.json')['config']
    analyse=p.load_module('thesis_event_analysis','06_analysis/analyse.py')
    rows=[]; hashes={}
    for key,meta in manifest['conditions'].items():
        truth=channels.truth(p.ROOT/'results/evaluation',meta)
        for detector in ['adwin','hddm_w','hellinger','d3_oof']:
            for kind in FILTERS:
                folder=source/'runs'/key/detector/kind
                summary=p.read(folder/'summary.json')
                digest=p.sha(folder/'events.parquet')
                assert digest==summary['files']['events.parquet']
                hashes[str((folder/'events.parquet').relative_to(p.ROOT))]=digest
                e=pd.read_parquet(folder/'events.parquet')
                scored=e[e.start_row.ge(summary['activation_row'])]
                assert int(scored.reset_committed.sum())==summary['scored_resets']
                record={k:summary[k] for k in ['condition',*KEYS,'activation_row']}
                record.update(blocks=len(scored),alarms=int(scored.raw_fire.sum()),
                              alarm_rate=float(scored.raw_fire.mean()),
                              resets=int(scored.reset_committed.sum()),
                              reset_requests=int(scored.reset_requested.sum()),
                              rejected_updates=int((~scored.accepted).sum()),
                              rejected_update_rate=float((~scored.accepted).mean()),
                              reset_rejections=int((scored.reset_requested & ~scored.accepted).sum()),
                              rows=summary['rows']-summary['activation_row'])
                record['resets_per_100k']=record['resets']/record['rows']*100000
                scores=analyse.event_scores(e,truth,meta,config)
                record.update(scores)
                record['event_recall']=1-scores['missed_detections']/scores['scheduled_events'] if scores['scheduled_events'] else None
                rows.append(record)
        print('Monitor traces:',key,flush=True)
    frame=pd.DataFrame(rows)
    assert len(frame)==732
    frame.to_csv(target,index=False)
    metrics=['alarm_rate','resets_per_100k','rejected_update_rate','event_recall','detection_delay','unexposed_false_alarm_rate','unexposed_blocks']
    cond=frame.groupby(KEYS)[metrics].mean().reset_index()
    cond.to_csv(OUT/'filter_monitor_conditions.csv',index=False)
    cond['population']=np.where(cond['mode'].eq('clean'),'clean','attacked')
    cond.groupby(['stream','detector','filter','population'])[metrics].mean().to_csv(OUT/'filter_monitor_overview.csv')
    saved={'traces':len(rows),'event_files':hashes,'csv_sha256':p.sha(target),
           'event_metric_source_sha256':p.sha(p.ROOT/'06_analysis/analyse.py'),
           'interpretation':'Operational monitors coupled to adaptation. RADAR has alarm rates only, without verified event truth.'}
    p.write(OUT/'filter_monitor_provenance.json',saved)
    return saved


def original_response_activity():
    source=p.ROOT/'results/evaluation/06_analysis/metrics.csv'
    data=pd.read_csv(source)
    # First average RL fitting seeds; then average source/attack assignments.
    metrics=['resets_per_100k','rejected_update_blocks','reset_requests','reset_rejections','unexposed_false_alarm_rate','unexposed_blocks']
    cohort=data.groupby(['stream','condition','mode','level','policy','guarded','detector'],dropna=False)[metrics].mean().reset_index()
    condition=cohort.groupby(['stream','mode','level','policy','guarded','detector'])[metrics].mean().reset_index()
    condition['population']=np.where(condition['mode'].eq('clean'),'clean','attacked')
    condition.groupby(['stream','policy','guarded','detector','population'])[metrics].mean().to_csv(OUT/'original_response_overview.csv')
    return {'source':str(source.relative_to(p.ROOT)),'sha256':p.sha(source),'rows':len(data)}


if __name__=='__main__':
    OUT.mkdir(parents=True,exist_ok=True)
    record={'fixed':fixed_comparisons(),'original_response':original_response_activity(),'filter_monitors':filter_monitor_activity()}
    p.write(OUT/'analysis_manifest.json',record)
    print('Additional existing-evidence comparisons complete')
