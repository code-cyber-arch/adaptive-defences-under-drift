"""Extract traceable narrative tables without rerunning any research model."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from common import protocol as p, channels
OUT = ROOT/'results/narrative'
TABLES = OUT/'tables'
SOURCES = {}


def csv(relative):
    file=ROOT/relative;SOURCES[relative]=p.sha(file)
    return pd.read_csv(file,float_precision='round_trip')


def record(relative):
    file=ROOT/relative;SOURCES[relative]=p.sha(file)
    return p.read(file)


def save(frame,name):
    frame.to_csv(TABLES/name,index=False)
    return frame


def finite_json(value):
    if isinstance(value,dict):return {str(k):finite_json(v) for k,v in value.items()}
    if isinstance(value,list):return [finite_json(v) for v in value]
    if isinstance(value,(float,np.floating)) and not np.isfinite(value):return None
    if isinstance(value,np.generic):return value.item()
    return value


def original():
    provenance=record('results/narrative/references/radar_release/provenance_check.json')
    assert provenance['status']=='published_feature_values_matched'
    assert provenance['rows']==484753 and provenance['exact_numeric_cells']==32963204
    raw_provenance=record('results/narrative/references/radar_release/raw_frequency_check.json')
    assert raw_provenance['all_rows_within_5e_7'] and raw_provenance['mapped_raw_labels_match']
    root='results/evaluation/06_analysis/'
    manifest=record(root+'manifest.json')
    for name,digest in manifest['outputs'].items():assert p.sha(ROOT/root/name)==digest
    verification=record('results/evaluation/verification.json')
    assert verification['status']=='passed' and verification['runs']==1464
    prefix=root+'presentation/'
    baseline=csv(prefix+'baseline_clean_vs_poisoned/plot_values.csv')
    metrics=['accuracy','macro_f1','macro_recall','roc_auc']
    baseline=baseline[baseline.metric.isin(metrics)].copy()
    save(baseline,'baseline_condition_metrics.csv')
    wide=baseline.pivot(index=['dataset','category','mode','level'],columns='metric',values='mean').reset_index()
    controls=wide[wide.category.eq(0)].set_index('dataset')
    for metric in ['accuracy','macro_f1','macro_recall']:
        wide[metric+'_change_pp']=100*(wide[metric]-wide.dataset.map(controls[metric]))
    save(wide,'baseline_condition_changes.csv')
    passive=csv(prefix+'detector_baseline/plot_values.csv')
    save(passive,'passive_detector_condition_metrics.csv')
    passive_overview=passive.groupby(['dataset','series','metric'],as_index=False).agg(
        mean_of_condition_means=('mean','mean'),minimum_condition_mean=('mean','min'),
        maximum_condition_mean=('mean','max'),defined_conditions=('mean','count'))
    save(passive_overview,'passive_detector_overview.csv')
    fixed=csv(prefix+'fixed_responses/tables/paired_screening_effects.csv')
    measurements=[c for c in fixed if c.endswith('_screened_minus_unscreened')]
    conditions=fixed.groupby(['dataset','category','detector','policy'],as_index=False)[measurements].mean()
    save(conditions,'screening_condition_differences.csv')
    overview=conditions.melt(id_vars=['dataset','category','detector','policy'],value_vars=measurements,
                            var_name='metric',value_name='difference')
    overview.metric=overview.metric.str.removesuffix('_screened_minus_unscreened')
    overview=overview.groupby(['dataset','detector','policy','metric'],as_index=False).agg(
        mean_difference=('difference','mean'),min_condition_difference=('difference','min'),
        max_condition_difference=('difference','max'),defined_conditions=('difference','count'),
        positive_conditions=('difference',lambda x:int(x.gt(0).sum())))
    save(overview,'screening_overview.csv')
    rl=csv(prefix+'rl_comparison/tables/improvement_values.csv')
    save(rl,'rl_condition_differences.csv')
    rlo=rl.groupby(['dataset','series','reference','metric'],as_index=False).agg(
        mean_difference=('mean','mean'),min_condition_difference=('mean','min'),
        max_condition_difference=('mean','max'),defined_conditions=('mean','count'),
        positive_conditions=('mean',lambda x:int(x.gt(0).sum())))
    save(rlo,'rl_overview.csv')
    paired=csv(prefix+'rl_comparison/tables/paired_improvements.csv')
    paired=paired[paired.reference.eq('confirmed')]
    key=['dataset','base_key','attack_seed','category','metric']
    error=paired[paired.series.eq('rl')].set_index(key)
    alarm=paired[paired.series.eq('rl_alarm')].set_index(key)
    numeric=[c for c in paired if c not in key+['series','reference']]
    # This table contains already seed-averaged, matched cohort differences.
    value='difference' if 'difference' in numeric else 'value'
    if value not in error:raise ValueError('Unexpected RL paired table schema: '+str(paired.columns.tolist()))
    state_delta=(error[value]-alarm[value]).rename('error_aware_minus_alarm').reset_index()
    save(state_delta,'rl_state_cohort_differences.csv')
    state_cond=state_delta.groupby(['dataset','category','metric'],as_index=False).error_aware_minus_alarm.mean()
    save(state_cond,'rl_state_condition_differences.csv')
    counts=csv(prefix+'fixed_responses/tables/run_scores.csv')[['dataset','base_key','attack_seed','common_host_rows']].drop_duplicates()
    save(counts,'original_host_denominators.csv')
    facts={'original_runs':1464,'passive_evaluations':244,
           'baseline':wide.to_dict('records'),'screening':overview.to_dict('records'),
           'passive':passive_overview.to_dict('records'),'rl':rlo.to_dict('records'),
           'rl_state_contrasts':state_cond.to_dict('records')}
    p.write(OUT/'original_facts.json',finite_json(facts))


def separability():
    frames=[]
    for stream in ['SEA_A','RBF_I','radar']:
        prefix='results/filter_study/models/'+stream+'/'
        meta=record(prefix+'manifest.json')
        for name,digest in meta['files'].items():assert p.sha(ROOT/prefix/name)==digest
        frames.append(csv(prefix+'separability.csv'))
    data=pd.concat(frames,ignore_index=True)
    assert len(data)==122
    save(data,'filter_separability_conditions.csv')
    group=data[data['mode'].ne('clean')].groupby(['stream','view','mode'])
    overview=group[['roc_auc','average_precision','poison_recall','poison_precision','legitimate_rejection']].mean().reset_index()
    save(overview,'filter_separability_overview.csv')
    p.write(OUT/'separability_facts.json',finite_json(overview.to_dict('records')))


def radar_context():
    """Class support and an untrained reference; no model fitting or reruns."""
    root=ROOT/'results/evaluation'
    manifest=record('results/evaluation/01_attacks/manifest.json')
    conditions=[c for c in manifest['conditions'].values() if c['stream']=='radar']
    assert len(conditions)==19
    loaded={};records=[]
    def references(c):
        if c['key'] not in loaded:
            truth=channels.truth(root,c)
            protected=channels.protected(root,c).row_id.to_numpy(int)
            loaded[c['key']]=(truth,protected)
        return loaded[c['key']]
    def scored(protocol,condition,mode,level,seed,labels,mask_hash=None):
        count=labels.value_counts().to_dict();n=len(labels);benign=int(count.get(7,0))
        return {'protocol':protocol,'condition':condition,'mode':mode,'level':level,
                'attack_seed':seed,'host_rows':n,'classes_present':len(count),
                **{f'class_{i}':int(count.get(i,0)) for i in range(8)},
                'always_goodware_accuracy':benign/n,
                'always_goodware_macro_f1':2*benign/(n+benign)/8,
                'always_goodware_malicious_recall':0.,'always_goodware_benign_fpr':0.,
                'mask_sha256':mask_hash}
    for c in conditions:
        file=ROOT/'results/filter_study/models/radar/heldout'/c['key']/'features.parquet'
        ids=pd.read_parquet(file,columns=['row_id']).row_id.to_numpy(int)
        truth,_=references(c)
        ids=ids[truth.host_eligible.to_numpy(bool)[ids]]
        summary_file=ROOT/'results/filter_study/runs'/c['key']/'adwin/none/summary.json'
        if summary_file.exists():assert len(ids)==p.read(summary_file)['host_rows']
        records.append(scored('filter_suffix',c['key'],c['mode'],c['level'],c.get('attack_seed'),
                              truth.iloc[ids].host_reference_label))
    originals=csv('results/evaluation/06_analysis/presentation/fixed_responses/tables/run_scores.csv')
    for seed in range(3):
        selected=[c for c in conditions if c['mode']=='clean' or c.get('attack_seed')==seed]
        assert len(selected)==7
        mask=np.arange(selected[0]['rows'])>=1000
        labels=None
        for c in selected:
            truth,protected=references(c)
            if labels is None:labels=truth.host_reference_label
            else:np.testing.assert_array_equal(labels,truth.host_reference_label)
            mask &= truth.host_eligible.to_numpy(bool);mask[protected]=False
        expected=originals[originals.dataset.eq('RADAR') & originals.attack_seed.eq(seed)]
        digest=hashlib.sha256(np.packbits(mask).tobytes()).hexdigest()
        assert expected.common_host_rows.eq(int(mask.sum())).all()
        assert expected.mask_sha256.eq(digest).all()
        records.append(scored('original_common_mask','radar_cohort_'+str(seed),'clean','none',seed,
                              labels[mask],digest))
    frame=save(pd.DataFrame(records),'radar_class_support_and_constant_reference.csv')
    original=frame[frame.protocol.eq('original_common_mask')]
    filter_rows=frame[frame.protocol.eq('filter_suffix') & frame['mode'].ne('clean')]
    metrics=['always_goodware_accuracy','always_goodware_macro_f1']
    overview={'original_common_mask':original[metrics].mean().to_dict(),
              'filter_attacked':filter_rows.groupby(['mode','level'])[metrics].mean().mean().to_dict(),
              'filter_minimum_class_support':int(filter_rows[[f'class_{i}' for i in range(8)]].min().min()),
              'filter_classes_present_minimum':int(filter_rows.classes_present.min()),
              'description':'Post hoc, untrained always-goodware scoring reference; not an additional learning run.'}
    p.write(OUT/'radar_context_facts.json',finite_json(overview))


def filters():
    prefix='results/filter_study/'
    binary=record('results/narrative/radar_binary_checks.json')
    assert binary['status']=='passed' and binary['runs']==38 and binary['binary_rates_checked']==114
    for check in binary['checks']:
        file=ROOT/prefix/'runs'/check['condition']/'adwin'/check['filter']/'summary.json'
        assert p.sha(file)==check['summary_sha256']
    verification=record(prefix+'verification.json')
    assert verification['status']=='passed' and verification['runs']==1708
    final=record('results/independent_audit/2026-09-12/post_completion/final_table_audit.json')
    trace=record('results/independent_audit/2026-09-12/post_completion/filter_trace_audit.json')
    assert final['status']=='passed' and final['unique_runs']==1708
    assert trace['status']=='passed_for_snapshot' and trace['checked_runs']==1708
    assert trace['summary_hashes']==verification['summary_hashes']
    report=record(prefix+'report_manifest.json')
    assert final['verification_sha256']==p.sha(ROOT/prefix/'verification.json')
    assert final['report_manifest_sha256']==p.sha(ROOT/prefix/'report_manifest.json')
    for name,digest in report['outputs'].items():assert p.sha(ROOT/prefix/name)==digest
    assert p.sha(ROOT/prefix/'metrics.csv')==verification['metrics_sha256']
    raw=csv(prefix+'metrics.csv')
    pairs=csv(prefix+'paired_differences.csv')
    measures=['accuracy','macro_f1','macro_recall','roc_auc','poison_admission_rate','legitimate_withhold_rate',
              'filter_poison_recall','filter_legitimate_rejection','scored_resets','total_seconds']
    keys=['stream','mode','level','detector','filter']
    counts=pairs.groupby(keys).size().rename('matched_assignments')
    condition=pairs.groupby(keys)[[m+'_difference' for m in measures]].mean().join(counts).reset_index()
    save(condition,'filter_condition_differences.csv')
    long=condition.melt(id_vars=keys+['matched_assignments'],value_vars=[m+'_difference' for m in measures],
                        var_name='metric',value_name='difference')
    long.metric=long.metric.str.removesuffix('_difference')
    # Separate clean from attacked. Six attack/severity cells receive equal
    # weight; controller/detector settings never become extra replicates.
    long['population']=np.where(long['mode'].eq('clean'),'clean','attacked')
    overview=long.groupby(['stream','detector','filter','population','metric'],as_index=False).agg(
        mean_difference=('difference','mean'),min_condition_difference=('difference','min'),
        max_condition_difference=('difference','max'),defined_conditions=('difference','count'),
        positive_conditions=('difference',lambda x:int(x.gt(0).sum())))
    save(overview,'filter_overview.csv')
    denom=raw[['condition','stream','mode','level','activation_row','host_rows']].drop_duplicates()
    assert len(denom)==61
    save(denom,'filter_host_denominators.csv')
    absolute=raw.groupby(keys)[measures].mean().reset_index()
    save(absolute,'filter_condition_metrics.csv')
    cyber_metrics=['malicious_precision','malicious_recall','benign_fpr']
    cyber=raw[raw.stream.eq('radar')].copy()
    reference=cyber[cyber['filter'].eq('none')][['condition','detector',*cyber_metrics]]
    cyber=cyber.merge(reference,on=['condition','detector'],suffixes=('','_no_filter'),validate='many_to_one')
    for metric in cyber_metrics:cyber[metric+'_difference']=cyber[metric]-cyber[metric+'_no_filter']
    save(cyber[['condition',*keys,*cyber_metrics,*[m+'_difference' for m in cyber_metrics]]],
         'filter_radar_cybersecurity_pairs.csv')
    paired_control=pairs.set_index(['condition','detector','filter'])
    controls=[]
    for condition_id,detector in raw[['condition','detector']].drop_duplicates().itertuples(index=False,name=None):
        for granularity in ['rows','blocks']:
            oracle=paired_control.loc[(condition_id,detector,'oracle_'+granularity)]
            random=paired_control.loc[(condition_id,detector,'random_'+granularity)]
            controls.append({'condition':condition_id,'stream':oracle['stream'],'mode':oracle['mode'],
                'level':oracle['level'],'detector':detector,'granularity':granularity,
                **{m+'_oracle_minus_random':oracle[m]-random[m] for m in measures}})
    oracle=save(pd.DataFrame(controls),'oracle_minus_random.csv')
    oracle_measures=[c for c in oracle if c.endswith('_oracle_minus_random')]
    oracle_conditions=oracle[oracle['mode'].ne('clean')].groupby(
        ['stream','detector','granularity','mode','level'])[oracle_measures].mean().reset_index()
    save(oracle_conditions.groupby(['stream','detector','granularity'])[oracle_measures].mean().reset_index(),
         'oracle_overview.csv')
    p.write(OUT/'filter_facts.json',finite_json({'runs':len(raw),'condition_differences':condition.to_dict('records'),
        'overview':overview.to_dict('records'),'host_denominators':denom.to_dict('records')}))


def main(full=False,output=None):
    global OUT,TABLES,SOURCES
    OUT=Path(output) if output is not None else ROOT/'results/narrative'
    TABLES=OUT/'tables';SOURCES={}
    TABLES.mkdir(parents=True,exist_ok=True)
    original();separability();radar_context()
    if full:filters()
    p.write(OUT/'evidence_manifest.json',{'created_utc':p.now(),'stage':'full_verified' if full else 'original_and_separability',
        'sources':SOURCES,'outputs':{f.relative_to(OUT).as_posix():p.sha(f) for f in [*TABLES.glob('*.csv'),*OUT.glob('*_facts.json')]},
        'aggregation':'Original prediction differences use common host masks; original overviews weight seven condition means equally. Filter overviews separate clean from six equally weighted attacked condition means. Descriptive only.'})
    print('Narrative evidence prepared:', 'full verified study' if full else 'original benchmark and completed separability')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--full',action='store_true')
    main(parser.parse_args().full)
