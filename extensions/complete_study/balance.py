"""Equal-weight comparison blocks with explicit shared RADAR clean references."""
import pandas as pd
from common import protocol as p

CATEGORIES=[('clean','none'),('instance','moderate'),('instance','severe'),('concept','moderate'),('concept','severe'),('splice','moderate'),('splice','severe')]


def comparison_index(spec,manifests):
    records=[]
    for ds in spec['datasets']:
        conditions=list(manifests[ds]['conditions'].values())
        repetitions=spec['radar']['evaluation_attack_assignments'] if ds=='radar' else spec['evaluation_seeds']
        assert len(repetitions)==5
        for repetition,seed in enumerate(repetitions,1):
            for mode,level in CATEGORIES:
                matches=[c for c in conditions if c['mode']==mode and c['level']==level and
                         ((ds=='radar' and (mode=='clean' or c['attack_assignment']==seed)) or
                          (ds!='radar' and c['base_seed']==seed))]
                assert len(matches)==1,(ds,repetition,mode,level)
                c=matches[0]
                records.append(dict(stream=ds,repetition=repetition,input_seed=seed,mode=mode,level=level,
                                    condition=c['key'],shared_clean_reference=ds=='radar' and mode=='clean',
                                    within_dataset_weight=1/35,dataset_weight=1/len(spec['datasets']),
                                    input_rows=c['rows']))
    return pd.DataFrame(records)


def write_index(out):
    spec=p.read(p.ROOT/'configs/study.json')
    manifests={ds:p.read(out/'stages/evaluation'/ds/'01_attacks/manifest.json') for ds in spec['datasets']}
    frame=comparison_index(spec,manifests)
    assert frame.groupby('stream').size().eq(35).all()
    assert frame.groupby(['stream','repetition']).size().eq(7).all()
    assert frame.groupby(['stream','mode','level']).size().eq(5).all()
    counts=frame.groupby('stream').condition.nunique().to_dict()
    assert counts=={'RBF_I':35,'SEA_A':35,'radar':31}
    folder=out/'analysis/full_study';folder.mkdir(parents=True,exist_ok=True)
    frame.to_csv(folder/'comparison_index.csv',index=False)
    p.write(folder/'comparison_balance.json',dict(status='passed',comparison_entries_per_dataset=35,
        repetitions_per_dataset=5,conditions_per_repetition=7,unique_input_conditions=counts,
        equal_condition_weights=True,shared_radar_clean_is_not_independent=True,
        index_sha256=p.sha(folder/'comparison_index.csv')))
    return frame


if __name__=='__main__':write_index(p.ROOT/'results/per_dataset_study')


def publish_comparison_entries(out):
    """Expose equal comparison blocks separately from unique execution records."""
    index=write_index(out)
    stages={'fixed_responses':'fixed_responses','passive_detectors':'passive_detectors',
            'filter_evaluation':'filter_evaluation','rl_evaluation':'evaluation','feedback_sensitivity':'sensitivity'}
    counts={}
    for label,stage in stages.items():
        path=out/'analysis'/stage/'metrics.csv'
        if not path.exists():raise ValueError(f'Missing complete comparison metrics: {path}')
        values=pd.read_csv(path)
        expanded=index.merge(values,on=['stream','condition','mode','level'],validate='many_to_many')
        sizes=expanded.groupby('stream').size()
        assert len(sizes)==3 and sizes.nunique()==1,(label,sizes.to_dict())
        expanded.to_csv(out/'analysis/full_study'/f'{label}_comparison_entries.csv',index=False)
        counts[label]=sizes.to_dict()
    p.write(out/'analysis/full_study/equal_coverage.json',dict(status='passed',comparison_counts=counts,
        shared_references_are_not_independent=True,utc=p.now()))
