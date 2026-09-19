"""Verify the complete experiment grid and publish integrated comparisons."""
from pathlib import Path
from functools import lru_cache
import numpy as np
import pandas as pd
from common import protocol as p, channels
from . import workers as w
check=p.load_module('full_independent_metrics','06_analysis/reporting/check_and_plot.py')
METRICS=['host_macro_f1','host_accuracy','poison_admission_rate','clean_withhold_rate','resets','training_visits','malicious_recall','benign_fpr']


def verify_and_report(out,sources,configs,conditions,seeds,detectors,pilot):
    contract=p.read(out/'full_protocol.json')
    for name,digest in contract['sources'].items():assert p.sha(p.ROOT/name)==digest,name
    for (role,ds),root in sources.items():
        recorded=contract['input_contracts'][f'{role}/{ds}']
        assert p.sha(root/'request.json')==recorded['request']
        assert p.sha(root/'01_attacks/manifest.json')==recorded['manifest']
        for name,digest in p.read(root/'01_attacks/manifest.json')['outputs'].items():assert p.sha(root/'01_attacks'/name)==digest
    for ds in ['SEA_A','RBF_I','radar']:
        train_keys={c['key'] for c in conditions[('training',ds)]};val_keys={c['key'] for c in conditions[('validation',ds)]};test_keys={c['key'] for c in conditions[('evaluation',ds)]}
        assert not (train_keys&val_keys or train_keys&test_keys or val_keys&test_keys)
        for seed in seeds:
            folder=out/'filters'/f'seed_{seed}'/'models'/ds;record=p.read(folder/'manifest.json');w.verify_files(folder,record)
            assert record['seed']==seed
            fitted=pd.read_parquet(folder/'fit_rows.parquet');calibration=pd.read_parquet(folder/'calibration_rows.parquet')
            assert set(fitted.condition)<=train_keys and set(calibration.condition)<=val_keys
            assert not set(fitted.fingerprint)&set(calibration.fingerprint)
    lookup={c['key']:c for (role,ds),items in conditions.items() if role=='evaluation' for c in items}
    @lru_cache(maxsize=2)
    def truth_for(key):
        c=lookup[key];return channels.truth(sources[('evaluation',c['stream'])],c)
    expected={stage:set() for stage in ['fixed_responses','passive_detectors','filter_evaluation']}
    for c in lookup.values():
        ds=c['stream']
        for arm in w.fixed_arms(detectors):expected['fixed_responses'].add((ds,c['key'],arm['detector'],arm['policy'],arm['guarded']))
        for detector in detectors:expected['passive_detectors'].add((ds,c['key'],detector))
        for detector in [*detectors,'none']:
            for kind,seed in w.filter_specs(seeds):
                if detector=='none' and kind!='none' and not kind.startswith('learned_'):continue
                expected['filter_evaluation'].add((ds,c['key'],detector,kind,seed))
    frames={};checked=[]
    for stage,path in [('fixed_responses','fixed_responses/runs'),('passive_detectors','passive_detectors/runs'),('filter_evaluation','filters/runs')]:
        rows=[];actual=set()
        for file in sorted((out/path).rglob('summary.json')):
            r=p.read(file);folder=file.parent;w.verify_files(folder,r)
            c=lookup[r['condition']];ds=c['stream'];config=configs[('evaluation',ds)]
            assert r['contract']['protocol_sha256']==p.sha(out/'full_protocol.json')
            assert r['contract']['config']==config
            source=sources[('evaluation',ds)];manifest=p.read(source/'01_attacks/manifest.json')
            for name,digest in r['contract']['inputs'].items():assert digest==manifest['outputs'][c[name]]
            truth=truth_for(c['key'])
            if stage=='passive_detectors':
                events=pd.read_parquet(folder/'alarms.parquet');assert len(events)==(len(truth)+config['block_size']-1)//config['block_size']-1
                assert r['alarm_blocks']==int(events.raw_fire.sum()) and r['learner_resets']==0
                check.close(r['alarm_rate'],float(events.raw_fire.mean()))
                actual.add((ds,c['key'],r['detector']))
            else:
                pred=pd.read_parquet(folder/'predictions.parquet');events=pd.read_parquet(folder/'events.parquet')
                assert pred.row_id.equals(truth.row_id)
                reserved=pred.audit_reserved
                assert not pred.loc[reserved,'scored'].any() and pred.loc[reserved,'candidate_training_visits'].eq(0).all()
                assert pred.loc[reserved,'first_admitted_block'].eq(-1).all()
                assert r['resets']==int(events.reset_committed.sum())
                assert r['training_visits']==int(pred.candidate_training_visits.sum())
                assert (events.loc[events.scored,'audit_latest_row']<events.loc[events.scored,'start_row']).all()
                assert (events.reset_committed==(events.reset_requested&events.accepted)).all()
                metric_truth=truth.copy()
                if stage=='filter_evaluation':
                    profile=out/'filters/profiles'/ds/c['key'];pr=p.read(profile/'summary.json');w.verify_files(profile,pr)
                    assert p.sha(profile/'summary.json')==r['contract']['profile_sha256']
                    key=w.filter_key(r['filter'],r['filter_seed']);masks=pd.read_parquet(profile/'masks.parquet')
                    assert np.array_equal(pred.filter_withheld,masks[key]) and np.array_equal(pred.comparison_scored,masks.comparison_scored)
                    assert pred.loc[pred.filter_withheld,'candidate_training_visits'].eq(0).all()
                    assert pred.loc[pred.filter_withheld,'first_admitted_block'].eq(-1).all()
                    assert not pred.loc[reserved,'comparison_scored'].any()
                    metric_truth['host_eligible'] &= pred.comparison_scored
                    actual.add((ds,c['key'],r['detector'],r['filter'],r['filter_seed']))
                else:actual.add((ds,c['key'],r['detector'],r['policy'],r['guarded']))
                for name,value in check.independent_metrics(pred,metric_truth,c['classes']).items():check.close(r[name],value)
                if r['policy']=='none':assert r['resets']==0 and not events.raw_fire.any()
                if r.get('reused'):
                    assert p.sha(p.ROOT/r['reused']['summary'])==r['reused']['sha256']
            rows.append({k:v for k,v in r.items() if k not in ['files','contract','reused']})
            checked.append({'path':file.relative_to(p.ROOT).as_posix(),'sha256':p.sha(file),'stage':stage})
        assert actual==expected[stage],(stage,len(actual),len(expected[stage]))
        assert len(rows)==len(actual)
        frames[stage]=pd.DataFrame(rows)
        frames[stage].to_csv(out/'analysis'/stage/'metrics.csv',index=False)
        print(f'Full-study audit: {stage} {len(rows)} passed',flush=True)
    dest=out/'analysis/full_study';dest.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(checked).to_csv(dest/'checked_runs.csv',index=False)
    verification={'status':'passed','counts':{stage:len(keys) for stage,keys in expected.items()},'filter_models':len(seeds)*3*2,
                  'shared_inputs_checked':True,'metrics_recomputed':True,'filter_partitions_checked':True,'utc':p.now()}
    p.write(dest/'verification.json',verification)
    baseline_report(out,frames['fixed_responses'],sources,conditions)
    render(out,frames,conditions,seeds,pilot)
    p.write(out/'full_completion.json',dict(status='complete',pilot=pilot,verification=verification,protocol_sha256=p.sha(out/'full_protocol.json'),utc=p.now()))


def baseline_report(out,fixed,sources,conditions):
    """Compare attack damage on the intersection of eligible rows per realization."""
    rows=[]
    for ds in ['SEA_A','RBF_I','radar']:
        selected=conditions[('evaluation',ds)]
        for base_key in {c['base_key'] for c in selected}:
            group=[c for c in selected if c['base_key']==base_key];mask=None
            for c in group:
                truth=channels.truth(sources[('evaluation',ds)],c)
                local=truth.host_eligible.to_numpy()&~truth.audit_reserved.to_numpy()&(truth.row_id.to_numpy()>=c['warmup_rows'])
                mask=local if mask is None else mask&local
            assert mask.any()
            for c in group:
                record=fixed[fixed.condition.eq(c['key'])&fixed.policy.eq('none')&~fixed.guarded].iloc[0]
                pred=pd.read_parquet(p.ROOT/record.folder/'predictions.parquet')
                truth=channels.truth(sources[('evaluation',ds)],c);truth['host_eligible']=mask
                scores=check.independent_metrics(pred,truth,c['classes'])
                rows.append(w.metadata(c)|{k:scores[k] for k in ['host_n','host_accuracy','host_macro_f1']})
    frame=pd.DataFrame(rows);dest=out/'analysis/full_study'
    ref=frame[frame['mode'].eq('clean')][['stream','base_key','host_macro_f1','host_accuracy']]
    frame=frame.merge(ref,on=['stream','base_key'],suffixes=('','_clean'),validate='many_to_one')
    frame['macro_f1_change_from_clean']=frame.host_macro_f1-frame.host_macro_f1_clean
    frame.to_csv(dest/'baseline_attack_damage.csv',index=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    order=[('clean','none'),('instance','moderate'),('instance','severe'),('concept','moderate'),('concept','severe'),('splice','moderate'),('splice','severe')]
    fig,axes=plt.subplots(1,3,figsize=(15,5))
    for ax,ds in zip(axes,['SEA_A','RBF_I','radar']):
        means=frame[frame.stream.eq(ds)].groupby(['mode','level'])[['host_accuracy','host_macro_f1']].mean().reindex(pd.MultiIndex.from_tuples(order))
        for metric,label in [('host_accuracy','Accuracy'),('host_macro_f1','Macro-F1')]:ax.plot(range(7),means[metric]*100,marker='o',label=label)
        ax.set_title(ds);ax.set_xticks(range(7),['Clean','Instance 15%','Instance 25%','Concept 15%','Concept 25%','Splice 15%','Splice 25%'],rotation=45,ha='right');ax.set_ylabel('Score (%)');ax.grid(alpha=.2)
    axes[0].legend();fig.tight_layout();folder=out/'analysis/figures';folder.mkdir(parents=True,exist_ok=True)
    for ext in ['pdf','png']:fig.savefig(folder/f'baseline_attack_damage.{ext}',dpi=170,bbox_inches='tight')
    plt.close(fig)


def balanced(frame,groups,metrics):
    return frame.groupby([*groups,'mode','level'],dropna=False)[metrics].mean().groupby(groups,dropna=False).mean().reset_index()


def render(out,frames,conditions,seeds,pilot):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    dest=out/'analysis/full_study';figures=out/'analysis/figures';figures.mkdir(parents=True,exist_ok=True)
    fixed=frames['fixed_responses'];filtered=frames['filter_evaluation'];passive=frames['passive_detectors']
    keys=['stream','condition','detector','policy']
    paired=fixed[fixed.guarded].merge(fixed[~fixed.guarded],on=keys,suffixes=('_screened','_unscreened'),validate='one_to_one')
    for metric in METRICS:paired[metric+'_difference']=paired[metric+'_screened']-paired[metric+'_unscreened']
    paired.to_csv(dest/'screening_effects.csv',index=False)
    fk=['stream','condition','detector'];ref=filtered[filtered['filter'].eq('none')][fk+METRICS]
    effects=filtered.merge(ref,on=fk,suffixes=('','_reference'),validate='many_to_one')
    for metric in METRICS:effects[metric+'_difference']=effects[metric]-effects[metric+'_reference']
    effects.to_csv(dest/'filter_effects.csv',index=False)
    rk=['stream','condition','filter','filter_seed'];noreset=filtered[filtered.detector.eq('none')][rk+METRICS]
    reset=filtered[~filtered.detector.eq('none')].merge(noreset,on=rk,suffixes=('','_no_reset'),validate='many_to_one')
    for metric in METRICS:reset[metric+'_reset_difference']=reset[metric]-reset[metric+'_no_reset']
    reset.to_csv(dest/'filter_reset_effects.csv',index=False)
    fixed_mean=balanced(fixed,['stream','detector','policy','guarded'],METRICS);fixed_mean.to_csv(dest/'fixed_summary.csv',index=False)
    # Equal condition weights within seed, then equal seed weights within method.
    filter_seed=balanced(filtered,['stream','detector','filter','filter_seed'],METRICS)
    filter_seed.to_csv(dest/'filter_seed_summary.csv',index=False)
    filter_mean=filter_seed.groupby(['stream','detector','filter'],dropna=False)[METRICS].mean().reset_index()
    filter_mean.to_csv(dest/'filter_summary.csv',index=False)
    passive_metrics=['alarm_rate','poison_alarm_rate','unexposed_alarm_rate','event_recall','detection_delay','seconds_per_100k']
    passive_mean=balanced(passive,['stream','detector'],passive_metrics);passive_mean.to_csv(dest/'passive_summary.csv',index=False)
    names={'SEA_A':'SEA','RBF_I':'RBF','radar':'RADAR'};detectors=['adwin','hddm_w','hellinger','d3_oof']
    for name,data,series,xkey,ykey in [
        ('fixed_response_comparison',fixed_mean,[f'{policy}/{guard}' for policy in ['immediate','confirmed'] for guard in [False,True]],'detector','host_macro_f1'),
        ('filter_comparison',filter_mean,['none','learned_features','learned_features_label','oracle_rows','random_rows','oracle_blocks','random_blocks'],'detector','host_macro_f1'),
        ('passive_detector_comparison',passive_mean,['alarm_rate','poison_alarm_rate','unexposed_alarm_rate'],'detector',None)]:
        fig,axes=plt.subplots(1,3,figsize=(15,5))
        for ax,(ds,label) in zip(axes,names.items()):
            sub=data[data.stream.eq(ds)]
            for item in series:
                if name.startswith('fixed'):
                    policy,guard=item.split('/');part=sub[sub.policy.eq(policy)&sub.guarded.eq(guard=='True')];metric=ykey
                    legend=f'{policy}, {"screened" if guard=="True" else "unscreened"}'
                elif name.startswith('filter'):
                    part=sub[sub['filter'].eq(item)];metric=ykey;legend=item.replace('_',' ')
                else:part=sub;metric=item;legend=item.replace('_',' ')
                part=part.set_index('detector').reindex(detectors)
                ax.plot(range(4),part[metric]*100,marker='o',label=legend)
            ax.set_title(label);ax.set_xticks(range(4),['ADWIN','HDDM-W','Hellinger','D3 OOF'],rotation=20)
            ax.set_ylabel('Macro-F1 (%)' if ykey else 'Alarm blocks (%)');ax.grid(alpha=.2)
        handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=3,frameon=False)
        fig.tight_layout(rect=(0,.2,1,1))
        for ext in ['pdf','png']:fig.savefig(figures/f'{name}.{ext}',dpi=170,bbox_inches='tight')
        plt.close(fig)
    if not pilot:
        rl_path=p.ROOT/'07_documentation/RESULTS.md'
        rl_text=rl_path.read_text().replace('Experiment 3 retains its existing three-dataset evidence.','Experiment 3 uses the same evaluation partitions and five-seed grid.')
        overview=['# Complete five-seed experimental results','',
                  'All three experiments use the same evaluation conditions: 35 SEA, 35 RBF and 31 RADAR conditions. RQ1 and RQ2 remain unchanged.','',
                  f'The study contains {len(fixed):,} fixed-response configurations, {len(passive):,} passive detector evaluations, 120 RL controllers, 30 fitted filters and {len(filtered):,} filter/control configurations. Shared no-reset references and reused trajectories are not independent replications.','',
                  'Synthetic evaluation uses five stream seeds. RADAR uses five poisoning assignments on one held-out capture period, with one shared clean condition. Learned filters use five fitting seeds for each of their two observation views. All methods share the same protected reservation and per-condition data; filter comparisons additionally exclude development-feature duplicates from host scoring. Training and validation data remain separate from evaluation.','',
                  '- [Baseline attack damage](../results/per_dataset_study/analysis/figures/baseline_attack_damage.pdf)',
                  '- [Fixed responses](../results/per_dataset_study/analysis/figures/fixed_response_comparison.pdf)',
                  '- [Passive detectors](../results/per_dataset_study/analysis/figures/passive_detector_comparison.pdf)',
                  '- [Filtering](../results/per_dataset_study/analysis/figures/filter_comparison.pdf)',
                  '- [Complete-study verification](../results/per_dataset_study/analysis/full_study/verification.json)','',rl_text]
        rl_path.write_text('\n'.join(overview))
