"""Audit all three datasets and render the matched detector/policy comparison."""
from pathlib import Path
import sys
import importlib.util
import hashlib
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from common import protocol as p
from common.replication import counts, training_request_matches, run_contract_matches
OUT=ROOT/'results/per_dataset_study'
spec=importlib.util.spec_from_file_location('separate_metric_audit',Path(__file__).with_name('check_and_plot.py'))
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)
METRICS=['host_macro_f1','host_accuracy','poison_admission_rate','clean_withhold_rate','resets','malicious_recall','benign_fpr']


def verify():
    cfg=p.read(ROOT/'configs/study.json')
    sizes=counts(cfg)
    contract=p.read(OUT/'protocol.json')
    assert contract['spec']==cfg
    for name,digest in contract['sources'].items(): assert p.sha(ROOT/name)==digest,name
    assert p.sha(ROOT/contract['radar_source']['snapshot'])==contract['radar_source']['source_sha256']
    manifests={}; configs={}
    for role in ['training','validation','evaluation']:
        for dataset in cfg['datasets']:
            source=OUT/'stages'/role/dataset
            m=p.read(source/'01_attacks/manifest.json'); manifests[(role,dataset)]=m
            configs[(role,dataset)]=p.read(source/'request.json')['config']
            assert m['data_role']==role and m['dataset']==dataset
            for name,digest in m['outputs'].items(): assert p.sha(source/'01_attacks'/name)==digest
            if dataset=='radar':
                expected={'training':(0,242000),'validation':(243000,339000),'evaluation':(340000,484753)}[role]
                part=m['source_partition']; assert (part['start_row'],part['end_row_exclusive'])==expected
                mapping=pd.read_parquet(source/'00_streams/row_map.parquet')
                assert np.array_equal(mapping.source_row_id,np.arange(*expected))
                for c in m['conditions'].values():
                    assert c['source_partition']==part
                    truth=pd.read_parquet(source/'01_attacks'/c['truth'])
                    replay=truth.replay_source_row.to_numpy()
                    assert ((replay>=-1)&(replay<expected[1]-expected[0])).all()
    models=p.read(OUT/'models.json')['models']
    expected={(ds,d,v,s) for ds in cfg['datasets'] for d in cfg['detectors']
              for v in ['alarm','alarm_persistence'] for s in cfg['rl_seeds']}
    assert len(models)==len(expected)==sizes['controllers']
    assert {(m['dataset'],m['detector'],m['variant'],m['seed']) for m in models}==expected
    orders={}; model_records={}
    for m in models:
        path=ROOT/m['path']; assert p.sha(path)==m['sha256']
        record=p.read(path); model_records[m['path']]=record
        for key in ['dataset','detector','variant','seed']: assert record[key]==m[key]
        source=OUT/'stages/training'/m['dataset']
        assert training_request_matches(record['request_sha256'],source)
        assert set(record['training_conditions'])==set(manifests[('training',m['dataset'])]['conditions'])
        for name,digest in record['files'].items(): assert p.sha(path.parent/name)==digest
        transitions=pd.read_parquet(path.parent/'transitions.parquet')
        assert transitions.dataset.eq(m['dataset']).all() and transitions.detector.eq(m['detector']).all()
        assert transitions.reward_block.eq(transitions.action_block+1).all()
        assert (transitions.audit_latest_row<transitions.reward_block*1000).all()
        n=configs[('training',m['dataset'])]['rows']
        assert len(transitions)==record['training_transitions']==7*cfg['training_epochs']*((n+999)//1000-1)
        episodes=pd.read_csv(path.parent/'episodes.csv'); assert len(episodes)==28
        order=episodes[['epoch','condition']].to_csv(index=False)
        assert orders.setdefault((m['dataset'],m['seed']),order)==order
    selected=p.read(OUT/'selected_policies.json')
    assert selected['protocol_sha256']==p.sha(OUT/'protocol.json')
    assert selected['models_sha256']==p.sha(OUT/'models.json')
    rows=[]; cached={}; masks={}; schedules={}
    for role,directory in [('validation','validation_runs'),('evaluation','evaluation_runs')]:
        for file in sorted((OUT/directory).rglob('summary.json')):
            r=p.read(file);ds=r['stream']; source=OUT/'stages'/role/ds
            c=manifests[(role,ds)]['conditions'][r['condition']]
            for name,digest in r['files'].items(): assert p.sha(file.parent/name)==digest
            for name,digest in r['contract']['inputs'].items():
                assert digest==manifests[(role,ds)]['outputs'][c[name]]
            current_contract=dict(r['contract'], config=configs[(role,ds)])
            assert run_contract_matches(r['contract'],current_contract)
            key=(role,ds,r['condition'])
            if key not in cached:
                cached[key]=(pd.read_parquet(source/'01_attacks'/c['truth']),pd.read_parquet(source/'01_attacks'/c['audit']))
            truth,reserved=cached[key]
            pred=pd.read_parquet(file.parent/'predictions.parquet')
            events=pd.read_parquet(file.parent/'events.parquet')
            feedback=pd.read_parquet(file.parent/'feedback_schedule.parquet')
            assert pred.row_id.equals(truth.row_id)
            assert set(pred.loc[pred.audit_reserved,'row_id'])==set(reserved.row_id)==set(feedback.row_id)
            assert not pred.loc[pred.audit_reserved,'scored'].any()
            assert pred.loc[pred.audit_reserved,'candidate_training_visits'].eq(0).all()
            assert pred.loc[pred.audit_reserved,'first_admitted_block'].eq(-1).all()
            for name,value in audit.independent_metrics(pred,truth,c['classes']).items(): audit.close(r[name],value)
            if ds=='radar':
                eligible=pred.scored&truth.host_eligible
                y=truth.loc[eligible,'host_reference_label'].to_numpy();yh=pred.loc[eligible,'prediction'].to_numpy()
                malicious=y!=c['benign_class']; predicted=(yh>=0)&(yh!=c['benign_class'])
                audit.close(r['malicious_recall'],float((malicious&predicted).sum()/malicious.sum()))
                audit.close(r['benign_fpr'],float((~malicious&predicted).sum()/(~malicious).sum()))
            mask=hashlib.sha256(pred.scored.to_numpy().tobytes()).hexdigest()
            assert masks.setdefault(key,mask)==mask
            schedule_key=(role,ds,c['base_key'],r['feedback_fraction'],r['delay_blocks'])
            assert schedules.setdefault(schedule_key,r['files']['feedback_schedule.parquet'])==r['files']['feedback_schedule.parquet']
            for event in events.itertuples():
                if not event.scored:continue
                released=feedback[(feedback.row_id//1000==event.block-r['delay_blocks'])&feedback.release_block.le(event.block)]
                assert event.audit_n==len(released)
                assert event.audit_latest_row==(int(released.row_id.max()) if len(released) else -1)
                assert event.reset_committed==bool(event.reset_requested and event.accepted)
                if not len(released):assert not event.accepted
            assert r['resets']==int(events.reset_committed.sum())
            assert r['training_visits']==int(pred.candidate_training_visits.sum())
            arm=r['contract']['arm'];assert arm['dataset']==ds and arm['detector']==r['detector']
            table=None
            if 'model_path' in arm:
                fitted=model_records[arm['model_path']]
                assert p.sha(ROOT/arm['model_path'])==arm['model_sha256']
                assert (fitted['dataset'],fitted['detector'],fitted['variant'],fitted['seed'])==(ds,r['detector'],arm['variant'],arm['seed'])
                table=fitted['table']
            elif 'table' in arm:table=arm['table']
            if table is not None:
                for event in events.loc[events.scored].itertuples():
                    expected_action='update' if table[event.rl_state][0]>=table[event.rl_state][1] else 'reset'
                    assert event.rl_action==expected_action
            if arm['name']=='selected_map':assert arm['table']==selected['policies'][ds][r['detector']]['table']
            rows.append({**{k:r[k] for k in ['stream','condition','detector','arm','feedback_fraction','delay_blocks',*METRICS]},
                         'stage':role,'prediction_sha256':r['files']['predictions.parquet'],
                         'summary_path':file.relative_to(ROOT).as_posix(),'summary_sha256':p.sha(file)})
        print(f'Checked {role}: {len(rows)} cumulative executions',flush=True)
    values=pd.DataFrame(rows); assert len(values)==sizes['executions']
    keys=['stream','condition','detector','arm','feedback_fraction','delay_blocks']
    expected_keys=set()
    for ds in cfg['datasets']:
        for detector in cfg['detectors']:
            for c in manifests[('validation',ds)]['conditions'].values():
                for bits in range(16):
                    expected_keys.add(('validation',ds,c['key'],detector,f'map_{bits:04b}',.05,1))
            arms=['no_reset','confirmed','error_rule','selected_map']+[f'q_{v}_{seed}' for v in ['alarm','alarm_persistence'] for seed in cfg['rl_seeds']]
            sensitivity_arms=[a for a in arms if not a.startswith('q_alarm_') or a.startswith('q_alarm_persistence_')]
            for c in manifests[('evaluation',ds)]['conditions'].values():
                for arm in arms:expected_keys.add(('evaluation',ds,c['key'],detector,arm,.05,1))
                if [c['mode'],c['level']] in cfg['sensitivity_conditions']:
                    for arm in sensitivity_arms:
                        for fraction in cfg['feedback_fractions']:
                            for delay in cfg['feedback_delays']:
                                expected_keys.add(('evaluation',ds,c['key'],detector,arm,fraction,delay))
    assert set(values[['stage',*keys]].itertuples(index=False,name=None))==expected_keys
    assert not values.duplicated(['stage',*keys]).any()
    controls=values[values.stage.eq('evaluation')&values.arm.eq('no_reset')]
    assert controls.groupby(['stream','condition','feedback_fraction','delay_blocks']).prediction_sha256.nunique().eq(1).all()
    for ds in cfg['datasets']:
        for detector in cfg['detectors']:
            sub=values[values.stage.eq('validation')&values.stream.eq(ds)&values.detector.eq(detector)]
            assert len(sub)==112
            rank=sub.groupby('arm').agg(f1=('host_macro_f1','mean'),resets=('resets','mean')).reset_index()
            winner=rank.sort_values(['f1','resets','arm'],ascending=[False,True,True]).iloc[0].arm
            assert winner==selected['policies'][ds][detector]['policy_id']
    for stage,count in [(stage,sizes[stage]) for stage in ['validation','evaluation','sensitivity']]:
        frame=pd.read_csv(OUT/'analysis'/stage/'metrics.csv');assert len(frame)==count
        sub=values[values.stage.eq('validation' if stage=='validation' else 'evaluation')]
        joined=frame.merge(sub,on=keys,validate='one_to_one',suffixes=('','_checked'));assert len(joined)==count
        for metric in METRICS:assert np.allclose(joined[metric],joined[metric+'_checked'],equal_nan=True,atol=1e-12,rtol=0)
    dest=OUT/'analysis/verification';dest.mkdir(parents=True,exist_ok=True)
    values.to_csv(dest/'checked_runs.csv',index=False)
    p.write(dest/'verification.json',{'status':'passed','trained_controllers':sizes['controllers'],'comparison_executions':sizes['executions'],
        'datasets':cfg['datasets'],'training_and_replay_partitions_checked':True,'frozen_policy_decisions_checked':True,
        'dataset_and_detector_identity_checked':True,'metrics_recalculated':True,'completed_utc':p.now(),
        'checker_sha256':p.sha(Path(__file__))})


def report():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    cfg=p.read(ROOT/'configs/study.json')
    sizes=counts(cfg)
    frame=pd.read_csv(OUT/'analysis/evaluation/metrics.csv')
    condition=frame.groupby(['stream','detector','arm','mode','level'])[METRICS].mean()
    means=condition.groupby(['stream','detector','arm']).mean()
    arms=['confirmed','error_rule','selected_map'] + [f'q_{v}_{seed}' for v in ['alarm','alarm_persistence'] for seed in cfg['rl_seeds']]
    labels=['Confirmed reset','Error rule','Selected mapping'] + [f'{n}-state Q / {seed}' for n in [2,4] for seed in cfg['rl_seeds']]
    names={'adwin':'ADWIN','hddm_w':'HDDM-W','hellinger':'Hellinger','d3_oof':'D3 OOF'}
    dataset_names={'SEA_A':'SEA','RBF_I':'RBF','radar':'RADAR'}
    folder=OUT/'analysis/figures';folder.mkdir(parents=True,exist_ok=True)
    fig,axes=plt.subplots(1,3,figsize=(16,8),sharey=True)
    for ax,ds in zip(axes,cfg['datasets']):
        for i,d in enumerate(cfg['detectors']):
            m=means.loc[(ds,d)]
            ax.plot([m.loc[a,'host_macro_f1']*100 for a in arms],np.arange(len(arms))+(i-1.5)*.14,
                    marker='o',linestyle='',label=names[d])
        ax.axvline(means.loc[(ds,cfg['detectors'][0],'no_reset'),'host_macro_f1']*100,
                   color='grey',linestyle='--',label='Shared no-reset reference')
        ax.set_title(dataset_names[ds]);ax.set_xlabel('Mean macro-F1 (%)');ax.grid(axis='x',alpha=.2)
        ax.set_yticks(range(len(arms)),labels)
    axes[0].invert_yaxis();handles,legends=axes[0].get_legend_handles_labels()
    fig.legend(handles,legends,loc='lower center',ncol=5,frameon=False)
    fig.suptitle('Detector–policy comparison: training and evaluation within each dataset')
    fig.tight_layout(rect=(0,.09,1,.96))
    for ext in ['pdf','png']:fig.savefig(folder/f'detector_policy_comparison.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)
    feedback=pd.read_csv(OUT/'analysis/sensitivity/metrics.csv')
    selected_arms=['no_reset','confirmed','error_rule','selected_map'] + [f'q_alarm_persistence_{seed}' for seed in cfg['rl_seeds']]
    for ds in cfg['datasets']:
        for delay in cfg['feedback_delays']:
            fig,axes=plt.subplots(2,2,figsize=(11,7),sharex=True)
            for ax,d in zip(axes.flat,cfg['detectors']):
                sub=feedback[feedback.stream.eq(ds)&feedback.detector.eq(d)&feedback.delay_blocks.eq(delay)]
                avg=sub.groupby(['arm','feedback_fraction','mode','level']).host_macro_f1.mean().groupby(['arm','feedback_fraction']).mean()
                for a in selected_arms:
                    line=avg.loc[a];ax.plot(line.index*100,line.values*100,marker='o',label=a)
                ax.set_title(names[d]);ax.set_xlabel('Usable protected feedback (%)');ax.set_ylabel('Macro-F1 (%)')
                ax.set_xticks([1,2,5]);ax.grid(alpha=.2)
            handles,legends=axes.flat[0].get_legend_handles_labels()
            fig.legend(handles,legends,loc='lower center',ncol=3,frameon=False)
            fig.suptitle(f'{dataset_names[ds]}: feedback delay {delay} block(s)')
            fig.tight_layout(rect=(0,.14,1,.96))
            for ext in ['pdf','png']:fig.savefig(folder/f'feedback_{ds}_delay_{delay}.{ext}',dpi=180,bbox_inches='tight')
            plt.close(fig)
    means.to_csv(OUT/'analysis/evaluation/balanced_condition_means.csv')
    lines=['# Detector and reset-policy comparison across all datasets','',
           'RQ1 and RQ2 remain unchanged. These controls and robustness analyses belong to Experiments 1 and 2. Experiment 3 retains its existing three-dataset evidence.','',
           'Phase 04 trained 120 controllers: three datasets, four detectors, two state representations and five training seeds. Every controller was trained on its own dataset development split and evaluated only on that dataset with its matching detector.','',
           'All 11,680 comparison executions passed trace checks: 1,344 validation, 5,656 default-setting evaluation and 4,680 additional feedback runs. The sensitivity table has 5,616 entries including 936 shared defaults. Training contains 3,360 episodes across 120 controllers. Repeated no-reset controls are not independent detector evidence.','',
           '| Dataset | Detector | Policy | Mean macro-F1 (%) | Difference from matched confirmed reset (pp) |',
           '| --- | --- | --- | ---: | ---: |']
    for ds in cfg['datasets']:
        for d in cfg['detectors']:
            m=means.loc[(ds,d)]
            for arm,label in zip(arms,labels):
                value=m.loc[arm,'host_macro_f1']*100; reference=m.loc['confirmed','host_macro_f1']*100
                lines.append(f'| {dataset_names[ds]} | {names[d]} | {label} | {value:.2f} | {value-reference:+.2f} |')
    lines+=['',
      'Grand means weight the seven clean/poisoning conditions equally. Synthetic condition means use five stream realizations. RADAR attacked-condition means use five attack assignments on one evaluation suffix; clean is a single shared realization. These are different replication units, and there is no pooled ranking across datasets. All RL seeds remain visible.','',
      'SEA and RBF use separate training, validation and evaluation seeds. RADAR uses rows [0,242000) for training, [243000,339000) for validation and [340000,484753) for evaluation, with 1,000-row gaps. Each partition has its own warm-up. Attacks are generated independently inside each partition, including replay donor selection. The RADAR results concern its evaluation suffix, not the original full-stream scoring population.','',
      'Each dataset/detector receives a separate validation-selected deterministic mapping. Selection uses macro-F1, whereas Q-learning uses protected accuracy; the comparison does not isolate optimisation algorithm alone or prove convergence. The four training passes are fixed. RADAR has longer episodes than the synthetic streams; training budgets are matched across detectors within each dataset.','',
      'Sensitivity varies usable feedback (1%, 2%, 5%) and delay (one or five blocks) on clean and severe label poisoning for every dataset and detector, retaining the same 5% reserve. Controllers stay frozen. RADAR malicious recall and benign false-positive rate are retained in the metric tables.','',
      'The RADAR vectors retain their known whole-input frequency and unresolved FastText provenance limitations; temporal policy splits do not establish causal feature extraction. Previously inspected benchmark outcomes are not presented as a wholly untouched independent replication. No claims about unseen real deployments or deep RL follow.','',
      '- [Final three-dataset detector comparison](../results/per_dataset_study/analysis/figures/detector_policy_comparison.pdf)',
      '- [Paired policy effects](../results/per_dataset_study/analysis/evaluation/paired_vs_confirmed.csv)',
      '- [Feedback effects](../results/per_dataset_study/analysis/sensitivity/paired_feedback_effects.csv)',
      '- [Trace verification](../results/per_dataset_study/analysis/verification/verification.json)','']
    (ROOT/'07_documentation/RESULTS.md').write_text('\n'.join(lines))
    p.write(OUT/'report_manifest.json',{'status':'complete','files':{f.relative_to(ROOT).as_posix():p.sha(f)
            for f in [ROOT/'07_documentation/RESULTS.md',*folder.glob('*')]},'script_sha256':p.sha(Path(__file__))})
    p.write(OUT/'completion.json',{'status':'complete','datasets':cfg['datasets'],'controllers':sizes['controllers'],
                                 'comparison_executions':sizes['executions'],'completed_utc':p.now(),'protocol_sha256':p.sha(OUT/'protocol.json')})


if __name__=='__main__':
    verify()
    report()
