"""Evaluate passive drift monitors on the completed, unscreened baseline streams."""
from pathlib import Path
import os
import sys
sys.dont_write_bytecode=True
for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[name]='1'
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from common import protocol as p
monitors=p.load_module('passive_monitors','02_detectors/monitors.py')
layout=p.load_module('passive_layout','06_analysis/dataset_summary/compact.py')
DATASETS={'SEA_A':'SEA','RBF_I':'RBF','radar':'RADAR'}
SERIES={'adwin':'ADWIN','hddm_w':'HDDM-W','hellinger':'Hellinger','d3_oof':'D3 OOF'}
CATEGORIES=[('clean','none'),('instance','moderate'),('instance','severe'),
            ('concept','moderate'),('concept','severe'),('splice','moderate'),('splice','severe')]
SPECS=[('alarm_rate','Alarm blocks (%)',100),
       ('poison_alarm_rate','Poison-exposed alarms (%)',100),
       ('unexposed_alarm_rate','Unexposed alarms (%)',100),
       ('event_recall','Drift-event recall (%)',100),
       ('detection_delay','Detection delay (blocks)',1),
       ('seconds_per_100k','Monitor time (s / 100k rows)',1)]


def figure_specs(dataset):
    """Choose four supported metrics without adding empty drift panels for RADAR."""
    names=['alarm_rate','poison_alarm_rate','unexposed_alarm_rate','seconds_per_100k'] if dataset=='RADAR' else [
           'alarm_rate','event_recall','detection_delay','seconds_per_100k']
    return [spec for spec in SPECS if spec[0] in names]


def fingerprint(file,expected,inputs):
    """Reject changed inputs and record the precise files used."""
    digest=p.sha(file)
    if digest!=expected:
        raise ValueError(f'Changed source: {file}')
    inputs[file.resolve().relative_to(ROOT).as_posix()]=digest


def observe(values,errors,reserved,config,detector):
    """Replay only observable features and errors, without labels from scoring truth."""
    records=[]
    size=config['block_size']
    for block,left in enumerate(range(0,len(values),size)):
        right=min(left+size,len(values))
        if left<config['warmup_rows']:
            continue
        ids=np.arange(left,right)[~reserved[left:right]]
        tick=time.perf_counter()
        fired=detector.update(values[ids],errors[ids])
        elapsed=time.perf_counter()-tick
        records.append({'block':block,'start_row':left,'end_row':right,
                        'raw_fire':bool(fired),'seconds':elapsed,'observed_rows':len(ids)})
    return pd.DataFrame(records)


def score(events,truth,meta,reserved,config):
    """Score saved alarms offline; exposure is not a verified poisoning diagnosis."""
    size=config['block_size']
    poison=(truth.atk_flip.to_numpy()|truth.atk_burst.to_numpy())&~reserved
    exposed=np.logical_or.reduceat(poison,np.arange(0,len(truth),size))
    # Match the original two-block carry-over exclusion for background alarms.
    excluded=exposed.copy()
    for shift in [1,2]:
        excluded[shift:] |= exposed[:-shift]
    windows=[(a,min(len(truth),b+20*size)) for a,b in meta['transition_intervals']]
    for a,b in windows:
        excluded[a//size:(b-1)//size+1]=True
    blocks=events.block.to_numpy(dtype=int)
    events=events.copy()
    events['poison_exposed']=exposed[blocks]
    events['unexposed']=~excluded[blocks]
    fired=events.raw_fire.to_numpy(dtype=bool)
    target=events.poison_exposed.to_numpy(dtype=bool)
    background=events.unexposed.to_numpy(dtype=bool)
    delays=[]
    if meta['supports_event_metrics']:
        available=list(events.loc[events.raw_fire,'end_row'])
        for a,b in windows:
            hit=next((t for t in available if a<t<=b),None)
            if hit is not None:
                available.remove(hit)
                delays.append((hit-a)/size)
    def rate(n,d):
        """Leave a missing denominator undefined."""
        return float(n/d) if d else None
    metrics={'alarm_rate':float(fired.mean()),'alarm_blocks':int(fired.sum()),
             'monitored_blocks':len(events),'poison_exposed_blocks':int(target.sum()),
             'poison_alarm_blocks':int((fired&target).sum()),
             'poison_alarm_rate':rate((fired&target).sum(),target.sum()),
             'unexposed_blocks':int(background.sum()),
             'unexposed_alarm_blocks':int((fired&background).sum()),
             'unexposed_alarm_rate':rate((fired&background).sum(),background.sum()),
             'event_recall':rate(len(delays),len(windows)) if meta['supports_event_metrics'] else None,
             'detected_events':len(delays) if meta['supports_event_metrics'] else None,
             'scheduled_events':len(windows) if meta['supports_event_metrics'] else None,
             'detection_delay':float(np.mean(delays)) if delays else None,
             'seconds_per_100k':float(events.seconds.sum()/len(truth)*100000)}
    return events,metrics


def condition_task(task):
    """Run all four independent monitors on one immutable baseline condition."""
    root=ROOT/'results/evaluation'
    meta,run,config,hashes,completion_digest,output=task
    output=Path(output)
    inputs={}
    folder=root/run['relative_folder']
    fingerprint(folder/'summary.json',completion_digest,inputs)
    summary=p.read(folder/'summary.json')
    if summary['policy']!='none' or summary['guarded'] or summary['resets'] or summary['detector']!='none':
        raise ValueError('Expected an unscreened baseline without a response')
    prediction_file=folder/'predictions.parquet'
    fingerprint(prediction_file,summary['files']['predictions.parquet'],inputs)
    for key in ['observations','audit','truth']:
        fingerprint(root/'01_attacks'/meta[key],hashes[meta[key]],inputs)
    observation=pd.read_parquet(root/'01_attacks'/meta['observations'])
    pred=pd.read_parquet(prediction_file,columns=['row_id','prediction','audit_reserved'])
    np.testing.assert_array_equal(pred.row_id,np.arange(meta['rows']))
    if len(observation)!=meta['rows']:
        raise ValueError('Incomplete stream')
    if 'row_id' in observation:
        np.testing.assert_array_equal(observation.row_id,pred.row_id)
    reserved=pred.audit_reserved.to_numpy(dtype=bool)
    audit_ids=pd.read_parquet(root/'01_attacks'/meta['audit'],columns=['row_id']).row_id.to_numpy()
    np.testing.assert_array_equal(np.flatnonzero(reserved),np.sort(audit_ids))
    features=[c for c in observation if c.startswith('f') and c[1:].isdigit()]
    values=observation[features].to_numpy(dtype=float)
    errors=pred.prediction.to_numpy()!=observation.label.to_numpy()
    alarm_tables={}
    for name in SERIES:
        detector=monitors.make(name,seed=p.seed('monitor',meta['key'],name))
        alarm_tables[name]=observe(values,errors,reserved,config,detector)
    # Evaluator-only truth is loaded after the detectors have finished.
    truth=pd.read_parquet(root/'01_attacks'/meta['truth'])
    np.testing.assert_array_equal(truth.row_id,pred.row_id)
    records=[]
    for name,events in alarm_tables.items():
        events,metrics=score(events,truth,meta,reserved,config)
        destination=output/'runs'/DATASETS[meta['stream']]/meta['mode']/meta['level']/meta['key']/name
        destination.mkdir(parents=True,exist_ok=True)
        events.to_parquet(destination/'alarms.parquet',index=False)
        record={'condition':meta['key'],'dataset':DATASETS[meta['stream']],
                'base_key':meta['base_key'],'attack_seed':meta['attack_seed'],
                'mode':meta['mode'],'level':meta['level'],'series':name,
                'category':CATEGORIES.index((meta['mode'],meta['level'])),
                'source_run':summary['run_id'],'rows':len(observation),
                'audit_rows':int(reserved.sum()),'policy':'none','learner_resets':0,
                'events_path':(destination/'alarms.parquet').relative_to(output).as_posix(),**metrics}
        p.write(destination/'summary.json',record)
        records.append(record)
    return records,inputs


def report(raw,output):
    """Aggregate independent conditions and draw one four-panel PDF per dataset."""
    long=raw.melt(id_vars=['dataset','category','mode','level','series'],
                  value_vars=[s[0] for s in SPECS],var_name='metric',value_name='value')
    points=long.groupby(['dataset','category','mode','level','series','metric']).agg(
        mean=('value','mean'),sd=('value','std'),defined=('value','count'),runs=('value','size')).reset_index()
    points.to_csv(output/'plot_values.csv',index=False)
    entries=[]
    for dataset in DATASETS.values():
        part=points[points.dataset.eq(dataset)]
        entry=layout.draw(part,figure_specs(dataset),'','',SERIES,output/f'{dataset}_detector_baseline','')
        entry['dataset']=dataset
        entries.append(entry)
    return entries


def main():
    """Complete the detector-only evaluation without rerunning or changing the learner."""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args()
    root=ROOT/'results/evaluation'
    request=p.read(root/'request.json')
    if p.read(root/'verification.json')['status']!='passed' or p.sources()!=request['sources']:
        raise ValueError('The original experiment must verify unchanged')
    metrics_file=root/'06_analysis/metrics.csv'
    if p.sha(metrics_file)!=p.read(root/'06_analysis/manifest.json')['outputs']['metrics.csv']:
        raise ValueError('Metrics changed')
    frame=pd.read_csv(metrics_file)
    frame=frame[frame.policy.eq('none')&~frame.guarded&frame.detector.eq('none')]
    if len(frame)!=61:
        raise ValueError('Expected 61 complete baseline conditions')
    manifest=p.read(root/'01_attacks/manifest.json')
    completion=p.read(root/'05_runs/completion.json')
    output=root/'06_analysis/presentation/detector_baseline'
    output.mkdir(parents=True,exist_ok=True)
    tasks=[(manifest['conditions'][run.condition],run.to_dict(),request['config'],manifest['outputs'],
            completion['summaries'][run.run_id],str(output)) for _,run in frame.iterrows()]
    records,inputs=[],{}
    p.write(output/'status.json',{'status':'running','conditions_complete':0,'conditions_total':61})
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(condition_task,task) for task in tasks]
        for count,future in enumerate(as_completed(futures),1):
            result,used=future.result()
            records.extend(result)
            inputs.update(used)
            p.write(output/'status.json',{'status':'running','conditions_complete':count,'conditions_total':61})
            print(f'{count}/61 conditions: {result[0]["condition"]}',flush=True)
    raw=pd.DataFrame(records).sort_values(['dataset','category','condition','series'])
    raw.to_csv(output/'run_metrics.csv',index=False)
    entries=report(raw,output)
    (output/'README.md').write_text('''# Detector baseline: clean versus six poisoned conditions

Each PDF contains four panels in a 2 x 2 layout and four detector lines: ADWIN, HDDM-W, Hellinger and D3 OOF. Dimensions are 170 x 128 mm, fitting the A4 text width and less than half a page in height. There are no surrounding titles or explanatory footers. The dataset is in the filename. Bars show one sample standard deviation, not confidence intervals. Points are categorical comparisons, not time trajectories.

This is an additional passive-detector evaluation on the completed unscreened, no-reset learner runs. It is separate from the 1,464-run policy benchmark. All 61 baseline conditions are processed by all four monitors (244 monitor evaluations). The learner is not rerun. Detector state evolves normally, including each detector's internal alarm handling; no response resets the learner or recreates the detector. Predictions and observed, potentially poisoned labels provide error feedback. Feature detectors receive the observed features. The protected 5% reservations and warm-up remain excluded exactly as in the original baseline. Evaluator truth is used only after all alarms are generated. All 484,753 RADAR positions are retained in the input stream.

SEA and RBF show alarm rate, drift-event recall, detection delay and monitor time. RADAR shows alarm rate, poison-exposed alarm rate, unexposed alarm rate and monitor time. Entirely unsupported metrics are omitted from the figures; occasional undefined condition points remain N/A. All six calculated metrics are retained in the tables and defined below:

- Alarm blocks: percentage of monitored blocks with at least one alarm.
- Poison-exposed alarms: percentage of blocks containing modified non-reserved observations that raise an alarm. Clean is undefined. This is an association with known exposure, not observation-level poison classification or causal evidence that an alarm was caused by poisoning.
- Unexposed alarms: percentage of eligible background blocks with an alarm. Exclude poison-exposed blocks, their next two blocks, and synthetic drift matching windows. For RADAR, natural drift is unknown, so these cannot be labelled false alarms. The same caution applies to delayed poisoning effects beyond the two-block exclusion.
- Drift-event recall: fraction of scheduled natural synthetic drift events matched by an alarm. Each event window starts at the transition start and ends 20 blocks after its end. Alarms are matched once in order. RADAR is undefined because there are no verified drift-event references.
- Detection delay: mean delay in 1,000-row block units from transition start to the first matched block-end alarm, conditional on detection. Missed events are omitted from delay and remain visible through recall. RADAR is undefined. Small delay alone does not establish better performance.
- Monitor time: detector update seconds per 100,000 source positions, excluding learner training, loading and offline scoring. Timing comes from this additional evaluation with four workers; it is not the original full pipeline runtime.

There are three independent synthetic stream realizations per plotted condition. RADAR uses one capture, with three attack assignments per poisoned condition and one clean run; its clean point has no repeatability error bar. A missing denominator or undetected event remains undefined rather than zero. Error bars use only defined runs, with counts in plot_values.csv.

Without an intervention, detector choice cannot change classifier accuracy, F1 or other predictive scores. Those remain in the separate baseline_clean_vs_poisoned figures. The next stages compare fixed response policies and then RL; they are not included in this detector baseline.

Reproduce from the project root with `python -B 06_analysis/dataset_summary/detector_baseline.py --workers 4`. run_metrics.csv records all 244 evaluations. The runs folder contains block alarm records and summaries by dataset, attack, severity, condition and detector. manifest.json fingerprints source inputs, analysis code and outputs. PNG files are previews; the requested deliverables are the three individual PDFs. No HTML is generated.
''',encoding='utf-8')
    p.write(output/'status.json',{'status':'complete','conditions_complete':61,'conditions_total':61,'monitor_evaluations':244})
    p.write(output/'manifest.json',{'created_utc':p.now(),'evaluation':'passive detectors on frozen no-response predictions',
            'workers':args.workers,'environment':p.environment(),'script_sha256':p.sha(__file__),
            'layout_sha256':p.sha(ROOT/'06_analysis/dataset_summary/compact.py'),
            'monitor_sha256':p.sha(ROOT/'02_detectors/monitors.py'),'request_sha256':p.sha(root/'request.json'),
            'inputs':inputs,'entries':entries,'outputs':{f.relative_to(output).as_posix():p.sha(f)
            for f in output.rglob('*') if f.is_file() and f.name!='manifest.json'}})
    print('Complete: 244 monitor evaluations and three four-panel PDFs.',flush=True)


if __name__=='__main__':
    main()
