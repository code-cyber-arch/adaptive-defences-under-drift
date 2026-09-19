"""Report-sized figures with the thesis's established colour and panel style."""
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from common import protocol as p
from . import style
from .balance import CATEGORIES
NAMES={'SEA_A':'SEA','RBF_I':'RBF','radar':'RADAR'}
DETECTORS={'adwin':'ADWIN','hddm_w':'HDDM-W','hellinger':'Hellinger','d3_oof':'D3 OOF'}
CORE=[('host_accuracy','Accuracy (%)',100),('host_macro_f1','Macro-F1 (%)',100),
      ('poison_admission_rate','Poison admission (%)',100),('clean_withhold_rate','Legitimate withholding (%)',100)]
PROTECTION=[('poison_admission_rate','Poison admission (%)',100),('clean_withhold_rate','Legitimate withholding (%)',100),
            ('resets_per_100k','Resets / 100k rows',1),('training_visits_per_100k','Training work (visits / 100k rows)',1)]
PALETTE=['#0072B2','#D55E00','#009E73','#CC79A7','#532b88','#64748b']
RL_SERIES={'no_reset':'No reset','confirmed':'Confirmed reset','error_rule':'Error rule',
           'selected_map':'Selected mapping','alarm':'Alarm-only RL','alarm_persistence':'Persistence RL'}
RL_STYLES={key:dict(color=color,marker=marker,linestyle=line,markerfacecolor=color)
           for key,color,marker,line in [('no_reset','#0072B2','o','--'),('confirmed','#009E73','^','--'),
           ('error_rule','#D55E00','s',':'),('selected_map','#64748b','P',':'),
           ('alarm','#CC79A7','D','-.'),('alarm_persistence','#532b88','v','-')]}


def category(frame):
    result=frame.copy();result['category']=[CATEGORIES.index((m,l)) for m,l in zip(result['mode'],result.level)]
    return result


def points(frame,specs):
    """Average fitting seeds within input condition before reporting input variation."""
    metrics=[m for m,_,_ in specs]
    within=frame.groupby(['condition','category','series'],dropna=False)[metrics].mean().reset_index()
    melted=within.melt(id_vars=['condition','category','series'],value_vars=metrics,var_name='metric',value_name='value')
    return melted.groupby(['category','series','metric'],dropna=False).agg(mean=('value','mean'),sd=('value','std'),runs=('value','size'),defined=('value','count')).reset_index()


def default_styles(series):
    return {key:dict(color=PALETTE[i%len(PALETTE)],marker=['o','s','^','D','v','P'][i%6],linestyle=['-','--','-.',':'][i%4]) for i,key in enumerate(series)}


def draw(frame,specs,series,path,title,book,entries,styles=None,labels=None,delta=False):
    values=points(frame,specs);values.to_csv(path.with_suffix('.csv'),index=False)
    limits=None
    if delta:
        limits={}
        for metric,_,scale in specs:
            group=values[values.metric.eq(metric)];a=group['mean']*scale;b=group.sd.fillna(0)*scale
            extent=max(abs(a-b).max(),abs(a+b).max(),.1)
            if np.isfinite(extent):limits[metric]=(-extent*1.15,extent*1.15)
    record=style.draw(values,specs,title,'',series,path,'',styles=styles or default_styles(series),limits=limits,labels=labels,book=book)
    record.update(title=title,source_table=path.with_suffix('.csv').relative_to(p.ROOT).as_posix(),path=path.with_suffix('.pdf').relative_to(p.ROOT).as_posix(),
                  error_bars='Descriptive SD across input realizations/attack assignments after averaging fitting seeds within each input; not confidence intervals. Shared RADAR clean input has one unique realization.')
    entries.append(record)


def render(out,pilot=False,rl_source=None):
    folder=out/'analysis/figures';folder.mkdir(parents=True,exist_ok=True)
    entries=[]
    fixed=category(pd.read_csv(out/'analysis/fixed_responses/metrics.csv'))
    filtered=category(pd.read_csv(out/'analysis/filter_evaluation/metrics.csv'))
    passive=category(pd.read_csv(out/'analysis/passive_detectors/metrics.csv'))
    metadata={}
    for ds in NAMES:
        root=out/'inputs/evaluation'/ds if pilot else out/'stages/evaluation'/ds
        metadata.update(p.read(root/'01_attacks/manifest.json')['conditions'])
    for frame in [fixed,filtered]:
        rows=frame.condition.map(lambda k:metadata[k]['rows'])
        frame['resets_per_100k']=frame.resets/rows*100000
        frame['training_visits_per_100k']=frame.training_visits/rows*100000
    fixed_series={};fixed_styles={}
    for policy,label,color,marker in [('none','No reset','#0072B2','o'),('immediate','Immediate','#D55E00','s'),('confirmed','Confirmed','#009E73','^')]:
        for guarded in [False,True]:
            key=f'{policy}_{guarded}';fixed_series[key]=label+(' / screened' if guarded else ' / unscreened')
            fixed_styles[key]=dict(color=color,marker=marker,linestyle='-' if guarded else '--',markerfacecolor=color if guarded else 'white')
    fixed['series']=fixed.policy+'_'+fixed.guarded.astype(str)
    first=None
    with PdfPages(folder/'fixed_response_comparison.pdf') as book:
        for ds,label in NAMES.items():
            for detector,det_label in DETECTORS.items():
                frame=fixed[fixed.stream.eq(ds)&fixed.detector.isin([detector,'none'])]
                for name,specs in [('prediction_screening',CORE),('protection_work',PROTECTION)]:
                    path=folder/f'{label}_{detector}_{name}';first=first or path
                    draw(frame,specs,fixed_series,path,f'{label} / {det_label}: fixed responses',book,entries,fixed_styles)
    shutil.copy2(first.with_suffix('.png'),folder/'fixed_response_comparison.png')
    passive['series']=passive.detector
    first=None
    with PdfPages(folder/'passive_detector_comparison.pdf') as book:
        for ds,label in NAMES.items():
            specs=[('alarm_rate','Alarm blocks (%)',100),('poison_alarm_rate','Poison-exposed alarms (%)',100),('unexposed_alarm_rate','Unexposed alarms (%)',100),('seconds_per_100k','Monitor time (s / 100k rows)',1)] if ds=='radar' else [
                ('alarm_rate','Alarm blocks (%)',100),('event_recall','Drift-event recall (%)',100),('detection_delay','Detection delay (blocks)',1),('seconds_per_100k','Monitor time (s / 100k rows)',1)]
            path=folder/f'{label}_detector_baseline';first=first or path
            draw(passive[passive.stream.eq(ds)],specs,DETECTORS,path,f'{label}: passive detectors',book,entries)
    shutil.copy2(first.with_suffix('.png'),folder/'passive_detector_comparison.png')
    operational={'none':'No filter','learned_features':'Features','learned_features_label':'Features + label'}
    controls={'none':'No filter','oracle_rows':'Oracle rows','random_rows':'Random rows','oracle_blocks':'Oracle blocks','random_blocks':'Random blocks'}
    filtered['series']=filtered['filter']
    first=None
    with PdfPages(folder/'filter_comparison.pdf') as book:
        for ds,label in NAMES.items():
            for detector,det_label in {**DETECTORS,'none':'No reset'}.items():
                frame=filtered[filtered.stream.eq(ds)&filtered.detector.eq(detector)]
                for suffix,series in [('learned',operational),('controls',controls)]:
                    if detector=='none' and suffix=='controls':continue
                    path=folder/f'{label}_{detector}_filter_{suffix}';first=first or path
                    draw(frame[frame.series.isin(series)],CORE,series,path,f'{label} / {det_label}: filtering',book,entries)
    shutil.copy2(first.with_suffix('.png'),folder/'filter_comparison.png')
    reset=category(pd.read_csv(out/'analysis/full_study/filter_reset_effects.csv'));reset['series']=reset['filter']
    specs=[(metric+'_reset_difference',label+' difference',scale) for metric,label,scale in CORE]
    with PdfPages(folder/'filter_reset_advantage.pdf') as book:
        for ds,label in NAMES.items():
            for detector,det_label in DETECTORS.items():
                part=reset[reset.stream.eq(ds)&reset.detector.eq(detector)]
                draw(part,specs,operational,folder/f'{label}_{detector}_filter_reset_advantage',f'{label} / {det_label}: confirmed minus no reset',book,entries,delta=True)
    baseline=category(pd.read_csv(out/'analysis/full_study/baseline_attack_damage.csv'));baseline['series']='baseline'
    if {'macro_precision','macro_recall'}.issubset(baseline):
        specs=[('host_accuracy','Accuracy (%)',100),('host_macro_f1','Macro-F1 (%)',100),('macro_precision','Macro precision (%)',100),('macro_recall','Macro recall (%)',100)]
        first=None
        with PdfPages(folder/'baseline_attack_damage.pdf') as book:
            for ds,label in NAMES.items():
                path=folder/f'{label}_clean_vs_poisoned';first=first or path
                draw(baseline[baseline.stream.eq(ds)],specs,{'baseline':'Baseline'},path,f'{label}: clean and poisoned baseline',book,entries)
        shutil.copy2(first.with_suffix('.png'),folder/'baseline_attack_damage.png')
    source=out if rl_source is None else Path(rl_source)
    rl_file=source/'analysis/evaluation/metrics.csv'
    if rl_file.exists():
        rl=category(pd.read_csv(rl_file));rl['series']=rl.arm.map(lambda a:'alarm_persistence' if a.startswith('q_alarm_persistence_') else 'alarm' if a.startswith('q_alarm_') else a)
        first=None
        with PdfPages(folder/'detector_policy_comparison.pdf') as book:
            for ds,label in NAMES.items():
                for detector,det_label in DETECTORS.items():
                    part=rl[rl.stream.eq(ds)&rl.detector.eq(detector)]
                    path=folder/f'{label}_{detector}_RL_comparison';first=first or path
                    draw(part,CORE,RL_SERIES,path,f'{label} / {det_label}: learned reset policies',book,entries,RL_STYLES)
                    for variant in ['alarm','alarm_persistence']:
                        selected=part[part.arm.str.startswith(f'q_{variant}_') & (part.series.eq(variant)) | part.arm.eq('confirmed')].copy()
                        selected['series']=selected.arm
                        series={'confirmed':'Confirmed reset'}|{a:'Seed '+a.rsplit('_',1)[-1] for a in sorted(selected.arm.unique()) if a!='confirmed'}
                        draw(selected,CORE,series,folder/f'{label}_{detector}_{variant}_seeds',f'{label} / {det_label}: {variant} fitting seeds',book,entries)
        shutil.copy2(first.with_suffix('.png'),folder/'detector_policy_comparison.png')
        feedback=pd.read_csv(source/'analysis/sensitivity/metrics.csv')
        feedback['series']=feedback.arm.map(lambda a:'alarm_persistence' if a.startswith('q_alarm_persistence_') else 'alarm' if a.startswith('q_alarm_') else a)
        series=RL_SERIES
        for ds,label in NAMES.items():
            for delay in [1,5]:
                first=None
                with PdfPages(folder/f'feedback_{ds}_delay_{delay}.pdf') as book:
                    for detector,det_label in DETECTORS.items():
                        part=feedback[feedback.stream.eq(ds)&feedback.detector.eq(detector)&feedback.delay_blocks.eq(delay)].copy()
                        # Equal clean/poison weighting and fitting-seed averaging before descriptive input SD.
                        cols=[m for m,_,_ in CORE]
                        part=part.groupby(['condition','mode','level','feedback_fraction','series'])[cols].mean().reset_index()
                        part=part.groupby(['feedback_fraction','series','mode','level'])[cols].mean().groupby(['feedback_fraction','series']).mean().reset_index()
                        part['condition']=part.feedback_fraction.astype(str);part['category']=part.feedback_fraction.map({.01:0,.02:1,.05:2})
                        path=folder/f'{label}_{detector}_feedback_delay_{delay}';first=first or path
                        draw(part,CORE,series,path,f'{label} / {det_label}: feedback delay {delay} blocks',book,entries,RL_STYLES,labels=['1%','2%','5%'])
                        entries[-1]['error_bars']='No error bars: equal-condition means after averaging fitting seeds and input realizations.'
                shutil.copy2(first.with_suffix('.png'),folder/f'feedback_{ds}_delay_{delay}.png')
    p.write(folder/'report_figure_index.json',dict(style='Existing compact thesis layout: 170 x 128 mm, 2 x 2 panels',figures=entries,
        book_note='Combined PDFs collect individual report figures; companion PNGs preview their first figure.',
        source_sha256=p.sha(Path(__file__))))
    captions=['# Report figure guide','',
              'Use the individual dataset/detector PDFs at 170 mm width. They follow the existing four-panel report style. Combined PDFs are figure collections; PNGs with the collection name preview the first figure.','',
              'Curves average all declared fitting seeds within each evaluation input. Error bars show descriptive standard deviation across input realizations (synthetic streams) or attack assignments (one RADAR capture), not confidence intervals. Clean RADAR has one unique input. Dedicated RL seed figures retain every seed separately. Feedback figures show equal-condition means without error bars.','']
    for entry in entries:captions.append(f"- **{entry['title']}**: [{Path(entry['path']).name}]({Path(entry['path']).name})")
    (folder/'FIGURE_GUIDE.md').write_text('\n'.join(captions)+'\n')
