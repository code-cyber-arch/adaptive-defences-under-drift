"""Put all four detectors on the same axes under matched response settings."""
from pathlib import Path
import sys
sys.dont_write_bytecode=True
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from common import protocol as p
fixed=p.load_module('detector_comparison_source','06_analysis/dataset_summary/fixed_responses.py')
PROFILES={'immediate_unscreened':'Immediate reset, unscreened',
          'confirmed_unscreened':'Confirmed reset, unscreened',
          'immediate_screened':'Immediate reset, screened',
          'confirmed_screened':'Confirmed reset, screened'}


def document(output,entries):
    """Provide one entry point to the detector comparison and its interpretation."""
    text='''# Compare the detectors directly

Each panel contains four detector lines: ADWIN, HDDM-W, Hellinger and D3 OOF. Colours, markers and line styles identify the same detectors throughout. The response and screening setting is fixed within each PDF. These figures compare classifier performance achieved with each detector under a matched response rule; they do not label classifier accuracy as accuracy of drift detection itself.

Start with **immediate reset, unscreened** to compare detectors under the simplest response. Next compare confirmed reset with screening still disabled. Finally inspect the screened versions. Every setting is reported; this order is for reading, not a selection of favourable outcomes. No-reset references belong to the preceding learner comparison because they have no detector.

Each PDF has the same four panels: accuracy, macro-F1, macro recall and ROC-AUC. The x-axis shows clean plus six poisoned conditions. Higher values are better for these four predictive metrics. Four detector lines now share each panel; there is no need to switch files to compare detectors. There are twelve individual PDFs: three datasets x two response rules x two screening settings. Dataset subfolders keep the files together; PNG previews are stored separately. Each figure is 170 x 128 mm, in a 2 x 2 layout, without surrounding headings or footers.

Points and SD bars are reused exactly from the verified fixed-response tables. All four detectors use identical scoring observations within each matched stream/attack-assignment cohort. Y-axis ranges are shared across every response/screening setting within each dataset and metric. No policies, severities or detectors are pooled into a single score. No learner, detector or controller is rerun.

Synthetic points average three stream realizations. RADAR points average three matched attack assignments on one capture; clean predictions are scored on the three common masks. Error bars are one sample SD, not confidence intervals or evidence of statistical significance. ROC-AUC is macro one-versus-rest for synthetic data and malicious-versus-benign for RADAR. Host scoring excludes protected reservations, warm-up and all replacement intervals in the matched seven-condition comparison. The full RADAR input remains 484,753 rows. This is comparative predictive performance under drift and poisoning, not observation-level poison classification.

Read across conditions to see sensitivity to attack type and severity. Compare detector lines vertically to assess detector choice while holding the response fixed. Then compare the corresponding screened/unscreened PDFs to see whether screening changes the detector ordering. Protection, withholding and computational costs remain available in the fixed_responses companion figures; predictive scores alone do not establish a universally best defence.

Reproduce with `python -B 06_analysis/dataset_summary/compare_detectors.py`. plot_values.csv includes every displayed mean, SD, source response profile and defined count. The manifest links the source tables, unchanged experiment request, rendering code and generated files.

| Response setting | SEA | RBF | RADAR |
|---|---|---|---|
'''
    for profile,label in PROFILES.items():
        links=[f'[{dataset} PDF]({dataset}/{dataset}_{profile}.pdf)' for dataset in fixed.DATASETS.values()]
        text+='| '+label+' | '+' | '.join(links)+' |\n'
    (output/'README.md').write_text(text,encoding='utf-8')


def main():
    """Transpose the verified policy comparisons into a detector-first presentation."""
    root=ROOT/'results/evaluation'
    request=p.read(root/'request.json')
    if p.sources()!=request['sources'] or p.read(root/'verification.json')['status']!='passed':
        raise ValueError('An unchanged, verified benchmark is required')
    source=root/'06_analysis/presentation/fixed_responses'
    source_manifest=p.read(source/'manifest.json')
    file=source/'tables/plot_values.csv'
    if p.sha(file)!=source_manifest['outputs']['tables/plot_values.csv']:
        raise ValueError('Fixed-response plot values changed')
    source_points=pd.read_csv(file)
    points=source_points[source_points.series.isin(PROFILES)&
                         source_points.metric.isin([s[0] for s in fixed.PREDICTIVE])].copy()
    points=points.rename(columns={'series':'profile','comparison_detector':'series'})
    if len(points)!=1344 or not points.defined.eq(3).all():
        raise ValueError('Incomplete detector comparison')
    output=root/'06_analysis/presentation/detector_comparison'
    output.mkdir(parents=True,exist_ok=True)
    points.to_csv(output/'plot_values.csv',index=False)
    entries=[]
    for dataset in fixed.DATASETS.values():
        dataset_points=points[points.dataset.eq(dataset)]
        limits=fixed.plot_limits(dataset_points,fixed.PREDICTIVE)
        folder=output/dataset
        previews=output/'previews'/dataset
        folder.mkdir(exist_ok=True)
        previews.mkdir(parents=True,exist_ok=True)
        for profile in PROFILES:
            selected=dataset_points[dataset_points.profile.eq(profile)]
            for metric,_,_ in fixed.PREDICTIVE:
                groups=selected[selected.metric.eq(metric)].groupby('series').category.apply(list)
                if set(groups.index)!=set(fixed.DETECTORS) or any(sorted(cats)!=list(range(7)) for cats in groups):
                    raise ValueError('Missing detector or condition in a panel')
            entry=fixed.layout.draw(selected,fixed.PREDICTIVE,'','',fixed.DETECTORS,
                                    folder/f'{dataset}_{profile}','',limits=limits)
            preview_source=(folder/entry['png']).resolve()
            preview_target=(previews/entry['png']).resolve()
            if not preview_source.is_relative_to(output.resolve()) or not preview_target.is_relative_to(output.resolve()):
                raise ValueError('Preview path outside the output folder')
            preview_source.replace(preview_target)
            entry.update(dataset=dataset,profile=profile,detectors=list(fixed.DETECTORS),
                         pdf=(folder/entry['pdf']).relative_to(output).as_posix(),
                         png=preview_target.relative_to(output).as_posix(),y_limits=limits)
            entries.append(entry)
    document(output,entries)
    p.write(output/'manifest.json',{'created_utc':p.now(),'script_sha256':p.sha(__file__),
            'layout_sha256':p.sha(ROOT/'06_analysis/dataset_summary/compact.py'),
            'source_reporter_sha256':p.sha(ROOT/'06_analysis/dataset_summary/fixed_responses.py'),
            'request_sha256':p.sha(root/'request.json'),
            'inputs':{file.relative_to(ROOT).as_posix():p.sha(file),
                      (source/'manifest.json').relative_to(ROOT).as_posix():p.sha(source/'manifest.json')},
            'entries':entries,'outputs':{f.relative_to(output).as_posix():p.sha(f) for f in output.rglob('*')
            if f.is_file() and f.name!='manifest.json'}})
    print('Created 12 detector-comparison PDFs, with four detector lines in each panel.',flush=True)


if __name__=='__main__':
    main()
