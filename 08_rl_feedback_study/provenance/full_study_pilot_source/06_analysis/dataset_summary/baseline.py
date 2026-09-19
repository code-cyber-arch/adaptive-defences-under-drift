"""Plot applicable baseline metrics for clean and six poisoned streams."""
from pathlib import Path
from html import escape
import sys
sys.dont_write_bytecode = True
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, average_precision_score
from common import protocol as p

DATASETS={'SEA_A':'SEA','RBF_I':'RBF','radar':'RADAR'}
CATEGORIES=[('clean','none'),('instance','moderate'),('instance','severe'),
            ('concept','moderate'),('concept','severe'),('splice','moderate'),('splice','severe')]
TICKS=['Clean','Instance\n15%','Instance\n25%','Concept\n15%','Concept\n25%',
       'Splice\n15%','Splice\n25%']

PREDICTIVE=[('accuracy','Accuracy (%)',100),('macro_f1','Macro-F1 (%)',100),
            ('macro_precision','Macro precision (%)',100),('macro_recall','Macro recall (%)',100),
            ('roc_auc','ROC-AUC',1),('average_precision','Average precision',1)]
DISPLAY_METRICS=[spec for spec in PREDICTIVE if spec[0] in
                 ['accuracy','macro_f1','macro_recall','roc_auc']]
RADAR=[('malicious_precision','Malicious precision (%)',100),
       ('malicious_recall','Malicious recall (%)',100),('benign_fpr','Benign FPR (%)',100)]
OPERATIONAL=[('poison_admission_rate','Poison admission (%)',100),
             ('clean_withhold_rate','Legitimate withholding (%)',100),
             ('training_visits_per_100k','Training work\n(k visits / 100k rows)',.001),
             ('seconds_per_100k','Runtime\n(seconds / 100k rows)',1)]


def predictive_metrics(labels, predictions, probabilities, classes, benign):
    """Calculate class-macro scores and probability metrics on common host rows."""
    precision,recall,f1,_=precision_recall_fscore_support(
        labels,predictions,labels=classes,average='macro',zero_division=0)
    result={'macro_precision':float(precision),'macro_recall':float(recall),'macro_f1':float(f1),
            'malicious_precision':np.nan,'malicious_recall':np.nan,'benign_fpr':np.nan}
    if benign is not None and not pd.isna(benign):
        positive=labels!=benign
        predicted_positive=(predictions>=0)&(predictions!=benign)
        true_positive=int(np.sum(positive&predicted_positive))
        result['malicious_precision']=true_positive/predicted_positive.sum() if predicted_positive.any() else np.nan
        result['malicious_recall']=true_positive/positive.sum() if positive.any() else np.nan
        result['benign_fpr']=float(np.sum(~positive&predicted_positive)/np.sum(~positive)) if (~positive).any() else np.nan
        score=probabilities[:,[index for index,c in enumerate(classes) if c!=benign]].sum(axis=1)
        result['roc_auc']=float(roc_auc_score(positive,score)) if np.unique(positive).size==2 else np.nan
        result['average_precision']=float(average_precision_score(positive,score)) if np.unique(positive).size==2 else np.nan
    else:
        areas,averages=[],[]
        for index,label in enumerate(classes):
            target=labels==label
            if np.unique(target).size!=2:
                areas.append(np.nan)
                averages.append(np.nan)
            else:
                areas.append(roc_auc_score(target,probabilities[:,index]))
                averages.append(average_precision_score(target,probabilities[:,index]))
        # Missing class support leaves the full-class macro quantity undefined.
        result['roc_auc']=float(np.mean(areas))
        result['average_precision']=float(np.mean(averages))
    return result


def figure(data,dataset,output):
    """Plot four predictive metrics in a wide 2 x 2 PDF without surrounding prose."""
    layout=p.load_module('compact_baseline','06_analysis/dataset_summary/compact.py')
    name=f'{dataset}_clean_vs_poisoned'
    entry=layout.draw(data.assign(series='baseline'),DISPLAY_METRICS,'','',
                      {'baseline':'Baseline'},output/name,'')
    return {'dataset':dataset,'table':name+'.csv',**entry}


def main():
    """Read only baseline predictions and compare all conditions on common rows."""
    root=ROOT/'results/evaluation'
    if p.read(root/'verification.json')['status']!='passed':
        raise ValueError('Verified results are required')
    metric=root/'06_analysis/metrics.csv'
    if p.sha(metric)!=p.read(root/'06_analysis/manifest.json')['outputs']['metrics.csv']:
        raise ValueError('Metrics changed')
    frame=pd.read_csv(metric)
    frame=frame[frame.policy.eq('none') & ~frame.guarded & frame.detector.eq('none')]
    if len(frame)!=61:
        raise ValueError('Expected the 61 unscreened, no-reset runs')
    request=p.read(root/'request.json')
    manifest=p.read(root/'01_attacks/manifest.json')
    conditions=pd.DataFrame(list(manifest['conditions'].values()))
    records,inputs=[],{}

    def verify(file,digest):
        """Verify and fingerprint every consumed source once."""
        key=file.resolve().relative_to(ROOT).as_posix()
        if key not in inputs:
            if p.sha(file)!=digest:
                raise ValueError(f'Changed input: {key}')
            inputs[key]=digest

    for stream,dataset in DATASETS.items():
        part=conditions[conditions.stream.eq(stream)&conditions['mode'].ne('clean')]
        for (base,seed),group in part.groupby(['base_key','attack_seed']):
            clean=conditions[conditions.base_key.eq(base)&conditions['mode'].eq('clean')]
            if len(clean)!=1 or len(group)!=6:
                raise ValueError('Expected clean and six poisoned conditions')
            selected=[clean.iloc[0].to_dict()]+[
                group[group['mode'].eq(mode)&group.level.eq(level)].iloc[0].to_dict()
                for mode,level in CATEGORIES[1:]]
            mask,labels=None,None
            for condition in selected:
                file=root/'01_attacks'/condition['truth']
                verify(file,manifest['outputs'][condition['truth']])
                truth=pd.read_parquet(file,columns=['row_id','host_eligible','host_reference_label'])
                np.testing.assert_array_equal(truth.row_id,np.arange(len(truth)))
                if mask is None:
                    mask=truth.row_id.to_numpy()>=request['config']['warmup_rows']
                    labels=truth.host_reference_label.to_numpy()
                else:
                    np.testing.assert_array_equal(labels,truth.host_reference_label)
                mask &= truth.host_eligible.to_numpy(dtype=bool)
                audit=root/'01_attacks'/condition['audit']
                verify(audit,manifest['outputs'][condition['audit']])
                mask[pd.read_parquet(audit,columns=['row_id']).row_id.to_numpy(dtype=int)]=False
            if not mask.any():
                raise ValueError('No common host observations')
            clean_accuracy=None
            for category,condition in enumerate(selected):
                matches=frame[frame.condition.eq(condition['key'])]
                if len(matches)!=1:
                    raise ValueError('Missing unique baseline run')
                run=matches.iloc[0]
                folder=root/run.relative_folder
                summary=p.read(folder/'summary.json')
                if summary['guarded'] or summary['resets'] or summary['detector']!='none':
                    raise ValueError('A response or screen entered the baseline comparison')
                file=folder/'predictions.parquet'
                verify(file,summary['files']['predictions.parquet'])
                classes=condition['classes']
                probability_columns=[f'p_{c}' for c in classes]
                pred=pd.read_parquet(file,columns=['row_id','scored','prediction']+probability_columns)
                np.testing.assert_array_equal(pred.row_id,np.arange(len(mask)))
                assert pred.scored.to_numpy()[mask].all()
                correct=int(np.sum(pred.prediction.to_numpy()[mask]==labels[mask]))
                accuracy=correct/int(mask.sum())
                probabilities=pred.loc[mask,probability_columns].to_numpy()
                if not np.isfinite(probabilities).all():
                    raise ValueError('Undefined prediction probabilities')
                np.testing.assert_allclose(probabilities.sum(axis=1),1,atol=1e-8)
                predictive=predictive_metrics(labels[mask],pred.prediction.to_numpy()[mask],
                                              probabilities,classes,condition['benign_class'])
                if category==0:
                    clean_accuracy=accuracy
                records.append({'dataset':dataset,'base_key':base,'attack_seed':int(seed),
                                'category':category,'mode':condition['mode'],'level':condition['level'],
                                'run_id':run.run_id,'common_host_rows':int(mask.sum()),'correct':correct,
                                'accuracy':accuracy,'change_from_clean_pp':100*(accuracy-clean_accuracy),
                                **predictive,**{key:float(run[key]) for key,_,_ in OPERATIONAL}})
        print(f'Read baseline predictions: {dataset}',flush=True)
    raw=pd.DataFrame(records)
    metric_columns=[spec[0] for spec in PREDICTIVE+RADAR+OPERATIONAL]
    long=raw.melt(id_vars=['dataset','base_key','attack_seed','category','mode','level'],
                  value_vars=metric_columns,var_name='metric',value_name='value')
    points=long.groupby(['dataset','category','mode','level','metric']).agg(
        mean=('value','mean'),sd=('value','std'),defined=('value','count'),
        realizations=('value','size')).reset_index()
    assert points.realizations.eq(3).all()
    output=root/'06_analysis/presentation/baseline_clean_vs_poisoned'
    output.mkdir(parents=True,exist_ok=True)
    raw.to_csv(output/'run_scores.csv',index=False)
    points.to_csv(output/'plot_values.csv',index=False)
    plt.rcParams.update({'font.family':'DejaVu Serif','font.size':10,'axes.labelsize':10,
                         'xtick.labelsize':9,'ytick.labelsize':9,'legend.fontsize':9,'pdf.fonttype':42})
    entries=[]
    for dataset in DATASETS.values():
        data=points[points.dataset.eq(dataset)].sort_values('category')
        entry=figure(data,dataset,output)
        data[data.metric.isin(entry['metrics'])].to_csv(output/entry['table'],index=False)
        entries.append(entry)
    text='''# Baseline: clean versus six poisoned streams

These three figures use only the Hoeffding Tree's unscreened no-reset runs. There is no detector-driven intervention, candidate screening, immediate reset, confirmed reset, or RL. The learner predicts each block before learning its observed labels, including poisoned labels/content in attacked runs.

Each figure contains exactly four predictive metrics in a two-row, two-column layout. Categories are 0: clean, 1: instance 15%, 2: instance 25%, 3: concept 15%, 4: concept 25%, 5: splice 15%, 6: splice 25%. The dashed line is the clean reference for each metric. Connected points are categorical comparisons, not a time sequence or continuous dose-response curve. The learner was evaluated in separate matched clean and poisoned runs; these are not before/after snapshots of one trained model.

All datasets show accuracy, macro-F1, macro recall and ROC-AUC. Macro precision and average precision remain available in the tables. These cover overall correctness, class-balanced performance and probability ranking. Higher is better. Synthetic AUC/AP are macro one-versus-rest summaries; RADAR uses malicious-versus-benign probability scores. Macro precision/recall and synthetic probability metrics are additional descriptive calculations from saved predictions, not previously selected primary outcomes. ROC-AUC is a scalar summary, not a ROC curve. Exposure and computational measures remain in the tables but are omitted from these four-panel figures.

Precision/recall/F1 give equal weight to the declared classes, with zero division set to zero. AUC/AP are undefined when the required positive/negative support is absent. Undefined values are omitted. The clean stream has no poison-admission denominator. Unscreened baseline poison admission is 100% and legitimate withholding is zero; those values remain in the underlying table. Detector alarm/delay and response metrics are not applicable to this baseline and are omitted. All baseline reset counts are zero.

All seven conditions within a base/attack-seed cohort are scored on identical host positions. The mask excludes warm-up, protected reservations, and all concept/splice replacement intervals across the six attacked conditions. This estimates effects on unchanged host observations, not accuracy on the replacement examples. The whole input stream was processed in the saved runs; common_host_rows is the scoring denominator, not a stream truncation.

The baseline shares the experiment's 5% reservation for comparability. Those reserved observations did not train this learner even though screening is disabled. Saved predictions are rescored; no learners were rerun. The common mask differs from the main report and earlier three-level comparisons, so their headline accuracies need not match these values.

Means and sample standard deviations use three synthetic realizations or three RADAR attack assignments. RADAR is one capture. Its clean predictions are reused on the three common masks; those bars are not variation across independent captures. Error bars are descriptive, not confidence intervals. Each dataset can have a different vertical axis range.

Splice copies a segment's features and labels from another archive location. Replay in the detailed reports names this same mechanism. The figures retain Splice.

Predictive metrics use the common scoring rows. Exposure, withholding, training work and runtime are read from the verified full-run metrics and do not use the smaller common host mask. Training work counts learner update visits per 100,000 source positions; the panel displays thousands of visits. Runtime is seconds per 100,000 source positions and reflects the original machine and concurrent execution. Lower FPR, exposure and runtime can be useful, but must be interpreted alongside predictive performance; low withholding preserves access to legitimate training data.

run_scores.csv records each source run, correct count, common scoring denominator, every metric and paired accuracy change from clean. plot_values.csv and the per-dataset tables contain each metric's mean, sample standard deviation and defined observation count. Each dataset has one single-page PDF at 170 mm width and 128 mm height, fitting the text width of A4 with 20 mm margins and less than half its page height. Direct condition labels replace the numeric key. There are no figure headings or explanatory footers; the dataset is in the filename. Bars are one sample standard deviation and the dashed horizontal line is the clean reference. PNG files are previews. No HTML is produced.
'''
    (output/'README.md').write_text(text,encoding='utf-8')
    p.write(output/'manifest.json',{'created_utc':p.now(),'width_mm':170,'height_mm':128,
            'script_sha256':p.sha(__file__),'layout_sha256':p.sha(ROOT/'06_analysis/dataset_summary/compact.py'),'request_sha256':p.sha(root/'request.json'),
            'inputs':inputs,'entries':entries,'outputs':{f.name:p.sha(f) for f in output.iterdir()
            if f.is_file() and f.name!='manifest.json'}})
    print(f'Created three baseline figures: {output}',flush=True)


if __name__=='__main__':
    main()
