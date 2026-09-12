"""Compare policy runs at 0%, 15% and 25% poisoning on common scoring rows."""
from pathlib import Path
from html import escape
import sys
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import protocol as p

DATASETS = {'SEA_A': 'SEA', 'RBF_I': 'RBF', 'radar': 'RADAR'}
ATTACKS = {'instance': 'Instance', 'concept': 'Concept', 'splice': 'Splice'}
PROFILES = [('none', False, 'No reset / U', '#666666', '--', 'o'),
            ('none', True, 'No reset / S', '#333333', '-', 'o'),
            ('immediate', False, 'Immediate / U', '#db893b', '--', 's'),
            ('immediate', True, 'Immediate / S', '#ad5816', '-', 's'),
            ('confirmed', False, 'Confirmed / U', '#6eaa72', '--', '^'),
            ('confirmed', True, 'Confirmed / S', '#23702d', '-', '^'),
            ('rl_alarm', True, 'RL alarm / S', '#9856a0', '-', 'D'),
            ('rl', True, 'RL persistence / S', '#1d66bb', '-', 'v')]


def calculate(root, frame, manifest, config):
    """Rescore saved predictions using identical positions across three severities."""
    records, fingerprints = [], {}
    conditions = pd.DataFrame(list(manifest['conditions'].values()))

    def verified(path, digest):
        """Verify each consumed artifact once and record its fingerprint."""
        key = path.resolve().relative_to(ROOT).as_posix()
        if key not in fingerprints:
            if p.sha(path) != digest:
                raise ValueError(f'Changed input: {key}')
            fingerprints[key] = digest

    def truth(condition):
        """Load only offline eligibility, labels and protected row identities."""
        file = root / '01_attacks' / condition['truth']
        verified(file, manifest['outputs'][condition['truth']])
        ref = pd.read_parquet(file, columns=['row_id', 'host_eligible', 'host_reference_label'])
        audit = root / '01_attacks' / condition['audit']
        verified(audit, manifest['outputs'][condition['audit']])
        ids = pd.read_parquet(audit, columns=['row_id']).row_id.to_numpy(dtype=int)
        eligible = ref.host_eligible.to_numpy(dtype=bool).copy()
        eligible[:config['warmup_rows']] = False
        eligible[ids] = False
        np.testing.assert_array_equal(ref.row_id, np.arange(len(ref)))
        return ref.host_reference_label.to_numpy(), eligible

    for stream, dataset in DATASETS.items():
        for attack in ATTACKS:
            part = conditions[conditions.stream.eq(stream) & conditions['mode'].eq(attack)]
            for (base, seed), pair in part.groupby(['base_key', 'attack_seed']):
                clean = conditions[conditions.base_key.eq(base) & conditions['mode'].eq('clean')]
                if len(clean) != 1 or len(pair) != 2:
                    raise ValueError('Every comparison needs clean, moderate and severe conditions')
                selected = [(0, clean.iloc[0].to_dict()),
                            (15, pair[pair.level.eq('moderate')].iloc[0].to_dict()),
                            (25, pair[pair.level.eq('severe')].iloc[0].to_dict())]
                references = [truth(c) for _, c in selected]
                common = np.logical_and.reduce([mask for _, mask in references])
                labels = references[0][0][common]
                for reference, _ in references[1:]:
                    np.testing.assert_array_equal(labels, reference[common])
                if not common.any():
                    raise ValueError('No shared host observations remain')
                cohort = f'{base}__attack_seed_{int(seed)}'
                for level, condition in selected:
                    runs = frame[frame.condition.eq(condition['key']) & frame.detector.isin(['none','adwin'])]
                    if len(runs) != 12:
                        raise ValueError('Expected six fixed arms and six frozen RL arms')
                    for _, run in runs.iterrows():
                        folder = root / run.relative_folder
                        summary = p.read(folder / 'summary.json')
                        file = folder / 'predictions.parquet'
                        verified(file, summary['files']['predictions.parquet'])
                        pred = pd.read_parquet(file, columns=['row_id','scored','prediction'])
                        np.testing.assert_array_equal(pred.row_id, np.arange(len(common)))
                        if not pred.scored.to_numpy()[common].all():
                            raise ValueError('Shared rows include unscored observations')
                        predictions = pred.prediction.to_numpy()[common]
                        records.append({'dataset': dataset, 'attack': attack, 'cohort': cohort,
                                        'base_key': base, 'attack_seed': int(seed), 'level_percent': level,
                                        'condition': condition['key'], 'run_id': run.run_id,
                                        'policy': run.policy, 'guarded': bool(run.guarded),
                                        'rl_seed': run.rl_seed, 'common_host_rows': int(common.sum()),
                                        'correct': int(np.sum(predictions == labels)),
                                        'accuracy': float(np.mean(predictions == labels))})
            print(f'Scored {dataset}, {ATTACKS[attack]} on common rows', flush=True)
    raw = pd.DataFrame(records)
    cohort_keys = ['dataset','attack','cohort','level_percent','policy','guarded']
    # Average RL seeds first; they are not independent stream realizations.
    cohorts = raw.groupby(cohort_keys, dropna=False).agg(
        accuracy=('accuracy','mean'), common_host_rows=('common_host_rows','first'),
        policy_runs=('run_id','count')).reset_index()
    grouped = cohorts.groupby(['dataset','attack','level_percent','policy','guarded']).agg(
        accuracy=('accuracy','mean'), sd=('accuracy','std'), cohorts=('cohort','count'),
        minimum=('accuracy','min'), maximum=('accuracy','max')).reset_index()
    if not grouped.cohorts.eq(3).all():
        raise ValueError('Expected three realizations or attack assignments per point')
    for _, group in raw.groupby(['dataset','attack','cohort']):
        if group.common_host_rows.nunique() != 1:
            raise ValueError('Scoring support differs across compared runs')
    return raw, cohorts, grouped, fingerprints


def panel(ax, data, title, limits):
    """Plot policy means with descriptive standard-deviation error bars."""
    for policy, guarded, label, color, style, marker in PROFILES:
        line = data[data.policy.eq(policy) & data.guarded.eq(guarded)].sort_values('level_percent')
        if line.level_percent.tolist() != [0,15,25]:
            raise ValueError('Unexpected severity grid')
        ax.errorbar(line.level_percent, line.accuracy * 100, yerr=line.sd * 100,
                    color=color, linestyle=style, marker=marker, markersize=3.5,
                    linewidth=1.1, elinewidth=.75, capsize=2, label=label)
    ax.set(title=title, xlabel='Nominal poisoning level (%)', ylabel='Host accuracy (%)',
           xticks=[0,15,25], xlim=(-1,26), ylim=limits)
    ax.grid(alpha=.18)
    ax.spines[['top','right']].set_visible(False)


def main():
    """Export both requested layouts from verified saved predictions."""
    root = ROOT / 'results/evaluation'
    request = p.read(root / 'request.json')
    if p.read(root / 'verification.json')['status'] != 'passed':
        raise ValueError('Completed verification is required')
    metric_file = root / '06_analysis/metrics.csv'
    digest = p.read(root / '06_analysis/manifest.json')['outputs']['metrics.csv']
    if p.sha(metric_file) != digest:
        raise ValueError('Metrics changed')
    frame = pd.read_csv(metric_file)
    manifest = p.read(root / '01_attacks/manifest.json')
    output = root / '06_analysis/presentation/line_comparisons'
    output.mkdir(parents=True, exist_ok=True)
    for name in ['combined','separate','tables']:
        (output / name).mkdir(exist_ok=True)
    raw, cohorts, grouped, fingerprints = calculate(root, frame, manifest, request['config'])
    raw.to_csv(output/'tables/run_scores.csv', index=False)
    cohorts.to_csv(output/'tables/cohort_scores.csv', index=False)
    grouped.to_csv(output/'tables/plot_values.csv', index=False)
    plt.rcParams.update({'font.family':'DejaVu Serif','font.size':10, 'axes.titlesize':10,
                         'axes.labelsize':10,'xtick.labelsize':9,'ytick.labelsize':9,
                         'legend.fontsize':9,'pdf.fonttype':42})
    entries = []
    for dataset in DATASETS.values():
        subset = grouped[grouped.dataset.eq(dataset)]
        low = float(((subset.accuracy-subset.sd)*100).min())
        high = float(((subset.accuracy+subset.sd)*100).max())
        limits = (max(0, np.floor((low-1)/5)*5), min(100, np.ceil((high+1)/5)*5))
        fig, axes = plt.subplots(3, 1, figsize=(6.1,9.2))
        fig.subplots_adjust(left=.14,right=.97,top=.825,bottom=.075,hspace=.58)
        for ax, (attack,label) in zip(axes, ATTACKS.items()):
            panel(ax,subset[subset.attack.eq(attack)],label,limits)
        fig.suptitle(f'{dataset}: accuracy versus poisoning level', y=.987,fontsize=12)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.959),ncol=3,
                   frameon=False,columnspacing=1,handlelength=2)
        fig.text(.5,.865,'ADWIN responses | S: screened, U: unscreened',ha='center',fontsize=9)
        fig.text(.5,.019,'Points: means | Bars: ±1 SD across matched realizations/assignments',ha='center',fontsize=9)
        name = f'combined/{dataset}_accuracy'
        fig.savefig(output/(name+'.pdf'))
        fig.savefig(output/(name+'.png'),dpi=180)
        plt.close(fig)
        entries.append({'dataset':dataset,'layout':'combined','attack':'all','pdf':name+'.pdf',
                        'png':name+'.png','width_mm':154.94,'height_mm':233.68,'plots':3})
        for attack,label in ATTACKS.items():
            fig,ax=plt.subplots(figsize=(6.1,4.6))
            fig.subplots_adjust(left=.14,right=.97,top=.69,bottom=.19)
            panel(ax,subset[subset.attack.eq(attack)],'',limits)
            fig.suptitle(f'{dataset}, {label}: accuracy versus poisoning level',y=.98,fontsize=11)
            handles,labels=ax.get_legend_handles_labels()
            fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.925),ncol=3,
                       frameon=False,columnspacing=1,handlelength=2)
            fig.text(.5,.04,'ADWIN | S: screened, U: unscreened | Bars: ±1 SD',ha='center',fontsize=9)
            name=f'separate/{dataset}_{attack}_accuracy'
            fig.savefig(output/(name+'.pdf'))
            fig.savefig(output/(name+'.png'),dpi=180)
            plt.close(fig)
            entries.append({'dataset':dataset,'layout':'separate','attack':attack,'pdf':name+'.pdf',
                            'png':name+'.png','width_mm':154.94,'height_mm':116.84,'plots':1})
    description = '''# Policy line comparisons

The combined folder contains one figure per dataset, with three vertically stacked attack panels. The separate folder contains one figure per dataset and attack. Each PDF is 154.94 mm wide, with a matching PNG. These two layouts use identical plotted values.

Each plot compares eight policy profiles. Immediate and confirmed responses use ADWIN. No-reset references have no detector. RL is screened and uses ADWIN. S means screened; U means unscreened. Only the evaluated nominal poisoning levels 0%, 15% and 25% are shown; connecting lines are visual guides.

For each base realization and attack assignment, all policies and all three levels are scored on the intersection of eligible host rows. Protected rows, warm-up rows, and concept/splice replacement intervals at either severity are excluded. Saved predictions are rescored; no learner is retrained. Consequently, values may differ from the broader host scores in the main report. The 0% reference may differ across attack panels because the shared scoring positions differ.

RL seeds are averaged within a condition first. Points then average three synthetic realizations or three RADAR attack assignments. Error bars show ±1 sample standard deviation, not confidence intervals. RADAR assignments reuse one capture and do not represent three independent datasets. Its 0% bars can reflect different scoring masks on the same saved clean predictions. RL training-seed variation remains available in tables/run_scores.csv and is not represented separately by these bars.

The vertical scale is shared across the three attacks within each dataset. It may differ between datasets to preserve readability. Host accuracy does not measure poison-identification accuracy.

Splice retains the original experiment name. It copies features and labels from another 250-row archive interval; existing detailed reports call that mechanism Replay. The archive may include later observations, so the attack is not limited to replay of past stream data.

Tables: run_scores.csv contains every rescored run, common_host_rows and the number correct. cohort_scores.csv averages RL seeds. plot_values.csv supplies every point and error bar.
'''
    (output/'README.md').write_text(description,encoding='utf-8')
    html=['<!doctype html><html lang="en"><meta charset="utf-8"><title>Policy line comparisons</title>',
          '<style>body{max-width:1000px;margin:auto;padding:24px;font:17px/1.5 Georgia}img{max-width:100%}article{margin-bottom:40px}</style>',
          '<h1>Policy line comparisons</h1><p><a href="README.md">Scoring, error bars and terminology</a> | <a href="tables/plot_values.csv">Plot values</a></p>']
    for layout,title in [('combined','Three dataset figures'),('separate','Nine separate attack figures')]:
        html.append(f'<h2>{title}</h2>')
        for e in entries:
            if e['layout']==layout:
                caption=e['dataset']+(': '+ATTACKS[e['attack']] if e['attack']!='all' else ': all three attacks')
                html.append(f'<article><h3>{escape(caption)}</h3><a href="{e["pdf"]}">PDF</a> | <a href="{e["png"]}">PNG</a><br><img src="{e["png"]}" alt="{escape(caption)}"></article>')
    (output/'index.html').write_text('\n'.join(html)+'</html>',encoding='utf-8')
    p.write(output/'manifest.json',{'created_utc':p.now(),'metrics_sha256':digest,
            'request_sha256':p.sha(root/'request.json'),'script_sha256':p.sha(__file__),
            'inputs':fingerprints,'entries':entries,
            'outputs':{f.relative_to(output).as_posix():p.sha(f) for f in output.rglob('*')
                       if f.is_file() and f.name!='manifest.json'}})
    print(f'Created 3 combined and 9 separate figures: {output}',flush=True)


if __name__=='__main__':
    main()
