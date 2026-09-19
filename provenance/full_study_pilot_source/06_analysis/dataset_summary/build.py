"""Create one A4-width accuracy summary for each verified dataset."""
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
CATEGORIES = [('clean', 'none'), ('instance', 'moderate'), ('instance', 'severe'),
              ('concept', 'moderate'), ('concept', 'severe'),
              ('splice', 'moderate'), ('splice', 'severe')]
COLUMN_LABELS = ['Clean', 'Instance\n15%', 'Instance\n25%', 'Concept\n15%',
                 'Concept\n25%', 'Splice\n15%', 'Splice\n25%']
METRICS = ['host_accuracy', 'host_macro_f1', 'poison_admission_rate',
           'clean_withhold_rate', 'resets', 'seconds_per_100k']


def profiles():
    """List fixed arms separately and combine only matched RL training seeds."""
    rows = [('none', 'none', False, 'No reset / U'), ('none', 'none', True, 'No reset / S')]
    for detector, label in [('adwin', 'ADWIN'), ('hddm_w', 'HDDM-W'),
                            ('hellinger', 'Hellinger'), ('d3_oof', 'D3 OOF')]:
        for policy, response in [('immediate', 'immediate'), ('confirmed', 'confirmed')]:
            for guarded, suffix in [(False, 'U'), (True, 'S')]:
                rows.append((detector, policy, guarded, f'{label} {response} / {suffix}'))
    rows += [('adwin', 'rl_alarm', True, 'RL alarm / S'),
             ('adwin', 'rl', True, 'RL persistence / S')]
    return rows


def aggregate(frame):
    """Average RL seeds within a condition before averaging matched realizations."""
    keys = ['condition', 'mode', 'level', 'detector', 'policy', 'guarded']
    conditions = frame.groupby(keys, dropna=False)[METRICS].mean().reset_index()
    records = []
    for detector, policy, guarded, label in profiles():
        arm = conditions[conditions.detector.eq(detector) & conditions.policy.eq(policy)
                         & conditions.guarded.eq(guarded)]
        for mode, level in CATEGORIES:
            group = arm[arm['mode'].eq(mode) & arm.level.eq(level)]
            if group.empty:
                raise ValueError(f'Missing comparison: {label}, {mode}, {level}')
            source = frame[frame.condition.isin(group.condition) & frame.detector.eq(detector)
                           & frame.policy.eq(policy) & frame.guarded.eq(guarded)]
            records.append({'profile': label, 'detector': detector, 'policy': policy,
                            'guarded': guarded, 'mode': mode, 'level': level,
                            'conditions': len(group), 'runs': len(source),
                            **{metric: group[metric].mean() for metric in METRICS}})
    return pd.DataFrame(records)


def narration(frame, dataset):
    """Describe two prespecified matched comparisons without choosing a winner."""
    fixed = frame[frame.detector.eq('adwin') & frame.policy.eq('confirmed')]
    gate = fixed[fixed.guarded].set_index('condition').host_accuracy - fixed[~fixed.guarded].set_index('condition').host_accuracy
    richer = frame[frame.policy.eq('rl')].set_index(['condition', 'rl_seed'])
    alarm = frame[frame.policy.eq('rl_alarm')].set_index(['condition', 'rl_seed'])
    difference = richer.host_accuracy - alarm.host_accuracy
    return (f'{dataset}: with ADWIN confirmed reset, screening changed mean host accuracy by '
            f'{gate.mean() * 100:+.2f} percentage points. Adding error history to RL changed '
            f'mean accuracy by {difference.mean() * 100:+.2f} points relative to alarm-only RL. '
            'These descriptive differences average matched conditions; they are not significance tests.')


def main():
    """Verify inputs, export three figures and record their independent provenance."""
    run = ROOT / 'results/evaluation'
    verification = p.read(run / 'verification.json')
    if verification['status'] != 'passed':
        raise ValueError('Benchmark verification must pass before reporting')
    metric_path = run / '06_analysis/metrics.csv'
    expected = p.read(run / '06_analysis/manifest.json')['outputs']['metrics.csv']
    if p.sha(metric_path) != expected:
        raise ValueError('Verified metrics changed')
    frame = pd.read_csv(metric_path)
    if len(frame) != verification['runs'] or frame.run_id.duplicated().any():
        raise ValueError('Run inventory does not match verification')
    output = run / '06_analysis/presentation/summary'
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Serif', 'font.size': 10,
                         'axes.titlesize': 12, 'xtick.labelsize': 9, 'ytick.labelsize': 9,
                         'pdf.fonttype': 42})
    entries = []
    text = ['# Dataset summary figures', '',
            'Each figure shows mean host accuracy (%) for all detector/response combinations. '
            'S means screened; U means unscreened. RL uses ADWIN and is screened in every run. '
            'Higher values are better; the blue scale is fixed from 0 to 100 across all datasets.', '',
            'Each column averages its available realizations: three per synthetic condition, '
            'one for RADAR clean and three attack assignments per RADAR attack/severity. '
            'RL seeds are averaged within each condition first. The accompanying table lists all supports.', '',
            'Splice replaces a 250-row interval with features and labels copied from another archive '
            'segment. Existing detailed plots label these same conditions Replay. Source segments '
            'can occur later in the archive; this is not exclusively replay of past observations.', '',
            'Host scoring excludes protected and warm-up rows. Concept/Splice replacement intervals '
            'are also excluded. Compare policies within a column; clean-versus-attack differences '
            'do not necessarily use identical scoring rows.', '']
    for stream, dataset in DATASETS.items():
        subset = frame[frame.stream.eq(stream)]
        table = aggregate(subset)
        labels = [item[3] for item in profiles()]
        matrix = np.array([[table.loc[table.profile.eq(label) & table['mode'].eq(mode)
                                      & table.level.eq(level), 'host_accuracy'].iloc[0] * 100
                            for mode, level in CATEGORIES] for label in labels])
        if not np.isfinite(matrix).all():
            raise ValueError('Summary contains undefined accuracy')
        fig, ax = plt.subplots(figsize=(6.1, 7.7))
        fig.subplots_adjust(left=.315, right=.985, top=.87, bottom=.135)
        ax.imshow(matrix, cmap='Blues', vmin=0, vmax=100, aspect='auto', interpolation='nearest')
        ax.set_xticks(np.arange(7), ['', '15%', '25%', '15%', '25%', '15%', '25%'])
        for center, label in [(0, 'Clean'), (1.5, 'Instance'), (3.5, 'Concept'), (5.5, 'Splice')]:
            ax.text(center, -.053, label, transform=ax.get_xaxis_transform(),
                    ha='center', va='top', fontsize=9)
        ax.set_yticks(np.arange(len(labels)), labels)
        ax.tick_params(axis='both', length=0, pad=6)
        for row in range(matrix.shape[0]):
            for col in range(matrix.shape[1]):
                ax.text(col, row, f'{matrix[row, col]:.1f}', ha='center', va='center',
                        fontsize=9, color='white' if matrix[row, col] >= 60 else '#172737')
        for edge in [1.5, 5.5, 9.5, 13.5, 17.5]:
            ax.axhline(edge, color='white', linewidth=2)
        for spine in ax.spines.values():
            spine.set_visible(False)
        fig.suptitle(f'{dataset}: classification performance', x=.5, y=.97, fontsize=12)
        fig.text(.5, .933, 'Mean host accuracy (%) | higher is better', ha='center', fontsize=10)
        fig.text(.5, .897, 'All detectors and responses; attack severity shown separately', ha='center', fontsize=9)
        fig.text(.04, .055, 'S: screened   U: unscreened   RL: ADWIN, mean of 3 training seeds', fontsize=9)
        fig.text(.04, .028, 'Darker blue indicates higher accuracy; common scale: 0–100%.', fontsize=9)
        if len(fig.axes) != 1:
            raise AssertionError('Each summary must contain exactly one plot')
        name = dataset + '_summary'
        fig.savefig(output / (name + '.pdf'))
        fig.savefig(output / (name + '.png'), dpi=200)
        plt.close(fig)
        table.to_csv(output / (name + '.csv'), index=False)
        sentence = narration(subset, dataset)
        caption = f'{dataset}: mean host accuracy by detector, response, screening and attack severity.'
        text += [f'## {dataset}', '', caption, '', sentence, '',
                 f'[Figure PDF]({name}.pdf) | [PNG]({name}.png) | [Data table]({name}.csv)', '']
        entries.append({'dataset': dataset, 'pdf': name+'.pdf', 'png': name+'.png',
                        'table': name+'.csv', 'caption': caption, 'narration': sentence})
    (output / 'README.md').write_text('\n'.join(text)+'\n', encoding='utf-8')
    html = ['<!doctype html><html lang="en"><meta charset="utf-8"><title>Dataset summaries</title>',
            '<style>body{max-width:900px;margin:auto;padding:24px;font:17px/1.5 Georgia}img{max-width:100%}</style>',
            '<h1>Dataset summary figures</h1><p><a href="README.md">Captions, aggregation and terminology</a></p>']
    for entry in entries:
        html += [f'<h2>{entry["dataset"]}</h2><p>{escape(entry["caption"])}</p>',
                 f'<p>{escape(entry["narration"])}</p><p><a href="{entry["pdf"]}">PDF</a> | '
                 f'<a href="{entry["table"]}">Data table</a></p><img src="{entry["png"]}" alt="{escape(entry["caption"])}">']
    (output / 'index.html').write_text('\n'.join(html)+'</html>', encoding='utf-8')
    p.write(output / 'manifest.json', {'created_utc': p.now(), 'figures': 3,
            'width_mm': 154.94, 'height_mm': 195.58, 'plots_per_figure': 1,
            'script': str(Path(__file__).relative_to(ROOT)), 'script_sha256': p.sha(__file__),
            'metrics_sha256': expected, 'request_sha256': p.sha(run/'request.json'),
            'verification_sha256': p.sha(run/'verification.json'), 'entries': entries,
            'outputs': {f.name: p.sha(f) for f in output.iterdir() if f.is_file() and f.name != 'manifest.json'}})
    print(f'Created three dataset summaries: {output}', flush=True)


if __name__ == '__main__':
    main()
