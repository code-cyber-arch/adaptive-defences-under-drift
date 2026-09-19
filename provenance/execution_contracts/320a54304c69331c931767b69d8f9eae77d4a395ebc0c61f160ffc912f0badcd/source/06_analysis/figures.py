"""Individual A4-width figures, complete run tables and comparative narration."""
from html import escape
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve
from common import protocol as p
from common import channels
LABEL = {'none': 'No reset', 'immediate': 'Immediate', 'confirmed': 'Confirmed', 'rl_alarm': 'RL alarm', 'rl': 'RL persistence', 'SEA_A': 'SEA', 'RBF_I': 'RBF', 'radar': 'RADAR', 'clean': 'Clean', 'instance': 'Instance', 'concept': 'Concept', 'splice': 'Replay', 'adwin': 'ADWIN', 'hddm_w': 'HDDM-W', 'hellinger': 'Hellinger', 'd3_oof': 'D3 OOF'}
COLORS = {'none': '#666666', 'immediate': '#bd4844', 'confirmed': '#39814b', 'rl_alarm': '#9763a5', 'rl': '#246bb0'}
SPECS = [('host_accuracy', 'Host accuracy', 'Accuracy (%)', 100), ('host_macro_f1', 'Macro-F1', 'Macro-F1 (%)', 100), ('poison_admission_rate', 'Poison admission', 'Admitted injected rows (%)', 100), ('clean_withhold_rate', 'Legitimate withholding', 'Legitimate rows withheld (%)', 100), ('resets_per_100k', 'Reset frequency', 'Resets per 100,000 rows', 1), ('training_visits_per_100k', 'Training work', 'Training visits per 100,000 rows', 1), ('seconds_per_100k', 'Execution time', 'Seconds per 100,000 rows', 1), ('detection_delay', 'Detection delay', 'Delay (blocks)', 1), ('commit_delay', 'Commitment delay', 'Delay (blocks)', 1), ('missed_detection_rate', 'Missed detections', 'Scheduled transitions missed (%)', 100), ('missed_commit_rate', 'Missed commitments', 'Scheduled transitions missed (%)', 100), ('unexposed_false_alarm_rate', 'Unexposed false alarms', 'False-alarm blocks (%)', 100), ('malicious_precision', 'Malicious precision', 'Precision (%)', 100), ('malicious_recall', 'Malicious recall', 'Recall (%)', 100), ('benign_fpr', 'Benign false positives', 'False-positive rate (%)', 100), ('roc_auc', 'ROC-AUC', 'ROC-AUC', 1), ('average_precision', 'Average precision', 'Average precision', 1)]

def report(root):
    """Export individual plots, source tables and short comparative narration."""
    stage = root / '06_analysis/presentation'
    if stage.exists():
        raise FileExistsError(stage)
    for folder in ['figures', 'tables', 'runs', 'conditions']:
        (stage / folder).mkdir(parents=True, exist_ok=True)
    m = pd.read_csv(root / '06_analysis/metrics.csv')
    config = p.read(root / 'request.json')['config']
    manifest = p.read(root / '01_attacks/manifest.json')
    plt.rcParams.update({'font.family': 'DejaVu Serif', 'font.size': 10, 'axes.labelsize': 10, 'axes.titlesize': 10, 'xtick.labelsize': 9, 'ytick.labelsize': 9, 'legend.fontsize': 9, 'pdf.fonttype': 42, 'axes.spines.top': False, 'axes.spines.right': False})
    entries = []

    def save(fig, name, title, caption, data, folder='figures'):
        """Save one A4-width plot and its underlying table."""
        if len(fig.axes) != 1:
            raise AssertionError('Each figure must contain one plot')
        (stage / folder).mkdir(parents=True, exist_ok=True)
        (stage / 'tables' / folder).mkdir(parents=True, exist_ok=True)
        fig.canvas.draw()
        fig.set_layout_engine('none')
        fig.savefig(stage / folder / (name + '.pdf'))
        fig.savefig(stage / folder / (name + '.png'), dpi=170)
        plt.close(fig)
        data.to_csv(stage / 'tables' / folder / (name + '.csv'), index=False)
        entries.append({'number': len(entries) + 1, 'title': title, 'caption': caption, 'pdf': folder + '/' + name + '.pdf', 'png': folder + '/' + name + '.png', 'data': 'tables/' + folder + '/' + name + '.csv', 'width_mm': 154.94, 'plots': 1, 'kind': folder})
    profiles = [(q, g) for q in p.POLICIES for g in [False, True]] + [(q, True) for q in p.RL_POLICIES if m.policy.eq(q).any()]
    for stream in m.stream.unique():
        for detector in config.get('detectors', ['adwin']):
            part = m[m.stream.eq(stream) & (m.detector.eq(detector) | m.policy.isin(['none', *p.RL_POLICIES]))]
            for metric, title, ylabel, scale in SPECS:
                data = part[part[metric].notna()]
                if data.empty:
                    continue
                table = data.groupby(['mode', 'level', 'policy', 'guarded']).agg(value=(metric, 'mean'), minimum=(metric, 'min'), maximum=(metric, 'max'), runs=(metric, 'count')).reset_index()
                categories = [(mode, level) for mode in ['clean', 'instance', 'concept', 'splice'] for level in ['none', 'moderate', 'severe'] if ((table['mode'] == mode) & (table.level == level)).any()]
                fig, ax = plt.subplots(figsize=(6.1, 4.3), layout='constrained')
                for policy, guarded in profiles:
                    t = table[table.policy.eq(policy) & table.guarded.eq(guarded)].set_index(['mode', 'level'])
                    vals = t.reindex(pd.MultiIndex.from_tuples(categories)).value.to_numpy() * scale
                    ax.plot(np.arange(len(categories)), vals, marker='o', markersize=3, linewidth=1, color=COLORS[policy], alpha=1 if guarded else 0.45, linestyle='-' if guarded else '--', label=LABEL[policy] + (', screened' if guarded else ', unscreened'))
                ax.set_xticks(np.arange(len(categories)), [LABEL[mode] + '\n' + {'none': '', 'moderate': '15%', 'severe': '25%'}[level] for mode, level in categories])
                ax.set(title=f'{LABEL[stream]}, {LABEL[detector]}: {title}', ylabel=ylabel)
                ax.grid(axis='y', alpha=0.15)
                fig.legend(loc='outside upper center', ncol=2, frameon=False)
                name = metric
                save(fig, name, f'{LABEL[stream]} / {LABEL[detector]}: {title}', 'Mean across available matched realizations; severity remains separate. No-reset and RL references are repeated across detector views. Undefined values are omitted.', table, 'figures/' + LABEL[stream] + '/' + detector)
    pairs = pd.read_csv(root / '06_analysis/paired_gate_differences.csv')
    comparisons = []
    for keys, g in m.groupby(['condition', 'detector', 'policy']):
        if p.is_rl(keys[2]):
            continue
        screened = g[g.guarded.eq(True)].iloc[0]
        for _, rl in m[m.condition.eq(keys[0]) & m.policy.isin(p.RL_POLICIES)].iterrows():
            comparisons.append({'condition': keys[0], 'stream': rl.stream, 'base_key': rl.base_key, 'detector': keys[1], 'reference_policy': keys[2], 'rl_policy': rl.policy, 'rl_seed': rl.rl_seed, **{metric: float(rl[metric] - screened[metric]) for metric in ['host_accuracy', 'host_macro_f1', 'poison_admission_rate', 'clean_withhold_rate', 'resets', 'training_visits', 'seconds']}})
    pd.DataFrame(comparisons).to_csv(stage / 'tables/rl_paired_differences.csv', index=False)
    state_pairs = []
    metrics = ['host_accuracy', 'host_macro_f1', 'poison_admission_rate',
               'clean_withhold_rate', 'resets', 'training_visits', 'seconds',
               'malicious_recall', 'roc_auc']
    for _, run in m[m.policy.eq('rl')].iterrows():
        reference = m[m.policy.eq('rl_alarm') & m.condition.eq(run.condition) & m.rl_seed.eq(run.rl_seed)]
        if len(reference) != 1:
            raise ValueError('Persistence RL requires one matching alarm-only reference')
        reference = reference.iloc[0]
        state_pairs.append({'condition': run.condition, 'stream': run.stream,
                            'mode': run['mode'], 'level': run.level, 'base_key': run.base_key,
                            'rl_seed': run.rl_seed,
                            **{metric: float(run[metric] - reference[metric]) for metric in metrics}})
    state_frame = pd.DataFrame(state_pairs)
    state_frame.to_csv(stage / 'tables/rl_state_paired_differences.csv', index=False)
    for stream in state_frame.stream.unique() if len(state_frame) else []:
        part = state_frame[state_frame.stream.eq(stream)]
        for metric, title, ylabel, scale in [
                ('host_accuracy', 'Accuracy difference', 'Accuracy difference (percentage points)', 100),
                ('host_macro_f1', 'Macro-F1 difference', 'Macro-F1 difference', 1),
                ('poison_admission_rate', 'Poison admission difference', 'Admission difference (percentage points)', 100),
                ('resets', 'Reset difference', 'Reset count difference', 1)]:
            table = part.groupby(['mode', 'level']).agg(
                value=(metric, 'mean'), minimum=(metric, 'min'), maximum=(metric, 'max'),
                pairs=(metric, 'count')).reset_index()
            table = table[table.value.notna()]
            if table.empty:
                continue
            fig, ax = plt.subplots(figsize=(6.1, 3.8), layout='constrained')
            values = table.value.to_numpy() * scale
            bounds = np.array([table.value - table.minimum, table.maximum - table.value]) * scale
            ax.errorbar(np.arange(len(table)), values, yerr=bounds, fmt='o', capsize=3, color=COLORS['rl'])
            ax.axhline(0, color='#777777', linewidth=0.8)
            ax.set_xticks(np.arange(len(table)), [LABEL[row['mode']] + '\n' +
                         {'none': '', 'moderate': '15%', 'severe': '25%'}[row.level]
                         for _, row in table.iterrows()])
            ax.set(title=f'{LABEL[stream]}: persistence RL minus alarm-only RL', ylabel=ylabel)
            ax.grid(axis='y', alpha=0.15)
            save(fig, metric, f'{LABEL[stream]}: {title}',
                 'Matched by condition and RL seed. Points show means; bars show observed ranges, not confidence intervals.',
                 table, 'figures/' + LABEL[stream] + '/rl_state_comparison')
    for keys in [['stream', 'detector', 'policy', 'guarded'], ['stream', 'mode', 'level', 'policy', 'guarded']]:
        cols = [s[0] for s in SPECS]
        m.groupby(keys, dropna=False)[cols].agg(['mean', 'std', 'min', 'max', 'count']).to_csv(stage / 'tables' / ('summary_' + '_'.join(keys) + '.csv'))
    m.to_csv(stage / 'tables/run_inventory.csv', index=False)
    full = bool(config.get('input_bank') or config.get('full_report'))
    if full:
        for condition, c in manifest['conditions'].items():
            group = m[m.condition.eq(condition)]
            truth = channels.truth(root, c)
            curves = []
            condition_rows = []
            for _, run in group.iterrows():
                folder = root / run.relative_folder
                pred = pd.read_parquet(folder / 'predictions.parquet')
                events = pd.read_parquet(folder / 'events.parquet')
                eligible = pred.scored & truth.host_eligible
                ids = pred.loc[eligible, 'row_id']
                correct = pred.loc[eligible, 'prediction'].to_numpy() == truth.loc[eligible, 'host_reference_label'].to_numpy()
                blocks = pd.DataFrame({'block': ids.to_numpy() // config['block_size'], 'correct': correct.astype(int), 'support': 1}).groupby('block').sum().reindex(events.block, fill_value=0)
                blocks['accuracy'] = blocks.correct.div(blocks.support.replace(0, np.nan))
                blocks['display_accuracy'] = blocks.correct.rolling(5, min_periods=1).sum().div(blocks.support.rolling(5, min_periods=1).sum().replace(0, np.nan))
                blocks.loc[blocks.support.eq(0), 'display_accuracy'] = np.nan
                table = events.merge(blocks.reset_index(), on='block')
                fig, ax = plt.subplots(figsize=(6.1, 3.6), layout='constrained')
                ax.plot(table.block, table.display_accuracy * 100, color=COLORS[run.policy], linewidth=1)
                commits = table[table.reset_committed]
                ax.scatter(commits.block, commits.display_accuracy * 100, marker='x', color='#b52525', s=16, label='Committed reset')
                ax.set(title=f'{LABEL[run.stream]}: {LABEL[run.policy]}, {LABEL.get(run.detector, run.detector)}', xlabel='Block (1,000 source rows)', ylabel='Host accuracy (%)', ylim=(0, 100))
                ax.grid(alpha=0.15)
                if len(commits):
                    ax.legend(frameon=False)
                name = 'accuracy_over_time'
                run_folder = 'runs/' + Path(run.relative_folder).relative_to('05_runs').as_posix()
                save(fig, name, run.run_id, 'Five-block support-weighted display; metrics use unsmoothed eligible rows.', table, run_folder)
                sentence = f'Host accuracy was {run.host_accuracy * 100:.2f}% and macro-F1 was {run.host_macro_f1:.3f}. The arm committed {int(run.resets)} resets and used {int(run.training_visits):,} candidate-training visits.'
                (stage / run_folder / 'README.md').write_text('# ' + run.run_id + '\n\n' + sentence + '\n\n[Individual trace](' + name + '.pdf)\n', encoding='utf-8')
                condition_rows.append({'run': run.run_id, 'summary': sentence, 'figure': run_folder + '/' + name + '.pdf'})
                if run.stream == 'radar':
                    y = truth.loc[eligible, 'host_reference_label'].to_numpy()
                    yh = pred.loc[eligible, 'prediction'].to_numpy()
                    labels = c['classes']
                    cm = confusion_matrix(y, yh, labels=labels)
                    pd.DataFrame(cm, index=labels, columns=labels).to_csv(stage / 'tables' / run_folder / 'confusion_matrix.csv')
                    scores = pred.loc[eligible, [f'p_{k}' for k in labels if k != c['benign_class']]].sum(axis=1).to_numpy()
                    positive = y != c['benign_class']
                    if len(np.unique(positive)) == 2:
                        fpr, tpr, _ = roc_curve(positive, scores)
                        precision, recall, _ = precision_recall_curve(positive, scores)

                        def thin(a, b):
                            """Reduce display points without changing the separately calculated metrics."""
                            ii = np.unique(np.linspace(0, len(a) - 1, min(1500, len(a)), dtype=int))
                            return (a[ii], b[ii])
                        fpr, tpr = thin(fpr, tpr)
                        recall, precision = thin(recall, precision)
                        curves.append((run, fpr, tpr, recall, precision))
            condition_folder = stage / 'conditions' / LABEL[c['stream']] / c['mode'] / c['level'] / ('capture' if c['base_seed'] is None else f'seed_{c["base_seed"]}') / ('clean' if c.get('attack_seed') is None else f'attack_seed_{c["attack_seed"]}')
            condition_folder.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(condition_rows).to_csv(condition_folder / 'comparison.csv', index=False)
            (condition_folder / 'README.md').write_text('# ' + condition + '\n\n' + '\n\n'.join((r['run'] + ': ' + r['summary'] for r in condition_rows)) + '\n', encoding='utf-8')
            if c['stream'] == 'radar':
                for detector in config['detectors']:
                    for kind in ['ROC', 'PR']:
                        fig, ax = plt.subplots(figsize=(6.1, 4.4), layout='constrained')
                        rows = []
                        for run, x, y, r, pr in curves:
                            if run.detector != detector and run.policy not in ['none', *p.RL_POLICIES]:
                                continue
                            xx, yy = (x, y) if kind == 'ROC' else (r, pr)
                            label = LABEL[run.policy] + (' S' if run.guarded else ' U') + (f' {int(run.rl_seed)}' if p.is_rl(run.policy) else '')
                            ax.plot(xx, yy, linewidth=1, color=COLORS[run.policy], alpha=1 if run.guarded else 0.4, linestyle='-' if run.guarded else '--', label=label)
                            rows.extend(({'run': run.run_id, 'x': float(a), 'y': float(b)} for a, b in zip(xx, yy)))
                        ax.set(xlabel='False-positive rate' if kind == 'ROC' else 'Recall', ylabel='True-positive rate' if kind == 'ROC' else 'Precision', title=f"RADAR {c['mode']}, {c['level']}: {kind}", xlim=(0, 1), ylim=(0, 1))
                        fig.legend(loc='outside upper center', ncol=3, frameon=False)
                        ax.grid(alpha=0.15)
                        save(fig, kind, condition + ' ' + kind, 'S: screened; U: unscreened. RL seeds shown separately. Display points are thinned; scalar AUC/AP use all saved scores.', pd.DataFrame(rows), 'figures/' + condition_folder.relative_to(stage / 'conditions').as_posix() + '/' + detector)
    narrative = ['# Experiment sequence', '', f"Purpose: {config['purpose']}.", '', f'First, I prepared {m.condition.nunique()} conditions. All arms used identical observation inputs and the same protected audit reservation.', '', f'Then, I completed {len(m)} policy runs. I compared no reset, immediate reset and confirmed reset with and without screening.']
    if m.policy.isin(p.RL_POLICIES).any():
        narrative += ['', 'I then compared frozen alarm-only RL with RL using both alarms and recent error deterioration. Each representation used the same three training seeds and chose continue learning or reset. Both proposals passed through the same protected accuracy check. Their Q-tables were not updated during evaluation.']
    narrative += ['', 'Next, I compared host accuracy, macro-F1, poisoning admission and legitimate withholding. These outcomes show whether reducing exposure also preserves useful learning.', '']
    for (stream, detector, policy), g in pairs[pairs.detector.eq('adwin') & pairs.policy.eq('confirmed')].groupby(['stream', 'detector', 'policy']):
        narrative.append(f'{LABEL[stream]}, {LABEL[detector]}, {LABEL[policy]}: screening changed mean accuracy by {g.host_accuracy.mean() * 100:+.2f} percentage points and poison admission by {g.poison_admission_rate.mean() * 100:+.2f} points.')
    if comparisons:
        rl = pd.DataFrame(comparisons)
        for (stream, policy, reference), g in rl[(rl.detector == 'adwin') & rl.reference_policy.isin(['immediate', 'confirmed'])].groupby(['stream', 'rl_policy', 'reference_policy']):
            narrative.append(f'{LABEL[stream]}: {LABEL[policy]} changed mean accuracy by {g.host_accuracy.mean() * 100:+.2f} points and reset count by {g.resets.mean():+.1f} relative to screened ADWIN {LABEL[reference].lower()}.')
    for stream, g in state_frame.groupby('stream') if len(state_frame) else []:
        narrative.append(f'{LABEL[stream]}: adding error history changed mean accuracy by {g.host_accuracy.mean() * 100:+.2f} points, macro-F1 by {g.host_macro_f1.mean():+.3f} and reset count by {g.resets.mean():+.1f} relative to alarm-only RL at matching seeds. These differences do not establish statistical equivalence or guaranteed improvement.')
    narrative += ['', 'Finally, I examined reset frequency, actual candidate-training work and execution time. Synthetic delay is interpreted with missed events; RADAR has no verified drift-event labels.', '', 'These are descriptive matched comparisons. Policy arms and RADAR attack assignments are not independent datasets. Gate acceptance is not proof that an update is poison-free. The supplied benchmark had been inspected before this evaluation.', '', 'The protected channel is a simulated assumption. It supplies original observations with a one-block delay and never contributes examples to learner training.']
    (stage / 'NARRATION.md').write_text('\n'.join(narrative) + '\n', encoding='utf-8')
    pd.DataFrame(entries).to_csv(stage / 'catalogue.csv', index=False)
    html = '<!doctype html><html lang="en"><meta charset="utf-8"><title>Experiment results</title><style>body{font:17px/1.5 Georgia;max-width:960px;margin:auto;padding:24px}img{max-width:100%}li{margin:10px 0}</style><h1>Comparative experiment results</h1><p>' + escape(config['purpose']) + '</p><p><a href="NARRATION.pdf">Narration PDF</a> | <a href="NARRATION.md">Narration text</a> | <a href="catalogue.csv">All figures</a> | <a href="tables/run_inventory.csv">All runs</a></p><ul>'
    for e in entries:
        html += '<li><a href="' + e['pdf'] + '">' + escape(e['title']) + '</a> - ' + escape(e['caption']) + '</li>'
    (stage / 'index.html').write_text(html + '</ul></html>', encoding='utf-8')

    def tex(s):
        """Escape text for the LaTeX narration."""
        return ''.join(({'&': '\\&', '%': '\\%', '_': '\\_', '#': '\\#'}.get(c, c) for c in s))
    paragraphs = [line for line in narrative if line and (not line.startswith('#'))]
    lines = ['\\documentclass[11pt,a4paper]{article}', '\\usepackage[margin=25mm]{geometry}', '\\usepackage[T1]{fontenc}', '\\usepackage{lmodern}', '\\setlength{\\parindent}{0pt}', '\\setlength{\\parskip}{6pt}', '\\begin{document}', '{\\Large\\bfseries Experiment sequence}\\par']
    lines.extend((tex(line) + '\\par' for line in paragraphs))
    lines.append('\\end{document}')
    (stage / 'NARRATION.tex').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    p.write(stage / 'manifest.json', {'figures': len(entries), 'width_mm': 154.94, 'plots_per_figure': 1, 'outputs': {f.relative_to(stage).as_posix(): p.sha(f) for f in stage.rglob('*') if f.is_file()}})
    print(f'Presentation: {len(entries)} separate single-plot figures', flush=True)
