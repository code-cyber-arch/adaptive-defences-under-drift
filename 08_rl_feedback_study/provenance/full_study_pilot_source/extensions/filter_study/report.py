"""Verify supplementary runs and publish descriptive tables and figures."""
import numpy as np
import pandas as pd
from . import protocol as s
from common import protocol as p, channels


def verify(out, source, config, settings):
    manifest = p.read(source / '01_attacks/manifest.json')
    checks = runs = 0
    for stream in {c['stream'] for c in manifest['conditions'].values()}:
        folder = out / 'models' / stream
        record = p.read(folder / 'manifest.json')
        s.verify_files(folder, record)
        fit = pd.read_parquet(folder / 'fit_rows.parquet')
        calibration = pd.read_parquet(folder / 'calibration_rows.parquet')
        if np.intersect1d(fit.fingerprint, calibration.fingerprint).size:
            raise AssertionError('Exact feature copies cross fit/calibration')
        excluded = np.load(folder / 'excluded_test_fingerprints.npy', allow_pickle=False)
        for file in (folder / 'heldout').rglob('*.parquet'):
            heldout = pd.read_parquet(file, columns=['fingerprint'])
            if np.isin(heldout.fingerprint, excluded).any():
                raise AssertionError('Exact feature copies cross training/heldout')
            checks += 1
    for meta in manifest['conditions'].values():
        truth = channels.truth(source, meta)
        common = None
        for detector in settings['detectors']:
            masks = {}
            for kind in (*s.CONTROLS, *('learned_' + v for v in s.VIEWS)):
                folder = out / 'runs' / meta['key'] / detector / kind
                record = p.read(folder / 'summary.json')
                s.verify_files(folder, record)
                pred = pd.read_parquet(folder / 'predictions.parquet')
                events = pd.read_parquet(folder / 'events.parquet')
                if not np.array_equal(pred.row_id, np.arange(meta['rows'])):
                    raise AssertionError('Incomplete source coverage')
                reserved, withheld = pred.audit_reserved.to_numpy(bool), pred.filter_withheld.to_numpy(bool)
                if pred.loc[reserved | withheld, 'candidate_training_visits'].any():
                    raise AssertionError('Protected or filtered observations reached training')
                if pred.loc[reserved, 'comparison_scored'].any():
                    raise AssertionError('Protected observations reached scoring')
                scored = pred.comparison_scored.to_numpy(bool)
                if common is None:
                    common = scored
                elif not np.array_equal(common, scored):
                    raise AssertionError('Filter comparisons use different host positions')
                if int(scored.sum()) != record['host_rows']:
                    raise AssertionError('Scoring denominator changed')
                for row in events.itertuples():
                    active = np.arange(row.start_row, row.end_row)
                    expected = int((~reserved[active] & ~withheld[active]).sum()) if row.scored else 0
                    if row.monitor_rows != expected:
                        raise AssertionError('Monitor received filtered observations')
                    if row.scored and row.audit_latest_row >= row.start_row:
                        raise AssertionError('Current/future protected feedback used')
                if (events.reset_committed & ~events.accepted).any():
                    raise AssertionError('Reset bypassed screening')
                poisoned = (truth.atk_flip | truth.atk_burst).to_numpy(bool)
                if kind in s.CONTROLS:
                    expected = s.withholding(kind, poisoned, reserved, config['block_size'], record['activation_row'],
                                             p.seed(settings['seed'], meta['key'], kind))
                    if not np.array_equal(expected, withheld):
                        raise AssertionError('Control mask differs from declared protocol')
                if kind.startswith('oracle') and record['poison_admission_rate'] not in (None, 0.0):
                    raise AssertionError('Oracle admitted poison after activation')
                masks[kind] = int(withheld.sum())
                runs += 1
                checks += 8
            if masks['oracle_rows'] != masks['random_rows'] or masks['oracle_blocks'] != masks['random_blocks']:
                raise AssertionError('Random withholding budget is unmatched')
    planned = len(manifest['conditions']) * len(settings['detectors']) * 7
    if runs != planned:
        raise AssertionError('Missing experiment arms')
    summaries = {f.relative_to(out).as_posix(): p.sha(f) for f in (out / 'runs').rglob('summary.json')}
    p.write(out / 'verification.json', {'status': 'passed', 'runs': runs, 'checks': checks,
        'request_sha256': p.sha(out / 'request.json'), 'metrics_sha256': p.sha(out / 'metrics.csv'),
        'summary_hashes': summaries, 'protected_training_overlap': 0, 'filtered_training_overlap': 0,
        'matched_scoring_masks': True, 'frozen_filter_hashes': True, 'copied_feature_split_overlap': 0})


def build(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    verification = p.read(out / 'verification.json')
    if verification['status'] != 'passed' or p.sha(out / 'metrics.csv') != verification['metrics_sha256']:
        raise ValueError('Verified supplementary metrics required')
    raw = pd.read_csv(out / 'metrics.csv')
    measures = ['accuracy', 'macro_f1', 'macro_recall', 'roc_auc', 'poison_admission_rate',
                'legitimate_withhold_rate', 'filter_poison_recall', 'filter_legitimate_rejection',
                'scored_resets', 'total_seconds']
    measures = [c for c in measures if c in raw]
    keys = ['stream', 'mode', 'level', 'detector', 'filter']
    long = raw.melt(id_vars=keys + ['condition'], value_vars=measures, var_name='metric', value_name='value')
    summary = long.groupby(keys + ['metric']).value.agg(['mean', 'std', 'count']).reset_index()
    summary.to_csv(out / 'summary.csv', index=False)
    baseline = raw[raw['filter'].eq('none')][['condition', 'detector'] + measures]
    paired = raw.merge(baseline, on=['condition', 'detector'], suffixes=('', '_no_filter'), validate='many_to_one')
    for metric in measures:
        paired[metric + '_difference'] = paired[metric] - paired[metric + '_no_filter']
    paired.to_csv(out / 'paired_differences.csv', index=False)
    pd.concat([pd.read_csv(f) for f in (out / 'models').glob('*/separability.csv')], ignore_index=True).to_csv(out / 'separability.csv', index=False)
    figures = out / 'figures'
    figures.mkdir(exist_ok=True)
    styles = {'none': 'No filter', 'oracle_rows': 'Oracle rows', 'random_rows': 'Random rows',
              'oracle_blocks': 'Oracle blocks', 'random_blocks': 'Random blocks',
              'learned_features': 'Learned: features', 'learned_features_label': 'Learned: features + label'}
    categories = [('clean', 'none'), ('instance', 'moderate'), ('instance', 'severe'),
                  ('concept', 'moderate'), ('concept', 'severe'), ('splice', 'moderate'), ('splice', 'severe')]
    for (stream, detector), group in summary.groupby(['stream', 'detector']):
        fig, axes = plt.subplots(2, 2, figsize=(8, 6), constrained_layout=True)
        for ax, metric, title in zip(axes.flat, ['accuracy', 'macro_f1', 'poison_admission_rate', 'legitimate_withhold_rate'],
                                     ['Accuracy (%)', 'Macro-F1 (%)', 'Poison admission (%)', 'Legitimate withholding (%)']):
            for kind, label in styles.items():
                part = group[group['filter'].eq(kind) & group.metric.eq(metric)].set_index(['mode', 'level']).reindex(categories)
                ax.errorbar(np.arange(7), part['mean'] * 100, yerr=part['std'].fillna(0) * 100,
                            marker='o', markersize=3, linewidth=1, label=label)
            ax.set_ylabel(title)
            ax.set_xticks(range(7), ['Clean', 'I15', 'I25', 'C15', 'C25', 'S15', 'S25'], rotation=40)
            ax.grid(alpha=.2)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='outside lower center', ncol=3, fontsize=8)
        fig.suptitle(f'{stream} - {detector} - confirmed reset with screening', fontsize=11)
        fig.savefig(figures / f'{stream}_{detector}.png', dpi=140)
        fig.savefig(figures / f'{stream}_{detector}.pdf')
        plt.close(fig)
    text = f'''# Supplementary filter experiment

Verification passed for {len(raw)} matched runs. This output is separate from the original 1,464-run benchmark.

Seven filters share confirmed reset and candidate screening: no filter, oracle rows, random rows, oracle blocks, random blocks, learned features, and learned features plus observed labels. Oracle variants use hidden intervention truth solely to construct ideal-information withholding masks. Learned filters use no hidden truth at inference. All arms predict and score the same eligible host rows even when a filter rejects those rows for detector updates and training. Exact oracle rejection is not a universal accuracy upper bound. Whole-block rejection can remove many legitimate rows.

Synthetic discriminators fit on seed 112, calibrate on seed 113, and evaluate benchmark seeds (or the separately labelled smoke profile). RADAR uses the first 50% for fitting, 50-70% for calibration, and the final 30% for evaluation, with a one-block embargo after each boundary. Source/destination splice partition checks and exact-feature duplicate exclusion prevent copied observations crossing fitting and reported heldout sets. RADAR filtering activates only after calibration; learner warm-up and prefix evolution are shared by every arm. Its operational scores cover the eligible heldout suffix, not the complete-capture benchmark. The supervised RADAR labels are an offline experimental assumption, not a demonstrated source of trusted operational poison labels.

Thresholds are frozen at a validation legitimate-score quantile targeting at most 5% validation rejection (strict greater-than comparison keeps ties). This does not guarantee a 5% test false-positive rate. The two observation views do not include row identity, attack metadata or protected samples. Models are trained on pooled attack types; test results remain separate by attack and severity. Separability tests poisoned-versus-unmodified observations, not verified genuine-drift-versus-poison labels on RADAR. Null/poor results do not prove impossibility.

summary.csv contains descriptive means, sample SDs and defined counts across matched assignments, without pooling datasets. Clean RADAR has one capture. paired_differences.csv subtracts the no-filter arm within condition and detector. Undefined metrics remain empty. Timing includes the learner/detector/gate engine and frozen-filter inference, with inference amortised across detector variants; fit/calibration time is recorded with each model. It excludes input I/O and report generation. No hypothesis test or equivalence claim is made.

Each figure shows predictive quality and exposure costs. I/C/S denote instance/concept/splice; 15/25 are nominal budgets. Error bars are one sample SD, not confidence intervals. The controls randomly withhold one reproducible assignment per condition; block controls match both exposed-block count and eligible-row count including the partial final block. Empty filtered blocks do not update detectors; missing errors cannot confirm a reset. This behaviour is covered by extension tests.

Read metrics.csv for per-run outcomes, separability.csv for classification metrics, models/ for frozen models and disjoint fit/calibration/heldout records, and verification.json for invariants and artifact hashes. Original benchmark results and these supplementary results must be reported as separate protocols.
'''
    (out / 'README.md').write_text(text)
    p.write(out / 'report_manifest.json', {'request_sha256': p.sha(out / 'request.json'),
        'outputs': {f.relative_to(out).as_posix(): p.sha(f) for f in [out / 'summary.csv', out / 'paired_differences.csv',
                   out / 'separability.csv', out / 'README.md', *figures.glob('*')]}})
