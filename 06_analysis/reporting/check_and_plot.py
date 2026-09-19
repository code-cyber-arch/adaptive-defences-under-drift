"""Independently reconcile complete traces, then report within existing RQ1/RQ2."""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from common import protocol as p
OUT = ROOT / 'results/policy_robustness'


def ratio(n, d):
    return float(n / d) if d else None


def close(actual, expected):
    if actual is None or expected is None:
        assert actual is None and expected is None
    else:
        assert abs(float(actual) - float(expected)) < 1e-12, (actual, expected)


def independent_metrics(pred, truth, classes):
    mask = pred.scored.to_numpy() & truth.host_eligible.to_numpy()
    y = truth.host_reference_label.to_numpy()[mask]
    z = pred.prediction.to_numpy()[mask]
    f1 = []
    for label in classes:
        tp = int(((y == label) & (z == label)).sum())
        fp = int(((y != label) & (z == label)).sum())
        fn = int(((y == label) & (z != label)).sum())
        f1.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.)
    train = pred.scored.to_numpy()
    poisoned = (truth.atk_flip | truth.atk_burst).to_numpy() & train
    clean = ~(truth.atk_flip | truth.atk_burst).to_numpy() & train
    admitted = pred.first_admitted_block.to_numpy() >= 0
    return {'host_n': int(mask.sum()), 'host_accuracy': float((y == z).mean()),
            'host_macro_f1': float(np.mean(f1)),
            'poison_rows': int(poisoned.sum()),
            'poison_rows_admitted': int((poisoned & admitted).sum()),
            'poison_admission_rate': ratio((poisoned & admitted).sum(), poisoned.sum()),
            'clean_rows': int(clean.sum()), 'clean_rows_withheld': int((clean & ~admitted).sum()),
            'clean_withhold_rate': ratio((clean & ~admitted).sum(), clean.sum())}


def verify():
    assert p.read(OUT / 'completion.json')['status'] == 'complete'
    spec = p.read(ROOT / 'configs/study.json')
    expected_validation = len(spec['streams']) * len(spec['validation_seeds']) * 7 * 16
    expected_evaluation = len(spec['streams']) * len(spec['evaluation_seeds']) * 7 * 10
    expected_extra = len(spec['streams']) * len(spec['evaluation_seeds']) * len(spec['sensitivity_conditions']) * 7 * 5
    checked, fingerprints, defaults = [], {}, {}
    cached_truth, cached_manifests = {}, {}
    for stage, directory in [('validation', 'validation_runs'), ('evaluation', 'evaluation_runs')]:
        source = OUT / 'inputs' / stage
        manifest = p.read(source / '01_attacks/manifest.json')
        cached_manifests[stage] = manifest
        for file in sorted((OUT / directory).rglob('summary.json')):
            record = p.read(file)
            for name, digest in record['files'].items():
                assert p.sha(file.parent / name) == digest
            condition = manifest['conditions'][record['condition']]
            for name, digest in record['contract']['inputs'].items():
                path = (source / '01_attacks' / condition[name]).resolve()
                if str(path) not in fingerprints:
                    fingerprints[str(path)] = p.sha(path)
                assert fingerprints[str(path)] == digest
            key = (stage, record['condition'])
            if key not in cached_truth:
                truth = pd.read_parquet(source / '01_attacks' / condition['truth'])
                base_audit = pd.read_parquet(source / '01_attacks' / condition['audit'])
                cached_truth[key] = (truth, base_audit)
            truth, base_audit = cached_truth[key]
            pred = pd.read_parquet(file.parent / 'predictions.parquet')
            events = pd.read_parquet(file.parent / 'events.parquet')
            schedule = pd.read_parquet(file.parent / 'feedback_schedule.parquet')
            assert pred.row_id.equals(truth.row_id)
            assert set(pred.loc[pred.audit_reserved, 'row_id']) == set(base_audit.row_id) == set(schedule.row_id)
            assert not pred.loc[pred.audit_reserved, 'scored'].any()
            assert pred.loc[pred.audit_reserved, 'first_admitted_block'].eq(-1).all()
            assert pred.loc[pred.audit_reserved, 'candidate_training_visits'].eq(0).all()
            for name, value in independent_metrics(pred, truth, condition['classes']).items():
                close(record[name], value)
            delay = record['delay_blocks']; fraction = record['feedback_fraction']
            selected = schedule.release_block.eq(schedule.row_id // 1000 + delay)
            assert int(selected.sum()) == round(record['rows'] * fraction)
            for block, group in schedule[selected].groupby(schedule.loc[selected, 'row_id'] // 1000):
                assert len(group) == round(1000 * fraction)
            for row in events.itertuples():
                if not row.scored:
                    continue
                source_block = row.block - delay
                released = schedule[(schedule.row_id // 1000 == source_block) & schedule.release_block.le(row.block)]
                assert row.audit_n == len(released)
                assert row.audit_latest_row == (int(released.row_id.max()) if len(released) else -1)
                assert row.reset_committed == bool(row.reset_requested and row.accepted)
                if not len(released):
                    assert not row.accepted and row.reason == 'no_protected_sample'
            assert record['resets'] == int(events.reset_committed.sum())
            assert record['training_visits'] == int(pred.candidate_training_visits.sum())
            close(record['accepted_training_visits'], int(pred.accepted_training_visits.sum()))
            mask_digest = hashlib.sha256(pred.scored.to_numpy().tobytes()).hexdigest()
            previous = defaults.setdefault(key, mask_digest)
            assert previous == mask_digest
            checked.append({'stage': stage, 'condition': record['condition'], 'arm': record['arm'],
                            'feedback_fraction': fraction, 'delay_blocks': delay,
                            'host_macro_f1': record['host_macro_f1'], 'resets': record['resets'],
                            'summary_sha256': p.sha(file), 'prediction_sha256': record['files']['predictions.parquet'], 'path': file.relative_to(ROOT).as_posix()})
        print(f'Checked {stage} traces; cumulative executions {len(checked)}', flush=True)
    values = pd.DataFrame(checked)
    assert len(values[values.stage.eq('validation')]) == expected_validation
    assert len(values[values.stage.eq('evaluation')]) == expected_evaluation + expected_extra
    assert not values.duplicated(['stage', 'condition', 'arm', 'feedback_fraction', 'delay_blocks']).any()
    baseline = values[values.stage.eq('evaluation') & values.feedback_fraction.eq(.05) & values.delay_blocks.eq(1)]
    identical = baseline.pivot(index='condition', columns='arm', values='prediction_sha256')
    assert identical.no_reset.eq(identical.q_alarm_17).all()
    assert identical.q_alarm_7.eq(identical.q_alarm_27).all()
    rank = values[values.stage.eq('validation')].groupby('arm').agg(
        mean_host_macro_f1=('host_macro_f1', 'mean'), mean_resets=('resets', 'mean')).reset_index()
    rank = rank.sort_values(['mean_host_macro_f1', 'mean_resets', 'arm'], ascending=[False, True, True])
    selected = p.read(OUT / 'selected_policy.json')
    assert selected['policy_id'] == rank.iloc[0].arm
    protocol = p.read(OUT / 'protocol.json')
    for name, digest in protocol['sources'].items():
        assert p.sha(ROOT / name) == digest
    for name, digest in protocol['models'].items():
        assert p.sha(ROOT / name) == digest
    # Reconcile each published metric table with the separately read run summaries.
    for stage, expected in [('validation', expected_validation), ('evaluation', expected_evaluation),
                            ('sensitivity', len(spec['streams']) * len(spec['evaluation_seeds']) * len(spec['sensitivity_conditions']) * 7 * 6)]:
        table = pd.read_csv(OUT / 'analysis' / stage / 'metrics.csv')
        assert len(table) == expected
        subset = values[values.stage.eq('validation' if stage == 'validation' else 'evaluation')]
        join = table.merge(subset, on=['condition', 'arm', 'feedback_fraction', 'delay_blocks'],
                           validate='one_to_one', suffixes=('', '_checked'))
        assert len(join) == expected
        assert np.allclose(join.host_macro_f1, join.host_macro_f1_checked, atol=1e-12, rtol=0)
    destination = OUT / 'analysis/verification'
    destination.mkdir(parents=True, exist_ok=True)
    values.to_csv(destination / 'checked_runs.csv', index=False)
    p.write(destination / 'verification.json', {'status': 'passed', 'unique_research_executions': len(values),
        'validation': expected_validation, 'policy_evaluation': expected_evaluation,
        'additional_sensitivity': expected_extra, 'checks': ['input and trace hashes', 'independent confusion-matrix metrics',
        'reserved-row exclusion', 'identical score masks', 'feedback release and sample counts', 'fail-closed gate',
        'reset accounting', 'validation-only policy selection', 'frozen sources and Q-tables', 'metric table reconciliation', 'equivalent frozen-policy predictions'],
        'checker_sha256': p.sha(Path(__file__)), 'completed_utc': p.now()})


def reporting():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                         'savefig.bbox': 'tight', 'pdf.fonttype': 42})
    folder = OUT / 'analysis/figures'; folder.mkdir(parents=True, exist_ok=True)
    comparison = pd.read_csv(OUT / 'analysis/evaluation/metrics.csv')
    sensitivity = pd.read_csv(OUT / 'analysis/sensitivity/metrics.csv')
    arms = ['no_reset', 'confirmed', 'error_rule', 'selected_map',
            'q_alarm_7', 'q_alarm_17', 'q_alarm_27',
            'q_alarm_persistence_7', 'q_alarm_persistence_17', 'q_alarm_persistence_27']
    labels = ['No reset', 'Confirmed reset', 'Error rule', 'Selected mapping',
              '2-state Q / 7', '2-state Q / 17', '2-state Q / 27',
              '4-state Q / 7', '4-state Q / 17', '4-state Q / 27']
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5), sharey=True)
    for ax, stream in zip(axes, ['SEA_A', 'RBF_I']):
        seed_means = comparison[comparison.stream.eq(stream)].groupby(['arm', 'base_seed']).host_macro_f1.mean()
        for i, arm in enumerate(arms):
            x = seed_means.loc[arm].to_numpy() * 100
            ax.scatter(x, np.full(len(x), i), color='#4C78A8', alpha=.7, s=25)
            ax.plot([x.mean()], [i], marker='|', markersize=15, color='black')
        ax.set_title(stream); ax.set_xlabel('Macro-F1 (%)')
        ax.set_yticks(range(len(arms)), labels); ax.grid(axis='x', alpha=.2)
    axes[0].invert_yaxis()
    fig.suptitle('Policy comparison: points are stream-seed means across seven conditions')
    fig.tight_layout(); fig.savefig(folder / 'policy_comparison.pdf'); fig.savefig(folder / 'policy_comparison.png', dpi=180); plt.close(fig)
    chosen = ['no_reset', 'confirmed', 'error_rule', 'selected_map',
              'q_alarm_persistence_7', 'q_alarm_persistence_17', 'q_alarm_persistence_27']
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    for i, stream in enumerate(['SEA_A', 'RBF_I']):
        for j, delay in enumerate([1, 5]):
            ax = axes[i, j]
            sub = sensitivity[sensitivity.stream.eq(stream) & sensitivity.delay_blocks.eq(delay)]
            for arm in chosen:
                mean = sub[sub.arm.eq(arm)].groupby('feedback_fraction').host_macro_f1.mean()
                ax.plot(mean.index * 100, mean.values * 100, marker='o', label=labels[arms.index(arm)])
            ax.set_title(f'{stream}, delay {delay} block(s)'); ax.set_xticks([1, 2, 5])
            ax.set_xlabel('Usable protected feedback (%)'); ax.set_ylabel('Macro-F1 (%)'); ax.grid(alpha=.2)
    handles, names = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, names, loc='lower center', ncol=4, frameon=False)
    fig.suptitle('Feedback robustness: mean over three stream seeds and two conditions')
    fig.tight_layout(rect=(0, .10, 1, .96)); fig.savefig(folder / 'feedback_robustness.pdf'); fig.savefig(folder / 'feedback_robustness.png', dpi=180); plt.close(fig)
    # Preserve leading zeros in states when presenting original training diagnostics.
    diagnostics = []
    for path in sorted((ROOT / 'results/training/04_rl_training').rglob('policy.csv')):
        frame = pd.read_csv(path, dtype={'state': str})
        frame['variant'], frame['seed'] = path.parent.parent.name, int(path.parent.name.split('_')[1])
        diagnostics.append(frame)
    pd.concat(diagnostics).to_csv(OUT / 'analysis/training_diagnostics/state_action_coverage_verified.csv', index=False)
    p.write(folder / 'manifest.json', {'files': {f.name: p.sha(f) for f in folder.iterdir() if f.suffix in ['.pdf', '.png']},
                                     'script_sha256': p.sha(Path(__file__))})


if __name__ == '__main__':
    verify()
    reporting()
