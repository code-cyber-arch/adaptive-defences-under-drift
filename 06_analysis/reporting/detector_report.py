"""Verify detector-specific training and trajectories, then render comparisons."""
from pathlib import Path
import sys
import json
import hashlib
import importlib.util
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from common import protocol as p
OUT = ROOT / 'results/detector_policy_study'
# This independent metric implementation uses confusion counts, not execution metrics.
spec = importlib.util.spec_from_file_location('independent_trace_metrics', Path(__file__).with_name('check_and_plot.py'))
audit = importlib.util.module_from_spec(spec); spec.loader.exec_module(audit)
METRICS = ['host_macro_f1', 'host_accuracy', 'poison_admission_rate', 'clean_withhold_rate', 'resets']


def verify():
    cfg = p.read(ROOT / 'configs/study.json')
    manifest = p.read(OUT / 'training/04_rl_training/manifest.json')
    models = manifest['models']
    expected = {(d, v, s) for d in cfg['detectors'] for v in ['alarm', 'alarm_persistence'] for s in cfg['rl_seeds']}
    assert len(models) == len(expected) == 24
    assert {(m['detector'], m['variant'], m['seed']) for m in models} == expected
    episode_orders = {}
    for m in models:
        path = ROOT / m['path']; assert p.sha(path) == m['sha256']
        record = p.read(path)
        assert record['detector'] == m['detector']
        assert record['training_base_seeds'] == cfg['training_seeds']
        assert record['request_sha256'] == p.sha(OUT / 'training/request.json')
        for name, digest in record['files'].items():
            assert p.sha(path.parent / name) == digest
        events = pd.read_parquet(path.parent / 'transitions.parquet')
        assert events.detector.eq(m['detector']).all()
        assert events.reward_block.eq(events.action_block + 1).all()
        assert (events.audit_latest_row < events.reward_block * 1000).all()
        assert len(events) == record['training_transitions'] == 14 * cfg['training_epochs'] * (cfg['rows'] // 1000 - 1)
        episodes = pd.read_csv(path.parent / 'episodes.csv')
        order = episodes[['epoch', 'condition']].to_csv(index=False)
        assert episode_orders.setdefault(m['seed'], order) == order
    source_contract = p.read(OUT / 'protocol.json')
    for path, digest in source_contract['sources'].items():
        assert p.sha(ROOT / path) == digest
    rows = []; cached = {}; hashes = {}; masks = {}; schedules = {}
    selected = p.read(OUT / 'selected_policies.json')
    assert selected['training_manifest_sha256'] == p.sha(OUT / 'training/04_rl_training/manifest.json')
    for stage, directory in [('validation', 'validation_runs'), ('evaluation', 'evaluation_runs')]:
        source = OUT / stage
        conditions = p.read(source / '01_attacks/manifest.json')['conditions']
        for file in sorted((OUT / directory).rglob('summary.json')):
            r = p.read(file); condition = conditions[r['condition']]
            for name, digest in r['files'].items():
                assert p.sha(file.parent / name) == digest
            for name, digest in r['contract']['inputs'].items():
                path = (source / '01_attacks' / condition[name]).resolve()
                if path not in hashes: hashes[path] = p.sha(path)
                assert hashes[path] == digest
            key = (stage, r['condition'])
            if key not in cached:
                cached[key] = (pd.read_parquet(source / '01_attacks' / condition['truth']),
                               pd.read_parquet(source / '01_attacks' / condition['audit']))
            truth, reserved = cached[key]
            pred = pd.read_parquet(file.parent / 'predictions.parquet')
            events = pd.read_parquet(file.parent / 'events.parquet')
            feedback = pd.read_parquet(file.parent / 'feedback_schedule.parquet')
            assert pred.row_id.equals(truth.row_id)
            assert set(pred.loc[pred.audit_reserved, 'row_id']) == set(reserved.row_id) == set(feedback.row_id)
            assert not pred.loc[pred.audit_reserved, 'scored'].any()
            assert pred.loc[pred.audit_reserved, 'candidate_training_visits'].eq(0).all()
            assert pred.loc[pred.audit_reserved, 'first_admitted_block'].eq(-1).all()
            for metric, actual in audit.independent_metrics(pred, truth, condition['classes']).items():
                audit.close(r[metric], actual)
            mask_hash = hashlib.sha256(pred.scored.to_numpy().tobytes()).hexdigest()
            assert masks.setdefault(key, mask_hash) == mask_hash
            schedule_key = (stage, condition['base_key'], r['feedback_fraction'], r['delay_blocks'])
            schedule_hash = r['files']['feedback_schedule.parquet']
            assert schedules.setdefault(schedule_key, schedule_hash) == schedule_hash
            for event in events.itertuples():
                if not event.scored: continue
                released = feedback[(feedback.row_id // 1000 == event.block - r['delay_blocks']) & feedback.release_block.le(event.block)]
                assert event.audit_n == len(released)
                assert event.audit_latest_row == (int(released.row_id.max()) if len(released) else -1)
                assert event.reset_committed == bool(event.reset_requested and event.accepted)
                if not len(released): assert not event.accepted
            assert r['resets'] == int(events.reset_committed.sum())
            assert r['training_visits'] == int(pred.candidate_training_visits.sum())
            arm = r['contract']['arm']; assert arm['detector'] == r['detector']
            if 'model_path' in arm:
                fitted = p.read(ROOT / arm['model_path'])
                assert p.sha(ROOT / arm['model_path']) == arm['model_sha256']
                assert (fitted['detector'], fitted['variant'], fitted['seed']) == (r['detector'], arm['variant'], arm['seed'])
            if arm['name'] == 'selected_map':
                assert arm['table'] == selected['policies'][r['detector']]['table']
            rows.append({**{k: r[k] for k in ['condition', 'detector', 'arm', 'feedback_fraction', 'delay_blocks', *METRICS]},
                         'stage': stage, 'prediction_sha256': r['files']['predictions.parquet'],
                         'summary_path': file.relative_to(ROOT).as_posix(), 'summary_sha256': p.sha(file)})
        print(f'Trace audit completed {stage}: {len(rows)} cumulative runs', flush=True)
    values = pd.DataFrame(rows)
    assert len(values) == 4256
    assert not values.duplicated(['stage', 'condition', 'detector', 'arm', 'feedback_fraction', 'delay_blocks']).any()
    # No-reset controls do not use a detector and are shared evidence across detector panels.
    controls = values[(values.stage == 'evaluation') & values.arm.eq('no_reset')]
    assert controls.groupby(['condition', 'feedback_fraction', 'delay_blocks']).prediction_sha256.nunique().eq(1).all()
    for detector in cfg['detectors']:
        subset = values[(values.stage == 'validation') & values.detector.eq(detector)]
        assert len(subset) == 224
        ranks = subset.groupby('arm').agg(mean_host_macro_f1=('host_macro_f1', 'mean'), mean_resets=('resets', 'mean')).reset_index()
        winner = ranks.sort_values(['mean_host_macro_f1', 'mean_resets', 'arm'], ascending=[False, True, True]).iloc[0].arm
        assert winner == selected['policies'][detector]['policy_id']
    for stage, count in [('validation', 896), ('evaluation', 1680), ('sensitivity', 2016)]:
        table = pd.read_csv(OUT / 'analysis' / stage / 'metrics.csv')
        assert len(table) == count
        keys = ['condition', 'detector', 'arm', 'feedback_fraction', 'delay_blocks']
        subset = values[values.stage.eq('validation' if stage == 'validation' else 'evaluation')]
        combined = table.merge(subset, on=keys, validate='one_to_one', suffixes=('', '_checked'))
        assert len(combined) == count
        for metric in METRICS:
            assert np.allclose(combined[metric], combined[metric + '_checked'], equal_nan=True, atol=1e-12, rtol=0)
    destination = OUT / 'analysis/verification'; destination.mkdir(parents=True, exist_ok=True)
    values.to_csv(destination / 'checked_runs.csv', index=False)
    p.write(destination / 'verification.json', {'status': 'passed', 'trained_controllers': 24,
            'validated_executions': len(values), 'training_episode_order_matched': True,
            'detector_model_identity_checked': True, 'reserved_rows_and_feedback_timing_checked': True,
            'metrics_recalculated_from_traces': True, 'checks_completed_utc': p.now(),
            'checker_sha256': p.sha(Path(__file__)), 'metric_checker_sha256': p.sha(Path(audit.__file__))})


def report():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42})
    cfg = p.read(ROOT / 'configs/study.json')
    frame = pd.read_csv(OUT / 'analysis/evaluation/metrics.csv')
    arms = ['confirmed', 'error_rule', 'selected_map', 'q_alarm_7', 'q_alarm_17', 'q_alarm_27',
            'q_alarm_persistence_7', 'q_alarm_persistence_17', 'q_alarm_persistence_27']
    labels = ['Confirmed reset', 'Error rule', 'Selected mapping', '2-state Q / 7', '2-state Q / 17',
              '2-state Q / 27', '4-state Q / 7', '4-state Q / 17', '4-state Q / 27']
    names = {'adwin': 'ADWIN', 'hddm_w': 'HDDM-W', 'hellinger': 'Hellinger', 'd3_oof': 'D3 OOF'}
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharey=True)
    for ax, stream in zip(axes, cfg['streams']):
        data = frame[frame.stream.eq(stream)]
        for i, detector in enumerate(cfg['detectors']):
            means = data[data.detector.eq(detector)].groupby('arm').host_macro_f1.mean()
            ax.plot([means[a] * 100 for a in arms], np.arange(len(arms)) + (i-1.5)*.14,
                    marker='o', linestyle='', label=names[detector])
        control = data[data.arm.eq('no_reset') & data.detector.eq(cfg['detectors'][0])].host_macro_f1.mean()*100
        ax.axvline(control, color='grey', linestyle='--', linewidth=1, label='Shared no-reset reference')
        ax.set_title(stream); ax.set_xlabel('Mean macro-F1 (%)'); ax.grid(axis='x', alpha=.2)
        ax.set_yticks(range(len(arms)), labels)
    axes[0].invert_yaxis()
    handles, legends = axes[0].get_legend_handles_labels()
    fig.legend(handles, legends, loc='lower center', ncol=5, frameon=False)
    fig.suptitle('Detector–policy comparison: matched training and evaluation')
    fig.tight_layout(rect=(0, .09, 1, .96))
    folder = OUT / 'analysis/figures'; folder.mkdir(parents=True, exist_ok=True)
    for suffix in ['pdf', 'png']:
        fig.savefig(folder / f'detector_policy_comparison.{suffix}', dpi=180, bbox_inches='tight')
    plt.close(fig)
    # Retain detector-specific seed variation, rather than treating policy seeds as data replications.
    seed_values = frame.groupby(['stream', 'detector', 'arm', 'base_seed'])[METRICS].mean().reset_index()
    seed_values.to_csv(OUT / 'analysis/evaluation/stream_seed_metrics.csv', index=False)
    lines = ['# Detector and reset-policy comparison', '',
             'The existing RQ1 and RQ2 remain unchanged. Policy controls sit within Experiment 2; feedback robustness supports Experiments 1 and 2.', '',
             'Phase 04 trained 24 controllers: four detectors, two state representations and three training seeds. Every detector used the same training streams, episode order for a matching seed, four-pass budget and accuracy reward. Phase 05 required the evaluation detector to match the trained controller.', '',
             'All 4,256 comparison executions completed and passed trace checks: 896 validation runs, 1,680 default-feedback evaluations and 1,680 additional feedback settings. The sensitivity table includes 336 shared defaults (2,016 entries). Training comprises 1,344 episodes across the 24 controllers. Repeated no-reset controls are shared evidence, not independent detector results.', '',
             '| Dataset | Detector | Response | Mean macro-F1 (%) | Difference from same-detector confirmed reset (pp) |',
             '| --- | --- | --- | ---: | ---: |']
    for (stream, detector), group in frame.groupby(['stream', 'detector']):
        means = group.groupby('arm').host_macro_f1.mean()*100
        for arm, label in zip(arms, labels):
            lines.append(f'| {stream} | {names[detector]} | {label} | {means[arm]:.2f} | {means[arm]-means["confirmed"]:+.2f} |')
    lines += ['', 'These are means across seven conditions and three stream seeds within each synthetic family. Controller seeds are shown separately; no best test seed is selected. The evaluation streams contain 60,000 observations each and are not pooled with the longer original benchmark.', '',
              'The four-state deterministic mapping is selected separately for each detector on validation seed 114. Mapping selection uses macro-F1; Q-learning uses delayed protected accuracy. This does not isolate optimisation algorithm alone or establish convergence. Feedback sensitivity varies usable feedback (1%, 2%, 5%) and delay (one or five blocks) while keeping the same 5% reserve and frozen controllers.', '',
              'The evaluation realizations are kept identical across detectors. The ADWIN-only implementation had already been assessed on these realizations; the corrected grid is not presented as a wholly untouched independent replication. Hyperparameters and budgets are fixed across the detector comparison.', '',
              '- [Final detector comparison](../results/detector_policy_study/analysis/figures/detector_policy_comparison.pdf)',
              '- [Paired policy effects](../results/detector_policy_study/analysis/evaluation/paired_vs_confirmed.csv)',
              '- [Paired feedback effects](../results/detector_policy_study/analysis/sensitivity/paired_feedback_effects.csv)',
              '- [Trace checks](../results/detector_policy_study/analysis/verification/verification.json)', '']
    (ROOT / '07_documentation/RESULTS.md').write_text('\n'.join(lines))
    p.write(OUT / 'report_manifest.json', {'status': 'complete', 'files': {
        f.relative_to(ROOT).as_posix(): p.sha(f) for f in [ROOT / '07_documentation/RESULTS.md', *folder.glob('*')]},
        'script_sha256': p.sha(Path(__file__))})
    p.write(OUT / 'completion.json', {'status': 'complete', 'trained_controllers': 24, 'comparison_executions': 4256,
                                    'completed_utc': p.now(), 'protocol_sha256': p.sha(OUT / 'protocol.json')})


if __name__ == '__main__':
    verify()
    report()
