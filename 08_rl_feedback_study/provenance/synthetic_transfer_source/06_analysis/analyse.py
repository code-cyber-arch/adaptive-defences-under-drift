"""Offline evaluation using references unavailable to the learner."""
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score
from common import protocol as p
from common import channels

def fraction(n, d):
    """Return a ratio, keeping a zero denominator undefined."""
    return float(n / d) if d else None

def event_scores(events, truth, meta, config):
    """Match synthetic reference events to alarms and committed resets."""
    result = {'event_applicable': meta['supports_event_metrics'], 'detection_delay': None, 'commit_delay': None, 'missed_detections': None, 'missed_commits': None, 'scheduled_events': None, 'unexposed_false_alarm_rate': None, 'unexposed_blocks': None}
    if not meta['supports_event_metrics']:
        return result
    block = config['block_size']
    n = meta['rows']
    windows = [(a, min(n, b + 20 * block)) for a, b in meta['transition_intervals']]
    for metric, col in [('detection', 'raw_fire'), ('commit', 'reset_committed')]:
        available = list(events.loc[events.scored & events[col], 'end_row'])
        delays = []
        misses = 0
        for a, b in windows:
            hit = next((t for t in available if a < t <= b), None)
            if hit is None:
                misses += 1
            else:
                available.remove(hit)
                delays.append((hit - a) / block)
        result[f'{metric}_delay'] = float(np.mean(delays)) if delays else None
        result['missed_detections' if metric == 'detection' else 'missed_commits'] = misses
    result['scheduled_events'] = len(windows)
    poison = (truth.atk_flip | truth.atk_burst) & ~truth.audit_reserved
    exposed = np.logical_or.reduceat(poison.to_numpy(), np.arange(0, n, block))
    touched = exposed.copy()
    for shift in [1, 2]:
        exposed[shift:] |= touched[:-shift]
    for a, b in windows:
        exposed[a // block:(b - 1) // block + 1] = True
    eligible = events.scored.to_numpy() & ~exposed
    result['unexposed_blocks'] = int(eligible.sum())
    result['unexposed_false_alarm_rate'] = fraction(int(events.loc[eligible, 'raw_fire'].sum()), int(eligible.sum()))
    return result

def analyse(root):
    """Calculate offline metrics and matched screening differences."""
    stage = root / '06_analysis'
    if stage.exists():
        raise FileExistsError(stage)
    stage.mkdir()
    manifest = p.read(root / '01_attacks/manifest.json')
    config = p.read(root / 'request.json')['config']
    completion = p.read(root / '05_runs/completion.json')
    records = []
    for run_id, digest in completion['summaries'].items():
        folder = root / completion['paths'][run_id]
        assert p.sha(folder / 'summary.json') == digest
        summary = p.read(folder / 'summary.json')
        for file, expected in summary['files'].items():
            assert p.sha(folder / file) == expected
        pred = pd.read_parquet(folder / 'predictions.parquet')
        events = pd.read_parquet(folder / 'events.parquet')
        meta = manifest['conditions'][summary['condition']]
        truth = channels.truth(root, meta)
        assert pred.row_id.equals(truth.row_id)
        eligible = pred.scored & truth.host_eligible
        y = truth.loc[eligible, 'host_reference_label'].to_numpy(dtype=int)
        yh = pred.loc[eligible, 'prediction'].to_numpy(dtype=int)
        row = {k: v for k, v in summary.items() if k != 'files'}
        row.update(host_n=int(eligible.sum()), host_accuracy=float(np.mean(y == yh)) if len(y) else None, host_macro_f1=float(f1_score(y, yh, labels=meta['classes'], average='macro', zero_division=0)) if len(y) else None, undefined_predictions=int(np.sum(yh < 0)))
        training_eligible = (truth.row_id >= config['warmup_rows']) & ~truth.audit_reserved
        attacked = (truth.atk_flip | truth.atk_burst) & training_eligible
        clean = ~(truth.atk_flip | truth.atk_burst) & training_eligible
        admitted = pred.first_admitted_block >= 0
        row.update(poison_rows=int(attacked.sum()), poison_rows_admitted=int((attacked & admitted).sum()), poison_admission_rate=fraction(int((attacked & admitted).sum()), int(attacked.sum())), clean_rows=int(clean.sum()), clean_rows_withheld=int((clean & ~admitted).sum()), clean_withhold_rate=fraction(int((clean & ~admitted).sum()), int(clean.sum())), poison_training_visits=int(pred.loc[attacked, 'accepted_training_visits'].sum()), rejected_update_blocks=int((events.scored & ~events.accepted & ~events.reason.eq('rl_hold')).sum()), insufficient_audit_blocks=int(events.reason.eq('insufficient_audit').sum()), reset_requests=int(events.reset_requested.sum()), reset_rejections=int((events.reset_requested & ~events.accepted).sum()), gate_seconds=float(events.gate_seconds.sum()), candidate_seconds=float(events.candidate_seconds.sum()), prediction_seconds=float(events.predict_seconds.sum()))
        latency = pred.first_admitted_block[training_eligible & admitted] - pred.row_id[training_eligible & admitted] // config['block_size']
        row['mean_admission_delay_blocks'] = float(latency.mean()) if len(latency) else None
        row.update(malicious_precision=None, malicious_recall=None, benign_fpr=None, roc_auc=None, average_precision=None)
        if meta['benign_class'] is not None and len(y):
            benign = meta['benign_class']
            positive = y != benign
            pred_positive = (yh >= 0) & (yh != benign)
            row['malicious_precision'] = fraction(int((positive & pred_positive).sum()), int(pred_positive.sum()))
            row['malicious_recall'] = fraction(int((positive & pred_positive).sum()), int(positive.sum()))
            row['benign_fpr'] = fraction(int((~positive & pred_positive).sum()), int((~positive).sum()))
            scores = pred.loc[eligible, [f'p_{c}' for c in meta['classes'] if c != benign]].sum(axis=1).to_numpy()
            if len(np.unique(positive)) == 2:
                row['roc_auc'] = float(roc_auc_score(positive, scores))
                row['average_precision'] = float(average_precision_score(positive, scores))
        row.update(event_scores(events, truth, meta, config))
        row['missed_detection_rate'] = fraction(row['missed_detections'], row['scheduled_events']) if row['scheduled_events'] else None
        row['missed_commit_rate'] = fraction(row['missed_commits'], row['scheduled_events']) if row['scheduled_events'] else None
        for name in ['resets', 'training_visits', 'accepted_training_visits', 'accepted_refit_rows', 'seconds', 'gate_seconds']:
            row[f'{name}_per_100k'] = row[name] / row['rows'] * 100000
        records.append(row)
    frame = pd.DataFrame(records).sort_values('run_id')
    frame.to_csv(stage / 'metrics.csv', index=False)
    outcomes = ['host_accuracy', 'host_macro_f1', 'poison_admission_rate', 'clean_withhold_rate', 'resets', 'training_visits', 'seconds', 'malicious_recall', 'benign_fpr']
    pairs = []
    for (condition, detector, policy), group in frame[~frame.policy.isin(p.RL_POLICIES)].groupby(['condition', 'detector', 'policy']):
        unguarded = group[group.guarded.eq(False)].iloc[0]
        guarded = group[group.guarded.eq(True)].iloc[0]
        entry = {'condition': condition, 'detector': detector, 'policy': policy, 'stream': guarded.stream, 'mode': guarded['mode'], 'level': guarded.level}
        entry.update({metric: float(guarded[metric] - unguarded[metric]) if pd.notna(guarded[metric]) and pd.notna(unguarded[metric]) else None for metric in outcomes})
        pairs.append(entry)
    pd.DataFrame(pairs).to_csv(stage / 'paired_gate_differences.csv', index=False)
    p.write(stage / 'manifest.json', {'protocol': p.VERSION, 'request_sha256': p.sha(root / 'request.json'), 'runs': len(frame), 'pairs': len(pairs), 'outputs': {f.name: p.sha(f) for f in stage.glob('*.csv')}})
    print(f'Analysis: {len(frame)} runs, {len(pairs)} paired gate comparisons', flush=True)
