"""Descriptive paired effects; seeds, not individual rows, are repetitions."""
import pandas as pd
from common import protocol as p
METRICS = ['host_macro_f1', 'host_accuracy', 'poison_admission_rate',
           'clean_withhold_rate', 'resets', 'training_visits', 'seconds']


def frame(records):
    return pd.DataFrame([{k: v for k, v in r.items()
                          if k not in ['contract', 'files', 'state_visits', 'action_counts']}
                         for r in records])


def select_policy(records, arms):
    values = frame(records)
    ranked = values.groupby('arm').agg(mean_host_macro_f1=('host_macro_f1', 'mean'),
                                      mean_resets=('resets', 'mean'), conditions=('condition', 'nunique'))
    ranked = ranked.reset_index().sort_values(['mean_host_macro_f1', 'mean_resets', 'arm'],
                                            ascending=[False, True, True])
    winner = ranked.iloc[0].arm
    arm = next(a for a in arms if a['name'] == winner)
    return {'policy_id': winner, 'table': arm['table'], 'selection': ranked.iloc[0].to_dict(),
            'scope': 'Validation-selected four-state deterministic mapping; not test-optimal'}, ranked


def report(records, folder):
    folder.mkdir(parents=True, exist_ok=True)
    values = frame(records)
    values.to_csv(folder / 'metrics.csv', index=False)
    keys = ['condition', 'feedback_fraction', 'delay_blocks']
    ref = values[values.arm.eq('confirmed')][keys + METRICS]
    pairs = values.merge(ref, on=keys, suffixes=('', '_reference'), validate='many_to_one')
    for metric in METRICS:
        pairs[metric + '_difference'] = pairs[metric] - pairs[metric + '_reference']
    pairs.to_csv(folder / 'paired_vs_confirmed.csv', index=False)
    group = ['stream', 'mode', 'level', 'arm', 'feedback_fraction', 'delay_blocks']
    values.groupby(group)[METRICS].agg(['mean', 'std', 'min', 'max']).to_csv(folder / 'descriptive_summary.csv')
    baseline = values[values.feedback_fraction.eq(.05) & values.delay_blocks.eq(1)]
    if len(baseline):
        feedback = values.merge(baseline[['condition', 'arm'] + METRICS],
                                on=['condition', 'arm'], suffixes=('', '_baseline'), validate='many_to_one')
        for metric in METRICS:
            feedback[metric + '_difference'] = feedback[metric] - feedback[metric + '_baseline']
        feedback.to_csv(folder / 'paired_feedback_effects.csv', index=False)
    p.write(folder / 'manifest.json', {'runs': len(values), 'inference': 'Descriptive; three synthetic stream seeds. RL seeds are not additional data realizations.',
            'files': {f.name: p.sha(f) for f in folder.glob('*.csv')}})
