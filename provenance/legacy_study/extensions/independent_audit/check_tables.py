"""Check original result coverage, matched arithmetic and publication tables."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from common import protocol as p


def equal_frame(actual, expected, keys):
    actual = actual.sort_values(keys).reset_index(drop=True)
    expected = expected.sort_values(keys).reset_index(drop=True)
    pd.testing.assert_frame_equal(actual[expected.columns], expected, check_dtype=False,
                                  check_exact=False, rtol=1e-9, atol=1e-10)


def main():
    root = ROOT / 'results/evaluation'; out = ROOT / 'results/independent_audit/2026-09-12'
    original = p.read(root / 'request.original_github.json')
    manifest = p.read(root / '01_attacks/manifest.json')
    raw = pd.read_csv(root / '06_analysis/metrics.csv')
    analysis = p.read(root / '06_analysis/manifest.json')
    for name, digest in analysis['outputs'].items():
        assert p.sha(root / '06_analysis' / name) == digest
    expected = {p.run_id(c['key'], a) for c in manifest['conditions'].values() for a in p.arms(original['config'])}
    assert set(raw.run_id) == expected and len(raw) == len(expected) == 1464
    assert raw.status.eq('complete').all()
    pairs = pd.read_csv(root / '06_analysis/paired_gate_differences.csv')
    keys = ['condition', 'detector', 'policy']
    values = ['host_accuracy','host_macro_f1','poison_admission_rate','clean_withhold_rate',
              'resets','training_visits','seconds','malicious_recall','benign_fpr']
    fixed = raw[raw.policy.isin(['none','immediate','confirmed'])]
    merged = fixed[fixed.guarded].merge(fixed[~fixed.guarded], on=keys, suffixes=('_g','_u'), validate='one_to_one')
    independent = merged[keys].copy()
    for name in values:
        independent[name] = merged[name + '_g'] - merged[name + '_u']
    equal_frame(pairs, independent, keys)
    presentation = root / '06_analysis/presentation'
    fixed_rows = pd.read_csv(presentation / 'fixed_responses/tables/run_scores.csv')
    assert len(fixed_rows) == 1134 and fixed_rows.run_id.nunique() == 1098
    cohort = ['dataset','base_key','attack_seed']
    for _, group in fixed_rows.groupby(cohort):
        assert len(group) == 126 and group.mask_sha256.nunique() == group.common_host_rows.nunique() == 1
    np.testing.assert_allclose(fixed_rows.correct / fixed_rows.common_host_rows, fixed_rows.accuracy, atol=1e-12)
    fixed_module = p.load_module('audit_fixed_tables', '06_analysis/dataset_summary/fixed_responses.py')
    fixed_points, fixed_effects = fixed_module.aggregate(fixed_rows)
    equal_frame(pd.read_csv(presentation / 'fixed_responses/tables/plot_values.csv'), fixed_points,
                ['dataset','comparison_detector','category','series','metric'])
    equal_frame(pd.read_csv(presentation / 'fixed_responses/tables/paired_screening_effects.csv'), fixed_effects,
                cohort + ['category','detector','policy'])
    rl = pd.read_csv(presentation / 'rl_comparison/tables/seed_scores.csv')
    assert len(rl) == 567 and rl.run_id.nunique() == 549
    for _, group in rl.groupby(cohort):
        assert len(group) == 63 and group.mask_sha256.nunique() == group.common_host_rows.nunique() == 1
    np.testing.assert_allclose(rl.correct / rl.common_host_rows, rl.accuracy, atol=1e-12)
    rl_module = p.load_module('audit_rl_tables', '06_analysis/dataset_summary/rl_comparison.py')
    rl_cohorts, rl_points, rl_pairs, rl_differences = rl_module.aggregate(rl)
    for table, name, key in [
        (rl_cohorts, 'cohort_scores', cohort + ['category','series']),
        (rl_points, 'plot_values', ['dataset','category','series','metric']),
        (rl_pairs, 'paired_improvements', cohort + ['category','series','reference','metric']),
        (rl_differences, 'improvement_values', ['dataset','category','series','reference','metric'])]:
        equal_frame(pd.read_csv(presentation / 'rl_comparison/tables' / (name + '.csv')), table, key)
    policies = []
    for model in original['config']['rl_models']:
        file = ROOT / model['path']; assert p.sha(file) == model['sha256']
        artifact = p.read(file)
        actions = {state: ('update','reset')[int(np.argmax(values))] for state, values in artifact['table'].items()}
        policies.append({'variant':model['variant'],'seed':model['seed'],'actions':actions,
                         'alarm_sensitive':any(actions[s] != actions['1'+s[1:]] for s in actions if s.startswith('0'))})
    p.write(out / 'table_audit.json', {'status':'passed','completed_utc':p.now(),'original_runs':1464,
            'paired_gate_comparisons':len(pairs),'fixed_cohort_rows':len(fixed_rows),'rl_cohort_rows':len(rl),
            'saved_table_aggregation_matches':True,'same_mask_within_cohorts':True,'controllers':policies,
            'limitation':'Original raw traces are unavailable; these checks validate saved-table arithmetic and provenance, not independent original trace regeneration.'})
    render = out / 'render_check'; render.mkdir(exist_ok=True)
    (render / 'fixed_responses').mkdir(exist_ok=True)
    (render / 'rl_comparison').mkdir(exist_ok=True)
    (render / 'rl_comparison/tables').mkdir(exist_ok=True)
    fixed_module.render(fixed_points, render / 'fixed_responses')
    rl_module.render(rl_points, rl_differences, render / 'rl_comparison')
    baseline = p.load_module('audit_baseline_render', '06_analysis/dataset_summary/baseline.py')
    baseline_points = pd.read_csv(presentation / 'baseline_clean_vs_poisoned/plot_values.csv')
    destination = render / 'baseline'; destination.mkdir(exist_ok=True)
    for dataset in ['SEA','RBF','RADAR']:
        baseline.figure(baseline_points[baseline_points.dataset.eq(dataset)], dataset, destination)
    p.write(out / 'render_audit.json', {'status':'passed','pdfs':len(list(render.rglob('*.pdf'))),
            'completed_utc':p.now(),'scope':'Original baseline, fixed-policy and RL renderer functions in isolated output folders'})
    print('Original coverage, paired arithmetic, scoring masks, RL aggregation and isolated rendering passed.')


if __name__ == '__main__':
    main()
