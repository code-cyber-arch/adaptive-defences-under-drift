"""Independent behavioural checks; no production artifacts are overwritten."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from common import protocol as p
from extensions.filter_study import protocol as filtering

engine = p.load_module('audit_causal_engine', '05_experiment/engine.py')
filtered = p.load_module('audit_filtered_engine', 'extensions/filter_study/engine.py')
metrics = p.load_module('audit_metric_functions', '06_analysis/dataset_summary/baseline.py')
monitors = p.load_module('audit_monitor_functions', '02_detectors/monitors.py')
Controller = p.load_module('audit_controller', '04_rl_training/controller.py').Controller


def example():
    n, block = 2053, 50
    rng = np.random.default_rng(621)
    x = rng.normal(size=n)
    y = np.where(np.arange(n) < 1000, x > 0, x < 0).astype(int)
    reserved = np.arange(n) % 17 == 0
    reserved[::block] = True  # Include protected feedback in the three-row tail.
    obs = pd.DataFrame(dict(row_id=np.arange(n), f0=x, f1=rng.normal(size=n),
                            label=y, audit_reserved=reserved))
    audit = obs.loc[reserved, ['row_id', 'f0', 'f1', 'label']].copy()
    audit['release_block'] = audit.row_id // block + 1
    config = p.read(ROOT / 'configs/thesis.json')
    config.update(block_size=block, warmup_rows=block)
    config['detector'] = {'name': 'adwin', 'seed': 6}
    return obs, audit, config


class CausalityTests(unittest.TestCase):
    def test_future_perturbation_cannot_change_past_predictions_or_decisions(self):
        obs, audits, config = example()
        altered, changed_audits = obs.copy(), audits.copy()
        future = altered.row_id >= 1500
        altered.loc[future, ['f0', 'f1']] *= -100
        altered.loc[future, 'label'] = 1 - altered.loc[future, 'label']
        changed_audits.loc[changed_audits.row_id >= 1500, 'label'] ^= 1
        fields = ['raw_fire', 'reset_requested', 'reset_committed', 'accepted',
                  'audit_latest_row', 'candidate_rows', 'feedback_error']
        for name in ['adwin', 'hddm_w', 'hellinger', 'd3_oof']:
            for response in ['immediate', 'confirmed']:
                with self.subTest(detector=name, response=response):
                    config['detector']['name'] = name
                    a, ea, _ = engine.execute(obs, audits, [0, 1], response, True, config)
                    b, eb, _ = engine.execute(altered, changed_audits, [0, 1], response, True, config)
                    pd.testing.assert_frame_equal(a[a.row_id < 1500], b[b.row_id < 1500])
                    pd.testing.assert_frame_equal(ea.loc[ea.end_row <= 1500, fields],
                                                  eb.loc[eb.end_row <= 1500, fields])

    def test_reserved_observation_channel_values_cannot_train_or_feed_monitors(self):
        obs, audits, config = example()
        altered = obs.copy()
        altered.loc[altered.audit_reserved, ['f0', 'f1']] = 1e6
        altered.loc[altered.audit_reserved, 'label'] = 99
        a, ea, _ = engine.execute(obs, audits, [0, 1], 'confirmed', True, config)
        b, eb, _ = engine.execute(altered, audits, [0, 1], 'confirmed', True, config)
        pd.testing.assert_frame_equal(a, b)
        pd.testing.assert_series_equal(ea.raw_fire, eb.raw_fire)

    def test_hidden_truth_is_rejected_by_engine_schema(self):
        obs, audits, config = example()
        with self.assertRaisesRegex(ValueError, 'unexpected metadata'):
            engine.execute(obs.assign(atk_flip=False), audits, [0, 1], 'none', False, config)

    def test_no_filter_equivalence_across_all_four_real_monitors(self):
        obs, audits, config = example()
        fields = ['raw_fire', 'reset_requested', 'reset_committed', 'accepted', 'candidate_rows']
        for name in ['adwin', 'hddm_w', 'hellinger', 'd3_oof']:
            with self.subTest(detector=name):
                config['detector']['name'] = name
                a, ea, _ = engine.execute(obs, audits, [0, 1], 'confirmed', True, config)
                b, eb, _ = filtered.execute(obs, audits, [0, 1], 'confirmed', True, config)
                pd.testing.assert_frame_equal(a, b[a.columns])
                pd.testing.assert_frame_equal(ea[fields], eb[fields])

    def test_all_filtered_and_sparse_tail_keep_complete_scoring(self):
        obs, audits, config = example()
        withheld = (obs.row_id >= config['warmup_rows']) & ~obs.audit_reserved
        for name in ['adwin', 'hddm_w', 'hellinger', 'd3_oof']:
            config['detector']['name'] = name
            pred, events, _ = filtered.execute(obs, audits, [0, 1], 'confirmed', True, config,
                                               withheld=withheld.to_numpy())
            self.assertEqual(len(pred), len(obs))
            self.assertEqual(int(events.end_row.iloc[-1]), len(obs))
            self.assertEqual(int(pred.loc[withheld, 'candidate_training_visits'].sum()), 0)
            self.assertTrue(pred.loc[withheld, 'scored'].all())
            self.assertTrue(pred.loc[withheld, 'prediction'].ge(0).all())

    def test_frozen_controllers_stay_frozen_after_real_execution(self):
        obs, audits, config = example()
        for entry in p.read(ROOT / 'results/training/04_rl_training/manifest.json')['models']:
            record = p.read(ROOT / entry['path'])
            agent = Controller(record['settings'], record['seed'], False, record['table'])
            policy = 'rl_alarm' if entry['variant'] == 'alarm' else 'rl'
            _, events, _ = engine.execute(obs, audits, [0, 1], policy, True, config, agent)
            self.assertEqual(agent.table, record['table'])
            self.assertEqual(len(agent.transitions), int(events.scored.sum()))
            self.assertTrue(all(t['q_before'] == t['q_after'] for t in agent.transitions))
            self.assertTrue(all(t['reward_block'] == t['action_block'] + 1 for t in agent.transitions))


class NumericAndControlTests(unittest.TestCase):
    def test_multiclass_and_security_metrics_match_hand_calculation(self):
        y = np.array([0, 1, 2, 2]); predicted = np.array([0, 2, 2, 0])
        prob = np.array([[.8,.1,.1], [.1,.2,.7], [.1,.1,.8], [.6,.1,.3]])
        result = metrics.predictive_metrics(y, predicted, prob, [0, 1, 2], 0)
        self.assertAlmostEqual(result['macro_f1'], (2/3 + 0 + 1/2)/3)
        self.assertAlmostEqual(result['malicious_precision'], 1)
        self.assertAlmostEqual(result['malicious_recall'], 2/3)
        self.assertAlmostEqual(result['benign_fpr'], 0)
        # Every malicious example has a higher malicious score than the benign one.
        self.assertAlmostEqual(result['roc_auc'], 1)

    def test_undefined_security_denominators_remain_undefined(self):
        result = metrics.predictive_metrics(np.zeros(4, int), np.zeros(4, int),
                                            np.tile([1., 0.], (4, 1)), [0, 1], 0)
        self.assertTrue(np.isnan(result['malicious_recall']))
        self.assertTrue(np.isnan(result['malicious_precision']))
        self.assertTrue(np.isnan(result['roc_auc']))

    def test_random_controls_match_row_and_block_size_budgets_over_100_layouts(self):
        for seed in range(100):
            rng = np.random.default_rng(seed)
            n, block = int(rng.integers(100, 2000)), int(rng.integers(10, 70))
            reserved, poison = rng.random(n) < .1, rng.random(n) < .03
            start = 2 * block
            for unit in ['rows', 'blocks']:
                a = filtering.withholding('oracle_' + unit, poison, reserved, block, start, seed)
                b = filtering.withholding('random_' + unit, poison, reserved, block, start, seed)
                self.assertEqual(int(a.sum()), int(b.sum()))
                self.assertFalse(b[:start].any())
                self.assertFalse(b[reserved].any())
                if unit == 'blocks':
                    sizes_a = np.unique(np.flatnonzero(a)//block, return_counts=True)[1]
                    sizes_b = np.unique(np.flatnonzero(b)//block, return_counts=True)[1]
                    np.testing.assert_array_equal(np.sort(sizes_a), np.sort(sizes_b))

    def test_hellinger_known_distributions(self):
        np.testing.assert_allclose(monitors.HellingerDetector.distance(np.array([5.,0.]), np.array([0.,7.])), 1)
        np.testing.assert_allclose(monitors.HellingerDetector.distance(np.array([5.,5.]), np.array([7.,7.])), 0)

    def test_d3_detects_separable_windows_with_out_of_fold_scores(self):
        rng = np.random.default_rng(345)
        old, new = rng.normal(-5, .2, (100, 2)), rng.normal(5, .2, (100, 2))
        self.assertGreater(monitors.D3Detector(seed=3).auc(old, new), .99)


if __name__ == '__main__':
    unittest.main()
