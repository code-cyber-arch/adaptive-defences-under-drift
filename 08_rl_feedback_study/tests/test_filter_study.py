"""Protocol regressions for upstream withholding and supervised holdouts."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import protocol as p
from extensions.filter_study import protocol as s, separability

original = p.load_module('filter_original_engine_test', '05_experiment/engine.py')
filtered = p.load_module('filter_extension_engine_test', 'extensions/filter_study/engine.py')


class MaskTests(unittest.TestCase):
    def test_random_controls_match_tail_and_reservations(self):
        n, block = 103, 20
        reserved = np.zeros(n, bool)
        reserved[::10] = True
        poison = np.zeros(n, bool)
        poison[[25, 42, 101]] = True
        for unit in ('rows', 'blocks'):
            oracle = s.withholding('oracle_' + unit, poison, reserved, block, 20, 7)
            random = s.withholding('random_' + unit, poison, reserved, block, 20, 7)
            self.assertEqual(int(oracle.sum()), int(random.sum()))
            self.assertFalse(random[reserved].any())
            self.assertFalse(random[:20].any())
            np.testing.assert_array_equal(random, s.withholding('random_' + unit, poison, reserved, block, 20, 7))
        self.assertTrue(s.withholding('oracle_blocks', poison, reserved, block, 20, 7)[102])

    def test_no_poison_controls_do_not_remove_anything(self):
        for kind in s.CONTROLS:
            self.assertFalse(s.withholding(kind, np.zeros(103, bool), np.zeros(103, bool), 20, 20, 1).any())

    def test_future_splice_source_is_excluded_from_radar_fitting(self):
        ids = np.array([100, 200, 520, 720, 900])
        source = np.array([-1, 900, -1, 100, -1])
        split, start = s.temporal_partitions(ids, source, 1000, {'block_size': 20, 'warmup_rows': 20})
        np.testing.assert_array_equal(split, [0, -1, 1, -1, 2])
        self.assertEqual(start, 720)

    def test_features_only_is_invariant_to_labels_and_hidden_metadata(self):
        frame = pd.DataFrame({'f0': [1., 2.], 'label': [0, 1], 'row_id': [10, 20], 'atk_flip': [True, False]})
        before = s.inputs(frame, 'features', [0, 1])
        changed = frame.assign(label=[1, 0], row_id=[100, 200], atk_flip=[False, True])
        np.testing.assert_array_equal(before, s.inputs(changed, 'features', [0, 1]))
        self.assertEqual(s.inputs(frame, 'features_label', [0, 1]).shape[1], 3)
        np.testing.assert_array_equal(s.fingerprints(frame), s.fingerprints(changed))

    def test_threshold_keeps_ties_and_uses_legitimate_validation_only(self):
        scores = np.array([.1, .2, .2, .9])
        labels = np.array([False, False, False, True])
        threshold = s.threshold_from_validation(scores, labels, .05)
        self.assertEqual(threshold, .2)
        self.assertEqual(separability.binary_scores(labels, scores, threshold)['fp'], 0)


class EngineTests(unittest.TestCase):
    def setUp(self):
        self.config = p.read(p.ROOT / 'configs/smoke.json')
        self.config.update(block_size=20, warmup_rows=20)
        self.config['detector'] = {'name': 'adwin'}
        n = 163
        rng = np.random.default_rng(10)
        reserved = np.zeros(n, bool)
        reserved[::10] = True
        x = rng.normal(size=n)
        self.obs = pd.DataFrame({'row_id': np.arange(n), 'f0': x, 'label': (x > 0).astype(int), 'audit_reserved': reserved})
        self.audit = self.obs.loc[reserved, ['row_id', 'f0', 'label']].copy()
        self.audit['release_block'] = self.audit.row_id // 20 + 1

    def test_no_filter_matches_original_predictions_and_decisions(self):
        for policy in ['none', 'immediate', 'confirmed']:
            for guarded in [False, True]:
                a, ea, _ = original.execute(self.obs, self.audit, [0, 1], policy, guarded, self.config)
                b, eb, _ = filtered.execute(self.obs, self.audit, [0, 1], policy, guarded, self.config)
                pd.testing.assert_frame_equal(a, b[a.columns])
                columns = ['raw_fire', 'reset_requested', 'reset_committed', 'accepted', 'candidate_rows', 'audit_latest_row']
                pd.testing.assert_frame_equal(ea[columns], eb[columns])

    def test_all_withheld_blocks_keep_scoring_but_do_not_train_or_monitor(self):
        mask = (self.obs.row_id >= 20).to_numpy() & ~self.obs.audit_reserved.to_numpy()
        with patch.object(filtered.detectors, 'update', side_effect=AssertionError('Detector must not be called')):
            pred, events, _ = filtered.execute(self.obs, self.audit, [0, 1], 'confirmed', True, self.config, withheld=mask)
        self.assertTrue(pred.loc[mask, 'scored'].all())
        self.assertTrue(pred.loc[mask, 'prediction'].ge(0).all())
        self.assertFalse(pred.loc[mask, 'candidate_training_visits'].any())
        self.assertFalse(events.loc[events.scored, 'monitor_rows'].any())
        self.assertEqual(len(pred), 163)

    def test_rejected_rows_are_absent_from_reset_refits(self):
        mask = np.zeros(len(self.obs), bool)
        mask[25:38] = True
        mask[self.obs.audit_reserved] = False
        with patch.object(filtered.detectors, 'update', return_value=True):
            pred, _, _ = filtered.execute(self.obs, self.audit, [0, 1], 'immediate', False, self.config, withheld=mask)
        self.assertFalse(pred.loc[mask, 'candidate_training_visits'].any())
        self.assertTrue(pred.loc[mask, 'scored'].all())

    def test_filter_cannot_withhold_warmup_or_audit(self):
        for row in [1, 30]:
            mask = np.zeros(len(self.obs), bool)
            mask[row] = True
            with self.assertRaisesRegex(ValueError, 'preserve'):
                filtered.execute(self.obs, self.audit, [0, 1], 'none', True, self.config, withheld=mask)


if __name__ == '__main__':
    unittest.main()
