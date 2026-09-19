"""Check frozen policies and feedback timing without accessing research results."""
from copy import deepcopy
import unittest
import numpy as np
import pandas as pd
import test_update_safety as safety
from common import protocol as p
policy_robustness = p.load_module('tested_policy_robustness', '05_experiment/policy_robustness.py')
Controller = policy_robustness.Controller


class PolicyRobustnessTests(unittest.TestCase):
    def setUp(self):
        fixture = safety.EngineTests(); fixture.setUp()
        self.obs, self.audit, self.config = fixture.obs, fixture.audit, fixture.config
        self.settings = dict(p.read(p.ROOT / 'configs/thesis.json')['rl'], variant='alarm_persistence')

    def test_enumeration_is_complete_and_error_rule_uses_same_state(self):
        arms = policy_robustness.enumerated_arms()
        mappings = {tuple(np.argmax(a['table'][s]) for s in policy_robustness.STATES) for a in arms}
        self.assertEqual(len(mappings), 16)
        table = policy_robustness.policy_table('0101')
        agent = Controller(self.settings, 7, table=table, collect_rewards=False)
        self.assertEqual([agent.choose(s) for s in policy_robustness.STATES], ['update', 'reset', 'update', 'reset'])

    def test_frozen_decision_only_matches_original_one_block_evaluation(self):
        table = policy_robustness.policy_table('0001')
        original = Controller(self.settings, 7, table=table)
        frozen = Controller(self.settings, 7, table=table, collect_rewards=False)
        a, events_a, _ = policy_robustness.engine.execute(self.obs, self.audit, [0, 1], 'rl', True, self.config, original)
        b, events_b, _ = policy_robustness.engine.execute(self.obs, self.audit, [0, 1], 'rl', True, self.config, frozen)
        pd.testing.assert_frame_equal(a, b)
        columns = ['rl_action', 'rl_state', 'accepted', 'reset_committed', 'audit_latest_row']
        pd.testing.assert_frame_equal(events_a[columns], events_b[columns])
        self.assertEqual(frozen.table, table)
        self.assertEqual(frozen.transitions, [])

    def test_delayed_frozen_evaluation_fails_closed_until_release(self):
        self.config['audit']['delay_blocks'] = 5
        self.audit['release_block'] = self.audit.row_id // 20 + 5
        table = policy_robustness.policy_table('1111')
        agent = Controller(self.settings, 7, table=table, collect_rewards=False)
        pred, events, _ = policy_robustness.engine.execute(self.obs, self.audit, [0, 1], 'rl', True, self.config, agent)
        early = events.scored & events.block.lt(5)
        self.assertTrue(events.loc[early, 'reason'].eq('no_protected_sample').all())
        self.assertFalse(events.loc[early, 'reset_committed'].any())
        self.assertTrue(events.loc[events.audit_n.gt(0), 'audit_latest_row'].lt(
            (events.loc[events.audit_n.gt(0), 'block'] - 4) * 20).all())
        self.assertFalse(pred.loc[pred.audit_reserved, 'scored'].any())
        self.assertEqual(agent.table, table)

    def test_training_cannot_disable_rewards_or_use_long_delay(self):
        with self.assertRaises(ValueError):
            Controller(self.settings, 7, training=True, collect_rewards=False)
        self.config['audit']['delay_blocks'] = 5
        with self.assertRaises(ValueError):
            policy_robustness.engine.execute(self.obs, self.audit, [0, 1], 'rl', True, self.config,
                                    Controller(self.settings, 7, training=True))

    def test_feedback_subsets_are_nested_and_preserve_reservation(self):
        ids = np.r_[np.arange(50), 1000 + np.arange(50)]
        audit = pd.DataFrame({'row_id': ids, 'f0': 0., 'label': ids % 2, 'release_block': ids // 1000 + 1})
        selected = []
        for fraction, expected in [(.01, 20), (.02, 40), (.05, 100)]:
            result = policy_robustness.feedback_channel(audit, 2000, 1000, 'same_source', fraction, 1)
            self.assertEqual(set(result.row_id), set(audit.row_id))
            ids_selected = set(result.loc[result.release_block.le(2), 'row_id'])
            self.assertEqual(len(ids_selected), expected)
            selected.append(ids_selected)
            changed = audit.copy(); changed['label'] = 42
            again = policy_robustness.feedback_channel(changed, 2000, 1000, 'same_source', fraction, 1)
            pd.testing.assert_series_equal(result.release_block, again.release_block)
            delayed = policy_robustness.feedback_channel(audit, 2000, 1000, 'same_source', fraction, 5)
            self.assertEqual(ids_selected, set(delayed.loc[delayed.release_block.le(6), 'row_id']))
        self.assertTrue(selected[0] < selected[1] < selected[2])

    def test_full_feedback_is_original_release_schedule(self):
        ids = np.r_[np.arange(50), 1000 + np.arange(50)]
        audit = pd.DataFrame({'row_id': ids, 'f0': 0., 'label': ids % 2, 'release_block': ids // 1000 + 1})
        actual = policy_robustness.feedback_channel(audit, 2000, 1000, 'base', .05, 1)
        pd.testing.assert_frame_equal(audit, actual)


if __name__ == '__main__':
    unittest.main()
