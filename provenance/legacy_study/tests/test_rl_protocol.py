"""Verify RL learning, frozen evaluation and delayed reward boundaries."""
import unittest
from copy import deepcopy
from unittest.mock import patch
import numpy as np
import pandas as pd
import test_update_safety as safety
from common import protocol as p
rl=p.load_module('tested_rl','04_rl_training/controller.py')


class RLTests(unittest.TestCase):
    def setUp(self):
        fixture=safety.EngineTests();fixture.setUp()
        self.obs,self.audit,self.config=fixture.obs,fixture.audit,fixture.config
        self.settings={'learning_rate':.1,'discount':.9,'epsilon':.3,'work_penalty':.002,'rejection_penalty':.002}

    def test_training_updates_q_table_with_delayed_evidence(self):
        agent=rl.Controller(self.settings,7,True)
        safety.engine.execute(self.obs,self.audit,[0,1],'rl',True,self.config,agent)
        self.assertTrue(agent.table)
        self.assertTrue(any(t['q_before']!=t['q_after'] for t in agent.transitions))
        self.assertTrue(all(t['reward_block']==t['action_block']+1 for t in agent.transitions))
        self.assertTrue(all(t['audit_latest_row']<20*t['reward_block'] for t in agent.transitions))

    def test_evaluation_keeps_q_table_frozen(self):
        agent=rl.Controller(self.settings,7,False,{'0':[1.,2.],'1':[3.,4.]})
        initial=deepcopy(agent.table)
        safety.engine.execute(self.obs,self.audit,[0,1],'rl',True,self.config,agent)
        self.assertEqual(initial,agent.table)
        self.assertTrue(all(t['q_before']==t['q_after'] for t in agent.transitions))

    def test_rl_has_only_two_states_and_two_actions(self):
        agent=rl.Controller(self.settings,7,False)
        self.assertEqual(set(agent.table),{'0','1'})
        self.assertEqual(rl.ACTIONS,('update','reset'))
        self.assertEqual(agent.state(False),'0')
        self.assertEqual(agent.state(True),'1')

    def test_rl_reset_cannot_bypass_rejection(self):
        agent=rl.Controller(self.settings,7,False)
        with patch.object(agent,'choose',return_value='reset'),patch.object(safety.engine,'assess',safety.EngineTests.reject):
            pred,events,stats=safety.engine.execute(self.obs,self.audit,[0,1],'rl',True,self.config,agent)
        self.assertEqual(stats['resets'],0)
        self.assertTrue(events.loc[events.scored,'reset_requested'].all())
        self.assertTrue(pred.loc[pred.scored,'first_admitted_block'].eq(-1).all())

    def test_rl_without_screening_is_rejected(self):
        with self.assertRaises(ValueError):
            safety.engine.execute(self.obs,self.audit,[0,1],'rl',False,self.config,rl.Controller(self.settings,7))

    def test_reward_cannot_use_current_block(self):
        agent=rl.Controller(self.settings,7)
        model=safety.FixedModel([{0:1.,1:0.},{0:0.,1:1.}])
        agent.record('0','update',model,1,0,False)
        with self.assertRaises(AssertionError):
            agent.settle(self.audit,['f0'],20,1,'next')

    def test_benchmark_grid_has_unique_no_reset_arms(self):
        config=p.read(p.ROOT/'configs/thesis.json');config['rl_models']=[{'seed':s, 'variant':v} for v in ['alarm','alarm_persistence'] for s in [7,17,27]]
        arms=p.arms(config)
        self.assertEqual(len(arms),24)
        self.assertEqual(sum(a['policy']=='none' for a in arms),2)
        self.assertEqual(len({p.run_id('input',a) for a in arms}),24)

    def test_persistence_state_uses_seven_completed_blocks(self):
        """The current error is usable only after prediction and before fitting."""
        agent=rl.Controller(dict(self.settings,variant='alarm_persistence'),7)
        self.assertEqual(set(agent.table),{'00','01','10','11'})
        for error in [.1]*5+[.2]:
            self.assertEqual(agent.state(False,error),'00')
            self.assertFalse(agent.evidence['rl_error_ready'])
        self.assertEqual(agent.state(True,.2),'11')
        self.assertAlmostEqual(agent.evidence['rl_error_baseline'],.1)
        self.assertAlmostEqual(agent.evidence['rl_error_recent'],.2)

    def test_error_history_uses_disjoint_windows_and_strict_margin(self):
        """The recent observations must not leak into their preceding baseline."""
        agent=rl.Controller(dict(self.settings,variant='alarm_persistence'),7)
        for error in [0.]*5+[.02,.02]:
            state=agent.state(False,error)
        self.assertEqual(state,'00')
        agent.begin_episode()
        for error in [0.]*5+[.02,.021]:
            state=agent.state(False,error)
        self.assertEqual(state,'01')
        self.assertEqual(agent.state(True,0.),'10')

    def test_new_episode_clears_history_but_preserves_q(self):
        """Separate stream episodes cannot share old error observations."""
        agent=rl.Controller(dict(self.settings,variant='alarm_persistence'),7)
        agent.table['11']=[2.,3.]
        for error in [.1]*5+[.9,.9]:
            agent.state(True,error)
        agent.begin_episode()
        self.assertEqual(agent.state(True,.9),'10')
        self.assertEqual(agent.table['11'],[2.,3.])

    def test_both_representations_obey_gate_for_either_action(self):
        """Continue and reset both retain the active model when rejected."""
        for variant in ['alarm','alarm_persistence']:
            for action in rl.ACTIONS:
                with self.subTest(variant=variant,action=action):
                    agent=rl.Controller(dict(self.settings,variant=variant),7)
                    with patch.object(agent,'choose',return_value=action),patch.object(safety.engine,'assess',safety.EngineTests.reject):
                        pred,events,stats=safety.engine.execute(self.obs,self.audit,[0,1],'rl',True,self.config,agent)
                    self.assertEqual(stats['resets'],0)
                    self.assertTrue(pred.loc[pred.scored,'first_admitted_block'].eq(-1).all())

    def test_frozen_four_state_controller_keeps_all_q_values(self):
        """The richer state does not enable evaluation-time learning."""
        agent=rl.Controller(dict(self.settings,variant='alarm_persistence'),7)
        original=deepcopy(agent.table)
        safety.engine.execute(self.obs,self.audit,[0,1],'rl',True,self.config,agent)
        self.assertEqual(original,agent.table)
        self.assertTrue(all(t['reward_block']==t['action_block']+1 for t in agent.transitions))


if __name__=='__main__':unittest.main()
