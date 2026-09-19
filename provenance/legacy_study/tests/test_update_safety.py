"""Tests for isolation, audit timing and update decisions."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import protocol as p
phase=p.load_module('test_gate','03_policies/gate.py')
engine=p.load_module('test_engine','05_experiment/engine.py')
preparation=p.load_module('test_preparation','01_attacks/prepare.py')


class RadarTests(unittest.TestCase):
    def setUp(self):
        self.source={'path':'data/streams/radar.parquet','sha256':'expected','rows':5,'full_source_rows':5}
        self.frame=pd.DataFrame({'f0':np.arange(5),'label':[0,1,0,1,0]})

    def load(self,config,frame=None):
        with patch.object(p,'read',return_value=self.source), patch.object(p,'sha',return_value='expected'), patch.object(preparation.pd,'read_parquet',return_value=self.frame if frame is None else frame):
            return preparation.load_radar(config)

    def test_full_radar_retains_last_row_and_order(self):
        frame,meta=self.load({'radar_rows':5})
        pd.testing.assert_frame_equal(frame,self.frame)
        self.assertEqual(meta['rows'],5)

    def test_radar_truncation_is_rejected(self):
        with self.assertRaisesRegex(ValueError,'truncation'):
            self.load({'radar_rows':3})

    def test_incomplete_radar_source_is_rejected(self):
        with self.assertRaisesRegex(ValueError,'complete source'):
            self.load({},self.frame.iloc[:-1])


class FixedModel:
    def __init__(self,probabilities):
        self.probabilities=probabilities
    def predict_proba_one(self,x):
        return self.probabilities[int(x['f0'])]


class GateTests(unittest.TestCase):
    def setUp(self):
        self.settings={'minimum_rows':4,'minimum_class_rows':2,'minimum_classes':2,'max_log_loss_increase':.02,
                       'max_balanced_error_increase':.01,'max_class_recall_loss':.05}
        self.x=[{'f0':i%2} for i in range(8)]
        self.y=np.array([i%2 for i in range(8)])
        self.good=FixedModel([{0:.9,1:.1},{0:.1,1:.9}])
        self.bad=FixedModel([{0:.1,1:.9},{0:.9,1:.1}])
    def test_harmful_update_is_rejected(self):
        self.assertFalse(phase.assess(self.good,self.bad,self.x,self.y,[0,1],self.settings)['accepted'])
    def test_beneficial_update_is_accepted(self):
        self.assertTrue(phase.assess(self.bad,self.good,self.x,self.y,[0,1],self.settings)['accepted'])
    def test_equal_update_is_accepted(self):
        self.assertTrue(phase.assess(self.good,self.good,self.x,self.y,[0,1],self.settings)['accepted'])
    def test_insufficient_audit_fails_closed(self):
        value=phase.assess(self.good,self.good,self.x[:0],self.y[:0],[0,1],self.settings)
        self.assertEqual(value['reason'],'no_protected_sample')
        self.assertFalse(value['accepted'])
    def test_single_class_sample_does_not_imply_coverage_of_other_classes(self):
        value=phase.assess(self.good,self.good,[{'f0':0}]*8,np.zeros(8,dtype=int),[0,1],self.settings)
        self.assertTrue(value['accepted'])
        self.assertEqual(value['audit_n'],8)
    def test_audit_reservation_is_reproducible_and_bounded(self):
        a=preparation.audit_positions(1003,100,.05,11)
        b=preparation.audit_positions(1003,100,.05,11)
        np.testing.assert_array_equal(a,b)
        self.assertEqual(int(a.sum()),51)
    def test_confirmation_waits_for_future_feedback(self):
        policy=phase.Response('confirmed',history=2,wait=2)
        policy.propose(0,.1,False);policy.propose(1,.1,False)
        self.assertIsNone(policy.propose(2,.8,True)[0])
        self.assertIsNone(policy.propose(3,.8,False)[0])
        self.assertEqual(policy.propose(4,.8,False)[0],[2,3,4])


class EngineTests(unittest.TestCase):
    def setUp(self):
        self.config=p.read(p.ROOT/'configs/smoke.json')
        self.config['block_size']=20;self.config['warmup_rows']=20
        self.config['audit']['fraction']=.1
        n=160;reserved=preparation.audit_positions(n,20,.1,123)
        labels=np.r_[np.zeros(20,dtype=int),np.ones(n-20,dtype=int)]
        self.obs=pd.DataFrame({'row_id':np.arange(n),'f0':np.zeros(n),'label':labels,'audit_reserved':reserved})
        self.audit=self.obs.loc[reserved,['row_id','f0','label']].copy()
        self.audit['release_block']=self.audit.row_id//20+1
    @staticmethod
    def reject(*args):
        return {'accepted':False,'reason':'validation_loss','audit_n':len(args[3]),'supported_classes':[]}
    def test_rejected_candidate_does_not_mutate_active_model(self):
        with patch.object(engine,'assess',self.reject):
            pred,events,stats=engine.execute(self.obs,self.audit,[0,1],'none',True,self.config)
        eligible=pred.scored
        self.assertTrue(pred.loc[eligible,'prediction'].eq(0).all())
        self.assertTrue(pred.loc[eligible,'first_admitted_block'].eq(-1).all())
        self.assertGreater(stats['training_visits'],stats['accepted_training_visits'])
    def test_audit_rows_never_train_or_score(self):
        pred,events,stats=engine.execute(self.obs,self.audit,[0,1],'none',False,self.config)
        a=pred.audit_reserved
        self.assertTrue(pred.loc[a,'accepted_training_visits'].eq(0).all())
        self.assertFalse(pred.loc[a,'scored'].any())
        self.assertTrue(pred.loc[a,'first_admitted_block'].eq(-1).all())
    def test_gate_cannot_read_attack_reference_metadata(self):
        self.obs['atk_flip']=False
        with self.assertRaises(ValueError):
            engine.execute(self.obs,self.audit,[0,1],'none',True,self.config)
    def test_audit_future_information_is_rejected(self):
        self.audit['release_block']=0
        with self.assertRaises(AssertionError):
            engine.execute(self.obs,self.audit,[0,1],'none',True,self.config)
    def test_reset_cannot_bypass_gate(self):
        with patch.object(engine,'assess',self.reject),patch.object(engine.detectors,'update',return_value=True):
            pred,events,stats=engine.execute(self.obs,self.audit,[0,1],'immediate',True,self.config)
        self.assertTrue(events.loc[events.scored,'reset_requested'].all())
        self.assertEqual(stats['resets'],0)
        self.assertFalse(events.reset_committed.any())


if __name__=='__main__':
    unittest.main()
