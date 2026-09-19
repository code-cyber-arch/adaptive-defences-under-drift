"""Replication changes may reuse matching runs, never changed scientific settings."""
import unittest
from copy import deepcopy
from common.replication import counts, run_contract_matches, effective_config
from common import protocol as p

class ReplicationTests(unittest.TestCase):
    def test_declared_grid_counts(self):
        self.assertEqual(counts(p.read(p.ROOT/'configs/study.json')),dict(controllers=120,validation=1344,
            evaluation=5656,sensitivity=5616,shared_defaults=936,additional_feedback=4680,executions=11680))

    def test_reuse_only_grid_metadata(self):
        a={'config':{'rl':{'seeds':[7,17,27],'epochs':4},'base_seeds':[115,116,117],
                     'radar_attack_assignments':[0,1,2],'block_size':1000},
           'inputs':{'truth':'abc'},'arm':{'seed':7,'model_sha256':'model'},'condition':'fixed',
           'feedback_fraction':.05,'delay_blocks':1}
        b=deepcopy(a);b['config']['rl']['seeds'] += [37,47];b['config']['base_seeds'] += [118,119]
        b['config']['radar_attack_assignments'] += [3,4]
        self.assertTrue(run_contract_matches(a,b))
        for section,key,value in [('config','block_size',500),('inputs','truth','changed'),('arm','seed',37),('arm','model_sha256','changed')]:
            changed=deepcopy(b);changed[section][key]=value
            self.assertFalse(run_contract_matches(a,changed))
        changed=deepcopy(b);changed['config']['rl']['epochs']=5
        self.assertFalse(run_contract_matches(a,changed))
        changed=deepcopy(b);changed['delay_blocks']=5
        self.assertFalse(run_contract_matches(a,changed))

    def test_synthetic_inputs_extend_without_changing_existing_conditions(self):
        import tempfile
        import json
        from pathlib import Path
        prep=p.load_module('replication_input_test','01_attacks/per_dataset.py')
        config=deepcopy(p.read(p.ROOT/'configs/thesis.json'))
        config.update(name='evaluation',data_role='evaluation',dataset='SEA_A',streams=['SEA_A'],
                      base_seeds=[115],rows=12000,persist_inputs=False)
        config.pop('input_bank',None)
        with tempfile.TemporaryDirectory(dir=p.ROOT/'tests') as tmp:
            root=Path(tmp)
            old=json.loads(json.dumps(prep.build(root,config)))
            config['base_seeds']=[115,118]
            new=json.loads(json.dumps(prep.build(root,config)))
            self.assertEqual(len(new['conditions']),14)
            for key,value in old['conditions'].items():self.assertEqual(new['conditions'][key],value)
            for key,value in old['outputs'].items():self.assertEqual(new['outputs'][key],value)
            prep.verify(root)

    def test_training_data_must_match(self):
        a={'rl':{'seeds':[7,17,27]},'base_seeds':[112],'radar_attack_assignments':[0]}
        b=deepcopy(a);b['rl']['seeds'] += [37,47]
        self.assertEqual(effective_config(a,True),effective_config(b,True))
        for key in ['base_seeds','radar_attack_assignments']:
            c=deepcopy(b);c[key].append(99)
            self.assertNotEqual(effective_config(a,True),effective_config(c,True))

if __name__=='__main__':unittest.main()
