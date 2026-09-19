"""Dataset identity and split-local poisoning are enforced before training."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from common import protocol as p
prepare=p.load_module('tested_dataset_partitions','01_attacks/per_dataset.py')
worker=p.load_module('tested_dataset_identity','05_experiment/policy_robustness.py')
training=p.load_module('tested_dataset_training','04_rl_training/train.py')


class DatasetPartitionTests(unittest.TestCase):
    def test_radar_boundaries_are_ordered_disjoint_and_embargoed(self):
        bounds=prepare.radar_bounds(484753)
        self.assertEqual(bounds,{'training':(0,242000),'validation':(243000,339000),'evaluation':(340000,484753)})
        self.assertLess(bounds['training'][1],bounds['validation'][0])
        self.assertLess(bounds['validation'][1],bounds['evaluation'][0])

    def test_attack_generation_and_row_map_stay_inside_each_split(self):
        with tempfile.TemporaryDirectory(dir=p.ROOT/'tests') as tmp:
            root=Path(tmp); n=50000
            clean=pd.DataFrame({'f0':np.arange(n,dtype=float),'f1':np.arange(n,dtype=float)+.25,'label':np.arange(n)%8})
            source=root/'full.parquet';clean.to_parquet(source,index=False)
            meta={'stream':'radar','rows':n,'classes':list(range(8)),'benign_class':7,
                  'segment_boundaries':[10000,20000,30000,40000]}
            record={'snapshot':source.relative_to(p.ROOT).as_posix(),'metadata':meta,'source_sha256':p.sha(source)}
            for role,(start,end) in prepare.radar_bounds(n).items():
                config=dict(p.read(p.ROOT/'configs/thesis.json'),dataset='radar',data_role=role,
                            radar_embargo_blocks=1,radar_attack_assignments=[0])
                stage=root/role
                manifest=prepare.build(stage,config,record)
                self.assertEqual(len(manifest['conditions']),7)
                mapping=pd.read_parquet(stage/'00_streams/row_map.parquet')
                self.assertEqual(mapping.source_row_id.tolist(),list(range(start,end)))
                for c in manifest['conditions'].values():
                    self.assertEqual(c['source_partition']['role'],role)
                    observed=pd.read_parquet(stage/'01_attacks'/c['observations'])
                    self.assertTrue(observed.f0.between(start,end-1).all())
                    truth=pd.read_parquet(stage/'01_attacks'/c['truth'])
                    donor=truth.replay_source_row[truth.replay_source_row.ge(0)]
                    self.assertTrue(donor.lt(end-start).all())
                    if len(donor):
                        ids=truth.index[truth.replay_source_row.ge(0)]
                        self.assertTrue(np.array_equal(observed.loc[ids,'f0'], donor.to_numpy()+start))
                self.assertEqual(prepare.verify(stage)['data_role'],role)
                config['radar_attack_assignments']=[0,1]
                extended=prepare.build(stage,config,record)
                self.assertEqual(len(extended['conditions']),13)
                for name,digest in manifest['outputs'].items():
                    self.assertEqual(extended['outputs'][name],digest)

    def test_model_from_another_dataset_is_rejected(self):
        with tempfile.TemporaryDirectory(dir=p.ROOT/'tests') as tmp:
            file=Path(tmp)/'model.json'
            p.write(file,{'dataset':'SEA_A','detector':'adwin','variant':'alarm','seed':7})
            arm={'dataset':'radar','detector':'adwin','variant':'alarm','seed':7,
                 'model_path':file.relative_to(p.ROOT).as_posix(),'model_sha256':p.sha(file)}
            with self.assertRaisesRegex(ValueError,'dataset'):
                worker.load_frozen(arm)

    def test_training_refuses_a_radar_evaluation_partition(self):
        with tempfile.TemporaryDirectory(dir=p.ROOT/'tests') as tmp:
            root=Path(tmp)
            p.write(root/'01_attacks/manifest.json',{'dataset':'radar','data_role':'evaluation','conditions':{}})
            config=dict(p.read(p.ROOT/'configs/thesis.json'),dataset='radar')
            with self.assertRaisesRegex(ValueError,'training partition'):
                training.train_one(root,config,7,'alarm','adwin')

    def test_controller_grid_is_specific_to_dataset_and_detector(self):
        models=[{'dataset':ds,'detector':d,'variant':v,'seed':s,'path':'unused','sha256':'unused'}
                for ds in ['SEA_A','RBF_I','radar'] for d in ['adwin','hddm_w','hellinger','d3_oof']
                for v in ['alarm','alarm_persistence'] for s in [7,17,27,37,47]]
        self.assertEqual(len(models),120)
        for ds in ['SEA_A','RBF_I','radar']:
            for d in ['adwin','hddm_w','hellinger','d3_oof']:
                arms=worker.comparison_arms({'table':worker.policy_table('0101')},models,d,ds)
                self.assertEqual(len(arms),14)
                self.assertEqual({(a['dataset'],a['detector']) for a in arms},{(ds,d)})


if __name__=='__main__':unittest.main()
