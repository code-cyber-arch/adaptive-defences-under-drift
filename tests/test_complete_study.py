"""Whole-study coverage and matched execution invariants."""
import unittest
from extensions.complete_study import workers as w, separability
from common import protocol as p
from unittest.mock import patch
from pathlib import Path
import tempfile
import numpy as np
import pandas as pd

class CompleteStudyTests(unittest.TestCase):
    def test_complete_arm_counts(self):
        ds=['adwin','hddm_w','hellinger','d3_oof'];seeds=[7,17,27,37,47]
        self.assertEqual(len(w.fixed_arms(ds)),18)
        self.assertEqual(len(w.filter_specs(seeds)),23)
        no_reset=[a for a in w.filter_specs(seeds) if a[0]=='none' or a[0].startswith('learned_')]
        self.assertEqual(len(no_reset),11)
        self.assertEqual((35+35+31)*(4*23+11),10403)
        self.assertEqual(len(set(w.filter_specs(seeds))),23)

    def test_radar_partition_not_split_again(self):
        n=20
        obs=pd.DataFrame({'row_id':np.arange(n),'f0':np.arange(n,dtype=float),'label':np.arange(n)%2,'audit_reserved':False})
        truth=pd.DataFrame({'row_id':np.arange(n),'atk_flip':False,'atk_burst':False})
        config={'warmup_rows':2}
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            p.write(root/'01_attacks/manifest.json',{'conditions':{'radar_training':{'stream':'radar','key':'radar_training','mode':'clean','level':'none','source_partition':{'role':'training'}}}})
            with patch.object(separability.channels,'observations',return_value=obs),patch.object(separability.channels,'truth',return_value=truth):
                rows=list(separability.collect(root,'radar','training',config))[0]
                self.assertEqual(rows.row_id.tolist(),list(range(2,20)))
                self.assertTrue(rows.split.eq(0).all())
                with self.assertRaisesRegex(ValueError,'partition'):
                    list(separability.collect(root,'radar','evaluation',config))

    def test_changed_cached_contract_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);p.write(folder/'summary.json',{'contract':{'seed':7},'files':{}})
            with self.assertRaisesRegex(ValueError,'contract'):w.cached(folder,{'seed':17})

if __name__=='__main__':unittest.main()
