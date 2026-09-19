"""Comparison repetition counts never change dataset or condition weights."""
import unittest
import pandas as pd
from extensions.complete_study.balance import comparison_index,CATEGORIES
from extensions.complete_study.report import balanced

class DatasetBalanceTests(unittest.TestCase):
    def test_shared_radar_clean_has_five_references_not_five_independent_inputs(self):
        spec={'datasets':['SEA_A','RBF_I','radar'],'evaluation_seeds':[115,116,117,118,119],
              'radar':{'evaluation_attack_assignments':[0,1,2,3,4]}}
        manifests={}
        for ds in spec['datasets']:
            items={}
            for mode,level in CATEGORIES:
                seeds=([None] if mode=='clean' else range(5)) if ds=='radar' else spec['evaluation_seeds']
                for seed in seeds:
                    key=f'{ds}_{mode}_{level}_{seed}'
                    items[key]={'key':key,'mode':mode,'level':level,'rows':1000,'base_seed':seed,'attack_assignment':seed}
            manifests[ds]={'conditions':items}
        result=comparison_index(spec,manifests)
        self.assertEqual(len(result),105)
        self.assertTrue(result.groupby('stream').size().eq(35).all())
        self.assertTrue(result.groupby(['stream','repetition']).size().eq(7).all())
        radar=result[result.stream.eq('radar')]
        self.assertEqual(radar.condition.nunique(),31)
        self.assertEqual(radar[radar['mode'].eq('clean')].condition.nunique(),1)
        self.assertTrue(result.groupby('stream').within_dataset_weight.sum().sub(1).abs().lt(1e-12).all())

    def test_single_clean_reference_has_equal_condition_weight(self):
        rows=[]
        for mode,level in CATEGORIES:
            for _ in range(1 if mode=='clean' else 5):
                rows.append(dict(stream='radar',mode=mode,level=level,score=1. if mode=='clean' else 0.))
        result=balanced(pd.DataFrame(rows),['stream'],['score'])
        self.assertAlmostEqual(result.score.iloc[0],1/7)

if __name__=='__main__':unittest.main()
