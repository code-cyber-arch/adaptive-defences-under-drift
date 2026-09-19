"""Paired comparisons preserve dataset, condition, policy and detector identities."""
import unittest
import pandas as pd
from extensions.complete_study.comparisons import delta_rows,METRICS

class ComparisonMatrixTests(unittest.TestCase):
    def test_all_six_detector_pairs_within_each_policy(self):
        rows=[]
        for ds in ['SEA_A','RBF_I','radar']:
            for method in ['confirmed','rl']:
                for i,detector in enumerate(['adwin','hddm_w','hellinger','d3_oof']):
                    rows.append(dict(stream=ds,condition='fixed',method=method,detector=detector,**{m:float(i) for m in METRICS}))
        result=delta_rows(pd.DataFrame(rows),['stream','condition','method'],'detector','detector')
        self.assertEqual(len(result),3*2*6)
        self.assertTrue(result.groupby(['stream','method']).size().eq(6).all())

    def test_cross_experiment_pairs_exclude_within_experiment_pairs(self):
        rows=[dict(stream='radar',condition='fixed',detector='adwin',experiment=e,comparison_arm=e+'/'+m,**{k:float(i) for k in METRICS})
              for i,(e,m) in enumerate([('E1','a'),('E1','b'),('E2','a'),('E3','a')])]
        result=delta_rows(pd.DataFrame(rows),['stream','condition','detector'],'comparison_arm','cross_experiment')
        self.assertEqual(len(result),5)
        self.assertTrue(all(a.split('/')[0]!=b.split('/')[0] for a,b in zip(result.left,result.right)))

    def test_full_feedback_covers_seven_conditions(self):
        self.assertEqual((35+35+31)*4*(4+2*5)*3*2,33936)
        self.assertEqual(33936-(35+35+31)*4*(4+2*5),28280)

if __name__=='__main__':unittest.main()
