"""Figure error bars describe input repetition, not inflated fitting-seed samples."""
import unittest
import pandas as pd
from extensions.complete_study.figures import points

class FigureAggregationTests(unittest.TestCase):
    def test_fit_seeds_are_averaged_before_input_spread(self):
        frame=pd.DataFrame([{'condition':c,'category':0,'series':'rl','score':v}
                            for c,values in [('a',[0.,1.]),('b',[.4,.6])] for v in values])
        row=points(frame,[('score','Score',1)]).iloc[0]
        self.assertEqual(row.runs,2)
        self.assertAlmostEqual(row['mean'],.5)
        self.assertAlmostEqual(row.sd,0.)

    def test_shared_clean_references_do_not_inflate_error_bar_repetition(self):
        frame=pd.DataFrame([dict(condition='shared_clean',category=0,series='fixed',score=.8)]*5)
        row=points(frame,[('score','Score',1)]).iloc[0]
        self.assertEqual(row.runs,1)
        self.assertTrue(pd.isna(row.sd))

if __name__=='__main__':unittest.main()
