"""Integration-only check of common-row comparisons across all three experiments."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pandas as pd
from common import protocol as p
from extensions.complete_study import comparisons,workers as w
training=p.load_module('comparison_pilot_training','04_rl_training/train.py')


def main():
    out=p.ROOT/'results/per_dataset_study/integration_full_study'
    sources={};configs={};conditions={};models=[];jobs=[]
    for ds in ['SEA_A','RBF_I','radar']:
        for role in ['training','validation','evaluation']:
            root=out/'inputs'/role/ds;sources[(role,ds)]=root
            configs[(role,ds)]=p.read(root/'request.json')['config']
            conditions[(role,ds)]=list(p.read(root/'01_attacks/manifest.json')['conditions'].values())
        models.extend(training.train(sources[('training',ds)],configs[('training',ds)],2))
    for ds in ['SEA_A','RBF_I','radar']:
        for detector in ['adwin','hddm_w','hellinger','d3_oof']:
            arms=w.metrics.comparison_arms({'table':w.metrics.policy_table('0001')},models,detector,ds)
            for c in conditions[('evaluation',ds)]:
                jobs.extend((str(sources[('evaluation',ds)]),str(out/'evaluation_runs'),c,a,configs[('evaluation',ds)],.05,1) for a in arms)
    with ProcessPoolExecutor(max_workers=2) as pool:list(pool.map(w.metrics.run_one,jobs))
    index=pd.DataFrame([dict(stream=ds,condition=c['key'],mode=c['mode'],level=c['level'],repetition=1)
                       for ds in ['SEA_A','RBF_I','radar'] for c in conditions[('evaluation',ds)]])
    index.to_csv(out/'analysis/full_study/comparison_index.csv',index=False)
    comparisons.build(out,sources,configs,conditions,['adwin','hddm_w','hellinger','d3_oof'],2)
    print('Cross-experiment integration passed on all datasets and detectors.',flush=True)

if __name__=='__main__':main()
