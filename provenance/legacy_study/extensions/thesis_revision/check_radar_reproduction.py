"""Full RADAR reproduction check for the largest archived baseline discrepancy."""
from pathlib import Path
import sys
from copy import deepcopy
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from common import protocol as p, channels

key='radar__instance__severe__a1'
inputs=ROOT/'results/filter_study/execution_inputs/results/evaluation'
meta=p.read(inputs/'01_attacks/manifest.json')['conditions'][key]
obs=channels.observations(ROOT/'results/evaluation',meta)
audit=channels.protected(ROOT/'results/evaluation',meta)
records=[]
for policy,engine_path,folder in [
    ('none','05_experiment/engine.py',ROOT/'results/filter_no_reset/runs'/key/'none'),
    ('confirmed','extensions/filter_study/engine.py',ROOT/'results/filter_study/runs'/key/'adwin/none')]:
    config=deepcopy(p.read(inputs/'request.json')['config'])
    config['detector']={'name':'none'} if policy=='none' else {'name':'adwin','seed':p.seed('monitor',key,'adwin')}
    engine=p.load_module('radar_reproduction_'+policy,engine_path)
    saved=pd.read_parquet(folder/'predictions.parquet')
    events=pd.read_parquet(folder/'events.parquet')
    with threadpool_limits(limits=1):
        pred,ev,stats=engine.execute(obs,audit,meta['classes'],policy,True,config)
    pd.testing.assert_frame_equal(pred,saved[pred.columns],check_exact=True)
    ec=[c for c in ev if not c.endswith('_seconds')]
    pd.testing.assert_frame_equal(ev[ec],events[ec],check_exact=True)
    records.append({'condition':key,'policy':policy,'rows':len(pred),'saved_trace_identical':True,
                    'source_predictions_sha256':p.sha(folder/'predictions.parquet'),
                    'source_events_sha256':p.sha(folder/'events.parquet'),'engine':engine_path,
                    'seconds':stats['seconds']})
    print(records[-1],flush=True)
p.write(ROOT/'results/methods_revision/radar_reproduction.json',{
    'status':'passed','scope':'One complete RADAR severe label-poisoning assignment; no-reset original engine versus saved new extension control, and confirmed-reset extension engine versus saved original Mac filter-study reference. Exact comparison excludes timing fields only.',
    'records':records,'environment':p.environment(),'script_sha256':p.sha(Path(__file__))})
