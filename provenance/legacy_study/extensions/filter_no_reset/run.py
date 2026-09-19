"""Matched post-hoc no-reset controls for the frozen filtering study.

Run from the project root: .venv/bin/python -B -m extensions.filter_no_reset.run --workers 2
Saved inference decisions are reused; no filter is refitted or recalibrated.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path
import fcntl
import hashlib
import uuid
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from common import protocol as p, channels

FILTERS = ('none', 'learned_features', 'learned_features_label')
SOURCE = p.ROOT / 'results/filter_study'
INPUT = SOURCE / 'execution_inputs/results/evaluation'
OUT = p.ROOT / 'results/filter_no_reset'


def mask_sha(mask):
    return hashlib.sha256(np.asarray(mask, dtype=np.bool_).tobytes()).hexdigest()


def verify_trace(pred, events, stats, mask, scoring, reserved):
    assert not events.reset_requested.any() and not events.reset_committed.any()
    assert not events.raw_fire.any() and stats['resets'] == 0
    assert stats['policy'] == 'none' and stats['guarded']
    assert np.array_equal(pred.filter_withheld, mask)
    assert np.array_equal(pred.comparison_scored, scoring)
    assert not pred.loc[reserved | mask, 'candidate_training_visits'].any()
    assert not pred.loc[reserved | mask, 'accepted_training_visits'].any()
    assert pred.loc[reserved | mask, 'first_admitted_block'].lt(0).all()
    assert not scoring[reserved].any()
    assert pred.loc[scoring, 'prediction'].ge(0).all()
    assert events.loc[events.scored, 'audit_latest_row'].lt(events.loc[events.scored, 'start_row']).all()
    assert not stats['accepted_refit_rows']


def condition_job(meta, config):
    engine = p.load_module('matched_filter_engine', 'extensions/filter_study/engine.py')
    scorer = p.load_module('matched_filter_scoring', '06_analysis/dataset_summary/baseline.py')
    root = p.ROOT / 'results/evaluation'
    obs, audit, truth = channels.observations(root, meta), channels.protected(root, meta), channels.truth(root, meta)
    reserved = obs.audit_reserved.to_numpy(bool)
    poisoned = (truth.atk_flip | truth.atk_burst).to_numpy(bool)
    records = []
    for kind in FILTERS:
        source_folder = SOURCE / 'runs' / meta['key'] / 'adwin' / kind
        source_summary = p.read(source_folder / 'summary.json')
        pred_file = source_folder / 'predictions.parquet'
        assert p.sha(pred_file) == source_summary['files']['predictions.parquet']
        reference = pd.read_parquet(pred_file, columns=['row_id', 'audit_reserved', 'filter_withheld', 'comparison_scored'])
        assert np.array_equal(reference.row_id, obs.row_id)
        assert np.array_equal(reference.audit_reserved, reserved)
        mask, scoring = reference.filter_withheld.to_numpy(bool), reference.comparison_scored.to_numpy(bool)
        start = source_summary['activation_row']
        assert int(scoring.sum()) == source_summary['host_rows']
        folder = OUT / 'runs' / meta['key'] / kind
        if (folder / 'summary.json').exists():
            record = p.read(folder / 'summary.json')
            assert record['source_summary_sha256'] == p.sha(source_folder / 'summary.json')
            for name, digest in record['files'].items():
                assert p.sha(folder / name) == digest
            records.append(record)
            continue
        local = deepcopy(config)
        local['detector'] = {'name': 'none'}
        with threadpool_limits(limits=1):
            pred, events, stats = engine.execute(obs, audit, meta['classes'], 'none', True, local, withheld=mask)
        pred['comparison_scored'] = scoring
        verify_trace(pred, events, stats, mask, scoring, reserved)
        y = truth.loc[scoring, 'host_reference_label'].to_numpy(int)
        yh = pred.loc[scoring, 'prediction'].to_numpy(int)
        probs = pred.loc[scoring, [f'p_{c}' for c in meta['classes']]].to_numpy()
        assert np.isfinite(probs).all()
        metrics = scorer.predictive_metrics(y, yh, probs, meta['classes'], meta['benign_class'])
        metrics = {k: float(v) if np.isfinite(v) else None for k, v in metrics.items()}
        exposure = (obs.row_id.to_numpy() >= start) & ~reserved
        attacked, legitimate = poisoned & exposure, ~poisoned & exposure
        admitted = pred.first_admitted_block.to_numpy() >= 0
        ratio = lambda a,b: float(a / b) if b else None
        record = {k: source_summary[k] for k in ('condition','stream','mode','level','base_key','attack_seed','activation_row','host_rows')}
        record.update(detector='none', filter=kind, accuracy=float(np.mean(y == yh)), **metrics,
                      poison_admission_rate=ratio((admitted & attacked).sum(), attacked.sum()),
                      legitimate_withhold_rate=ratio((~admitted & legitimate).sum(), legitimate.sum()),
                      filter_poison_recall=ratio((mask & attacked).sum(), attacked.sum()),
                      filter_legitimate_rejection=ratio((mask & legitimate).sum(), legitimate.sum()),
                      **stats, scored_resets=0, source_summary_sha256=p.sha(source_folder/'summary.json'),
                      source_predictions_sha256=p.sha(pred_file), scoring_mask_sha256=mask_sha(scoring),
                      withholding_mask_sha256=mask_sha(mask), reused_frozen_inference=True)
        temporary = OUT / 'work_in_progress' / uuid.uuid4().hex
        temporary.mkdir(parents=True)
        pred.to_parquet(temporary/'predictions.parquet', index=False)
        events.to_parquet(temporary/'events.parquet', index=False)
        record['files'] = {f.name: p.sha(f) for f in temporary.glob('*.parquet')}
        p.write(temporary/'summary.json', record)
        folder.parent.mkdir(parents=True, exist_ok=True)
        temporary.rename(folder)
        records.append(record)
    return records


def run(workers):
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT/'.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        old = p.read(SOURCE/'request.json')
        for name,key in [('request.json','request_sha256'), ('01_attacks/manifest.json','manifest_sha256')]:
            assert p.sha(INPUT/name) == old['inputs']['evaluation'][key]
        for name,digest in old['original_sources'].items():
            assert p.sha(p.ROOT/name) == digest, name
        for name,digest in old['extension_sources'].items():
            assert p.sha(p.ROOT/'extensions/filter_study'/name) == digest, name
        manifest = p.read(INPUT/'01_attacks/manifest.json')
        for name,digest in manifest['outputs'].items():
            assert p.sha(p.ROOT/'results/evaluation/01_attacks'/name) == digest, name
        contract = {'version':1, 'design':'Post-hoc matched continuous learning without resets; screening retained',
                    'planned':183, 'filters':list(FILTERS), 'environment':p.environment(),
                    'source_request_sha256':p.sha(SOURCE/'request.json'),
                    'code':{f.name:p.sha(f) for f in Path(__file__).parent.glob('*.py')},
                    'config':p.read(INPUT/'request.json')['config']}
        assert contract['environment']['packages'] == old['environment']['packages']
        if (OUT/'request.json').exists():
            assert p.read(OUT/'request.json') == contract, 'Contract changed'
        p.write(OUT/'request.json', contract)
        conditions = list(manifest['conditions'].values())
        assert len(conditions)*len(FILTERS) == 183
        p.write(OUT/'status.json', {'phase':'running','completed':0,'planned':183,'updated_utc':p.now()})
        records = []
        try:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                futures = [pool.submit(condition_job,c,contract['config']) for c in conditions]
                for future in as_completed(futures):
                    records.extend(future.result())
                    p.write(OUT/'status.json', {'phase':'running','completed':len(records),'planned':183,'updated_utc':p.now()})
                    print(f'Matched no-reset controls: {len(records)}/183', flush=True)
            frame = pd.DataFrame([{k:v for k,v in r.items() if k!='files'} for r in records])
            assert len(frame)==183 and not frame.duplicated(['condition','filter']).any()
            frame.sort_values(['condition','filter']).to_csv(OUT/'metrics.csv', index=False)
            p.write(OUT/'status.json', {'phase':'runs_complete_awaiting_independent_audit','completed':183,'planned':183,
                                      'metrics_sha256':p.sha(OUT/'metrics.csv'),'updated_utc':p.now()})
        except BaseException as error:
            p.write(OUT/'status.json', {'phase':'failed','error':repr(error),'completed':len(records),'planned':183,'updated_utc':p.now()})
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    if args.workers < 1: parser.error('workers must be positive')
    run(args.workers)
