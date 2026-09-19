"""Run the supplementary study without changing the frozen benchmark contract.

From drift-response-2: .venv/bin/python -B -m extensions.filter_study.run --workers 2
Use --smoke for a separate small integration run. Completed jobs resume by hash.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path
import os
import time
import uuid
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from . import protocol as s, separability
from common import protocol as p, channels


def condition_job(root, out, meta, config, settings):
    """One condition at a time amortises input loading across matched arms."""
    engine = p.load_module('upstream_filter_engine', 'extensions/filter_study/engine.py')
    scoring = p.load_module('filter_predictive_metrics', '06_analysis/dataset_summary/baseline.py')
    obs, audit = channels.observations(root, meta), channels.protected(root, meta)
    truth = channels.truth(root, meta)
    poisoned = (truth.atk_flip | truth.atk_burst).to_numpy(bool)
    reserved = obs.audit_reserved.to_numpy(bool)
    start = config['warmup_rows']
    split = np.full(len(obs), 2)
    if meta['stream'] == 'radar':
        split, start = s.temporal_partitions(obs.row_id.to_numpy(), truth.replay_source_row.to_numpy(), len(obs), config)
    hashes = np.load(out / 'models' / meta['stream'] / 'excluded_test_fingerprints.npy', allow_pickle=False)
    holdout = (split == 2) & ~np.isin(s.fingerprints(obs), hashes)
    score_mask = ((obs.row_id.to_numpy() >= start) & ~reserved & truth.host_eligible.to_numpy(bool) & holdout)
    # Score all matched filters on identical positions, including rows they reject.
    exposure_mask = (obs.row_id.to_numpy() >= start) & ~reserved
    masks = {kind: s.withholding(kind, poisoned, reserved, config['block_size'], start,
                              p.seed(settings['seed'], meta['key'], kind)) for kind in s.CONTROLS}
    inference = {kind: 0.0 for kind in masks}
    for view in s.VIEWS:
        bundle = separability.load_bundle(out, meta['stream'], view)
        if s.features(obs) != bundle['feature_names']:
            raise ValueError('Filter feature schema changed')
        tick = time.perf_counter()
        scores = np.zeros(len(obs))
        eligible_ids = np.flatnonzero(exposure_mask)
        with threadpool_limits(limits=1):
            for left in range(0, len(eligible_ids), 50000):
                ids = eligible_ids[left:left+50000]
                scores[ids] = bundle['model'].predict_proba(s.inputs(obs.iloc[ids], view, bundle['classes']))[:, 1]
        name = 'learned_' + view
        masks[name] = (scores > bundle['threshold']) & exposure_mask
        inference[name] = time.perf_counter() - tick
    records = []
    for detector in settings['detectors']:
        for kind, mask in masks.items():
            folder = out / 'runs' / meta['key'] / detector / kind
            summary_file = folder / 'summary.json'
            if summary_file.exists():
                record = p.read(summary_file)
                s.verify_files(folder, record)
                records.append(record)
                continue
            local = deepcopy(config)
            local['detector'] = {'name': detector, 'seed': p.seed('monitor', meta['key'], detector)}
            with threadpool_limits(limits=1):
                pred, events, stats = engine.execute(obs, audit, meta['classes'],
                    settings['response'], settings['screened'], local, withheld=mask)
            ids = score_mask
            probabilities = pred.loc[ids, [f'p_{c}' for c in meta['classes']]].to_numpy()
            y = truth.loc[ids, 'host_reference_label'].to_numpy(int)
            yh = pred.loc[ids, 'prediction'].to_numpy(int)
            if len(y) and not np.isfinite(probabilities).all():
                raise AssertionError('Nonfinite probabilities on scoring rows')
            predictive = scoring.predictive_metrics(y, yh, probabilities, meta['classes'], meta['benign_class']) if len(y) else {}
            predictive = {k: float(v) if np.isfinite(v) else None for k, v in predictive.items()}
            admitted = pred.first_admitted_block.to_numpy() >= 0
            attacked, legitimate = poisoned & exposure_mask, ~poisoned & exposure_mask
            ratio = lambda a, b: float(a / b) if b else None
            record = {'condition': meta['key'], 'stream': meta['stream'], 'mode': meta['mode'],
                      'level': meta['level'], 'base_key': meta['base_key'], 'attack_seed': meta.get('attack_seed'),
                      'detector': detector, 'filter': kind, 'activation_row': int(start),
                      'host_rows': int(ids.sum()), 'accuracy': float(np.mean(y == yh)) if len(y) else None,
                      **predictive, 'poison_admission_rate': ratio((admitted & attacked).sum(), attacked.sum()),
                      'legitimate_withhold_rate': ratio((~admitted & legitimate).sum(), legitimate.sum()),
                      'filter_poison_recall': ratio((mask & attacked).sum(), attacked.sum()),
                      'filter_legitimate_rejection': ratio((mask & legitimate).sum(), legitimate.sum()),
                      'filter_inference_seconds': inference[kind], **stats,
                      'total_seconds': stats['seconds'] + inference[kind],
                      'scored_resets': int(events.loc[events.start_row.ge(start), 'reset_committed'].sum())}
            temp = out / 'work_in_progress' / uuid.uuid4().hex
            temp.mkdir(parents=True)
            pred['comparison_scored'] = score_mask
            pred.to_parquet(temp / 'predictions.parquet', index=False)
            events.to_parquet(temp / 'events.parquet', index=False)
            record['files'] = {f.name: p.sha(f) for f in temp.glob('*.parquet')}
            p.write(temp / 'summary.json', record)
            folder.parent.mkdir(parents=True, exist_ok=True)
            temp.rename(folder)
            records.append(record)
    return records


def input_contract(roots):
    result = {}
    for role, root in roots.items():
        request = root / 'request.json'
        manifest = root / '01_attacks/manifest.json'
        result[role] = {'root': str(root.relative_to(p.ROOT)), 'request_sha256': p.sha(request),
                        'manifest_sha256': p.sha(manifest)}
        for name, digest in p.read(manifest)['outputs'].items():
            if p.sha(root / '01_attacks' / name) != digest:
                raise ValueError(f'Changed input: {name}')
    return result


def run(args):
    settings = deepcopy(s.DEFAULTS)
    if args.smoke:
        settings.update(max_fit_rows=2000, max_calibration_rows=2000, detectors=['adwin'])
        settings['classifier']['max_iter'] = 5
    if args.detectors:
        settings['detectors'] = args.detectors
    roots = {name: p.ROOT / 'results' / name for name in ('training', 'validation', 'evaluation')}
    if args.smoke:
        roots['evaluation'] = p.ROOT / 'results/smoke_validation'
    out = p.output(p.ROOT / 'results' / ('filter_study_smoke' if args.smoke else 'filter_study'))
    out.mkdir(parents=True, exist_ok=True)
    # A process-owned advisory lock survives no crash and prevents duplicate writers.
    import fcntl
    with (out / '.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Another filter-study process owns this output') from None
        config = p.read(roots['evaluation'] / 'request.json')['config']
        contract = {'settings': settings, 'inputs': input_contract(roots), 'environment': p.environment(),
                    'original_sources': p.sources(),
                    'extension_sources': {f.name: p.sha(f) for f in Path(__file__).parent.glob('*.py')}}
        request = out / 'request.json'
        if request.exists() and p.read(request) != contract:
            raise ValueError('Supplementary study contract changed; use a separate output/version')
        p.write(request, contract)
        meta = p.read(roots['evaluation'] / '01_attacks/manifest.json')['conditions']
        streams = sorted({c['stream'] for c in meta.values()})
        try:
            for stream in streams:
                p.write(out / 'status.json', {'phase': 'separability', 'stream': stream, 'updated_utc': p.now()})
                separability.fit_stream(out, roots, stream, config, settings)
                print(f'Frozen filters and heldout separability: {stream}', flush=True)
            if args.stage == 'fit':
                p.write(out / 'status.json', {'phase': 'models_complete', 'updated_utc': p.now()})
                return
            planned = len(meta) * len(settings['detectors']) * (len(s.CONTROLS) + len(s.VIEWS))
            records = []
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                futures = [pool.submit(condition_job, roots['evaluation'], out, c, config, settings) for c in meta.values()]
                for future in as_completed(futures):
                    rows = future.result()
                    records.extend(rows)
                    p.write(out / 'status.json', {'phase': 'filter_evaluation', 'completed': len(records),
                                                 'planned': planned, 'updated_utc': p.now()})
                    print(f'Filter evaluation: {len(records)}/{planned}', flush=True)
            pd.DataFrame([{k: v for k, v in row.items() if k != 'files'} for row in records]).to_csv(out / 'metrics.csv', index=False)
            from .report import build, verify
            verify(out, roots['evaluation'], config, settings)
            build(out)
            p.write(out / 'status.json', {'phase': 'complete', 'completed': len(records), 'planned': planned,
                                         'verification': 'passed', 'updated_utc': p.now()})
        except BaseException as error:
            p.write(out / 'status.json', {'phase': 'failed', 'error': f'{type(error).__name__}: {error}', 'updated_utc': p.now()})
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--stage', choices=['all', 'fit'], default='all')
    parser.add_argument('--detectors', nargs='+', choices=s.DEFAULTS['detectors'])
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be positive')
    os.environ.setdefault('MPLCONFIGDIR', str(p.ROOT / 'results/.matplotlib'))
    run(args)


if __name__ == '__main__':
    main()
