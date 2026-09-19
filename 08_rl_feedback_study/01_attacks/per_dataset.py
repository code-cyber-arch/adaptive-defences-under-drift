"""Disjoint per-dataset inputs; RADAR attacks cannot cross split boundaries."""
from pathlib import Path
import json
from copy import deepcopy
import shutil
import numpy as np
import pandas as pd
from common import protocol as p
legacy = p.load_module('per_dataset_legacy_prepare', '01_attacks/prepare.py')
attack = p.load_module('per_dataset_attack', '01_attacks/interventions.py')


def radar_bounds(rows, block=1000, embargo=1):
    a = int(rows * .5) // block * block
    b = int(rows * .7) // block * block
    return {'training': (0, a), 'validation': (a + embargo * block, b),
            'evaluation': (b + embargo * block, rows)}


def snapshot_radar(spec):
    metadata_path = (p.ROOT / spec['radar']['source_metadata']).resolve()
    metadata = p.read(metadata_path)
    source = p.ROOT.parent / metadata['path']
    if metadata['rows'] != spec['radar']['source_rows'] or p.sha(source) != metadata['sha256']:
        raise ValueError('RADAR source identity differs')
    destination = p.ROOT / 'data/clean/RADAR/full_stream.parquet'
    if not destination.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    if p.sha(destination) != metadata['sha256']:
        raise ValueError('Local RADAR snapshot differs')
    return {'source_metadata_sha256': p.sha(metadata_path), 'source_sha256': metadata['sha256'],
            'snapshot': destination.relative_to(p.ROOT).as_posix(), 'metadata': metadata}


def partition_metadata(metadata, role, bounds):
    start, end = bounds
    result = deepcopy(metadata)
    result.update(key=f'radar__{role}', rows=end-start, base_seed=None, warmup_rows=1000,
                  transition_intervals=[], nominal_centers=[], supports_event_metrics=False,
                  segment_boundaries=[b-start for b in metadata['segment_boundaries'] if start < b < end],
                  source_partition={'role': role, 'start_row': start, 'end_row_exclusive': end,
                                    'full_source_rows': metadata['rows']})
    return result


def verify(root):
    manifest = p.read(root / '01_attacks/manifest.json')
    for name, digest in manifest['outputs'].items():
        if p.sha(root / '01_attacks' / name) != digest:
            raise ValueError(f'Input changed: {name}')
    return manifest


def build(root, config, source_record=None):
    if config['dataset'] != 'radar':
        manifest = legacy.build(root, config, allow_existing=True)
        manifest.update(dataset=config['dataset'], data_role=config['data_role'])
        p.write(root / '01_attacks/manifest.json', manifest)
        return manifest
    if source_record is None:
        raise ValueError('RADAR requires a verified source snapshot')
    previous = verify(root) if (root/'01_attacks/manifest.json').exists() else None
    clean_full = pd.read_parquet(p.ROOT / source_record['snapshot'])
    metadata = source_record['metadata']
    if len(clean_full) != metadata['rows']:
        raise ValueError('Incomplete RADAR snapshot')
    role = config['data_role']
    bounds = radar_bounds(len(clean_full), config['block_size'], config['radar_embargo_blocks'])[role]
    meta = partition_metadata(metadata, role, bounds)
    clean = clean_full.iloc[bounds[0]:bounds[1]].reset_index(drop=True).copy()
    if config.get('pilot_rows'):
        clean = clean.iloc[:config['pilot_rows']].copy()
        meta['rows'] = len(clean)
        meta['source_partition']['end_row_exclusive'] = bounds[0]+len(clean)
        # Pilot-only positional segments make replay executable on the small fixture.
        meta['segment_boundaries'] = [len(clean)//2]
    n = len(clean)
    stream_dir = root / '00_streams'; stream_dir.mkdir(parents=True, exist_ok=True)
    clean_file = stream_dir / 'stream.parquet'; clean.to_parquet(clean_file, index=False)
    p.write(stream_dir / 'metadata.json', meta)
    pd.DataFrame({'row_id': np.arange(n), 'source_row_id': np.arange(bounds[0], bounds[0]+n)}).to_parquet(stream_dir/'row_map.parquet', index=False)
    mask = legacy.audit_positions(n, config['block_size'], .05, p.seed('audit', config['audit']['seed'], meta['key']))
    protected = clean.loc[mask].copy(); protected.insert(0, 'row_id', np.flatnonzero(mask))
    protected['release_block'] = protected.row_id // config['block_size'] + 1
    audit_file = root / 'audit/protected/sample.parquet'; audit_file.parent.mkdir(parents=True, exist_ok=True)
    protected.to_parquet(audit_file, index=False)
    stage = root / '01_attacks'; stage.mkdir(parents=True, exist_ok=True)
    relative = lambda file: legacy.relative(file, stage)
    manifest = {'dataset': 'radar', 'data_role': role, 'conditions': {}, 'outputs': {},
                'source_partition': meta['source_partition'], 'source_sha256': source_record['source_sha256']}
    for file in [clean_file, stream_dir/'row_map.parquet', audit_file]:
        manifest['outputs'][relative(file)] = p.sha(file)
    variants = [('clean', 'none', 0., None)]
    assignments = config['radar_attack_assignments']
    for mode in config['attack_modes']:
        for level, rate in config['levels'].items():
            for assignment in assignments:
                variants.append((mode, level, rate, assignment))
    for mode, level, rate, assignment in variants:
        key = f"radar__{role}__{mode}__{level}__a{assignment}"
        seed = None if assignment is None else p.seed('per-dataset-attack', role, mode, level, assignment, config['attack_seed'])
        observed, truth, details = attack.make_attack(clean, meta, mode, rate, seed)
        folder = stage / key; folder.mkdir(parents=True, exist_ok=True)
        obs_file = clean_file if mode == 'clean' else folder/'observations.parquet'
        if mode != 'clean': observed.to_parquet(obs_file, index=False)
        truth_file = folder/'truth.parquet'; truth.to_parquet(truth_file, index=False)
        replay = truth.replay_source_row.to_numpy()
        if ((replay >= n) | (replay < -1)).any():
            raise AssertionError('Replay crossed a chronological partition')
        c = {k: meta[k] for k in ['stream', 'base_seed', 'rows', 'classes', 'benign_class',
                                 'transition_intervals', 'supports_event_metrics', 'warmup_rows', 'source_partition']}
        c.update(key=key, base_key=meta['key'], mode=mode, level=level, attack_seed=seed,
                 attack_assignment=assignment, observations=relative(obs_file), truth=relative(truth_file),
                 audit=relative(audit_file), attack=details, source_observation_sha256=p.sha(obs_file))
        manifest['conditions'][key] = c
        for file in [obs_file, truth_file]: manifest['outputs'][relative(file)] = p.sha(file)
    if previous is not None:
        for name, digest in previous['outputs'].items():
            if manifest['outputs'].get(name) != digest:
                raise ValueError(f'Existing RADAR input changed: {name}')
        for key, condition in previous['conditions'].items():
            if json.loads(json.dumps(manifest['conditions'].get(key))) != condition:
                raise ValueError(f'Existing RADAR condition changed: {key}')
    p.write(stage/'manifest.json', manifest)
    return manifest
