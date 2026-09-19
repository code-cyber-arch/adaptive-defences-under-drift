"""Prepare documented input paths and keep the three evidence channels separate."""
from pathlib import Path
import json
import os
import numpy as np
import pandas as pd
from common import protocol as p
DATASETS = {'SEA_A': 'SEA', 'RBF_I': 'RBF', 'radar': 'RADAR'}

def relative(file, stage):
    """Express a project file relative to a phase record."""
    return Path(os.path.relpath(file, stage)).as_posix()

def audit_positions(rows, block, fraction, audit_seed):
    """Reserve a reproducible 5% sample independently within each block."""
    if not 0 < fraction < 1:
        raise ValueError('Audit fraction must be between zero and one')
    rng = np.random.default_rng(audit_seed)
    chosen = np.zeros(rows, dtype=bool)
    for left in range(0, rows, block):
        right = min(left + block, rows)
        count = min(right - left - 1, max(1, int(round((right - left) * fraction))))
        if count > 0:
            chosen[rng.choice(np.arange(left, right), size=count, replace=False)] = True
    return chosen

def load_radar(config):
    """Require the complete RADAR source and verify its recorded hash."""
    source = p.read(p.ROOT / 'data/clean/RADAR/metadata.json')
    file = p.ROOT / source['path']
    if p.sha(file) != source['sha256']:
        raise ValueError('RADAR source hash mismatch')
    clean = pd.read_parquet(file)
    expected = source['full_source_rows']
    if len(clean) != expected or source['rows'] != expected:
        raise ValueError('RADAR must contain the complete source stream')
    if config.get('radar_rows', expected) != expected:
        raise ValueError('RADAR truncation is not permitted')
    return (clean, {k: v for k, v in source.items() if k not in ['path', 'truth_path', 'sha256', 'truth_sha256']})

def generated_inputs(root, config):
    """Generate development realizations once and reuse verified cached files."""
    foundation = p.load_module('stream_foundation', '00_streams/build.py')
    attack = p.load_module('attack_interventions', '01_attacks/interventions.py')
    bases = {}
    conditions = {}
    persistent = config.get('persist_inputs', False)
    role = 'training' if config['name'] == 'training' else 'validation' if config['name'] == 'validation' else 'check'
    for stream in config['streams']:
        if stream not in DATASETS:
            raise ValueError('Only SEA, RBF and RADAR are allowed')
        seeds = [None] if stream == 'radar' else config['base_seeds']
        for seed in seeds:
            key = 'radar' if seed is None else f'{stream}__s{seed}'
            seed_name = 'capture' if seed is None else f'{role}_seed_{seed}'
            folder = (p.ROOT / 'data/clean' if persistent else root / '00_streams') / DATASETS[stream] / seed_name
            metadata_file = folder / 'metadata.json'
            if metadata_file.exists():
                meta = p.read(metadata_file)
                file = p.ROOT / meta['path']
                if p.sha(file) != meta['sha256']:
                    raise ValueError('Cached stream changed')
                clean = pd.read_parquet(file)
                if len(clean) != config['rows']:
                    raise ValueError('Cached stream row count differs')
            else:
                clean, meta = load_radar(config) if stream == 'radar' else (lambda a: (a[0], a[2]))(foundation.synthetic(stream, config['rows'], seed))
                folder.mkdir(parents=True, exist_ok=True)
                file = folder / 'stream.parquet'
                clean.to_parquet(file, index=False)
                meta.update(key=key, path=file.relative_to(p.ROOT).as_posix(), sha256=p.sha(file), role=role)
                p.write(metadata_file, meta)
            meta['key'] = key
            bases[key] = meta
            jobs = [('clean', 'none', 0.0, None)]
            for mode in config['attack_modes']:
                for level, value in config['levels'].items():
                    jobs.append((mode, level, value, p.seed('intervention', config['attack_seed'], key, mode, level)))
            for mode, level, value, attack_seed in jobs:
                condition = f'{key}__{mode}__{level}'
                observed, reference, details = attack.make_attack(clean, meta, mode, value, attack_seed)
                sub = Path(DATASETS[stream]) / mode / level / seed_name / f'attack_seed_{attack_seed}'
                obs_folder = (p.ROOT / 'data/poisoned' if persistent else root / '01_attacks') / sub
                truth_folder = (p.ROOT / 'data/evaluation_truth' if persistent else root / 'audit/evaluator') / sub
                if mode == 'clean':
                    obs_file = file
                else:
                    obs_folder.mkdir(parents=True, exist_ok=True)
                    obs_file = obs_folder / 'stream.parquet'
                    observed.to_parquet(obs_file, index=False)
                truth_folder.mkdir(parents=True, exist_ok=True)
                truth_file = truth_folder / 'truth.parquet'
                reference.to_parquet(truth_file, index=False)
                c = dict(meta)
                c.update(key=condition, base_key=key, mode=mode, level=level, attack_seed=attack_seed, path=obs_file.relative_to(p.ROOT).as_posix(), truth_path=truth_file.relative_to(p.ROOT).as_posix(), sha256=p.sha(obs_file), truth_sha256=p.sha(truth_file))
                c.update(details)
                conditions[condition] = c
                if mode != 'clean':
                    p.write(obs_folder / 'metadata.json', c)
    return {'bases': bases, 'conditions': conditions}

def build(root, config, allow_existing=False):
    """Record exact stream inputs and create delayed protected samples."""
    stage = root / '01_attacks'
    previous = verify(root) if stage.exists() and allow_existing else None
    if stage.exists() and not allow_existing:
        raise FileExistsError(stage)
    bank = p.read(p.ROOT / config['input_bank']) if config.get('input_bank') else generated_inputs(root, config)
    stage.mkdir(parents=True, exist_ok=True)
    (root / '00_streams').mkdir(exist_ok=True)
    manifest = {'protocol': p.VERSION, 'bases': {}, 'conditions': {}, 'outputs': {}, 'audit_assumption': config['audit']}
    for key, meta in bank['bases'].items():
        if meta['stream'] not in config['streams']:
            continue
        file = p.ROOT / meta['path']
        if p.sha(file) != meta['sha256']:
            raise ValueError('Source stream hash mismatch')
        clean = pd.read_parquet(file)
        n = len(clean)
        if n != meta['rows'] or (meta['stream'] == 'radar' and n != 484753):
            raise ValueError('Incomplete stream')
        mask = audit_positions(n, config['block_size'], config['audit']['fraction'], p.seed('audit', config['audit']['seed'], key))
        audit = clean.loc[mask].copy()
        audit.insert(0, 'row_id', np.flatnonzero(mask))
        audit['release_block'] = audit.row_id // config['block_size'] + config['audit']['delay_blocks']
        audit_file = root / 'audit/protected' / DATASETS[meta['stream']] / ('capture' if meta['base_seed'] is None else f"seed_{meta['base_seed']}") / 'sample.parquet'
        audit_file.parent.mkdir(parents=True, exist_ok=True)
        audit.to_parquet(audit_file, index=False)
        manifest['bases'][key] = {'metadata': meta, 'clean': relative(file, stage), 'audit': relative(audit_file, stage), 'audit_rows': int(mask.sum())}
        for f in [file, audit_file]:
            manifest['outputs'][relative(f, stage)] = p.sha(f)
    for key, meta in bank['conditions'].items():
        if meta['base_key'] not in manifest['bases']:
            continue
        file = p.ROOT / meta['path']
        truth_file = p.ROOT / meta['truth_path']
        if p.sha(file) != meta['sha256'] or p.sha(truth_file) != meta['truth_sha256']:
            raise ValueError('Condition hash mismatch')
        c = {name: meta[name] for name in ['stream', 'base_seed', 'mode', 'level', 'rows', 'classes', 'benign_class', 'transition_intervals', 'supports_event_metrics', 'warmup_rows']}
        c.update(key=key, base_key=meta['base_key'], attack_seed=meta.get('attack_seed'), observations=relative(file, stage), truth=relative(truth_file, stage), audit=manifest['bases'][meta['base_key']]['audit'], attack=meta, source_observation_sha256=meta['sha256'])
        manifest['conditions'][key] = c
        for f in [file, truth_file]:
            manifest['outputs'][relative(f, stage)] = p.sha(f)
    if previous is not None:
        for name, digest in previous['outputs'].items():
            if manifest['outputs'].get(name) != digest:
                raise ValueError(f'Existing input changed: {name}')
        for key, condition in previous['conditions'].items():
            if json.loads(json.dumps(manifest['conditions'].get(key))) != condition:
                raise ValueError(f'Existing condition changed: {key}')
    p.write(stage / 'manifest.json', manifest)
    p.write(root / '00_streams/catalogue.json', manifest['bases'])
    (root / '00_streams/README.md').write_text('# Stream inputs\n\nThe catalogue points to the authoritative clean stream files. Data are not duplicated in this results folder.\n', encoding='utf-8')
    (stage / 'README.md').write_text('# Observation conditions\n\nThe manifest names each dataset, attack, severity and seed and links its exact input file. Protected samples and evaluator references are separate.\n', encoding='utf-8')
    print(f"Prepared {len(manifest['bases'])} sources and {len(manifest['conditions'])} conditions", flush=True)
    return manifest

def verify(root):
    """Reject a run if any referenced input has changed."""
    manifest = p.read(root / '01_attacks/manifest.json')
    for name, digest in manifest['outputs'].items():
        if p.sha(root / '01_attacks' / name) != digest:
            raise ValueError(f'Input changed: {name}')
    return manifest
