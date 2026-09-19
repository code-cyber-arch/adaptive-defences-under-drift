"""Run development, frozen RL training and the complete comparative evaluation."""
from pathlib import Path
from copy import deepcopy
import argparse
import subprocess
import sys
import os
sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import protocol as p
training = p.load_module('thesis_rl_training', '04_rl_training/train.py')

def stage(config, workers, build_only=False):
    """Execute one recorded pipeline stage using its frozen configuration."""
    root = p.ROOT / 'results' / config['name']
    file = p.ROOT / 'results/configurations' / (config['name'] + '.json')
    if file.exists() and p.read(file) != config:
        raise ValueError(f'Configuration changed: {file}')
    p.write(file, config)
    command = [sys.executable, '-u', str(p.ROOT / 'tools/run_experiment.py'), '--config', str(file), '--workers', str(workers)]
    if (root / 'request.json').exists():
        command.append('--resume')
    if build_only:
        command.append('--build-only')
    subprocess.run(command, cwd=p.ROOT, check=True)
    return root

def status(phase, **values):
    """Write the current pipeline phase for the user."""
    p.write(p.ROOT / 'results/status.json', {'phase': phase, 'updated_utc': p.now(), **values})

def main():
    """Run training, validation, evaluation and final documentation in order."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--workers', type=int, default=4)
    args = ap.parse_args()
    base = p.read(p.ROOT / 'configs/thesis.json')
    protocol_file = p.ROOT / 'results/protocol.json'
    contract = {'config': base, 'sources': p.sources(), 'bank_sha256': p.sha(p.ROOT / base['input_bank']), 'environment': p.environment()}
    if protocol_file.exists() and p.read(protocol_file) != contract:
        raise ValueError('Thesis source or data contract changed')
    p.write(protocol_file, contract)
    status('rl_training')
    c = deepcopy(base)
    c.pop('input_bank')
    c.update(name='training', persist_inputs=True, streams=['SEA_A', 'RBF_I'], rows=base['development']['rows'], base_seeds=[base['development']['training_seed']], detectors=['adwin'], purpose='Matched fitting of alarm-only and persistence RL on separate synthetic development data')
    root = stage(c, args.workers, True)
    models = training.train(root, c, args.workers)
    status('validation')
    validation = deepcopy(c)
    validation.update(name='validation', base_seeds=[base['development']['validation_seed']], rl_models=models, purpose='Separate synthetic validation; no automatic setting selection or retraining')
    stage(validation, args.workers)
    final = deepcopy(base)
    final.update(name='evaluation', rl_models=models, purpose='Frozen-controller comparison on the fixed benchmark and complete RADAR; benchmark previously inspected')
    status('benchmark_evaluation', planned_conditions=61, planned_arms_per_condition=len(p.arms(final)), rl_models=models)
    result = stage(final, args.workers)
    documentation = p.load_module('thesis_documentation', '07_documentation/build.py')
    documentation.build(result)
    status('complete', verification=p.read(result / 'verification.json'), results=str(result.relative_to(p.ROOT)))
    print('THESIS PIPELINE COMPLETE', flush=True)
if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        status('failed', error=f'{type(error).__name__}: {error}')
        raise
