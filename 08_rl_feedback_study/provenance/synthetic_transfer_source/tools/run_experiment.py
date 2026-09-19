"""Execute the guarded-update experiment phases."""
from pathlib import Path
import argparse
import sys
import os
sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import protocol as p
data = p.load_module('stream_preparation', '01_attacks/prepare.py')
runner = p.load_module('experiment_runner', '05_experiment/run_grid.py')

def main():
    """Run the requested input, execution, analysis and verification phases."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=p.ROOT / 'configs/smoke.json')
    parser.add_argument('--out', type=Path)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--build-only', action='store_true')
    args = parser.parse_args()
    config = p.read(args.config)
    if config['block_size'] != 1000 or config['warmup_rows'] != 1000:
        raise ValueError('The declared generation protocol uses 1,000-row blocks and warmup')
    if config['audit']['delay_blocks'] < 1:
        raise ValueError('Protected audit observations require at least one block of delay')
    root = p.output(args.out or p.ROOT / 'results' / config['name'])
    contract = {'protocol': p.VERSION, 'config': config, 'sources': p.sources(), 'environment': p.environment()}
    request = root / 'request.json'
    if request.exists():
        if not args.resume or p.read(request) != contract:
            raise ValueError('Existing request requires --resume and an unchanged contract')
    else:
        if root.exists() and any(root.iterdir()):
            raise FileExistsError(root)
        p.write(request, contract)
    if (root / '01_attacks/manifest.json').exists():
        manifest = data.verify(root)
    else:
        manifest = data.build(root, config)
    if args.build_only:
        return
    runner.run(root, manifest, config, workers=args.workers, resume=args.resume)
    analyse = p.load_module('experiment_analysis', '06_analysis/analyse.py').analyse
    report = p.load_module('experiment_figures', '06_analysis/figures.py').report
    if not (root / '06_analysis/manifest.json').exists():
        analyse(root)
    if not (root / '06_analysis/presentation/manifest.json').exists():
        report(root)
    verifier = p.load_module('experiment_verifier', 'tools/verify_experiment.py')
    verifier.verify(root)
    print(f'Experiment complete: {root}', flush=True)
if __name__ == '__main__':
    main()
