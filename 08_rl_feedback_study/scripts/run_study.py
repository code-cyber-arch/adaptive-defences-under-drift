"""Run a check or a fresh study while protecting the retained thesis snapshot."""
from pathlib import Path
import argparse
import json
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]


def run(*args):
    subprocess.run([sys.executable,'-B',*args],cwd=ROOT,check=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['smoke','benchmark','filters','no-reset'])
    parser.add_argument('--workers',type=int,default=2)
    args=parser.parse_args()
    if args.workers<1:
        parser.error('--workers must be positive')
    if args.stage=='smoke':
        if (ROOT/'results/quickstart').exists():
            parser.error('results/quickstart already exists. Keep it for inspection or move it before a fresh smoke check.')
        run('tools/run_experiment.py','--config','configs/quickstart.json','--workers',str(args.workers))
        return
    if not (ROOT/'run-workspace.json').is_file():
        parser.error('The root results are retained evidence. Create an isolated run with scripts/create_run.py first.')
    if args.stage=='benchmark':
        run('tools/run_thesis.py','--workers',str(args.workers))
    elif args.stage=='filters':
        if not (ROOT/'results/evaluation/verification.json').exists():
            parser.error('Complete this workspace\'s benchmark first.')
        run('-m','extensions.filter_study.run','--workers',str(args.workers))
        # The no-reset extension consumes the exact filter execution metadata.
        base=ROOT/'results/filter_study/execution_inputs'
        for name in ['training','validation','evaluation']:
            for rel in ['request.json','01_attacks/manifest.json']:
                source=ROOT/'results'/name/rel
                target=base/'results'/name/rel
                if target.exists() and target.read_bytes()!=source.read_bytes():
                    raise RuntimeError(f'Existing filter execution metadata differs: {target}')
                target.parent.mkdir(parents=True,exist_ok=True)
                shutil.copy2(source,target)
    else:
        if not (ROOT/'results/filter_study/verification.json').exists():
            parser.error('Complete this workspace\'s filter study first.')
        run('-m','extensions.filter_no_reset.run','--workers',str(args.workers))
        run('-m','extensions.thesis_revision.audit_no_reset')


if __name__=='__main__':
    main()
