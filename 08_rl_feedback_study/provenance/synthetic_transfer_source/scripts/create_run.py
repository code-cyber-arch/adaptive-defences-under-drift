"""Create a named code snapshot without modifying retained research results."""
from pathlib import Path
import argparse
import hashlib
import json
import re
import shutil
import subprocess
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
FOLDERS = ['00_streams','01_attacks','02_detectors','03_policies','04_rl_training',
           '05_experiment','06_analysis','07_documentation','common','configs','tools',
           'extensions','tests','scripts']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('name')
    parser.add_argument('--with-data', action='store_true', help='Copy the complete local input bank.')
    args = parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}', args.name):
        parser.error('Use 1–64 letters, digits, underscores or hyphens; start with a letter or digit.')
    destination = ROOT / 'runs' / args.name
    if destination.exists():
        parser.error(f'Destination already exists: {destination}')
    manifest = json.loads((ROOT / 'provenance/local-artifacts.json').read_text())
    inputs = [r for r in manifest['files'] if r['path'].startswith('data/')]
    if args.with_data:
        missing = [r['path'] for r in inputs if not (ROOT/r['path']).is_file()]
        if missing:
            parser.error(f'Missing {len(missing)} input files. See docs/DATA_AVAILABILITY.md.')
    destination.mkdir(parents=True)
    for folder in FOLDERS:
        shutil.copytree(ROOT/folder, destination/folder,
                        ignore=shutil.ignore_patterns('__pycache__','*.pyc','latex'))
    for name in ['requirements.txt','requirements.lock.txt','setup.sh','run.sh','.gitignore']:
        shutil.copy2(ROOT/name, destination/name)
    # Preserve required relative paths; no links to the old thesis repository.
    for source in (ROOT/'data').rglob('*'):
        if not source.is_file() or 'downloads' in source.relative_to(ROOT/'data').parts:
            continue
        if source.suffix=='.parquet' and not args.with_data:
            continue
        target = destination/source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.suffix=='.parquet':
            copied = subprocess.run(['cp','-c',str(source),str(target)],
                                    stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
            if copied.returncode:
                shutil.copy2(source,target)
        else:
            shutil.copy2(source,target)
    if (ROOT/'.venv').is_dir():
        (destination/'.venv').symlink_to('../../.venv',target_is_directory=True)
    (destination/'docs').mkdir()
    record = {'created_utc':datetime.now(timezone.utc).isoformat(),'name':args.name,
              'input_files_copied':args.with_data,'parent_repository':'../..',
              'code_sha256':{p.relative_to(destination).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
                             for folder in FOLDERS for p in (destination/folder).rglob('*.py')}}
    (destination/'run-workspace.json').write_text(json.dumps(record,indent=2)+'\n')
    print(f'Created {destination}\nRun from there: bash run.sh --benchmark --workers 2')


if __name__=='__main__':
    main()
