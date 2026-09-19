"""Create a fresh reproduction workspace without changing retained evidence."""
from pathlib import Path
import argparse
import hashlib
import json
import re
import shutil

ROOT = Path(__file__).resolve().parents[1]

def create(name, radar_input):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', name):
        raise ValueError('Use a simple run name containing letters, digits, underscores or hyphens.')
    destination = ROOT / 'runs' / name
    if destination.exists():
        raise FileExistsError(f'Workspace already exists: {destination}')
    metadata_path = ROOT / 'data/clean/RADAR/metadata.json'
    metadata = json.loads(metadata_path.read_text())
    source = Path(radar_input).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f'Provide the verified RADAR input file: {source}')
    with source.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    if digest != metadata['sha256']:
        raise ValueError('RADAR input hash differs from the recorded study input.')
    destination.mkdir(parents=True)
    folders = ['00_streams','01_attacks','02_detectors','03_policies','04_rl_training',
               '05_experiment','06_analysis','07_documentation','common','configs',
               'docs','extensions','scripts','tests','tools']
    for name in folders:
        shutil.copytree(ROOT/name, destination/name,
                        ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    for name in ['requirements.txt','requirements.lock.txt','setup.sh','run.sh','README.md','CITATION.cff']:
        shutil.copy2(ROOT/name, destination/name)
    data = destination / 'data/clean/RADAR'
    data.mkdir(parents=True)
    shutil.copy2(source, data/'full_stream.parquet')
    shutil.copy2(metadata_path, data/'metadata.json')
    (destination/'run-workspace.json').write_text(json.dumps({
        'purpose':'Fresh reproduction of the declared study',
        'radar_input_sha256':digest,'retained_results_copied':False},indent=2))
    return destination

if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('name')
    parser.add_argument('--radar-input',required=True)
    args=parser.parse_args()
    try:
        destination=create(args.name,args.radar_input)
    except (ValueError,FileNotFoundError,FileExistsError) as error:
        parser.exit(1,str(error)+'\n')
    print(f'Created {destination}\nRun bash setup.sh, then bash run.sh --full --workers 4 inside it.')
