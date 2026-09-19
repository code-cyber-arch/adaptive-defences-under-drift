"""Restore separately supplied local artifacts after verifying their exact hashes."""
from pathlib import Path
import argparse
import hashlib
import json
import shutil

ROOT=Path(__file__).resolve().parents[1]


def sha(path):
    with path.open('rb') as handle:return hashlib.file_digest(handle,'sha256').hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--from',dest='source',required=True,type=Path,
                        help='Directory with the same data/ and results/ paths as this repository.')
    parser.add_argument('--data-only',action='store_true')
    args=parser.parse_args()
    records=json.loads((ROOT/'provenance/local-artifacts.json').read_text())['files']
    if args.data_only:records=[r for r in records if r['path'].startswith('data/')]
    # Validate all sources and existing destinations before copying anything.
    for row in records:
        source=args.source/row['path'];destination=ROOT/row['path']
        if not source.is_file() or sha(source)!=row['sha256']:
            parser.error('Missing or mismatched source: '+row['path'])
        if destination.exists() and sha(destination)!=row['sha256']:
            parser.error('Refusing to overwrite a different local file: '+row['path'])
    copied=0
    for row in records:
        destination=ROOT/row['path']
        if destination.exists():continue
        destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(args.source/row['path'],destination)
        assert sha(destination)==row['sha256']
        copied+=1
    print(f'Verified {len(records)} artifacts; restored {copied}.')


if __name__=='__main__':main()
