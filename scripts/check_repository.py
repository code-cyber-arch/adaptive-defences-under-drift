"""Check the retained-file manifest without installing experiment dependencies."""
from pathlib import Path
import argparse
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--with-local-artifacts', action='store_true',
                        help='Also hash large local data, models and traces; requires those files.')
    args = parser.parse_args()
    manifest = json.loads((ROOT / 'provenance/retained-files.json').read_text())
    records = list(manifest['files'])
    if args.with_local_artifacts:
        records += json.loads((ROOT / 'provenance/local-artifacts.json').read_text())['files']
    failures = []
    for item in records:
        path = ROOT / item['path']
        if not path.is_file():
            failures.append({'path': item['path'], 'problem': 'missing'})
        elif sha(path) != item['sha256']:
            failures.append({'path': item['path'], 'problem': 'hash differs'})
    result = {'status': 'failed' if failures else 'passed', 'files_checked': len(records),
              'scope': 'File identity, not proof of scientific correctness or full original-run reproduction.',
              'failures': failures}
    print(json.dumps(result, indent=2))
    raise SystemExit(1 if failures else 0)


if __name__ == '__main__':
    main()
