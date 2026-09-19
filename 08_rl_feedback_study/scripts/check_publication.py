"""Verify the published study files without private data or dependencies."""
from pathlib import Path
import hashlib
import json

root = Path(__file__).resolve().parents[1]
manifest = json.loads((root / 'publication_manifest.json').read_text())
failures = []
for entry in manifest['files']:
    path = root / entry['path']
    if not path.is_file():
        failures.append({'path': entry['path'], 'problem': 'missing'})
    else:
        with path.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        if digest != entry['sha256']:
            failures.append({'path': entry['path'], 'problem': 'hash differs'})
print(json.dumps({'status': 'failed' if failures else 'passed',
                  'files_checked': len(manifest['files']), 'failures': failures,
                  'scope': 'Published file identity, not a rerun of local trace audits.'}, indent=2))
raise SystemExit(bool(failures))
