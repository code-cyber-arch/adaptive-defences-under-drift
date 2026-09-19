"""Compile the methodology or a result narration with a local LaTeX compiler."""
from pathlib import Path
import argparse
import shutil
import os
import subprocess
import sys
import os
sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import protocol as p

def main():
    """Compile the requested document and update its artifact record."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tex', type=Path, default=p.ROOT / 'docs/METHODOLOGY.tex')
    parser.add_argument('--compiler', default=None)
    args = parser.parse_args()
    bundled = p.ROOT / 'tools/latex/tectonic.exe'
    compiler = args.compiler or (str(bundled) if bundled.exists() else None) or shutil.which('tectonic') or shutil.which('pdflatex')
    os.environ['TECTONIC_CACHE_DIR'] = str(p.ROOT / 'tools/latex/cache')
    if not compiler:
        raise SystemExit('Install a LaTeX compiler or open the supplied PDF. Editable LaTeX sources are included.')
    source = args.tex.resolve()
    if 'tectonic' in Path(compiler).name.lower():
        command = [compiler, '--only-cached', '--untrusted', '--outdir', str(source.parent), str(source)]
    else:
        command = [compiler, '-interaction=nonstopmode', '-halt-on-error', '-output-directory', str(source.parent), str(source)]
    subprocess.run(command, check=True, cwd=source.parent)
    manifest = source.parent / 'manifest.json'
    if manifest.exists():
        record = p.read(manifest)
        record['outputs'][source.with_suffix('.pdf').name] = p.sha(source.with_suffix('.pdf'))
        p.write(manifest, record)
    print(source.with_suffix('.pdf'))
if __name__ == '__main__':
    main()
