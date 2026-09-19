"""Compatibility entry point for the current three-dataset research pipeline."""
from pathlib import Path
import subprocess
import sys
if __name__=='__main__':
    root=Path(__file__).resolve().parents[1]
    subprocess.run([sys.executable,'-B','-u',str(root/'scripts/run_research.py'),'all',*sys.argv[1:]],cwd=root,check=True)
