#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PYTHON_BIN="${PYTHON:-python3.14}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  if command -v uv >/dev/null 2>&1; then
    uv python install 3.14
    PYTHON_BIN="$(uv python find 3.14)"
  else
    echo "Python 3.14 is required. Install it, or set PYTHON to its executable."
    exit 1
  fi
fi
"$PYTHON_BIN" -c 'import sys; assert sys.version_info[:2] == (3, 14), "Python 3.14 is required"'
if [[ ! -x .venv/bin/python ]]; then
  "$PYTHON_BIN" -m venv .venv
fi
.venv/bin/python -m pip install -r requirements.lock.txt
.venv/bin/python -m pip check
echo "Ready: bash run.sh --tests, --check-evidence or --smoke."
