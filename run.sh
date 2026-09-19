#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="$PWD/results/.matplotlib"
COMMAND="${1:---help}"
if [[ $# -gt 0 ]]; then shift; fi
case "$COMMAND" in
  --tests) exec .venv/bin/python -B -m unittest discover -s tests -v "$@" ;;
  --pilot) exec .venv/bin/python -B -u scripts/run_research.py pilot "$@" ;;
  --run) exec .venv/bin/python -B -u scripts/run_research.py all "$@" ;;
  --training) exec .venv/bin/python -B -u scripts/run_research.py training "$@" ;;
  --validation) exec .venv/bin/python -B -u scripts/run_research.py validation "$@" ;;
  --evaluation) exec .venv/bin/python -B -u scripts/run_research.py evaluation "$@" ;;
  --sensitivity) exec .venv/bin/python -B -u scripts/run_research.py sensitivity "$@" ;;
  --help) echo 'Usage: bash run.sh --tests | --pilot | --run | --training | --validation | --evaluation | --sensitivity [--workers 2]' ;;
  *) echo "Unknown command: $COMMAND"; exit 2 ;;
esac
