#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [[ $# -eq 0 || "${1:-}" == "--help" ]]; then
  cat <<'HELP'
Usage: bash run.sh COMMAND [arguments]
  --tests              Behavioural tests; no research runs.
  --check-evidence     Reconcile retained tables and summaries; no raw data needed.
  --smoke              Small synthetic pipeline check under results/quickstart/.
  --benchmark          Full benchmark, only in a fresh scripts/create_run.py workspace.
  --filters            Filter study in that new workspace after its benchmark.
  --no-reset           Matched no-reset controls after that workspace's filter study.
  --help               Show this help.
HELP
  exit 0
fi
if [[ ! -x .venv/bin/python ]]; then
  echo "Run bash setup.sh first."
  exit 1
fi
export PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="$PWD/results/.matplotlib"
COMMAND="$1"
shift
case "$COMMAND" in
  --tests)
    .venv/bin/python -B -m unittest discover -s tests -v
    if [[ -f results/training/04_rl_training/manifest.json ]]; then
      .venv/bin/python -B -m unittest discover -s extensions/independent_audit -p test_contracts.py -v
    fi
    ;;
  --check-evidence) exec .venv/bin/python -B scripts/verify_evidence.py "$@" ;;
  --smoke) exec .venv/bin/python -B scripts/run_study.py smoke "$@" ;;
  --benchmark) exec .venv/bin/python -B scripts/run_study.py benchmark "$@" ;;
  --filters) exec .venv/bin/python -B scripts/run_study.py filters "$@" ;;
  --no-reset) exec .venv/bin/python -B scripts/run_study.py no-reset "$@" ;;
  *) echo "Unknown command: $COMMAND. Use --help."; exit 2 ;;
esac
