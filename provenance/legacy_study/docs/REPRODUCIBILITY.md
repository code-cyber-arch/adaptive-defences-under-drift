# Reproduction and continued work

## Environment

Use Python 3.14 and the package versions in `requirements.lock.txt`. `setup.sh` creates a local virtual environment and installs the lock file. The original benchmark records Windows/Python 3.14.5; the supplementary studies record macOS/Python 3.14.7. Execution contracts retain their original version and platform records. This migration does not relabel them as a new execution.

```bash
bash setup.sh
python3 scripts/check_repository.py
bash run.sh --tests
bash run.sh --check-evidence
bash run.sh --smoke
```

The smoke configuration is a small synthetic integration check under `results/quickstart/`. It is excluded from research conclusions. A fresh smoke directory is required. No archived benchmark output is overwritten.

## Verification levels

| Check | Required files | What it establishes |
| --- | --- | --- |
| `scripts/check_repository.py` | Public repository | Retained file hashes match the migration manifest |
| `run.sh --tests` | Environment and public code/model JSONs | Behavioural tests of update safety, timing, filtering and metrics |
| `run.sh --check-evidence` | Environment and public result tables/summaries | Coverage and numerical reconciliation of the retained evidence |
| `run.sh --smoke` | Environment; no RADAR download | Fresh small synthetic end-to-end execution |
| `check_repository.py --with-local-artifacts` | Complete separate local artifact set | Identity of input files, binary models and supplementary traces |
| Existing full trace auditors | Local data and trace artifacts | Observation-level checks for the supplementary studies |

The public archive does not support full re-verification of the original raw execution because its raw traces were not retained. [LIMITATIONS.md](LIMITATIONS.md) distinguishes this from the complete supplementary trace record.

## Fresh experiments

```bash
python3 scripts/create_run.py followup-01 --with-data
cd runs/followup-01
bash run.sh --benchmark --workers 2
bash run.sh --filters --workers 2
bash run.sh --no-reset --workers 2
```

The new workspace contains a snapshot of the current code and a separate copy of the input files. Its virtual environment links to this repository's local environment when available. It has no dependency on the former `drift-response-2` folder. To move a workspace elsewhere, run `setup.sh` there to provide its own environment.

The full pipeline retrains development controllers and produces a new execution record. It can be computationally expensive. The commands above are instructions for future work; the migration did not rerun the full study. The no-reset implementation currently expects the declared 61-condition study and its 183 controls. Changing the condition grid requires an explicit extension to that implementation and its verification.

Use a new name for each changed setting or code version. Existing execution contracts reject incompatible resumes. Keep the original `results/` tables as the reported thesis snapshot. Add new findings only after separately checking their metrics and provenance.

## Scientific source identity

The numbered phases, common modules, original tools, benchmark configurations and filter engines were copied without changes. New repository scripts live under `scripts/`, outside the original source-contract enumeration. Editorial authoring builders, platform compiler caches and stopped-run coordinators were excluded. Retained analysis and audit helpers remain in their existing extension paths so their recorded identities remain traceable.

The original Git history is linked in [provenance/README.md](../provenance/README.md). The standalone repository starts a clean history; it is not a rewrite of the original repository.

The retained RADAR source-inspection helper was made portable: supply `--archive /path/to/RADAR-release.zip`. It reads the engineering notebook directly from the verified archive and writes fresh checks under `results/repository_checks/radar_provenance/`. Its new hash differs from the historical inspection script; the archived provenance records retain their original script hash. No research input or engine was changed.
