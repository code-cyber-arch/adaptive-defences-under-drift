# Adaptive Defences under Drift

Controlled experiments on **update screening, reset control and observation filtering** under natural or scheduled change and injected poisoning. The shared classifier is a Hoeffding Tree. The datasets are SEA, Random RBF and one assembled RADAR ransomware telemetry stream.

The current research component is [08_rl_feedback_study](08_rl_feedback_study/README.md). It evaluates three datasets, four detectors and seven conditions, with five evaluation repetitions and five fitting seeds for learned methods. RADAR repetitions are attack assignments on one capture, not independent deployments.

| Experiment | Comparison | Current evidence |
| --- | --- | --- |
| 1. Fixed responses and screening | Screened/unscreened updates and reset policies | [Fixed-response summaries](08_rl_feedback_study/results/per_dataset_study/analysis/full_study/fixed_summary.csv) |
| 2. Learned reset control | Two Q-learning representations, fixed and same-information controls, feedback sensitivity | [Common-row comparisons](08_rl_feedback_study/results/per_dataset_study/analysis/comparisons/dataset_method_summary.csv) |
| 3. Observation filtering | Two learned views, oracle/random controls and matched no-reset responses | [Filter summaries](08_rl_feedback_study/results/per_dataset_study/analysis/full_study/filter_summary.csv) |

The verified grid contains **1,818 fixed-response configurations, 404 passive evaluations, 120 fitted controllers, 30 fitted filters, 10,403 filter/control configurations and 33,936 RL/control feedback evaluations** (including default feedback). See the [complete design](08_rl_feedback_study/docs/FULL_STUDY.md), [completion record](08_rl_feedback_study/results/per_dataset_study/full_completion.json) and [thesis evidence](08_rl_feedback_study/07_documentation/thesis_evidence/README.md). These are configuration counts, not independent statistical sample sizes.

The root-level implementation and results preserve the earlier benchmark and its [provenance](provenance/README.md). Current five-seed findings are under `08_rl_feedback_study/`; the earlier records remain separately identifiable.

## Current study

- [Study instructions](08_rl_feedback_study/README.md)
- [Methods](08_rl_feedback_study/07_documentation/METHODS.md) and [results](08_rl_feedback_study/07_documentation/RESULTS.md)
- [Public evidence and local artifact limits](08_rl_feedback_study/docs/PUBLICATION.md)

## Retained benchmark instructions

- [Experiment design and diagrams](docs/EXPERIMENT.md)
- [Results and figure guide](docs/RESULTS.md)
- [Reproduction and new-run instructions](docs/REPRODUCIBILITY.md)
- [Data availability](docs/DATA_AVAILABILITY.md)
- [Evidence limits](docs/LIMITATIONS.md)
- [Repository map](docs/FOLDER_MAP.md)
- [Citation instructions](docs/CITING.md)

```bash
# Verify the retained public files using only Python's standard library.
python3 scripts/check_repository.py

# Install the pinned experiment environment (Python 3.14).
bash setup.sh

# Run behavioural tests and reconcile saved tables with their run summaries.
bash run.sh --tests
bash run.sh --check-evidence

# Run a small synthetic integration check, separate from research results.
bash run.sh --smoke
```

The extensions support macOS and Linux. On Windows, use WSL for the complete study; the original PowerShell environment setup is retained for reference. No training is needed to inspect the saved results.

## Continue with a new experiment

```bash
# Copy current code and local inputs into an isolated working directory.
# This command requires the input Parquet files described in DATA_AVAILABILITY.md.
python3 scripts/create_run.py followup-01 --with-data
cd runs/followup-01
bash run.sh --benchmark --workers 2
bash run.sh --filters --workers 2
bash run.sh --no-reset --workers 2
```

New work goes under `runs/`, which is excluded from Git. The root `results/` directory contains the retained thesis evidence. Make code changes in the repository, then create a named run. The wrapper refuses to resume the archived root benchmark. Each new run has its own code snapshot, configuration and result folders.

## What is shared

Git contains code, settings, input metadata, aggregate tables, all available run summaries, selected figures and verification records. Large input files, fitted binary models and complete supplementary traces are retained locally at their original relative paths and excluded from Git. Their hashes are listed in [the local-artifact manifest](provenance/local-artifacts.json). The public repository alone does not contain these large files.

Old partial runs, compiler caches, temporary logs, duplicate presentation exports and editorial backups were excluded from this standalone copy. Private thesis material is retained locally under `.local/thesis/`, which is excluded from Git. [The migration record](provenance/README.md) states the boundary explicitly.

No reuse licence has been granted for the project code. Third-party software and datasets retain their own terms; see [RIGHTS.md](RIGHTS.md).
