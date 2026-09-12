# Adaptive Defences under Drift

Controlled experiments on **update screening, reset control and observation filtering** under natural or scheduled change and injected poisoning. The shared classifier is a Hoeffding Tree. The datasets are SEA, Random RBF and one assembled RADAR ransomware telemetry stream.

This standalone repository continues the `drift-response-2` study. It preserves the implemented scientific code and retained measurements. [Provenance](provenance/README.md) identifies the original repository and commit. The original working folder remains unchanged.

| Experiment | Purpose | Implementation | Retained evidence |
| --- | --- | --- | --- |
| 1. Fixed responses and update screening | Separate resetting from candidate-update validation | `00_streams`–`06_analysis` | [Original evaluation](results/evaluation/) |
| 2. Learned reset control | Compare two frozen tabular Q-learning controllers with a fixed screened response | `04_rl_training`, `05_experiment` | [RL comparisons](results/evaluation/06_analysis/presentation/rl_comparison/) |
| 3. Observation filtering | Test filtering before monitoring and learning, including matched no-reset controls | `extensions/filter_study`, `extensions/filter_no_reset` | [Filter study](results/filter_study/) and [no-reset controls](results/filter_no_reset/) |

The completed evidence contains **1,464 original policy records, 244 passive-monitor evaluations, 1,708 filter runs and 183 matched no-reset controls**. A no-reset model continues learning; it is not frozen. The filter controls are paired with four monitors, giving 732 comparisons without treating reused controls as independent runs.

## Start here

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
