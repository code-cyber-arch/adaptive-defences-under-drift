# Repository map

```text
00_streams/             Synthetic stream generation
01_attacks/             Attack construction and input preparation
02_detectors/           Four drift monitors
03_policies/            Candidate-update validation
04_rl_training/         Tabular controller fitting
05_experiment/          Original online engine and run grid
06_analysis/            Metrics, paired comparisons and plots
07_documentation/       Original experimental document generator
common/                 Shared contracts, paths and channel separation
configs/                Frozen benchmark, original smoke and quickstart settings
extensions/             Filtering, no-reset controls and retained audit/analysis tools
scripts/                Standalone repository checks and future-run management
tests/                  Behavioural regression tests
docs/                   Experiment, results, availability and citation guides
provenance/             Migration record and hash manifests
data/                   Public metadata; large inputs are local and ignored
results/                Retained reported evidence; large traces are local and ignored
runs/                   New named experiments; local and ignored
.local/                 Author's thesis and migration records; local and ignored
```

The numbered phases are implementation stages, not eight separate experiments. The thesis's three-experiment numbering is described in [EXPERIMENT.md](EXPERIMENT.md).
