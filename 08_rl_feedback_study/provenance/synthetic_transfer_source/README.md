# Detector–policy comparison and feedback robustness

This component belongs to Experiments 1 and 2 and supports the existing RQ1 and RQ2. It does not add a research question.

**Phase 04 trains controllers for all four detectors:** ADWIN, HDDM-W, Hellinger and D3 OOF. Two state representations and three RL seeds yield 24 detector-specific controllers. Phase 05 compares each controller with fixed screened responses and same-information controls using the matching detector. Feedback robustness varies usable feedback and delay with frozen controllers.

| Folder | Responsibility |
| --- | --- |
| `00_streams/` | Matched SEA and RBF realizations |
| `01_attacks/` | Poisoning and separate observation/protected/evaluator channels |
| `02_detectors/` | The four detector implementations |
| `03_policies/` | Fixed screened response rules |
| `04_rl_training/` | Train and record all 24 detector-specific controllers |
| `05_experiment/` | Matched detector/policy evaluation and feedback sensitivity |
| `06_analysis/` | Paired effects, detector comparison and trace verification |
| `07_documentation/` | Methods and results under the existing research questions |

```sh
bash run.sh --tests
bash run.sh --pilot --workers 4
bash run.sh --run --workers 4
```

The integration pilot trains eight small fixture controllers (all four detectors and both state representations) and executes 192 comparisons. Pilot observations never enter research results.

The full pipeline trains 24 controllers, validates 16 deterministic mappings separately for each detector, evaluates ten arms per detector, and runs the declared feedback grid. It then checks complete traces and automatically generates the final detector comparison. Code and settings are frozen before execution; completed jobs resume only under matching contracts and hashes.

Current execution records are in `results/detector_policy_study/`:

- `training/04_rl_training/status.json`: phase-04 progress.
- `training/04_rl_training/manifest.json`: completed model identities.
- `validation_status.json`, `evaluation_status.json`, `sensitivity_status.json`: subsequent stages.
- `completion.json`: marked complete only after trace verification and reporting.
- `analysis/figures/detector_policy_comparison.pdf`: final four-detector comparison when complete.
- `analysis/evaluation/paired_vs_confirmed.csv`: comparisons against the matching detector's fixed reference.
- `analysis/sensitivity/paired_feedback_effects.csv`: comparisons against the same detector/policy at default feedback.

[Protocol](docs/PROTOCOL.md) · [Training](04_rl_training/README.md) · [Methods](07_documentation/METHODS.md) · [Results/status](07_documentation/RESULTS.md)

The completed ADWIN-only artifacts remain under `results/policy_robustness/`. Their exact source/reporting snapshot is preserved under `provenance/adwin_only_source/`. Those artifacts are not the final four-detector comparison. The corrected grid retrains all detectors with matched budgets rather than relabelling ADWIN-trained controllers as other-detector controllers.

The local `.venv` symlink shares the parent environment. Generated input and trace Parquet files remain local under the existing ignore rules. Do not pool these 60,000-row synthetic results with the longer original benchmark or interpret repeated no-reset controls as independent detector evidence.
