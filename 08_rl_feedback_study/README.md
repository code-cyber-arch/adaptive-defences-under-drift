# Per-dataset detector and policy study

**The complete five-seed experiment covers SEA, RBF and RADAR with all four detectors.** This component covers Experiments 1–3 under the existing RQ1 and RQ2. Baselines, passive detectors, fixed screened/unscreened policies, RL and filtering share the five-seed evaluation grid.

Phase 04 trains **120 controllers**: three datasets × four detectors × two state representations × five RL seeds. Each controller is trained on its own dataset's development data and evaluated only with the matching dataset and detector. The complete grid uses ADWIN, HDDM-W, Hellinger and D3 OOF.

| Folder | Responsibility |
| --- | --- |
| `00_streams/` | Synthetic streams and the retained complete RADAR source |
| `01_attacks/per_dataset.py` | Disjoint inputs and poisoning confined to each chronological RADAR split |
| `02_detectors/` | Four monitor implementations |
| `03_policies/` | Fixed screened response rules |
| `04_rl_training/` | Per-dataset/per-detector fitting and diagnostics |
| `05_experiment/` | Matched policy evaluation and feedback sensitivity on all three datasets |
| `06_analysis/` | Paired metrics, three-dataset figures and trace verification |
| `07_documentation/` | Methods and results supporting the existing questions |

```sh
bash run.sh --tests
bash run.sh --pilot --workers 4
bash run.sh --run --workers 4
```

The pilot trains 24 small controllers and executes 288 comparisons across all three datasets and four detectors. Pilot rows and outcomes are excluded from research findings. The full RL/control pipeline trains 120 controllers and executes 1,344 validation configurations and 33,936 feedback/evaluation configurations (including defaults), followed by trace checks and reporting. Runs reuse verified artifacts only when their per-run scientific settings and input hashes match.

## Partitions

SEA and RBF each use training seed 112, validation seed 114 and evaluation seeds 115–119, with 60,000 observations per realization. They have separate controllers; their training episodes are not pooled.

RADAR retains its original 484,753-row source and partitions it chronologically:

| Role | Original row interval, zero-based and end-exclusive | Rows |
| --- | --- | ---: |
| Training | `[0, 242000)` | 242,000 |
| Validation | `[243000, 339000)` | 96,000 |
| Evaluation | `[340000, 484753)` | 144,753 |

The two 1,000-row gaps separate the periods. Each period has its own warm-up. Poisoning is generated separately in each partition, so training cannot replay donor rows or sample feature values from evaluation. Original-row maps and hashes are retained. Chronological policy partitions do not remove the known provenance limitations of the supplied RADAR feature vectors.

See [the complete experimental design](docs/FULL_STUDY.md) for phase-by-phase coverage, filter fitting seeds and matched controls.

The complete-study runner is `python -B -m extensions.complete_study.run --wait --workers 4`. It waits for the RL grid and then executes the remaining shared-input phases.

## Current outputs

All current outputs are under `results/per_dataset_study/`:

- `training_status.json` and `stages/training/<dataset>/04_rl_training/status.json`: training progress.
- `models.json`: the 120 fitted model identities when training completes.
- `validation_status.json`, `evaluation_status.json`, `sensitivity_status.json`: execution progress.
- `completion.json`: completion of the RL/control and feedback grid.
- `full_status.json` and `full_completion.json`: progress and verified completion of all three experiments.
- `analysis/full_study/`: baseline, screening, filter and matched no-reset effects.
- `analysis/figures/detector_policy_comparison.pdf`: final comparison with SEA, RBF and RADAR panels.
- `analysis/evaluation/paired_vs_confirmed.csv`: policy effects within each dataset and detector.
- `analysis/sensitivity/paired_feedback_effects.csv`: feedback effects within each dataset, detector and policy.
- `analysis/verification/verification.json`: dataset identity, temporal splits, training transitions and prediction-trace checks.

[Comparison matrix](docs/COMPARISON_MATRIX.md) · [Figure style](docs/FIGURES.md) · [Protocol](docs/PROTOCOL.md) · [Training phase](04_rl_training/README.md) · [Methods](07_documentation/METHODS.md) · [Results/status](07_documentation/RESULTS.md)

The ADWIN-only and synthetic-transfer computations remain identifiable under `results/policy_robustness/` and `results/detector_policy_study/`, with exact source snapshots under `provenance/`. They are not the current per-dataset result. The synthetic-transfer process was stopped when the requested training design changed; its completed outputs were preserved.

The `.venv` symlink shares the parent Python environment. The complete local RADAR snapshot, generated inputs and trace Parquet files are excluded from Git by the existing rules. See [publication scope](docs/PUBLICATION.md) for the published evidence and current verification command. Repository publication alone does not publish those inputs or traces.
