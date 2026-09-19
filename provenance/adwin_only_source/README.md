# Policy comparison and feedback robustness

This component provides policy controls for Experiment 2 and feedback-sensitivity analysis for Experiments 1 and 2. It supports the existing RQ1 and RQ2 without changing their wording or adding a research question.

The study retains the numbered pipeline used by the main repository. Implementation and outputs live here; the original study's source and retained measurements remain separately identifiable.

| Folder | Role |
| --- | --- |
| `00_streams/` | SEA and Random RBF generation with scheduled changes |
| `01_attacks/` | Clean, label-poisoned, generated-concept and splice conditions; separate observation, feedback and evaluator channels |
| `02_detectors/` | Shared ADWIN monitor |
| `03_policies/` | Screened no-reset and confirmed-reset responses |
| `04_rl_training/` | Frozen two-state and four-state controllers; training diagnostics |
| `05_experiment/policy_robustness.py` | Deterministic policies, matched execution and feedback availability |
| `06_analysis/policy_robustness.py` | Validation selection, paired effects and descriptive summaries |
| `07_documentation/` | Integrated methods text and reporting requirements |
| `configs/study.json` | Declared study grid |
| `scripts/run_research.py` | Preparation, validation selection, evaluation and sensitivity orchestration |
| `results/policy_robustness/` | Protocol, inputs, complete row/event traces, status and analysis |

Read [the results](07_documentation/RESULTS.md) and [the protocol](docs/PROTOCOL.md). The complete grid and trace checks passed. Validation chooses the deterministic mapping before evaluation. Evaluation never updates a Q-table. The execution pilot measures cost and checks operation; it is excluded from research findings.

```sh
bash run.sh --tests
bash run.sh --pilot --workers 2
bash run.sh --run --workers 2
```

`--run` executes validation, evaluation and feedback sensitivity in order. Completed runs are reused only after their inputs and output hashes are verified. Code, settings, environments and frozen controller hashes are fixed by the protocol record. A changed protocol must not be mixed with existing results.

This checkout shares the parent Python environment through `.venv`. Install the pinned dependencies with `setup.sh` if using a standalone copy. Six frozen controllers and training diagnostics are retained under `results/training/04_rl_training/`. No new RL training is part of this comparison.

This is not a new cybersecurity dataset evaluation. It uses independently generated synthetic realizations, with three evaluation stream seeds. Do not merge its absolute scores with the longer original benchmark or treat controller seeds as independent datasets.

## Evidence checks and figures

After execution, run the separate trace audit and plotting script:

```sh
MPLCONFIGDIR="$PWD/results/.matplotlib" .venv/bin/python -B 06_analysis/reporting/check_and_plot.py
```

It recalculates macro-F1 directly from confusion counts, checks exposure and withholding against evaluator truth, verifies feedback timing and reserved-row exclusion, and reconciles the published tables with complete run summaries. Its implementation is separate from the execution metric function. The script also checks that identical frozen action mappings give identical prediction traces.

- Research protocol: `docs/PROTOCOL.md`
- Integrated methods wording: `07_documentation/METHODS.md`
- Frozen mapping: `results/policy_robustness/selected_policy.json`
- Full grid status: `results/policy_robustness/completion.json`
- Independent checks: `results/policy_robustness/analysis/verification/verification.json`
- Paired policy effects: `results/policy_robustness/analysis/evaluation/paired_vs_confirmed.csv`
- Paired feedback effects: `results/policy_robustness/analysis/sensitivity/paired_feedback_effects.csv`
- Figures: `results/policy_robustness/analysis/figures/`
- Training state/action records with preserved state labels: `results/policy_robustness/analysis/training_diagnostics/state_action_coverage_verified.csv`

The row traces and generated input Parquet files remain local under the repository's existing ignore rules. A public code checkout without those files can regenerate inputs, but cannot inspect absent row traces. Keep the saved evidence in place when preparing a separate reproduction copy; do not overwrite its protocol or mix new settings into its completed run folders.
