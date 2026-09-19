# Complete comparison matrix

The evaluation design has three datasets, four detectors, five comparison repetitions and seven clean/attack conditions: 420 comparison entries per method before fitting-seed and feedback-setting dimensions. Each dataset contributes 35 entries. RADAR's five clean references share one input and are not independent repetitions.

| Comparison | Held fixed | Varied | Coverage |
| --- | --- | --- | --- |
| Detector against detector | Dataset, input condition, policy/filter, screening | ADWIN, HDDM-W, Hellinger, D3 OOF | All six detector pairs for each applicable method |
| Policy against policy | Dataset, input condition, detector | Fixed responses, learned reset methods or filter/response combinations within the experiment | All method pairs within each experiment |
| Dataset against dataset | Experiment, method, detector, condition weighting | SEA, RBF, RADAR | All three dataset pairs; descriptive comparisons |
| Experiment against experiment | Dataset, detector, input condition, common host scoring rows | Fixed responses, learned reset control, observation filtering | All cross-experiment method pairs |
| Passive detector comparison | Dataset, input condition, unchanged no-reset learner predictions | Four detectors | All six detector pairs; alarm metrics |
| Feedback sensitivity | Dataset, detector, method, input condition | 1%, 2%, 5% usable feedback; one/five-block delay | All seven conditions, both RL representations, all five fitting seeds |

| Study component | Dataset coverage | Condition repetitions | Detectors | Methods |
| --- | --- | --- | --- | --- |
| Fixed responses and screening | SEA, RBF, RADAR | Five × seven | All four for reset policies | No reset, immediate, confirmed; screened/unscreened |
| Learned reset control | SEA, RBF, RADAR | Five × seven | All four | No reset, confirmed, error rule, selected mapping, two-state RL, four-state RL |
| Observation filtering | SEA, RBF, RADAR | Five × seven | All four for confirmed reset | No filter; two learned views; oracle/random row and block controls; matched no-reset controls |
| Passive monitoring | SEA, RBF, RADAR | Five × seven | All four | Alarms measured without modifying the learner |

RL and learned filter methods have five fitting seeds in addition to the five evaluation repetitions. Fixed deterministic controls are not duplicated solely to attach a fitting-seed label. No-reset trajectories are shared references in detector panels because monitoring is disabled in that control.

The feedback grid contains 33,936 configurations: 101 unique conditions × four detectors × fourteen methods × three feedback fractions × two delays. It includes 5,656 shared default evaluations and 28,280 other feedback configurations. Together with 1,344 policy-validation configurations, the RL/control/feedback grid contains 35,280 unique configurations. The complete study also contains the fixed-response, passive-detector and filter grids.

## Matched comparisons and weighting

Within-dataset differences use the same input condition and host rows. Cross-experiment predictive metrics are recalculated from saved predictions on the filter study's common copy-disjoint host mask, so exclusion of development-feature duplicates cannot create an unfair scoring advantage. Protection and learning-work measures retain matching exposure definitions. Re-scoring does not retrain a model or select a preferred seed.

Averages first combine fitting seeds within each input condition, then evaluation repetitions within each clean/attack category. The seven categories have equal weight. Dataset results are retained separately; the optional descriptive overall table gives each dataset weight one third. Dataset differences are not paired statistical tests and do not establish generalisation to independent real captures. Passive alarm metrics are reported separately from predictive-policy metrics.

## Visual guide

[Visual comparison matrices (vector PDF)](graphics/comparison_heatmap.pdf) ([PNG](graphics/comparison_heatmap.png)) show coverage and comparison pairs with minimal text. Colours indicate comparison groups, not measured performance.

[Comparison-to-answer diagram](graphics/comparison_to_answers.svg) ([PNG](graphics/comparison_to_answers.png)) links each comparison to the evidence it provides for the existing research questions.

## Outputs

All tables are under `results/per_dataset_study/analysis/comparisons/`:

- `common_row_seed_metrics.csv`: every fitted-seed result and its source trace.
- `common_row_method_metrics.csv`: matched methods after fitting-seed averaging.
- `detector_pairs.csv`: the six detector-pair differences within each method and condition.
- `policy_pairs.csv`: method-pair differences within each experiment and detector.
- `dataset_method_summary.csv` and `dataset_differences.csv`: equally weighted summaries and descriptive dataset differences.
- `cross_experiment_pairs.csv`: differences across experiments on common host rows.
- `passive_detector_pairs.csv`: passive alarm comparisons.
- `equal_dataset_summary.csv`: descriptive overall means with equal dataset weights.
- `balanced_comparison_entries.csv`: equal five-by-seven comparison groups, with shared references identified.
- `verification.json`: coverage and common-scoring-mask checks.

`complete_feedback_status.json` tracks full-condition feedback; `comparison_status.json` tracks cross-experiment re-scoring. `full_completion.json` is complete only after all experiment grids and comparison checks pass. These analyses remain within the existing RQ1 and RQ2.
