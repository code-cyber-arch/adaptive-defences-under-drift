# Results and figure guide

The results are descriptive comparisons within the declared datasets, masks and feedback assumptions. Positive and less favourable outcomes are both retained. The public tables can be checked without rerunning the models.

| Question | Saved table | Figure location |
| --- | --- | --- |
| Clean versus six poisoned conditions | [Baseline values](../results/evaluation/06_analysis/presentation/baseline_clean_vs_poisoned/plot_values.csv) | [Baseline figures](../results/evaluation/06_analysis/presentation/baseline_clean_vs_poisoned/) |
| Passive detector behaviour | [Monitor values](../results/evaluation/06_analysis/presentation/detector_baseline/plot_values.csv) | [Passive-monitor figures](../results/evaluation/06_analysis/presentation/detector_baseline/) |
| Benefit and cost of screening | [Paired screening effects](../results/evaluation/06_analysis/presentation/fixed_responses/tables/paired_screening_effects.csv) | [Screening effects](figures/screening_tradeoffs.pdf) |
| Added value of fixed resets | [Matched pairs](../results/methods_revision/fixed_no_reset_pairs.csv) | [Reset advantage](../results/methods_revision/figures/fixed_reset_advantage.pdf) |
| Learned versus fixed reset control | [RL improvements](../results/evaluation/06_analysis/presentation/rl_comparison/tables/improvement_values.csv) | [RL heatmaps](figures/evaluation_rl_heatmaps.pdf) |
| Filter discrimination on held-out data | [Separability](../results/filter_study/separability.csv) | [Filter-study figures](../results/filter_study/figures/) |
| Monitor behaviour before and after filtering | [Monitor overview](../results/methods_revision/filter_monitor_overview.csv) | [Monitor activity](../results/methods_revision/figures/filter_detector_activity.pdf) |
| Predictive and learning effects of filtering | [Paired filter differences](../results/filter_study/paired_differences.csv) | [Filter tradeoffs](figures/filter_tradeoffs.pdf) |
| Reset versus continuous learning with the same filter | [732 paired comparisons](../results/filter_no_reset/paired_comparisons.csv) | [Filter/reset comparison](../results/methods_revision/figures/filter_reset_advantage.pdf) |
| Committed reset activity | [Original response overview](../results/methods_revision/original_response_overview.csv) | [Response activity](../results/methods_revision/figures/response_activity.pdf) |

Screening reduced poisoning exposure, but lower exposure did not always improve prediction. Learned filtering provided useful gains in the tested RADAR/ADWIN setting. Its benefits varied across datasets, interventions and monitors. Small tabular reset controllers did not consistently outperform the fixed screened reference.

The matched no-reset controls show why filtering and resetting must be interpreted separately. Confirmed reset improved SEA's attacked-condition mean macro-F1 in the filter study. RBF and RADAR achieved higher means through continuous learning without resets. These averages do not imply the same outcome for every condition.

The original benchmark and the filter study have different scoring protocols, including a later RADAR assessment suffix in the filter study. Compare treatments within their matched protocol. Do not pool the two sets of absolute scores as a common leaderboard.

## Verify the saved evidence

```bash
bash run.sh --check-evidence
```

This checks original run coverage and paired arithmetic, reconciles all 1,708 filter summaries with their tables, and verifies the 183 no-reset summaries and 732 comparisons. It writes fresh check records under `results/repository_checks/`. It does not overwrite the archived verifier records or substitute for missing original raw traces.
