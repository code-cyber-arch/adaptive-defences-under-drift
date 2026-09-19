# RL versus fixed responses

Every line figure compares five screened strategies: no reset, immediate reset, confirmed reset, alarm-only RL and persistence RL. The detector-driven strategies all use ADWIN; no reset has no detector. The two RL variants each use the three frozen training seeds 7, 17 and 27. No controller is trained, tuned or updated by this reporting step.

Read the performance figure first (accuracy, macro-F1, macro recall and ROC-AUC). Then inspect protection_and_cost (poison admission, legitimate withholding, committed resets per 100,000 rows and full-run runtime per 100,000 rows). Finally inspect the improvement heatmap, which shows each RL variant minus screened confirmed-reset ADWIN. That is one named descriptive reference for all datasets and conditions, not a best-per-condition selection. Comparisons against screened no reset and immediate reset are also retained in the paired tables.

Each heatmap has 56 signed values: two RL variants x seven conditions x four metrics. Blue is positive, red is negative and white is zero. Percentage-score changes are percentage points (pp). To keep small differences readable, the ROC-AUC panel displays thousandths (×10⁻³): +4 means +0.004 AUC; paired tables retain original AUC units. Colour scales are symmetric around zero and shared across datasets for each metric. Display rounding can show zero for a small nonzero difference; unrounded differences remain in the tables. The reference is named in the filename and this guide; captions should retain it. No surrounding figure heading or explanatory footer is added.

All predictive metrics use exactly the same common host-row masks as the fixed-response and baseline comparisons. This excludes warm-up, protected reservations and all concept/splice replacement intervals in each matched seven-condition cohort. All 484,753 RADAR rows remain in the source runs. Protected observations do not enter predictive scoring. Synthetic AUC is macro one-versus-rest; RADAR AUC contrasts malicious with benign scores.

RL training seeds are averaged within each stream/attack-assignment cohort first. The five line points then average three matched cohorts, with one sample SD across those cohort means. The three training seeds are not nine independent datasets. seed_scores.csv retains all seed-level outcomes, while cohort_scores.csv includes within-cohort training-seed SDs. The heatmaps subtract the same fixed reference within each cohort before averaging; paired SDs remain in improvement_values.csv. Error bars and differences are descriptive, not significance or equivalence tests.

RADAR is one capture with three attack assignments. Its clean predictions are scored on three common masks. Repeated clean fixed-run costs and repeated RL clean seed averages do not establish independent timing repetitions. Synthetic means use three stream realizations. No baseline is replicated to inflate the number of independent cohorts.

Exposure and cost measures retain their original full-run denominators. Clean poison admission is undefined and marked N/A. Admission measures exposure, not proof that every admitted observation caused damage. Low withholding preserves legitimate learning opportunities. Runtime includes the original machine's concurrent execution. A score improvement alone is insufficient to establish a better defence; compare admission and costs as well.

The report uses 366 unique RL runs and 183 unique screened fixed runs (549 total). Matching RADAR clean predictions to three masks gives 567 seed/run records, reduced to 315 cohort/strategy records after RL seed averaging. All 61 conditions are included, with no selection of favourable attacks or outcomes. This is presentation of the previously inspected benchmark, not a new untouched evaluation.

All nine PDFs are separate, 170 x 128 mm, in a 2 x 2 layout. PNG previews are kept separately. The performance/protection figures use dashed fixed-policy lines and distinct RL lines and markers. All five strategies are screened; the legend labels identify their response strategies. No HTML or combined multi-page PDF is generated.

Reproduce with `python -B 06_analysis/dataset_summary/rl_comparison.py --workers 3`. After completion, `--render-only` redraws verified tables without rescoring. tables/seed_scores.csv contains source IDs, RL seeds, frozen-model hashes and scoring-mask hashes; cohort_scores.csv contains seed-averaged scores and training-seed variation; plot_values.csv records plotted means/SDs; paired_improvements.csv retains within-cohort differences against all three fixed references; improvement_values.csv records their mean and SD. The manifest binds sources, models, calculations and figures.

| Dataset | Performance | Protection and cost | RL minus screened confirmed reset |
|---|---|---|---|
| SEA | [PDF](performance/SEA_RL_performance.pdf) | [PDF](protection_and_cost/SEA_RL_protection_and_cost.pdf) | [PDF](improvement/SEA_RL_minus_screened_confirmed_ADWIN.pdf) |
| RBF | [PDF](performance/RBF_RL_performance.pdf) | [PDF](protection_and_cost/RBF_RL_protection_and_cost.pdf) | [PDF](improvement/RBF_RL_minus_screened_confirmed_ADWIN.pdf) |
| RADAR | [PDF](performance/RADAR_RL_performance.pdf) | [PDF](protection_and_cost/RADAR_RL_protection_and_cost.pdf) | [PDF](improvement/RADAR_RL_minus_screened_confirmed_ADWIN.pdf) |
