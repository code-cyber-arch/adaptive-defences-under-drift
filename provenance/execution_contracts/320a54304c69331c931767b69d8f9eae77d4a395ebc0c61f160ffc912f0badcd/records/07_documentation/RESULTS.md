# Detector and reset-policy comparison across all datasets

RQ1 and RQ2 remain unchanged. These controls and robustness analyses belong to Experiments 1 and 2. Experiment 3 retains its existing three-dataset evidence.

Phase 04 trained 72 controllers: three datasets, four detectors, two state representations and three training seeds. Every controller was trained on its own dataset development split and evaluated only on that dataset with its matching detector.

All 6,024 comparison executions passed trace checks: 1,344 validation, 2,440 default-setting evaluation and 2,240 additional feedback runs. The sensitivity table has 2,688 entries including 448 shared defaults. Training contains 2,016 episodes across 72 controllers. Repeated no-reset controls are not independent detector evidence.

| Dataset | Detector | Policy | Mean macro-F1 (%) | Difference from matched confirmed reset (pp) |
| --- | --- | --- | ---: | ---: |
| SEA | ADWIN | Confirmed reset | 89.53 | +0.00 |
| SEA | ADWIN | Error rule | 89.90 | +0.37 |
| SEA | ADWIN | Selected mapping | 89.95 | +0.42 |
| SEA | ADWIN | 2-state Q / 7 | 88.92 | -0.61 |
| SEA | ADWIN | 2-state Q / 17 | 88.92 | -0.61 |
| SEA | ADWIN | 2-state Q / 27 | 90.24 | +0.71 |
| SEA | ADWIN | 4-state Q / 7 | 88.92 | -0.61 |
| SEA | ADWIN | 4-state Q / 17 | 89.19 | -0.34 |
| SEA | ADWIN | 4-state Q / 27 | 90.13 | +0.61 |
| SEA | HDDM-W | Confirmed reset | 89.55 | +0.00 |
| SEA | HDDM-W | Error rule | 89.90 | +0.35 |
| SEA | HDDM-W | Selected mapping | 90.35 | +0.80 |
| SEA | HDDM-W | 2-state Q / 7 | 88.92 | -0.63 |
| SEA | HDDM-W | 2-state Q / 17 | 90.16 | +0.61 |
| SEA | HDDM-W | 2-state Q / 27 | 90.35 | +0.80 |
| SEA | HDDM-W | 4-state Q / 7 | 89.37 | -0.18 |
| SEA | HDDM-W | 4-state Q / 17 | 90.12 | +0.57 |
| SEA | HDDM-W | 4-state Q / 27 | 90.18 | +0.63 |
| SEA | Hellinger | Confirmed reset | 89.11 | +0.00 |
| SEA | Hellinger | Error rule | 89.90 | +0.79 |
| SEA | Hellinger | Selected mapping | 89.99 | +0.88 |
| SEA | Hellinger | 2-state Q / 7 | 88.92 | -0.20 |
| SEA | Hellinger | 2-state Q / 17 | 89.64 | +0.53 |
| SEA | Hellinger | 2-state Q / 27 | 90.16 | +1.05 |
| SEA | Hellinger | 4-state Q / 7 | 88.92 | -0.20 |
| SEA | Hellinger | 4-state Q / 17 | 90.12 | +1.01 |
| SEA | Hellinger | 4-state Q / 27 | 90.11 | +1.00 |
| SEA | D3 OOF | Confirmed reset | 88.85 | +0.00 |
| SEA | D3 OOF | Error rule | 89.90 | +1.05 |
| SEA | D3 OOF | Selected mapping | 89.91 | +1.06 |
| SEA | D3 OOF | 2-state Q / 7 | 88.92 | +0.06 |
| SEA | D3 OOF | 2-state Q / 17 | 90.16 | +1.30 |
| SEA | D3 OOF | 2-state Q / 27 | 90.16 | +1.30 |
| SEA | D3 OOF | 4-state Q / 7 | 88.92 | +0.06 |
| SEA | D3 OOF | 4-state Q / 17 | 88.92 | +0.06 |
| SEA | D3 OOF | 4-state Q / 27 | 90.12 | +1.27 |
| RBF | ADWIN | Confirmed reset | 53.53 | +0.00 |
| RBF | ADWIN | Error rule | 53.01 | -0.52 |
| RBF | ADWIN | Selected mapping | 53.70 | +0.17 |
| RBF | ADWIN | 2-state Q / 7 | 53.32 | -0.21 |
| RBF | ADWIN | 2-state Q / 17 | 53.70 | +0.17 |
| RBF | ADWIN | 2-state Q / 27 | 53.70 | +0.17 |
| RBF | ADWIN | 4-state Q / 7 | 53.03 | -0.50 |
| RBF | ADWIN | 4-state Q / 17 | 53.01 | -0.52 |
| RBF | ADWIN | 4-state Q / 27 | 53.70 | +0.17 |
| RBF | HDDM-W | Confirmed reset | 52.77 | +0.00 |
| RBF | HDDM-W | Error rule | 53.01 | +0.23 |
| RBF | HDDM-W | Selected mapping | 53.70 | +0.93 |
| RBF | HDDM-W | 2-state Q / 7 | 53.70 | +0.93 |
| RBF | HDDM-W | 2-state Q / 17 | 53.25 | +0.48 |
| RBF | HDDM-W | 2-state Q / 27 | 52.94 | +0.16 |
| RBF | HDDM-W | 4-state Q / 7 | 53.70 | +0.93 |
| RBF | HDDM-W | 4-state Q / 17 | 53.03 | +0.26 |
| RBF | HDDM-W | 4-state Q / 27 | 53.70 | +0.93 |
| RBF | Hellinger | Confirmed reset | 53.77 | +0.00 |
| RBF | Hellinger | Error rule | 53.01 | -0.76 |
| RBF | Hellinger | Selected mapping | 53.70 | -0.06 |
| RBF | Hellinger | 2-state Q / 7 | 53.70 | -0.06 |
| RBF | Hellinger | 2-state Q / 17 | 53.18 | -0.59 |
| RBF | Hellinger | 2-state Q / 27 | 53.70 | -0.06 |
| RBF | Hellinger | 4-state Q / 7 | 52.64 | -1.13 |
| RBF | Hellinger | 4-state Q / 17 | 53.05 | -0.71 |
| RBF | Hellinger | 4-state Q / 27 | 53.70 | -0.06 |
| RBF | D3 OOF | Confirmed reset | 54.47 | +0.00 |
| RBF | D3 OOF | Error rule | 53.01 | -1.46 |
| RBF | D3 OOF | Selected mapping | 53.70 | -0.76 |
| RBF | D3 OOF | 2-state Q / 7 | 54.51 | +0.04 |
| RBF | D3 OOF | 2-state Q / 17 | 53.70 | -0.76 |
| RBF | D3 OOF | 2-state Q / 27 | 53.70 | -0.76 |
| RBF | D3 OOF | 4-state Q / 7 | 53.77 | -0.70 |
| RBF | D3 OOF | 4-state Q / 17 | 53.01 | -1.46 |
| RBF | D3 OOF | 4-state Q / 27 | 52.37 | -2.10 |
| RADAR | ADWIN | Confirmed reset | 45.52 | +0.00 |
| RADAR | ADWIN | Error rule | 45.23 | -0.29 |
| RADAR | ADWIN | Selected mapping | 45.08 | -0.44 |
| RADAR | ADWIN | 2-state Q / 7 | 44.44 | -1.08 |
| RADAR | ADWIN | 2-state Q / 17 | 47.37 | +1.85 |
| RADAR | ADWIN | 2-state Q / 27 | 47.37 | +1.85 |
| RADAR | ADWIN | 4-state Q / 7 | 44.33 | -1.19 |
| RADAR | ADWIN | 4-state Q / 17 | 44.33 | -1.19 |
| RADAR | ADWIN | 4-state Q / 27 | 45.23 | -0.29 |
| RADAR | HDDM-W | Confirmed reset | 44.83 | +0.00 |
| RADAR | HDDM-W | Error rule | 45.23 | +0.40 |
| RADAR | HDDM-W | Selected mapping | 46.21 | +1.38 |
| RADAR | HDDM-W | 2-state Q / 7 | 44.17 | -0.65 |
| RADAR | HDDM-W | 2-state Q / 17 | 44.82 | -0.00 |
| RADAR | HDDM-W | 2-state Q / 27 | 44.44 | -0.38 |
| RADAR | HDDM-W | 4-state Q / 7 | 43.61 | -1.21 |
| RADAR | HDDM-W | 4-state Q / 17 | 46.21 | +1.38 |
| RADAR | HDDM-W | 4-state Q / 27 | 44.44 | -0.38 |
| RADAR | Hellinger | Confirmed reset | 45.95 | +0.00 |
| RADAR | Hellinger | Error rule | 45.23 | -0.72 |
| RADAR | Hellinger | Selected mapping | 45.86 | -0.09 |
| RADAR | Hellinger | 2-state Q / 7 | 44.17 | -1.77 |
| RADAR | Hellinger | 2-state Q / 17 | 44.44 | -1.51 |
| RADAR | Hellinger | 2-state Q / 27 | 44.17 | -1.77 |
| RADAR | Hellinger | 4-state Q / 7 | 44.17 | -1.77 |
| RADAR | Hellinger | 4-state Q / 17 | 44.17 | -1.77 |
| RADAR | Hellinger | 4-state Q / 27 | 45.88 | -0.06 |
| RADAR | D3 OOF | Confirmed reset | 45.96 | +0.00 |
| RADAR | D3 OOF | Error rule | 45.23 | -0.73 |
| RADAR | D3 OOF | Selected mapping | 45.22 | -0.74 |
| RADAR | D3 OOF | 2-state Q / 7 | 44.44 | -1.52 |
| RADAR | D3 OOF | 2-state Q / 17 | 44.44 | -1.52 |
| RADAR | D3 OOF | 2-state Q / 27 | 44.44 | -1.52 |
| RADAR | D3 OOF | 4-state Q / 7 | 44.62 | -1.34 |
| RADAR | D3 OOF | 4-state Q / 17 | 44.62 | -1.34 |
| RADAR | D3 OOF | 4-state Q / 27 | 44.44 | -1.52 |

Grand means weight the seven clean/poisoning conditions equally. Synthetic condition means use three stream realizations. RADAR attacked-condition means use three attack assignments on one evaluation suffix; clean is a single shared realization. These are different replication units, and there is no pooled ranking across datasets. All RL seeds remain visible.

SEA and RBF use separate training, validation and evaluation seeds. RADAR uses rows [0,242000) for training, [243000,339000) for validation and [340000,484753) for evaluation, with 1,000-row gaps. Each partition has its own warm-up. Attacks are generated independently inside each partition, including replay donor selection. The RADAR results concern its evaluation suffix, not the original full-stream scoring population.

Each dataset/detector receives a separate validation-selected deterministic mapping. Selection uses macro-F1, whereas Q-learning uses protected accuracy; the comparison does not isolate optimisation algorithm alone or prove convergence. The four training passes are fixed. RADAR has longer episodes than the synthetic streams; training budgets are matched across detectors within each dataset.

Sensitivity varies usable feedback (1%, 2%, 5%) and delay (one or five blocks) on clean and severe label poisoning for every dataset and detector, retaining the same 5% reserve. Controllers stay frozen. RADAR malicious recall and benign false-positive rate are retained in the metric tables.

The RADAR vectors retain their known whole-input frequency and unresolved FastText provenance limitations; temporal policy splits do not establish causal feature extraction. Previously inspected benchmark outcomes are not presented as a wholly untouched independent replication. No claims about unseen real deployments or deep RL follow.

- [Final three-dataset detector comparison](../results/per_dataset_study/analysis/figures/detector_policy_comparison.pdf)
- [Paired policy effects](../results/per_dataset_study/analysis/evaluation/paired_vs_confirmed.csv)
- [Feedback effects](../results/per_dataset_study/analysis/sensitivity/paired_feedback_effects.csv)
- [Trace verification](../results/per_dataset_study/analysis/verification/verification.json)
