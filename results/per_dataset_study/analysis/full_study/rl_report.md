# Detector and reset-policy comparison across all datasets

RQ1 and RQ2 remain unchanged. These controls and robustness analyses belong to Experiments 1 and 2. Experiment 3 retains its existing three-dataset evidence.

Phase 04 trained 120 controllers: three datasets, four detectors, two state representations and five training seeds. Every controller was trained on its own dataset development split and evaluated only on that dataset with its matching detector.

All 11,680 comparison executions passed trace checks: 1,344 validation, 5,656 default-setting evaluation and 4,680 additional feedback runs. The sensitivity table has 5,616 entries including 936 shared defaults. Training contains 3,360 episodes across 120 controllers. Repeated no-reset controls are not independent detector evidence.

| Dataset | Detector | Policy | Mean macro-F1 (%) | Difference from matched confirmed reset (pp) |
| --- | --- | --- | ---: | ---: |
| SEA | ADWIN | Confirmed reset | 89.62 | +0.00 |
| SEA | ADWIN | Error rule | 89.96 | +0.33 |
| SEA | ADWIN | Selected mapping | 89.92 | +0.30 |
| SEA | ADWIN | 2-state Q / 7 | 88.93 | -0.69 |
| SEA | ADWIN | 2-state Q / 17 | 88.93 | -0.69 |
| SEA | ADWIN | 2-state Q / 27 | 90.21 | +0.59 |
| SEA | ADWIN | 2-state Q / 37 | 88.93 | -0.69 |
| SEA | ADWIN | 2-state Q / 47 | 90.21 | +0.59 |
| SEA | ADWIN | 4-state Q / 7 | 88.93 | -0.69 |
| SEA | ADWIN | 4-state Q / 17 | 89.30 | -0.32 |
| SEA | ADWIN | 4-state Q / 27 | 90.03 | +0.41 |
| SEA | ADWIN | 4-state Q / 37 | 89.90 | +0.28 |
| SEA | ADWIN | 4-state Q / 47 | 90.03 | +0.41 |
| SEA | HDDM-W | Confirmed reset | 89.54 | +0.00 |
| SEA | HDDM-W | Error rule | 89.96 | +0.42 |
| SEA | HDDM-W | Selected mapping | 90.26 | +0.72 |
| SEA | HDDM-W | 2-state Q / 7 | 88.93 | -0.61 |
| SEA | HDDM-W | 2-state Q / 17 | 90.06 | +0.53 |
| SEA | HDDM-W | 2-state Q / 27 | 90.26 | +0.72 |
| SEA | HDDM-W | 2-state Q / 37 | 88.93 | -0.61 |
| SEA | HDDM-W | 2-state Q / 47 | 90.26 | +0.72 |
| SEA | HDDM-W | 4-state Q / 7 | 89.45 | -0.09 |
| SEA | HDDM-W | 4-state Q / 17 | 90.03 | +0.49 |
| SEA | HDDM-W | 4-state Q / 27 | 90.09 | +0.55 |
| SEA | HDDM-W | 4-state Q / 37 | 89.91 | +0.38 |
| SEA | HDDM-W | 4-state Q / 47 | 90.09 | +0.55 |
| SEA | Hellinger | Confirmed reset | 89.19 | +0.00 |
| SEA | Hellinger | Error rule | 89.96 | +0.77 |
| SEA | Hellinger | Selected mapping | 90.03 | +0.84 |
| SEA | Hellinger | 2-state Q / 7 | 88.93 | -0.26 |
| SEA | Hellinger | 2-state Q / 17 | 89.64 | +0.45 |
| SEA | Hellinger | 2-state Q / 27 | 90.07 | +0.88 |
| SEA | Hellinger | 2-state Q / 37 | 90.07 | +0.88 |
| SEA | Hellinger | 2-state Q / 47 | 88.93 | -0.26 |
| SEA | Hellinger | 4-state Q / 7 | 88.93 | -0.26 |
| SEA | Hellinger | 4-state Q / 17 | 90.03 | +0.84 |
| SEA | Hellinger | 4-state Q / 27 | 90.02 | +0.84 |
| SEA | Hellinger | 4-state Q / 37 | 88.93 | -0.26 |
| SEA | Hellinger | 4-state Q / 47 | 90.02 | +0.84 |
| SEA | D3 OOF | Confirmed reset | 88.96 | +0.00 |
| SEA | D3 OOF | Error rule | 89.96 | +1.00 |
| SEA | D3 OOF | Selected mapping | 89.95 | +1.00 |
| SEA | D3 OOF | 2-state Q / 7 | 88.93 | -0.03 |
| SEA | D3 OOF | 2-state Q / 17 | 90.06 | +1.10 |
| SEA | D3 OOF | 2-state Q / 27 | 90.06 | +1.10 |
| SEA | D3 OOF | 2-state Q / 37 | 88.93 | -0.03 |
| SEA | D3 OOF | 2-state Q / 47 | 88.93 | -0.03 |
| SEA | D3 OOF | 4-state Q / 7 | 88.93 | -0.03 |
| SEA | D3 OOF | 4-state Q / 17 | 88.93 | -0.03 |
| SEA | D3 OOF | 4-state Q / 27 | 90.03 | +1.07 |
| SEA | D3 OOF | 4-state Q / 37 | 88.93 | -0.03 |
| SEA | D3 OOF | 4-state Q / 47 | 90.03 | +1.07 |
| RBF | ADWIN | Confirmed reset | 53.66 | +0.00 |
| RBF | ADWIN | Error rule | 52.95 | -0.72 |
| RBF | ADWIN | Selected mapping | 53.83 | +0.16 |
| RBF | ADWIN | 2-state Q / 7 | 53.14 | -0.52 |
| RBF | ADWIN | 2-state Q / 17 | 53.83 | +0.16 |
| RBF | ADWIN | 2-state Q / 27 | 53.83 | +0.16 |
| RBF | ADWIN | 2-state Q / 37 | 52.94 | -0.72 |
| RBF | ADWIN | 2-state Q / 47 | 52.94 | -0.72 |
| RBF | ADWIN | 4-state Q / 7 | 52.60 | -1.06 |
| RBF | ADWIN | 4-state Q / 17 | 53.16 | -0.50 |
| RBF | ADWIN | 4-state Q / 27 | 53.83 | +0.16 |
| RBF | ADWIN | 4-state Q / 37 | 53.83 | +0.16 |
| RBF | ADWIN | 4-state Q / 47 | 53.83 | +0.16 |
| RBF | HDDM-W | Confirmed reset | 52.83 | +0.00 |
| RBF | HDDM-W | Error rule | 52.95 | +0.11 |
| RBF | HDDM-W | Selected mapping | 53.83 | +0.99 |
| RBF | HDDM-W | 2-state Q / 7 | 53.83 | +0.99 |
| RBF | HDDM-W | 2-state Q / 17 | 52.85 | +0.02 |
| RBF | HDDM-W | 2-state Q / 27 | 52.78 | -0.05 |
| RBF | HDDM-W | 2-state Q / 37 | 52.78 | -0.05 |
| RBF | HDDM-W | 2-state Q / 47 | 52.94 | +0.11 |
| RBF | HDDM-W | 4-state Q / 7 | 53.83 | +0.99 |
| RBF | HDDM-W | 4-state Q / 17 | 52.89 | +0.05 |
| RBF | HDDM-W | 4-state Q / 27 | 53.83 | +0.99 |
| RBF | HDDM-W | 4-state Q / 37 | 51.42 | -1.41 |
| RBF | HDDM-W | 4-state Q / 47 | 53.83 | +0.99 |
| RBF | Hellinger | Confirmed reset | 53.75 | +0.00 |
| RBF | Hellinger | Error rule | 52.95 | -0.80 |
| RBF | Hellinger | Selected mapping | 53.83 | +0.08 |
| RBF | Hellinger | 2-state Q / 7 | 53.83 | +0.08 |
| RBF | Hellinger | 2-state Q / 17 | 52.90 | -0.85 |
| RBF | Hellinger | 2-state Q / 27 | 53.83 | +0.08 |
| RBF | Hellinger | 2-state Q / 37 | 53.83 | +0.08 |
| RBF | Hellinger | 2-state Q / 47 | 52.90 | -0.85 |
| RBF | Hellinger | 4-state Q / 7 | 52.20 | -1.55 |
| RBF | Hellinger | 4-state Q / 17 | 52.97 | -0.78 |
| RBF | Hellinger | 4-state Q / 27 | 53.83 | +0.08 |
| RBF | Hellinger | 4-state Q / 37 | 52.10 | -1.64 |
| RBF | Hellinger | 4-state Q / 47 | 53.83 | +0.08 |
| RBF | D3 OOF | Confirmed reset | 54.27 | +0.00 |
| RBF | D3 OOF | Error rule | 52.95 | -1.33 |
| RBF | D3 OOF | Selected mapping | 53.83 | -0.45 |
| RBF | D3 OOF | 2-state Q / 7 | 54.18 | -0.09 |
| RBF | D3 OOF | 2-state Q / 17 | 53.83 | -0.45 |
| RBF | D3 OOF | 2-state Q / 27 | 53.83 | -0.45 |
| RBF | D3 OOF | 2-state Q / 37 | 53.83 | -0.45 |
| RBF | D3 OOF | 2-state Q / 47 | 52.94 | -1.33 |
| RBF | D3 OOF | 4-state Q / 7 | 53.77 | -0.50 |
| RBF | D3 OOF | 4-state Q / 17 | 52.95 | -1.32 |
| RBF | D3 OOF | 4-state Q / 27 | 52.16 | -2.11 |
| RBF | D3 OOF | 4-state Q / 37 | 52.16 | -2.11 |
| RBF | D3 OOF | 4-state Q / 47 | 52.16 | -2.11 |
| RADAR | ADWIN | Confirmed reset | 45.34 | +0.00 |
| RADAR | ADWIN | Error rule | 45.67 | +0.33 |
| RADAR | ADWIN | Selected mapping | 45.72 | +0.38 |
| RADAR | ADWIN | 2-state Q / 7 | 44.83 | -0.51 |
| RADAR | ADWIN | 2-state Q / 17 | 47.51 | +2.17 |
| RADAR | ADWIN | 2-state Q / 27 | 47.51 | +2.17 |
| RADAR | ADWIN | 2-state Q / 37 | 43.90 | -1.44 |
| RADAR | ADWIN | 2-state Q / 47 | 44.83 | -0.51 |
| RADAR | ADWIN | 4-state Q / 7 | 44.69 | -0.64 |
| RADAR | ADWIN | 4-state Q / 17 | 44.69 | -0.64 |
| RADAR | ADWIN | 4-state Q / 27 | 45.67 | +0.33 |
| RADAR | ADWIN | 4-state Q / 37 | 43.90 | -1.44 |
| RADAR | ADWIN | 4-state Q / 47 | 44.69 | -0.64 |
| RADAR | HDDM-W | Confirmed reset | 44.63 | +0.00 |
| RADAR | HDDM-W | Error rule | 45.67 | +1.04 |
| RADAR | HDDM-W | Selected mapping | 46.71 | +2.08 |
| RADAR | HDDM-W | 2-state Q / 7 | 44.31 | -0.32 |
| RADAR | HDDM-W | 2-state Q / 17 | 45.36 | +0.74 |
| RADAR | HDDM-W | 2-state Q / 27 | 44.83 | +0.20 |
| RADAR | HDDM-W | 2-state Q / 37 | 45.36 | +0.74 |
| RADAR | HDDM-W | 2-state Q / 47 | 47.39 | +2.76 |
| RADAR | HDDM-W | 4-state Q / 7 | 44.11 | -0.52 |
| RADAR | HDDM-W | 4-state Q / 17 | 46.71 | +2.08 |
| RADAR | HDDM-W | 4-state Q / 27 | 44.83 | +0.20 |
| RADAR | HDDM-W | 4-state Q / 37 | 45.48 | +0.86 |
| RADAR | HDDM-W | 4-state Q / 47 | 44.11 | -0.52 |
| RADAR | Hellinger | Confirmed reset | 45.09 | +0.00 |
| RADAR | Hellinger | Error rule | 45.67 | +0.58 |
| RADAR | Hellinger | Selected mapping | 45.96 | +0.88 |
| RADAR | Hellinger | 2-state Q / 7 | 44.31 | -0.78 |
| RADAR | Hellinger | 2-state Q / 17 | 44.83 | -0.26 |
| RADAR | Hellinger | 2-state Q / 27 | 44.31 | -0.78 |
| RADAR | Hellinger | 2-state Q / 37 | 44.31 | -0.78 |
| RADAR | Hellinger | 2-state Q / 47 | 44.83 | -0.26 |
| RADAR | Hellinger | 4-state Q / 7 | 44.31 | -0.78 |
| RADAR | Hellinger | 4-state Q / 17 | 44.31 | -0.78 |
| RADAR | Hellinger | 4-state Q / 27 | 46.30 | +1.21 |
| RADAR | Hellinger | 4-state Q / 37 | 44.31 | -0.78 |
| RADAR | Hellinger | 4-state Q / 47 | 44.69 | -0.40 |
| RADAR | D3 OOF | Confirmed reset | 45.60 | +0.00 |
| RADAR | D3 OOF | Error rule | 45.67 | +0.06 |
| RADAR | D3 OOF | Selected mapping | 45.66 | +0.06 |
| RADAR | D3 OOF | 2-state Q / 7 | 44.83 | -0.78 |
| RADAR | D3 OOF | 2-state Q / 17 | 44.83 | -0.78 |
| RADAR | D3 OOF | 2-state Q / 27 | 44.83 | -0.78 |
| RADAR | D3 OOF | 2-state Q / 37 | 44.83 | -0.78 |
| RADAR | D3 OOF | 2-state Q / 47 | 44.31 | -1.29 |
| RADAR | D3 OOF | 4-state Q / 7 | 45.58 | -0.02 |
| RADAR | D3 OOF | 4-state Q / 17 | 45.58 | -0.02 |
| RADAR | D3 OOF | 4-state Q / 27 | 44.83 | -0.78 |
| RADAR | D3 OOF | 4-state Q / 37 | 44.83 | -0.78 |
| RADAR | D3 OOF | 4-state Q / 47 | 44.31 | -1.29 |

Grand means weight the seven clean/poisoning conditions equally. Synthetic condition means use five stream realizations. RADAR attacked-condition means use five attack assignments on one evaluation suffix; clean is a single shared realization. These are different replication units, and there is no pooled ranking across datasets. All RL seeds remain visible.

SEA and RBF use separate training, validation and evaluation seeds. RADAR uses rows [0,242000) for training, [243000,339000) for validation and [340000,484753) for evaluation, with 1,000-row gaps. Each partition has its own warm-up. Attacks are generated independently inside each partition, including replay donor selection. The RADAR results concern its evaluation suffix, not the original full-stream scoring population.

Each dataset/detector receives a separate validation-selected deterministic mapping. Selection uses macro-F1, whereas Q-learning uses protected accuracy; the comparison does not isolate optimisation algorithm alone or prove convergence. The four training passes are fixed. RADAR has longer episodes than the synthetic streams; training budgets are matched across detectors within each dataset.

Sensitivity varies usable feedback (1%, 2%, 5%) and delay (one or five blocks) on clean and severe label poisoning for every dataset and detector, retaining the same 5% reserve. Controllers stay frozen. RADAR malicious recall and benign false-positive rate are retained in the metric tables.

The RADAR vectors retain their known whole-input frequency and unresolved FastText provenance limitations; temporal policy splits do not establish causal feature extraction. Previously inspected benchmark outcomes are not presented as a wholly untouched independent replication. No claims about unseen real deployments or deep RL follow.

- [Final three-dataset detector comparison](../results/per_dataset_study/analysis/figures/detector_policy_comparison.pdf)
- [Paired policy effects](../results/per_dataset_study/analysis/evaluation/paired_vs_confirmed.csv)
- [Feedback effects](../results/per_dataset_study/analysis/sensitivity/paired_feedback_effects.csv)
- [Trace verification](../results/per_dataset_study/analysis/verification/verification.json)
