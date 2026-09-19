# Phase 04: train within every dataset and detector

Train **120 controllers** across SEA, RBF and RADAR: four detectors × two state representations × five seeds for each dataset. Training episodes are not pooled across datasets.

```text
results/per_dataset_study/stages/training/
  SEA_A/04_rl_training/<detector>/<variant>/seed_<seed>/
  RBF_I/04_rl_training/<detector>/<variant>/seed_<seed>/
  radar/04_rl_training/<detector>/<variant>/seed_<seed>/
```

Detectors are `adwin`, `hddm_w`, `hellinger` and `d3_oof`; variants are `alarm` and `alarm_persistence`; seeds are 7, 17, 27, 37 and 47. Each folder contains its Q-table, selected actions, visit/reward diagnostics, episodes and transitions. The combined `results/per_dataset_study/models.json` identifies all 120 fitted models.

SEA and RBF each train on their own seed-112 realization. RADAR trains only on original rows `[0,242000)`. Validation and evaluation use disjoint seeds or later RADAR periods with embargoes. Poisoning is generated within each partition, including splice donors. Phase 04 rejects a manifest that is not a training partition of the matching dataset.

All detectors within a dataset use the same four passes over seven conditions, episode order for a matching RL seed, reward and gate. This is a fixed budget, not a convergence claim. Phase 05 requires the model's dataset and detector to match its inputs and never updates the fitted Q-table.

Run `bash run.sh --training --workers 4` for training only, or `bash run.sh --run --workers 4` for training through final verification and reporting.
