# Experiment data

Only three dataset families are present: **SEA, RBF and RADAR**.

- `clean/`: authoritative clean stream files, grouped by dataset and seed.
- `poisoned/`: observation files grouped by dataset, attack, severity, base seed and attack seed.
- `evaluation_truth/`: original scoring labels and intervention references; never supplied to learner training.
- `catalogue.json`: relative paths, metadata and hashes for the 61 fixed benchmark conditions.

Clean conditions reuse the authoritative clean file. They are not duplicated under poisoned data. Each stream folder includes short metadata.

The benchmark uses synthetic seeds 0, 1 and 2. Clearly labelled training seed 112 and validation seed 113 support RL development; these are the same SEA/RBF dataset families, not additional types of data.
