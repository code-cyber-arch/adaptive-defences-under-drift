# Poisoned observation streams

Navigate in this order: **dataset / attack / severity / base seed / attack seed**.

`instance` changes training labels. `concept` replaces intervals with generated concept-based observations. `splice` replaces intervals with replayed observations. `moderate` is the nominal 15% budget; `severe` is 25%.

Each final folder contains `stream.parquet` and `metadata.json`. No hidden clean-label or poison-marker columns are added to the observation file. Scoring references are stored separately in `../evaluation_truth`.

Training and validation conditions use explicit `training_seed_112` and `validation_seed_113` folder names. They do not overwrite benchmark inputs.
