# Clean streams

Open `SEA`, `RBF` or `RADAR`. Each `stream.parquet` contains numeric features and the observed class label. `metadata.json` records its row count, seed, feature information and hash.

Synthetic `seed_0`, `seed_1` and `seed_2` folders are the exact 250,000-row benchmark realizations. `training_seed_112` and `validation_seed_113` are separate 60,000-row development realizations. They appear during their preparation phases. RADAR has one complete 484,753-row file.
