# Reproduce the experiment

A clone contains code and retained evidence, but not the RADAR input. SEA and RBF inputs are generated from the declared seeds. To rerun all three datasets, obtain the exact `full_stream.parquet` and accompanying metadata from the author. The input is approximately 6.6 MiB; the creator below checks its SHA-256 against the metadata already published in `data/clean/RADAR/metadata.json`. An arbitrary upstream export is not interchangeable with this verified input.

Use Python 3.14 on macOS or Linux, or Linux through WSL on Windows. The complete-study runner uses POSIX file locking. Install the pinned dependencies with `bash setup.sh`; results may vary across platforms, so successful execution does not guarantee byte-identical outputs or identical scores on different numerical libraries.

## Inspect published evidence

```sh
python3 scripts/check_publication.py
bash setup.sh
bash run.sh --tests
```

The publication check verifies retained file identity. Tests exercise behaviour; neither reruns the full experiment.

## Create a fresh reproduction

From the clone, with the supplied input extracted:

```sh
python3 scripts/create_run.py jabed_reproduction --radar-input /path/to/full_stream.parquet
cd runs/jabed_reproduction
bash setup.sh
bash run.sh --full --workers 4
```

The creator copies code and settings into a new workspace, verifies and copies the RADAR input, and omits all retained results. The `--full` command runs dataset-specific training, validation, RL/control evaluation and feedback sensitivity, followed by fixed responses, passive detectors, filters and cross-experiment verification. Both stages must succeed. The final record is `results/per_dataset_study/full_completion.json` with `status: complete` and passed verification.

For a smaller integration check, create a separate workspace and run `bash run.sh --pilot --workers 4`. That pilot checks the RL/control path and is not the full three-experiment grid. Do not reuse pilot results as research evidence.

Allow substantial runtime and disk space: the retained complete local study occupies approximately 126 GB. Runtime depends on hardware and workers. To inspect or independently recompute metrics from the original execution instead of producing a new one, the original excluded prediction traces and fitted binary filters must also be obtained from the author.

## What has been checked

The published root-layout checkout passed 65 existing tests using the pinned local environment. Two additional regression tests cover repository-local RADAR loading and rejection of an incorrect input. The updated 67-test suite passed, and an isolated workspace successfully prepared pilot training inputs for SEA, RBF and RADAR using the verified full RADAR input. The complete experiment has not been rerun as a new independent reproduction after this packaging change. These checks establish the tested entry points and data preparation, not full cross-platform replication.
