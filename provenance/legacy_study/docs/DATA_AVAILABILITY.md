# Data and artifact availability

The current five-seed study has its own [publication scope](../08_rl_feedback_study/docs/PUBLICATION.md) and file-identity manifest. The legacy local-artifact manifest below applies to the earlier root benchmark.

## Public Git contents

The repository contains the frozen settings, source code, input metadata and hashes, completed metric tables, available run summaries, selected figures and verification records. It supports code inspection, small synthetic pipeline checks and saved-table reconciliation without downloading the full research data.

## Large local artifacts

The author's standalone working copy also retains the input Parquet files, protected channels, fitted filter artifacts and full supplementary prediction/event traces at their original relative paths. They are excluded from Git, not deleted. [provenance/local-artifacts.json](../provenance/local-artifacts.json) identifies each file by path, size and SHA-256.

The large artifacts have not been deposited at a new public download location. A Git clone therefore cannot reproduce every thesis run or repeat the complete trace audits by itself. The original benchmark's missing traces and 432 absent summaries are not reconstructed by this migration.

To restore an independently supplied artifact directory whose `data/` and `results/` paths match this project:

```bash
# Restore inputs only, or omit --data-only to restore all retained artifacts.
python3 scripts/restore_artifacts.py --from /path/to/artifact-directory --data-only
python3 scripts/check_repository.py --with-local-artifacts
```

The restore command verifies all source hashes before copying and refuses to overwrite a different local file. The full check requires the entire artifact set, not only the inputs.

## RADAR source

The study used the [RADAR v0.0.1-beta source release](https://zenodo.org/records/14564541), DOI `10.5281/zenodo.14564541`. Its published feature values and mapped labels were checked against the local bank. The source dataset is third-party material and retains its own terms.

Downloading the upstream archive alone does not install this project's input bank. The bank also contains declared preprocessing, synthetic streams and poisoned variants. Its exact file hashes are recorded. No claim is made that newly exporting Parquet under another software version will produce identical file bytes.

The [source review](../results/narrative/references/radar_release/PROVENANCE.md) records the feature provenance and its limits. A public release of the derived input bank and supplementary traces would improve access, but has not been completed in this repository migration.
