# Public study evidence

Version 1.1.2 presents the completed five-seed study in the main numbered research folders. The repository has one authoritative stream, attack, detector, policy, training, experiment and analysis structure.

Published material includes scientific code, pinned dependencies, configuration and input metadata, fitted Q-table JSON, run summaries, aggregate CSV tables, vector figures, completion and verification records, and retained execution provenance. Current conclusions use `results/per_dataset_study/analysis/comparisons/` for common-row adaptation comparisons. Earlier studies, pilots and integration checks remain identifiable and are not additional thesis evidence.

The complete local study occupies approximately 126 GB. Input and trace Parquet files, binary filter models, downloaded archives, the local Python environment and the private thesis working folder remain excluded from Git. A public clone supports inspection and verification of retained files, but does not contain everything required to repeat the full trace audits. `full_completion.json` records the completed local verification; it is not a claim that those checks can run without the excluded artifacts.

To check public file identity using Python's standard library:

```sh
python3 scripts/check_publication.py
```

To run behavioural tests with the pinned environment:

```sh
bash setup.sh
bash run.sh --tests
```

Historical benchmark utilities are retained under `provenance/legacy_study/`. Use the publication check above for the current study. Files exported for the thesis are documented in [the evidence guide](../07_documentation/thesis_evidence/README.md).

The thesis cites an exact commit containing version 1.1.2, so later changes to `main` cannot silently change its reference. `CITATION.cff` at the repository root supplies author, title, version and release date. The project author is Abdurahman Mahammedsied; 2026 is the publication year.

A new full reproduction requires only the verified RADAR input in addition to the code and dependencies; synthetic streams and outputs are generated. See [reproduction instructions](REPRODUCIBILITY.md). Original traces are required for re-auditing the original execution, not for generating a new one.
