# Evidence used in the thesis

The vector figures and LaTeX tables match the current thesis evidence. The thesis draft itself is not published here. The completed study uses shared adaptation scoring rows, equal condition weighting and all declared fitting seeds. Baseline damage and feedback sensitivity use their separately declared populations.

`paired_variability.csv` and `paired_values.json` distinguish evaluation-repetition and fitting-seed variation. `filter_discrimination.csv` contains held-out ROC AUC means. `radar_scoring_support.csv` retains counts for every RADAR evaluation condition. These are descriptive summaries, not significance tests.

`generate_evidence.py` regenerates the three supporting evidence tables and CSVs. It requires the local Parquet masks and truth channels at their recorded paths; a public clone alone is insufficient. Run it with the pinned environment from the repository root. `evidence_hashes.json` identifies the underlying files. The other tables and figures are retained report exports from the verified study.
