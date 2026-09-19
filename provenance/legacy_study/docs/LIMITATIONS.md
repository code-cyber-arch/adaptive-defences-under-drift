# Scope and evidence limits

- RADAR is one assembled laboratory stream, not several independent deployments. Its repeat attack assignments are conditional repeats on that stream.
- One supplied feature uses whole-input frequency statistics. FastText fitting provenance remains unresolved. Online decisions were checked on supplied vectors; end-to-end past-only feature extraction was not established.
- Protected feedback is simulated, reliable, fixed at 5%, and delayed by one block. Other quantities, delays and error rates were not tested.
- The original GitHub archive lacks prediction/event Parquet traces and 432 instance-poisoning summaries. Aggregate results and available summaries remain inspectable. The supplementary filter and no-reset studies retain their full local traces.
- Archived Windows baseline scores were not reproduced exactly on the Mac. Selected complete Mac trajectories did reproduce exactly. These are different claims; see [the reproduction record](../results/filter_no_reset/archived_baseline_reproduction.json).
- Two small tabular Q-learning controllers were evaluated. There is no fixed error-rule ablation that isolates additional state information from policy learning.
- The full run count measures coverage of the comparison grid. It is not an independent statistical sample size. Reported condition ranges and standard deviations are descriptive.
- Alarms and committed resets do not establish attacker intent or false adaptation. RADAR lacks verified natural-drift event ground truth.

These limits remain attached to the retained results after migration. A new repository name and a passed file-integrity check do not change the scientific scope.
