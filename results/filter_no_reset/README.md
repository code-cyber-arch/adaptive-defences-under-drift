# Matched continuous-learning controls for Experiment 3

This post-hoc extension adds 183 controls: 61 input conditions with no filter, the features-only filter, and the features-plus-label filter. Each model continues learning with the protected update gate enabled and never resets. The baseline is not a frozen classifier. The fitted poisoning filters are reused without refitting or recalibration.

Each control uses the original filter-study observations, protected positions, saved withholding decisions and scoring mask. Its four pairings with confirmed-reset monitors give 732 comparisons, not 732 independent runs. RADAR processing starts at the beginning of the stream; the reported assessment uses the original filter suffix. The comparison therefore includes the effect of different reset histories before filtering starts.

## Evidence and commands

Run from `drift-response-2`:

```sh
.venv/bin/python -B -m extensions.filter_no_reset.run --workers 4
.venv/bin/python -B extensions/thesis_revision/check_archived_no_reset.py --require-complete
.venv/bin/python -B -m extensions.thesis_revision.audit_no_reset
```

The first command resumes completed controls after verifying their hashes and the frozen request contract. Do not change the frozen filter engine or input files to force a resume. `status.json` records execution and audit completion. `verification.json` records the separate saved-trace checks. The audit recomputes main metrics, checks zero resets and protected-row isolation, and verifies exact mask equality against each matched monitor arm.

- `runs/`: predictions, block events and summary for each control.
- `metrics.csv`: all 183 control records.
- `paired_comparisons.csv`: each confirmed-reset result minus its matched control.
- `condition_comparisons.csv` and `condition_descriptive_spread.csv`: condition means and descriptive variation.
- `overview.csv`: clean results separately from the equally weighted six attacked-condition means.
- `within_no_reset_filter_overview.csv`: the added effect of filtering while resets remain absent.

## Archived baseline reproduction assessment

`archived_baseline_reproduction.csv` rescores the new no-filter controls on the original archived scoring masks. Those scores differ from the original Windows archive despite identical input hashes and relevant settings. The precise cause was not isolated because the original prediction traces are unavailable. The JSON assessment explicitly distinguishes completed checking from exact reproduction.

Two full 250,000-row synthetic streams were additionally checked on the current Mac. The original and extension no-reset engines agreed exactly on predictions, probabilities, training visits and non-timing events. They also agreed with the separately saved new controls. Two saved Mac no-filter confirmed-reset ADWIN trajectories were reproduced exactly. A full 484,753-row RADAR severe label-poisoning assignment also reproduced its saved no-reset and ADWIN reset references exactly. These checks are recorded in `results/methods_revision/engine_reproduction.json`, `saved_filter_reproduction.json` and `radar_reproduction.json`. They support the matched Mac comparison, but are not a proof of exact Windows reproduction or independent generalisation.

The original `results/evaluation/` and `results/filter_study/` evidence is preserved. New results must not be substituted into archived original comparisons with different scoring populations.
