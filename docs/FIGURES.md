# Report figure style

All final figures use the thesis's established compact style: 170 × 128 mm, four panels in a 2 × 2 layout, DejaVu Sans, embedded TrueType fonts, light backgrounds and grids, and consistent colour/marker combinations. Screened and unscreened versions of a fixed policy retain the same colour, with filled/solid versus open/dashed marks.

Use the individual dataset/detector PDF files from `results/per_dataset_study/analysis/figures/` when inserting figures into the report. These are vector exports sized for a full-width report figure. PNG files are previews. Combined PDFs collect related figures across datasets and detectors; a combined PDF's PNG shows its first figure.

- Baseline: `<dataset>_clean_vs_poisoned.pdf`.
- Passive detection: `<dataset>_detector_baseline.pdf`.
- Fixed responses: `<dataset>_<detector>_prediction_screening.pdf` and `_protection_work.pdf`.
- RL: `<dataset>_<detector>_RL_comparison.pdf`, with individual fitting-seed figures for both representations.
- Feedback: `<dataset>_<detector>_feedback_delay_<delay>.pdf`.
- Filtering: `<dataset>_<detector>_filter_learned.pdf`, `_filter_controls.pdf` and `_filter_reset_advantage.pdf`.

The curves average fitting seeds within each input condition. Error bars are descriptive standard deviations across evaluation stream realizations or RADAR attack assignments; they are not confidence intervals. Reused RADAR clean references do not increase the number of independent observations. Dedicated RL seed figures show every fitting seed. Feedback figures use equal-condition averages without error bars. The generated `FIGURE_GUIDE.md`, plot-value CSV files and `report_figure_index.json` specify each figure's meaning and provenance.

Final figures are regenerated only from completed results. Integration-test previews are excluded from thesis findings.

The comparison framework in `graphics/comparison_heatmap.pdf` retains its single combined layout, as requested. Its canvas is 170 × 145 mm for A4 placement, with 10-point panel titles and 8.5-point labels. Insert at 170 mm width. This design diagram is an exception to the results figures’ 2 × 2 layout. The separate coverage export is superseded.
