# Dataset summary figures

For RL versus screened fixed responses, run `python -B 06_analysis/dataset_summary/rl_comparison.py --workers 3` after the fixed-response tables exist. Open `results/evaluation/06_analysis/presentation/rl_comparison/README.md`. Nine individual PDFs cover performance, protection/cost and paired improvement against screened confirmed-reset ADWIN for all three datasets. RL seeds are averaged within each cohort before descriptive SDs across cohorts are calculated. `--render-only` redraws the checked tables without rescoring.

For annotated detector heatmaps, run `python -B 06_analysis/dataset_summary/detector_heatmaps.py` after the direct detector comparisons exist. Open `results/evaluation/06_analysis/presentation/detector_heatmaps/README.md`. Each PDF contains four 4-by-7 metric panels, displaying 112 mean scores. Colour ranges are shared across response settings within each dataset and metric. Use the matching line plots and tables for uncertainty.

For direct detector comparisons under a matched response, run `python -B 06_analysis/dataset_summary/compare_detectors.py` after the fixed-response tables exist. Open `results/evaluation/06_analysis/presentation/detector_comparison/README.md`. Each four-panel PDF has four detector lines on the same axes, with response and screening held constant. Dataset subfolders contain the individual PDFs; previews are separate. The fixed_responses figures instead compare policies within a detector.

Start with the unprotected learner comparison: run `python -B 06_analysis/dataset_summary/baseline.py`. Its three PDFs are under `results/evaluation/06_analysis/presentation/baseline_clean_vs_poisoned/`. They compare clean data with the six poisoning conditions using only unscreened, no-reset predictions. Each dataset has four panels: accuracy, macro-F1, macro recall and ROC-AUC.

Next, run `python -B 06_analysis/dataset_summary/detector_baseline.py --workers 4`. This evaluates the four detectors on the saved baseline features and predictions without applying a response policy. Its three PDFs are under `results/evaluation/06_analysis/presentation/detector_baseline/`, with four supported detection and monitoring metrics per dataset and one line per detector. RADAR has no verified drift-event references, so recall and delay are unavailable. The passive evaluation is separate from the completed policy benchmark.

Both sets use `compact.py` for a 170 x 128 mm layout: A4 text width and less than half an A4 page in height. Each PDF has four panels in a 2 x 2 layout with directly labelled conditions, no heading and no explanatory footer. PDF is the presentation format; PNG copies are previews. The following existing comparisons show later policy and RL stages; their four-panel layout will follow separately.

Run `python -B 06_analysis/dataset_summary/build.py` from the project root after benchmark verification completes.

For line plots with error bars, run `python -B 06_analysis/dataset_summary/lines.py`. Open `results/evaluation/06_analysis/presentation/line_comparisons/index.html` to compare both requested layouts: three dataset figures with stacked attack panels and nine separate attack figures. These plots rescore saved predictions on common eligible rows across severities; their README explains the error bars and scoring support.

This standalone reporting step reads the verified metrics and writes three individual summary figures to `results/evaluation/06_analysis/presentation/summary/`. Each PDF is 154.94 mm wide and contains one accuracy heatmap. PNG copies, the underlying tables, short captions and an HTML index are included.

Rows retain individual detector/response/screening combinations. Columns keep clean data and attack severities separate. RL training seeds are averaged within each condition before averaging stream/attack realizations. No synthetic accuracy is pooled with RADAR accuracy.

The attack name is **Splice**. In this implementation, splicing replaces a 250-row interval with features and labels copied from another archive segment. The existing detailed reports call this mechanism Replay; both names refer to the same stored `splice` conditions. The archive can include future source segments, so this is not restricted to replay of previously observed stream data.

The summary has its own manifest recording its script, input hashes, aggregation and output hashes. The completed experiment's frozen source files and existing detailed reports remain intact.

Redraw the two completed summary sets without rerunning learners or detectors with `python -B 06_analysis/dataset_summary/render_summaries.py`. Input table hashes are checked before rendering.

For the fixed-response stage, run `python -B 06_analysis/dataset_summary/fixed_responses.py --workers 3`. Open `results/evaluation/06_analysis/presentation/fixed_responses/README.md`. The performance folder contains twelve four-panel PDFs (three datasets x four detectors), and protection_and_cost contains twelve companion PDFs. Six lines compare no reset, immediate reset and confirmed reset with and without screening. `--render-only` redraws verified tables without rescoring. RL is the following presentation stage.
