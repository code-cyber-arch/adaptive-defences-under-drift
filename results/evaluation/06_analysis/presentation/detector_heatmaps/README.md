# Detector-performance heatmaps

Each PDF shows 112 mean scores: four metric panels x four detectors x seven conditions. Read down a column to compare detectors under the same condition; read across a row to see how one detector changes across attack types and severities. Accuracy, macro-F1, macro recall and ROC-AUC are shown. Darker blue means a higher score. Values are printed inside every cell.

Each figure fixes one dataset, response rule and screening setting. Start with immediate reset without screening, then confirmed reset, followed by the screened versions. The figures compare the classifier performance obtained with each detector and response; they are not measurements of drift-detection accuracy itself. All four settings are included without choosing only favourable conditions.

Colour ranges are identical across the four settings within each dataset and metric. Each panel has its own labelled colour bar. Ranges may differ between datasets or metrics and do not necessarily begin at zero. Use the printed numbers and bars when comparing figures. Percentage scores are shown to one decimal place; ROC-AUC is shown to three decimals, omitting the leading zero inside cells (for example, .804 means 0.804). Display rounding does not change the underlying scores. The selected colour map progresses from light yellow through green to dark blue, with contrasting cell text.

The means are copied exactly from the verified detector-comparison tables. No experiment was rerun and no conditions were pooled. All predictive comparisons use the existing common host-row masks, excluding warm-up, protected reservations and concept/splice replacement intervals across all seven conditions. The original full streams, including all 484,753 RADAR rows, remain unchanged.

Synthetic means use three stream realizations. RADAR means use three matched attack assignments on one capture; its clean predictions are reused on the corresponding masks. The heatmaps show means only. Standard deviations, defined counts and unrounded values remain in plot_values.csv. Use the matching detector_comparison line plots to inspect error bars. Neither colour differences nor rounded differences establish statistical significance.

Synthetic ROC-AUC is macro one-versus-rest; RADAR ROC-AUC contrasts malicious with benign scores. No observation-level poison-identification metric is implied. Higher predictive scores must still be considered alongside poisoning admission, legitimate-data withholding and computational cost in the companion fixed-response figures.

All PDFs are individual, vector figures in a 170 x 128 mm, 2 x 2 layout. They have metric names, direct row/column labels, cell values and colour bars, with no surrounding heading or explanatory footer. Dataset folders contain PDFs; PNG previews are separate. No HTML or combined PDF is produced.

Reproduce with `python -B 06_analysis/dataset_summary/detector_heatmaps.py`. plot_values.csv preserves the source mean, SD and sample count. displayed_cells.csv records every displayed number and its dataset, setting, metric, detector and condition. manifest.json fingerprints the source tables, reporting code and outputs.

| Response setting | SEA | RBF | RADAR |
|---|---|---|---|
| Immediate reset, unscreened | [SEA PDF](SEA/SEA_immediate_unscreened_heatmap.pdf) | [RBF PDF](RBF/RBF_immediate_unscreened_heatmap.pdf) | [RADAR PDF](RADAR/RADAR_immediate_unscreened_heatmap.pdf) |
| Confirmed reset, unscreened | [SEA PDF](SEA/SEA_confirmed_unscreened_heatmap.pdf) | [RBF PDF](RBF/RBF_confirmed_unscreened_heatmap.pdf) | [RADAR PDF](RADAR/RADAR_confirmed_unscreened_heatmap.pdf) |
| Immediate reset, screened | [SEA PDF](SEA/SEA_immediate_screened_heatmap.pdf) | [RBF PDF](RBF/RBF_immediate_screened_heatmap.pdf) | [RADAR PDF](RADAR/RADAR_immediate_screened_heatmap.pdf) |
| Confirmed reset, screened | [SEA PDF](SEA/SEA_confirmed_screened_heatmap.pdf) | [RBF PDF](RBF/RBF_confirmed_screened_heatmap.pdf) | [RADAR PDF](RADAR/RADAR_confirmed_screened_heatmap.pdf) |

[Matching line plots and uncertainty](../detector_comparison/README.md)
