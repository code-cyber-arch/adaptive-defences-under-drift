# Compare the detectors directly

Each panel contains four detector lines: ADWIN, HDDM-W, Hellinger and D3 OOF. Colours, markers and line styles identify the same detectors throughout. The response and screening setting is fixed within each PDF. These figures compare classifier performance achieved with each detector under a matched response rule; they do not label classifier accuracy as accuracy of drift detection itself.

Start with **immediate reset, unscreened** to compare detectors under the simplest response. Next compare confirmed reset with screening still disabled. Finally inspect the screened versions. Every setting is reported; this order is for reading, not a selection of favourable outcomes. No-reset references belong to the preceding learner comparison because they have no detector.

Each PDF has the same four panels: accuracy, macro-F1, macro recall and ROC-AUC. The x-axis shows clean plus six poisoned conditions. Higher values are better for these four predictive metrics. Four detector lines now share each panel; there is no need to switch files to compare detectors. There are twelve individual PDFs: three datasets x two response rules x two screening settings. Dataset subfolders keep the files together; PNG previews are stored separately. Each figure is 170 x 128 mm, in a 2 x 2 layout, without surrounding headings or footers.

Points and SD bars are reused exactly from the verified fixed-response tables. All four detectors use identical scoring observations within each matched stream/attack-assignment cohort. Y-axis ranges are shared across every response/screening setting within each dataset and metric. No policies, severities or detectors are pooled into a single score. No learner, detector or controller is rerun.

Synthetic points average three stream realizations. RADAR points average three matched attack assignments on one capture; clean predictions are scored on the three common masks. Error bars are one sample SD, not confidence intervals or evidence of statistical significance. ROC-AUC is macro one-versus-rest for synthetic data and malicious-versus-benign for RADAR. Host scoring excludes protected reservations, warm-up and all replacement intervals in the matched seven-condition comparison. The full RADAR input remains 484,753 rows. This is comparative predictive performance under drift and poisoning, not observation-level poison classification.

Read across conditions to see sensitivity to attack type and severity. Compare detector lines vertically to assess detector choice while holding the response fixed. Then compare the corresponding screened/unscreened PDFs to see whether screening changes the detector ordering. Protection, withholding and computational costs remain available in the fixed_responses companion figures; predictive scores alone do not establish a universally best defence.

Reproduce with `python -B 06_analysis/dataset_summary/compare_detectors.py`. plot_values.csv includes every displayed mean, SD, source response profile and defined count. The manifest links the source tables, unchanged experiment request, rendering code and generated files.

| Response setting | SEA | RBF | RADAR |
|---|---|---|---|
| Immediate reset, unscreened | [SEA PDF](SEA/SEA_immediate_unscreened.pdf) | [RBF PDF](RBF/RBF_immediate_unscreened.pdf) | [RADAR PDF](RADAR/RADAR_immediate_unscreened.pdf) |
| Confirmed reset, unscreened | [SEA PDF](SEA/SEA_confirmed_unscreened.pdf) | [RBF PDF](RBF/RBF_confirmed_unscreened.pdf) | [RADAR PDF](RADAR/RADAR_confirmed_unscreened.pdf) |
| Immediate reset, screened | [SEA PDF](SEA/SEA_immediate_screened.pdf) | [RBF PDF](RBF/RBF_immediate_screened.pdf) | [RADAR PDF](RADAR/RADAR_immediate_screened.pdf) |
| Confirmed reset, screened | [SEA PDF](SEA/SEA_confirmed_screened.pdf) | [RBF PDF](RBF/RBF_confirmed_screened.pdf) | [RADAR PDF](RADAR/RADAR_confirmed_screened.pdf) |
