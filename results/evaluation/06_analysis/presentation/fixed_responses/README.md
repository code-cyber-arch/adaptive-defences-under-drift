# Fixed responses: performance, protection and cost

Start with performance/SEA_adwin_performance.pdf, then compare the other detectors within SEA before moving to RBF and RADAR. Each dataset/detector has a four-panel performance PDF and a companion under protection_and_cost/. All 24 PDFs are 170 x 128 mm, with a 2 x 2 layout, direct condition labels and a six-line legend. No HTML or combined multi-page PDF is produced. PNG files are previews.

Blue circles denote no reset, orange squares immediate reset, and green triangles confirmed reset. Dashed lines with hollow markers are unscreened; solid lines with filled markers are screened. Immediate and confirmed use the detector named in the filename. No-reset runs have no detector and the same reference is reused across detector figures.

The performance panels show accuracy, macro-F1, macro recall and ROC-AUC. All seven conditions and all 18 fixed arms within a stream/attack-assignment cohort are rescored on exactly the same host observations. Exclude warm-up, protected reservations and the union of all concept/splice replacement intervals. This is the same common mask as the preceding learner-baseline figures. It measures effects on unchanged host observations, not classification of the replacement examples. All original input rows were processed, including 484,753 RADAR rows; a smaller scoring denominator is not stream truncation.

Class-macro metrics weight declared classes equally; zero-division cases use zero. Synthetic ROC-AUC is macro one-versus-rest, while RADAR ROC-AUC contrasts malicious with benign scores. Required missing class support leaves AUC undefined. Additional descriptive metrics computed from stored predictions do not change the original experiment's primary outcomes.

The companion panels show poisoned-row admission, legitimate-data withholding, committed resets per 100,000 source rows and full-run runtime per 100,000 source rows. These use verified original full-run metrics, not the common scoring mask. Clean poison admission has no denominator and remains N/A. Admitted poisoned rows measure exposure; admission is not proof that every such row caused damage, and screening does not guarantee poison-free training. Low withholding retains legitimate learning opportunities. Runtime includes the original concurrent execution environment and should be interpreted alongside predictive performance.

Means and sample-standard-deviation bars use three synthetic realizations or three RADAR attack assignments. The RADAR clean predictions are reused on three common masks; predictive variation is not variation across independent captures. For full-run RADAR clean costs, the same run is repeated as a matched reference and zero variation does not establish runtime repeatability. Bars are descriptive, not confidence intervals. Undefined values are not zero. Y ranges are shared across detectors within each dataset and metric; they can differ between datasets. Lines join categorical conditions, not timestamps or a continuous dose-response sweep.

There are 1,098 unique source fixed-policy runs (61 conditions x 18 arms). Rescoring creates 1,134 cohort/run rows because the 18 RADAR clean runs are evaluated on three matched masks. The two no-reset arms are reused for each detector comparison without increasing the number of independent realizations. RL is excluded from this stage.

Read the figures in this order: first assess poisoning damage on the no-reset unscreened reference; next compare immediate and confirmed responses with that reference; then compare screened and unscreened versions of the same response; finally use the companion figures to assess exposure, withholding, resets and runtime. The next stage compares frozen RL policies with these fixed references.

tables/run_scores.csv lists every source run, scoring-mask hash, scoring denominator and metric. tables/plot_values.csv contains each plotted mean, SD and defined count. tables/paired_screening_effects.csv contains within-cohort screened-minus-unscreened differences; rate differences are percentage points, AUC differences are in AUC units, and costs retain their panel units. All comparisons are reported without selecting only favourable conditions.

Reproduce with `python -B 06_analysis/dataset_summary/fixed_responses.py --workers 3`. After completion, `--render-only` redraws the PDFs from verified saved tables without rescoring predictions. The manifest fingerprints all inputs, reporting code and outputs. The frozen benchmark, observations, learners and controllers are unchanged.

## Figure files

| Dataset | Detector | Performance | Protection and cost |
|---|---|---|---|
| SEA | ADWIN | [PDF](performance/SEA_adwin_performance.pdf) | [PDF](protection_and_cost/SEA_adwin_protection_and_cost.pdf) |
| SEA | HDDM-W | [PDF](performance/SEA_hddm_w_performance.pdf) | [PDF](protection_and_cost/SEA_hddm_w_protection_and_cost.pdf) |
| SEA | Hellinger | [PDF](performance/SEA_hellinger_performance.pdf) | [PDF](protection_and_cost/SEA_hellinger_protection_and_cost.pdf) |
| SEA | D3 OOF | [PDF](performance/SEA_d3_oof_performance.pdf) | [PDF](protection_and_cost/SEA_d3_oof_protection_and_cost.pdf) |
| RBF | ADWIN | [PDF](performance/RBF_adwin_performance.pdf) | [PDF](protection_and_cost/RBF_adwin_protection_and_cost.pdf) |
| RBF | HDDM-W | [PDF](performance/RBF_hddm_w_performance.pdf) | [PDF](protection_and_cost/RBF_hddm_w_protection_and_cost.pdf) |
| RBF | Hellinger | [PDF](performance/RBF_hellinger_performance.pdf) | [PDF](protection_and_cost/RBF_hellinger_protection_and_cost.pdf) |
| RBF | D3 OOF | [PDF](performance/RBF_d3_oof_performance.pdf) | [PDF](protection_and_cost/RBF_d3_oof_protection_and_cost.pdf) |
| RADAR | ADWIN | [PDF](performance/RADAR_adwin_performance.pdf) | [PDF](protection_and_cost/RADAR_adwin_protection_and_cost.pdf) |
| RADAR | HDDM-W | [PDF](performance/RADAR_hddm_w_performance.pdf) | [PDF](protection_and_cost/RADAR_hddm_w_protection_and_cost.pdf) |
| RADAR | Hellinger | [PDF](performance/RADAR_hellinger_performance.pdf) | [PDF](protection_and_cost/RADAR_hellinger_protection_and_cost.pdf) |
| RADAR | D3 OOF | [PDF](performance/RADAR_d3_oof_performance.pdf) | [PDF](protection_and_cost/RADAR_d3_oof_protection_and_cost.pdf) |
