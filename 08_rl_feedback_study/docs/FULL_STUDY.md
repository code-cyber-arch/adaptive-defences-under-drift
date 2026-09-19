# Five-seed design across the complete experiment

The study addresses the existing RQ1 and RQ2. Stream preparation, poisoning, baseline prediction, passive detection, fixed responses, learned reset policies, protected-feedback sensitivity and observation filtering use one declared evaluation grid.

## Shared inputs and replication

SEA and RBF each have five independent evaluation streams (seeds 115–119), with 60,000 observations per stream. Each realization has clean data and six poisoning conditions: instance, concept and splice poisoning at 15% and 25%. This gives five matched seven-condition repetitions: 35 comparison entries per dataset.

RADAR uses the same chronological training, validation and evaluation partitions throughout the study. Evaluation contains 144,753 observations, corresponding to original rows `[340000,484753)`. Five attack assignments (0–4) are applied independently to each of the six poisoning conditions. The unmodified evaluation period is shared across the five comparison repetitions, giving the same 35 comparison entries as SEA and RBF. These entries refer to 31 unique input conditions. Reusing the clean reference does not create additional independent replication.

RL and learned observation filters each use fitting seeds 7, 17, 27, 37 and 47. Synthetic development uses separate training seed 112 and validation seed 114; five fitting seeds measure algorithmic randomness on these fixed development data, not five independent training populations. RADAR has one chronological training period and one validation period. These replication units are reported separately.

Each dataset has the same five-by-seven comparison structure and four-detector coverage. `analysis/full_study/comparison_index.csv` records the mapping from comparison entries to unique input conditions. Every condition has weight 1/7 within a dataset, averaging its five comparison entries first. Dataset-level results are reported separately; any overall summary must weight each dataset 1/3, not by row count or number of configurations.

All methods use matching observations, attack realizations, warm-up and protected reservations within an evaluation condition. The five-percent protected sample is excluded from training and host scoring in both screened and unscreened arms. Attacks are prepared inside each chronological RADAR partition.

## Experimental comparisons

| Phase | Design | Configurations |
| --- | --- | ---: |
| Baseline prediction and fixed responses | No reset, immediate reset and confirmed reset; screened and unscreened; four detectors for reset policies | 1,818 |
| Passive detection | Four detectors replay the fixed unscreened, no-reset predictions, without changing the learner | 404 |
| Learned reset control | Three datasets × four detectors × two representations × five fitting seeds | 120 controllers |
| Deterministic policy validation | Sixteen mappings for every dataset and detector | 1,344 |
| RL/control evaluation | Four controls and ten learned controllers per dataset/detector | 5,656 |
| Feedback sensitivity | Three usable fractions × two delays; fixed controls and both RL representations with five fitting seeds each | 33,936, including shared defaults |
| Filter fitting | Three datasets × two observation views × five fitting seeds | 30 filters |
| Filtering with confirmed reset | Four detectors × 23 filter/control configurations per condition | 9,292 |
| Filtering with no reset | Shared unfiltered reference and ten learned filters per condition | 1,111 |

The 23 filter configurations are unfiltered, two oracle controls, five seeds each for two random matched-withholding controls, and five seeds each for feature-only and feature-plus-observed-label learned filters. Oracle controls describe ideal-information bounds and do not represent deployable filters. The no-reset filter runs are shared across detector comparisons; they are not counted as four independent runs.

## Filter fitting and scoring

The discriminator remains the existing histogram gradient-boosted classifier. Fitting and threshold calibration use only their declared development partitions, with at most 60,000 sampled rows per stage and seed. The threshold targets five-percent rejection of legitimate validation observations. Features-plus-label uses the ordinary observed label, which may be poisoned. Attack annotations are offline supervised fitting targets and oracle-control information, not an operational inference channel.

Development feature fingerprints exclude exact copies from validation and evaluation host scoring. Every filter and its corresponding no-filter/no-reset control uses the same host scoring mask within an evaluation condition, including observations rejected by the filter. This additional copy exclusion means filter host scores must be compared against their matched controls rather than directly against unrestricted host scores from another phase.

The filter engine receives a fixed withholding mask before monitoring and training. Protected rows and warm-up are never withheld by a filter. No-reset controls retain screening and continuous learning, without monitoring or resets. RADAR partitions are already sliced; filter training does not partition them a second time.

## Analysis and outputs

All five seeds appear together in the same method tables and figures. Averages first combine repeats within each attack/severity condition, then give the seven conditions equal weight. Filter fitting seeds are averaged equally after this condition balancing. Per-seed results remain available. Baseline clean-versus-poisoned damage uses common eligible host rows within each stream realization.

The results are under `results/per_dataset_study/`: fixed responses, passive detectors, filters, RL evaluations and feedback sensitivity share the input manifests. `analysis/figures/` contains their figures; `analysis/full_study/` contains matched effects and verification. `full_completion.json` indicates completion of all experiments; `completion.json` refers to the RL/control and feedback grid.

The synthetic stream seeds, algorithm fitting seeds and RADAR attack assignments are different sources of variation. Five attack assignments do not create five independent real captures. RADAR feature-provenance limitations, offline attacker access within partitions, and the distinction between accuracy-based RL rewards and macro-F1 mapping selection remain explicit. RQ1 and RQ2 are unchanged.

The [comparison matrix](COMPARISON_MATRIX.md) defines all detector, policy, dataset and cross-experiment comparisons. Feedback sensitivity covers all seven conditions and both RL representations. Cross-experiment prediction scores use a common host scoring mask.
