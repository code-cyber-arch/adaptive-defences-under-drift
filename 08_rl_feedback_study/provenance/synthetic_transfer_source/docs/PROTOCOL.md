# Protocol: matched detector and reset-policy comparison

The existing questions remain unchanged:

- **RQ1:** How does screening proposed model updates affect poisoning exposure, legitimate learning and predictive performance?
- **RQ2:** Do learned reset policies or upstream observation filters improve on a fixed screened response?

The policy controls belong to Experiment 2 (RQ2). Feedback robustness supports interpretation of the screening assumption in Experiments 1 and 2 (RQ1/RQ2). Experiment 3 is unchanged.

## Detector-specific training

ADWIN, HDDM-W, Hellinger and D3 OOF each receive two-state and four-state Q-learning controllers for seeds 7, 17 and 27. Phase 04 trains all 24 controllers on the same SEA/RBF seed 112, 60,000 rows per stream, clean plus six poisoned conditions, and four passes. Equal episode order for matching RL seeds, reward, learning settings and data prevent unequal training budgets. There are 56 episodes per controller, 1,344 total. Controllers record their detector and training split; evaluation requires exact matching and frozen Q-tables.

All arms use the same Hoeffding Tree, 1,000-row blocks and warm-up, screening rule and observation channels. Fixed confirmed reset retains its buffered refit; learned and enumerated mappings reset from the current block. Ordinary error feedback uses the existing potentially poisoned labels. Protected feedback is simulated and reliable.

## Selection and evaluation

Validate all 16 four-state/two-action mappings separately for each detector using seed 114. State order is `00,01,10,11` for `(alarm,elevated error)`; action 0 continues learning and action 1 proposes reset. Select by mean macro-F1, then fewer mean committed resets, then lexical mapping ID. Freeze four selected mappings before evaluating them.

Evaluate on identical seeds 115, 116 and 117 for SEA/RBF, each with clean and six poisoned conditions. For each detector compare screened continuous learning, confirmed reset, an error-only rule (`0101`), its selected mapping and its six learned controllers. No-reset controls disable monitoring and are shared evidence across detector panels, not independent detector performances.

Mapping selection uses macro-F1; RL uses delayed protected accuracy. This compares implemented procedures, not optimisation algorithm alone. Four passes do not establish convergence. Selection uses one validation realization per family; three evaluation stream seeds support descriptive variation, not row-level significance claims.

The evaluation realizations are kept identical across detectors. Their ADWIN-only outcomes have already been inspected, so the corrected comparison is not claimed to be a wholly untouched independent replication. Settings and budgets are fixed across detectors; no best test seed is selected.

## Feedback robustness

Keep the same 5% protected reservation at every setting, excluding all reserved observations from learning and scoring. Release label-independent nested subsets corresponding to 1%, 2% and 5% of source rows, with one- or five-block delay and a one-source-block feedback window. Unused protected rows are not returned to training. Longer-delay screening fails closed until evidence arrives; ordinary labels keep their original timing.

Use clean and severe label poisoning on all six evaluation realizations. For each detector compare no reset, confirmed reset, the error rule, the selected mapping and all three four-state controllers. All policies remain frozen. This sensitivity isolates usable protected information under a fixed reservation; it does not re-estimate the screened/unscreened treatment effect or model noisy protected labels.

## Counts, checks and reporting

The full comparison contains 896 validation runs, 1,680 default-setting evaluations and 1,680 additional feedback runs: 4,256 executed comparison configurations. The sensitivity table has 2,016 entries because 336 defaults are reused. Repeated no-reset controls are not independent experimental units. The separate 12,000-row pilot fits eight controllers with one pass and seed 9917, then runs 192 checks; pilot outcomes are excluded.

Retain row predictions, admission history, per-block decisions, feedback schedules, training transitions and hashes. Verify model/detector identity, matched training order, delayed rewards, frozen evaluation, protected exclusion, feedback timing and metrics recalculated from traces. Pair policy effects within detector/condition and feedback effects within detector/policy/condition. Show all detectors and RL seeds, macro-F1, accuracy, poison admission, legitimate withholding, reset activity, training work and time.

The dataset scope is the declared synthetic streams. The original RADAR evidence remains separately available; this component does not claim new RADAR or additional-dataset validation.
