# Experimental protocol: policy comparison and feedback robustness

## Existing research questions and experimental placement

The thesis retains its existing research questions verbatim:

- **RQ1:** How does screening proposed model updates affect poisoning exposure, legitimate learning and predictive performance?
- **RQ2:** Do learned reset policies or upstream observation filters improve on a fixed screened response?

The deterministic-policy comparisons are controls within Experiment 2, supporting the learned-reset part of RQ2. Feedback sensitivity is a robustness analysis of the protected screening assumption used in Experiments 1 and 2, supporting interpretation of RQ1 and RQ2. It does not create a fourth experiment or a new research question. Observation filtering remains Experiment 3 and is not changed here.

All completed conditions and all learned seeds are reported; improvement is not assumed. The sensitivity grid compares screened responses and does not independently re-estimate the screened-versus-unscreened treatment effect.

## Shared system

Use the existing Hoeffding Tree, 1,000-row blocks, 1,000-row warm-up, ADWIN, candidate isolation, screening criterion and poisoning generators. Predictions precede candidate learning. Protected feedback retains original features and labels; ordinary error-state inputs use the existing observed-label semantics. The feedback sensitivity changes the protected gate, not the timeliness or correctness of ordinary labels.

For fixed confirmed reset, retain its existing buffered refit and confirmation timing. For all four-state mappings and both learned controllers, reset refits the current block. Consequently the same-state mapping comparison isolates policy choice within that policy class; comparisons with confirmed reset concern the whole response mechanism, including its different refit window.

## Development and assessment

- Existing RL training: synthetic seed 112; four training passes; controller seeds 7, 17 and 27; no retraining or new convergence claim.
- Deterministic policy selection: new synthetic seed 114 for both SEA and RBF, 60,000 observations each.
- Evaluation: new seeds 115, 116 and 117 for each synthetic family, 60,000 observations per realization. These are independent generated realizations, not independent real deployments.
- Each stream has clean plus three existing intervention families at 15% and 25%, producing 14 validation and 42 evaluation conditions.
- All attacks, settings, policy-selection rules and evaluation seeds are specified before inspecting these new evaluation outcomes. The design is informed by the already inspected original study; this is not a claim of retrospective preregistration.
- Runtime pilot: seed 9917, both synthetic families, clean and 25% label poisoning. Pilot outputs are excluded from the research tables.

## Policy comparison

The four states are ordered `(alarm, elevated error) = 00, 01, 10, 11`. Error elevation uses the existing disjoint five-block baseline and two-block recent window, with a strict 0.02 margin. Each mapping has two choices, update or reset, so there are 16 deterministic mappings. `0` means update and `1` means reset.

Evaluate all mappings using 5% protected feedback and one-block delay on validation only. Select by mean macro-F1 with equal condition weight (balanced across the two families); break exact ties by fewer mean committed resets, then lexical mapping ID. Freeze the selected mapping and its input/protocol hashes before evaluating it. It is the validation-selected mapping, not a test oracle or globally optimal policy.

Evaluation compares ten arms: screened continuous learning, screened confirmed reset, a same-state error rule (`0101`), the validation-selected mapping, and both RL representations for each of the three saved seeds. Identical selected and rule mappings remain labelled as conceptual comparators but must not be treated as independent evidence. Main endpoints are macro-F1 and paired differences against confirmed reset; also report accuracy, poison admission, legitimate withholding, reset counts, training visits and runtime. Clean-condition poison admission is undefined, not zero.

The mapping-selection objective is validation macro-F1, whereas the retained Q-learning reward is delayed protected-sample accuracy. The comparison therefore evaluates the implemented procedures; it does not isolate optimisation algorithm alone or establish convergence.

## Feedback robustness

Retain the same five percent reserved positions at every setting. Release a deterministic, nested subset corresponding to 1%, 2% or 5% of each complete block. The selected subset depends only on source identity and row position, never labels, attacks or outcomes. Unreleased reserved rows remain excluded from both ordinary training and final scoring. This isolates the effect of usable feedback; it does not estimate a deployment that returns the unused reserve to training.

Use delays of one and five blocks with a one-source-block feedback window. At five-block delay, updates after warm-up fail closed until feedback first arrives; report those no-feedback blocks. The selected policy and saved Q-tables remain frozen. Reward logging is disabled, so there is no training or fictitious one-block reward release under delayed feedback. Prediction and gate decisions at the default setting are checked against the original reward-logging evaluation mode.

Use clean and severe (25%) label poisoning on all six evaluation realizations: 12 conditions. The seven arms are continuous learning, confirmed reset, the error rule, the selected mapping and the three four-state Q controllers. Run all six feedback settings. Reuse the 84 matching default-setting evaluations already present in the policy comparison; do not count them twice.

## Grid and reporting

There are 224 validation executions, 420 policy-evaluation executions and 420 additional sensitivity executions: 1,064 unique research executions. The sensitivity table contains 504 entries, including the 84 shared defaults. The pilot has 24 separate executions.

Retain prediction traces, admission histories, per-block events, feedback-release schedules, metrics and hashes for every execution. Report paired condition effects and per-stream-seed variability. Three evaluation seeds permit descriptive uncertainty summaries but do not support treating rows, policy seeds, feedback settings or reused controls as independent observations. No significance claim or confidence interval is introduced by this implementation.

The scope is 60-block synthetic streams and the declared interventions. It does not establish generalisation to additional real datasets, deep RL, noisy protected labels or adaptive adversaries. A four-state map that matches or exceeds Q-learning is a valid result about the tested learning and representation, not evidence against RL generally.
