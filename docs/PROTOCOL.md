# Protocol: all datasets, all detectors, per-dataset training

The existing research questions remain unchanged:

- **RQ1:** How does screening proposed model updates affect poisoning exposure, legitimate learning and predictive performance?
- **RQ2:** Do learned reset policies or upstream observation filters improve on a fixed screened response?

Policy controls belong to Experiment 2. Feedback sensitivity supports the protected-screening assumptions in Experiments 1 and 2. Experiment 3 uses the same five-seed evaluation inputs; its fitting, controls and scoring are specified in [the complete design](../docs/FULL_STUDY.md).

## Dataset partitions and preparation

SEA and RBF use independent synthetic seeds for training (112), validation (114) and evaluation (115, 116, 117, 118, 119), with 60,000 rows per realization. Each dataset is fitted separately.

The complete RADAR source contains 484,753 rows. Training uses original rows `[0,242000)`, validation `[243000,339000)` and evaluation `[340000,484753)`. The periods are chronological with one 1,000-row embargo between adjacent periods. A separate 1,000-row warm-up is used in each period. All eight labels occur in each full partition. Row maps connect local model input rows to immutable original row indices.

RADAR poisoning is regenerated independently inside each period. Label-target construction, generated-concept feature sampling and splice donor selection use only that period's labelled archive. Positional segment boundaries are rebased to the period. Within-period future reference remains an explicit offline attacker capability; future evaluation observations cannot enter training. The original attacks spanning the entire archive are not sliced and reused for fitting.

Training and validation have seven conditions per dataset: clean plus three attack families at 15% and 25%. RADAR uses one attack assignment per training/validation condition. Evaluation retains five attack assignments per poisoned RADAR condition (31 conditions total including one clean), and five stream seeds per synthetic family (35 conditions each).

## Phase 04: 120 controllers

Train separate controllers for every dataset × detector × representation × RL seed combination: three datasets, ADWIN/HDDM-W/Hellinger/D3 OOF, two/four states, and seeds 7/17/27/37/47. Each model sees only its dataset's declared training partition. Record dataset, detector, input contract, state definition and seed in its artifact.

Use four passes over seven conditions, so there are 28 episodes per controller and 3,360 episodes in total. Episode order is identical across detectors and state representations for a matching dataset and seed. Training budgets are matched across detectors within a dataset. RADAR episodes are longer than synthetic episodes; this is not a compute-matched comparison across datasets or a convergence claim.

The host is a Hoeffding Tree. Blocks and warm-up contain 1,000 observations. Candidate updates retain the original protected-accuracy gate. The reward remains one-block-delayed protected accuracy. Ordinary errors retain the existing observed, potentially poisoned labels. The two actions continue learning or propose reset. Q-tables are frozen for evaluation, which rejects dataset or detector mismatches.

## Validation and policy evaluation

For each of the 12 dataset/detector combinations, validate all 16 deterministic mappings of the four alarm/error states to the two actions. Select by mean macro-F1, then fewer mean committed resets, then mapping ID. Freeze the twelve selected mappings before evaluation. Selection maximises macro-F1 while Q-learning rewards accuracy; the comparison therefore concerns the implemented procedures, not optimisation algorithm alone.

For every evaluation dataset and detector, compare screened no reset, screened confirmed reset, the same-information error rule (`0101`), the selected mapping and all ten learned controllers. State order is `00,01,10,11` for `(alarm,elevated error)`. No-reset references disable monitoring and are shared evidence across detector panels. Confirmed reset retains its buffered refit; mapped and learned policies reset from the current block. This distinction remains attached to comparisons against confirmed reset.

There are 1,344 validation configurations and 5,656 default-feedback evaluation configurations. All evaluation methods use matching inputs, protected reservations and scoring rows within a condition. No controller seed is selected by evaluation outcomes. Previously inspected benchmark data are not called a wholly untouched independent replication.

## Feedback sensitivity on every dataset and detector

Use all seven clean/attack conditions for each dataset: five evaluation seeds for SEA and RBF; one shared clean suffix and five assignments per attack condition for RADAR. Compare no reset, confirmed reset, the error rule, the selected mapping and both RL representations, each with five fitting seeds, with all four detectors.

Keep a fixed 5% reservation, releasing nested, label-independent subsets corresponding to 1%, 2% or 5% of rows after one or five blocks. Unreleased reserved rows remain excluded from both training and scoring. Missing feedback rejects the candidate. Ordinary labels keep their original timing. Controllers are frozen and reward logging is disabled during sensitivity evaluation.

The sensitivity design contains 33,936 entries, including 5,656 shared defaults and 28,280 additional configurations. Together with policy validation, the RL/control/feedback grid contains 35,280 unique configurations. The fixed-response, passive-detector and filter grids are specified in [the complete design](FULL_STUDY.md). Reused defaults and repeated no-reset controls are not independent experimental units. The pilot uses 12,000-row fixtures, one training pass and one RL seed: 24 controllers and 288 comparisons, all excluded from research results.

## Metrics, verification and limits

Report macro-F1, accuracy, poison admission, legitimate withholding, reset activity, learning work and runtime. RADAR additionally reports malicious recall and benign false-positive rate. Grand summaries first average repeats within an attack/severity condition, then weight the seven conditions equally; a single clean RADAR realization must not receive one-fifth the weight of each attacked condition.

Retain input/trace hashes, source row maps, training transitions, per-block events, prediction/admission traces and release schedules. Verify split-local replay, dataset/detector model identity, matched training order, delayed rewards, frozen actions, protected-row exclusion, feedback timing and independently recalculated metrics. The final detector figure contains all three datasets; feedback figures and tables cover all three as well.

Synthetic stream seeds and conditional attack assignments on one RADAR capture are different replication units. RADAR evaluation covers its final temporal partition and is not directly pooled with the original full-stream benchmark or the filter study's host trajectories. The supplied whole-input frequency feature and unresolved FastText provenance remain limitations; chronological RL partitions do not establish past-only feature extraction. No inference about deep RL, adaptive attackers or independent real deployments is made.

The complete baseline, passive-detector, fixed-response and filtering design is specified in [FULL_STUDY.md](../docs/FULL_STUDY.md). All three experiments use the same evaluation partitions.
