# Experiment sequence

Purpose: Separate synthetic validation; no automatic setting selection or retraining.

First, I prepared 14 conditions. All arms used identical observation inputs and the same protected audit reservation.

Then, I completed 168 policy runs. I compared no reset, immediate reset and confirmed reset with and without screening.

I then compared frozen alarm-only RL with RL using both alarms and recent error deterioration. Each representation used the same three training seeds and chose continue learning or reset. Both proposals passed through the same protected accuracy check. Their Q-tables were not updated during evaluation.

Next, I compared host accuracy, macro-F1, poisoning admission and legitimate withholding. These outcomes show whether reducing exposure also preserves useful learning.

RBF, ADWIN, Confirmed: screening changed mean accuracy by +0.23 percentage points and poison admission by -39.71 points.
SEA, ADWIN, Confirmed: screening changed mean accuracy by +0.41 percentage points and poison admission by -35.40 points.
RBF: RL persistence changed mean accuracy by -0.99 points and reset count by +12.4 relative to screened ADWIN confirmed.
RBF: RL persistence changed mean accuracy by +0.78 points and reset count by +8.9 relative to screened ADWIN immediate.
RBF: RL alarm changed mean accuracy by -0.52 points and reset count by +14.3 relative to screened ADWIN confirmed.
RBF: RL alarm changed mean accuracy by +1.24 points and reset count by +10.8 relative to screened ADWIN immediate.
SEA: RL persistence changed mean accuracy by +0.19 points and reset count by +15.6 relative to screened ADWIN confirmed.
SEA: RL persistence changed mean accuracy by +0.17 points and reset count by +12.9 relative to screened ADWIN immediate.
SEA: RL alarm changed mean accuracy by -0.09 points and reset count by +17.7 relative to screened ADWIN confirmed.
SEA: RL alarm changed mean accuracy by -0.10 points and reset count by +15.0 relative to screened ADWIN immediate.
RBF: adding error history changed mean accuracy by -0.47 points, macro-F1 by -0.004 and reset count by -1.9 relative to alarm-only RL at matching seeds. These differences do not establish statistical equivalence or guaranteed improvement.
SEA: adding error history changed mean accuracy by +0.27 points, macro-F1 by +0.003 and reset count by -2.1 relative to alarm-only RL at matching seeds. These differences do not establish statistical equivalence or guaranteed improvement.

Finally, I examined reset frequency, actual candidate-training work and execution time. Synthetic delay is interpreted with missed events; RADAR has no verified drift-event labels.

These are descriptive matched comparisons. Policy arms and RADAR attack assignments are not independent datasets. Gate acceptance is not proof that an update is poison-free. The supplied benchmark had been inspected before this evaluation.

The protected channel is a simulated assumption. It supplies original observations with a one-block delay and never contributes examples to learner training.
