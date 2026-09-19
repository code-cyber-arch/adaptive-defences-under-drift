"""Write integrated results for the existing research questions from verified tables."""
from pathlib import Path
import json
import pandas as pd
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/policy_robustness'
check = json.loads((OUT / 'analysis/verification/verification.json').read_text())
assert check['status'] == 'passed'
x = pd.read_csv(OUT / 'analysis/evaluation/metrics.csv')
s = pd.read_csv(OUT / 'analysis/sensitivity/metrics.csv')
selected = json.loads((OUT / 'selected_policy.json').read_text())
arms = ['no_reset', 'confirmed', 'error_rule', 'selected_map', 'q_alarm_7', 'q_alarm_17', 'q_alarm_27',
        'q_alarm_persistence_7', 'q_alarm_persistence_17', 'q_alarm_persistence_27']
labels = ['Screened no reset', 'Screened confirmed reset', 'Same-state error rule', 'Validation-selected mapping',
          'Two-state Q, seed 7', 'Two-state Q, seed 17', 'Two-state Q, seed 27',
          'Four-state Q, seed 7', 'Four-state Q, seed 17', 'Four-state Q, seed 27']
means = x.groupby(['arm', 'stream']).host_macro_f1.mean().unstack() * 100
lines = ['# Policy comparison and feedback robustness: results', '',
'These controls and robustness analyses support the existing RQ1 and RQ2 within Experiments 1 and 2. They do not introduce a new research question or alter Experiment 3.', '',
'## Evidence and validation', '',
'All 1,064 unique research executions completed: 224 validation runs, 420 policy-comparison runs and 420 additional sensitivity runs. The sensitivity table contains 504 entries because 84 default-setting controls are shared with the policy comparison. The 24 runtime-pilot runs are excluded. All 1,064 executions passed the separate trace audit; the 42 behavioural tests passed, and the parent repository retained-file check passed for 3,878 files.', '',
'The policy comparison uses 42 conditions: two synthetic families, three evaluation stream seeds, and seven clean/poisoning conditions. Each realization contains 60,000 observations. The feedback analysis uses clean and severe label poisoning on the same six realizations. These scores belong to this declared grid; they are not pooled with the longer original benchmark.', '',
'## Experiment 2: learned reset policies and same-information controls', '',
'| Policy | SEA macro-F1 (%) | Difference from confirmed (pp) | RBF macro-F1 (%) | Difference from confirmed (pp) |',
'| --- | ---: | ---: | ---: | ---: |']
for arm, label in zip(arms, labels):
    sea, rbf = means.loc[arm, 'SEA_A'], means.loc[arm, 'RBF_I']
    lines.append(f'| {label} | {sea:.2f} | {sea-means.loc["confirmed", "SEA_A"]:+.2f} | {rbf:.2f} | {rbf-means.loc["confirmed", "RBF_I"]:+.2f} |')
lines += ['', 'Values average equally across the seven conditions and three stream seeds within each family. Controller seeds remain separate rows. Percentage-point differences are calculated before rounding.', '',
'The selected deterministic mapping improved mean macro-F1 relative to confirmed reset by 0.71 percentage points on SEA and reduced it by 0.21 points on RBF. Its per-stream-seed mean differences ranged from +0.56 to +0.91 points on SEA and from -0.57 to +0.47 points on RBF. Thus the mean improvement was not consistent across both stream families.', '',
'No learned controller exceeded the selected mapping on the SEA family mean. On RBF, the two-state seed-17 controller had the highest learned mean, but its frozen policy always continued learning and produced exactly the same prediction traces as screened no reset. Two-state seeds 7 and 27 both always proposed reset and also produced identical prediction traces. These are repeated effective policies, not three distinct learned strategies.', '',
'The validation-selected mapping is `1100` in state order `00, 01, 10, 11`, where each state is `(alarm, elevated error)` and action 1 proposes reset. It therefore proposes reset without an alarm and continues learning with an alarm; it ignores the error bit. This is a result of the declared validation search and shared screening gate, not a general recommendation to reset when there is no alarm. The full sixteen-policy ranking is retained.', '',
'For RQ2, the evidence supports a conditional answer: the tested Q-learning policies do not provide a consistent advantage over fixed screened responses and simple mappings with the same information. Small differences between the better SEA policies should not be interpreted as statistically established superiority.', '',
'## Experiments 1 and 2: sensitivity to protected feedback', '',
'The following table shows the screened confirmed-reset reference. Macro-F1 and legitimate withholding average over clean and severely label-poisoned conditions across three stream seeds. Poison admission averages only over the three poisoned conditions, because it is undefined for clean data.', '',
'| Family | Usable feedback (%) | Delay (blocks) | Macro-F1 (%) | Poison admitted (%) | Legitimate withheld (%) |',
'| --- | ---: | ---: | ---: | ---: | ---: |']
for (stream, fraction, delay), group in s[s.arm.eq('confirmed')].groupby(['stream', 'feedback_fraction', 'delay_blocks']):
    lines.append(f'| {stream} | {fraction*100:.0f} | {delay} | {group.host_macro_f1.mean()*100:.2f} | {group.poison_admission_rate.mean()*100:.2f} | {group.clean_withhold_rate.mean()*100:.2f} |')
lines += ['',
'At one-block delay, reducing usable feedback from 5% to 1% increased mean poison admission for confirmed reset from 74.18% to 95.62% on SEA and from 70.61% to 81.73% on RBF. Lower feedback did not uniformly lower prediction scores: the corresponding RBF macro-F1 mean increased while its poisoning exposure also increased. This supports RQ1\'s joint assessment of exposure, legitimate learning and prediction rather than treating any one metric as sufficient.', '',
'At 5% feedback, extending delay from one to five blocks reduced confirmed-reset SEA macro-F1 by about 0.91 points and increased legitimate withholding in both families. Delay has no monotonic benefit or cost across every policy and condition; all paired setting effects are retained. The five-block setting also rejects updates during four post-warm-up blocks before the first protected sample arrives.', '',
'All settings retain the same 5% reservation. The unused protected rows are not returned to training or scoring. These results isolate available feedback under a fixed reservation and do not estimate the effect of reserving only 1% of a deployment stream. Ordinary labels retain their existing timing and poisoning semantics.', '',
'## Interpretation boundaries', '',
'- There are three evaluation stream seeds per synthetic family. Ranges and standard deviations are descriptive; rows, controller seeds and reused controls are not independent datasets.',
'- The deterministic mapping was selected on one separate validation realization per family. Its ranking may depend on those realizations.',
'- Mapping selection maximises validation macro-F1; the existing Q-learning reward is delayed protected-sample accuracy. The comparison assesses the implemented procedures and does not isolate optimisation algorithm alone or establish Q-learning convergence/failure.',
'- Confirmed reset retains a buffered refit; learned and enumerated policies reset from the current block. The same-state mapping comparisons control this distinction, while comparisons against confirmed reset include it.',
'- No new real-world dataset, noisy protected-feedback condition, adaptive attacker or deep-RL method is evaluated.', '',
'The policy figure shows individual stream-seed means and a black tick for their overall mean. In the feedback figure, overlapping lines indicate identical aggregate outcomes, not missing arms.', '',
'## Artifacts', '',
'- [Policy figure](../results/policy_robustness/analysis/figures/policy_comparison.png)',
'- [Feedback figure](../results/policy_robustness/analysis/figures/feedback_robustness.png)',
'- [Policy paired effects](../results/policy_robustness/analysis/evaluation/paired_vs_confirmed.csv)',
'- [Feedback paired effects](../results/policy_robustness/analysis/sensitivity/paired_feedback_effects.csv)',
'- [Validation ranking](../results/policy_robustness/analysis/validation/policy_ranking.csv)',
'- [Trace verification](../results/policy_robustness/analysis/verification/verification.json)',
'- [Protocol](../docs/PROTOCOL.md)', '']
(ROOT / '07_documentation/RESULTS.md').write_text('\n'.join(lines))
# Keep stream-seed means available alongside the aggregate presentation.
x.groupby(['stream', 'base_seed', 'arm']).host_macro_f1.mean().reset_index().to_csv(
    OUT / 'analysis/evaluation/stream_seed_macro_f1.csv', index=False)
print('Wrote integrated results and per-stream-seed summary.')
