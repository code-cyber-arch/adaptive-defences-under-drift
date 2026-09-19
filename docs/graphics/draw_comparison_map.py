"""Render the study's comparison-to-answer map; no empirical claims."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = Path(__file__).resolve().parent
plt.rcParams.update({'font.family':'DejaVu Sans', 'svg.fonttype':'none', 'text.color':'#243447'})
fig = plt.figure(figsize=(200/25.4, 255/25.4), facecolor='white')
ax = fig.add_axes([0,0,1,1]); ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis('off')
blue='#0072B2'; green='#009E73'; muted='#526477'
def text(x,y,s,size=8.4,weight='normal',color='#243447',va='top'):
    return ax.text(x,y,s,fontsize=size,fontweight=weight,color=color,va=va,linespacing=1.4)
def box(x,y,w,h,fc,ec='none'):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.006,rounding_size=0.009',facecolor=fc,edgecolor=ec,linewidth=.7))
text(.045,.967,'What each comparison tells us',17,'bold')
text(.045,.929,'Study design → matched comparison → evidence for RQ1 and RQ2',9.1,color=muted)
box(.045,.825,.91,.078,'#eaf3f9')
text(.062,.893,'SHARED EVALUATION DESIGN',8.2,'bold',blue)
text(.062,.871,'SEA · RBF · RADAR   |   7 clean/attack conditions   |   5 evaluation repetitions',8.4)
text(.062,.849,'ADWIN · HDDM-W · Hellinger · D3 OOF   |   5 fitting seeds for learned methods',8.4)
text(.055,.803,'COMPARE',8.4,'bold',blue)
text(.368,.803,'KEEP MATCHED',8.4,'bold',muted)
text(.659,.803,'THE ANSWER IT SUPPORTS',8.4,'bold',green)
rows=[
('Screened vs unscreened\nmodel updates', 'Dataset, detector,\nresponse policy, condition', 'Does screening reduce poisoning\nwithout losing useful learning?'),
('Four detectors\nAll six detector pairs', 'Dataset, policy,\ncondition', 'Which detector best supports\nthis defence, and at what cost?'),
('Reset policies\nFixed, rule-based and RL', 'Dataset, detector,\ncondition', 'Does learned reset control\nimprove on a fixed response?'),
('Observation filters\nLearned and control filters', 'Dataset, detector, response,\ncondition, scoring rows', 'Does filtering improve protection\nand predictive performance?'),
('SEA vs RBF vs RADAR\nAll three dataset pairs', 'Method, detector,\ncondition weighting', 'Are the observed benefits\nconsistent across these datasets?'),
('Across experiments\nFixed responses, RL, filtering', 'Dataset, detector, condition,\ncommon scoring rows', 'Which approach offers the best\nprotection–performance trade-off?'),
('Feedback amount and delay\n1%, 2%, 5%; 1 or 5 blocks', 'Dataset, detector, method,\ncondition', 'How much does the defence\ndepend on timely feedback?'),
('Passive detector alarms\nAll six detector pairs', 'Dataset, condition,\nunchanged learner predictions', 'How do detectors differ in\nalarm activity and drift response?'),
]
top=.779; step=.0665
for i,(compare,matched,answer) in enumerate(rows):
    y=top-i*step
    box(.045,y-.055,.91,.057,'#f1f6fa' if i%2==0 else '#fafbfd')
    text(.058,y-.006,compare,8.2,'bold')
    text(.368,y-.006,matched,7.9,color=muted)
    ax.add_patch(FancyArrowPatch((.624,y-.028),(.649,y-.028),arrowstyle='-|>',mutation_scale=9,color=green,linewidth=1))
    text(.659,y-.006,answer,8.05)
box(.045,.130,.91,.095,'#edf6f2')
text(.062,.214,'EVIDENCE USED TO ANSWER THE EXISTING RESEARCH QUESTIONS',8.1,'bold',green)
text(.062,.191,'RQ1  Screening: poisoning exposure, legitimate learning and predictive performance.',8.1)
text(.062,.169,'RQ2  Learned resets or upstream filters: improvement over a fixed screened response.',8.1)
text(.045,.110,'Read outcomes together: accuracy / macro-F1 · poisoning admitted · legitimate learning',7.6,color=muted)
text(.045,.092,'withheld · reset activity. Passive alarm measures are assessed separately.',7.6,color=muted)
text(.045,.068,'Equal condition and dataset weights. Shared RADAR clean and no-reset references are not',7.2,color=muted)
text(.045,.051,'independent repetitions. Dataset comparisons describe consistency, not proof of generalisation.',7.2,color=muted)
text(.045,.027,'Questions shown here describe the evidence sought; they do not assume a favourable result.',7.1,color=muted)
for ext in ('svg','png'):
    fig.savefig(OUT/f'comparison_to_answers.{ext}',dpi=220,facecolor='white')
plt.close(fig)
