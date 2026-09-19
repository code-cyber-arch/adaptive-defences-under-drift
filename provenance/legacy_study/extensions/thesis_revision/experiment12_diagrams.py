"""Draw the executed Experiment 1 and 2 pipelines without regenerating result figures."""
from pathlib import Path
import hashlib,json,os
ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/'results/examiner_followthrough/figures'
OUT.mkdir(parents=True,exist_ok=True); os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'results/.matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42})
BLUE='#294D70'; GREEN='#2B7850'; GREY='#485465'; ORANGE='#AB6A12'
def canvas(h=4.1):
 f,a=plt.subplots(figsize=(6.02,h)); a.set_xlim(0,100);a.set_ylim(0,100);a.axis('off');f.subplots_adjust(left=.01,right=.99,top=.99,bottom=.01);return f,a
def box(a,x,y,w,h,t,fill='#edf3f9',edge=BLUE,size=8.4):
 a.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.3,rounding_size=1.4',facecolor=fill,edgecolor=edge,lw=1.2)); a.text(x+w/2,y+h/2,t,ha='center',va='center',fontsize=size,linespacing=1.25)
def arrow(a,x1,y1,x2,y2,color=BLUE,dashed=False):
 a.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle='-|>',mutation_scale=10,color=color,lw=1.2,linestyle='--' if dashed else '-'))
def save(f,name,sources):
 p=OUT/(name+'.pdf');f.savefig(p,bbox_inches='tight',pad_inches=.06,metadata={'CreationDate':None,'ModDate':None});f.savefig(OUT/(name+'.png'),dpi=190,bbox_inches='tight',pad_inches=.06);plt.close(f)
 return {'output':p.name,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'source_files':{s:hashlib.sha256((ROOT/s).read_bytes()).hexdigest() for s in sources},'reference':'experiment_report.pdf: box-and-arrow style only; current thesis grouping retained'}
records=[]
f,a=canvas(4.0)
box(a,2,83,36,15,'Data stream\nSEA / RBF / RADAR',size=9)
box(a,48,83,50,15,'Clean or six poisoned conditions\nLabel / concept / splice; 15% or 25%',fill='#fff2df',edge=ORANGE,size=8.8)
arrow(a,20,83,20,70);a.plot([72,72],[83,76],color=ORANGE,lw=1.1);arrow(a,72,76,20,76,ORANGE)
for x,w,t in [(2,23,'Predict before\nlearning'),(29,21,'Drift monitor\n(when used)'),(54,21,'Fixed response\nNo / immediate /\nconfirmed reset'),(79,19,'Train candidate\nlearner')]:box(a,x,51,w,18,t)
for u,v in [(25,29),(50,54),(75,79)]:arrow(a,u,60,v,60)
a.text(50,45,'Reset arms: ADWIN, HDDM-W, Hellinger or D3-inspired',ha='center',fontsize=8.1)
a.text(50,39,'No-reset control: continue learning without a response monitor',ha='center',fontsize=8.1)
# Candidate enters one of two separately evaluated screening settings.
a.plot([98.6,99.4,99.4,50],[59,59,34,34],color=BLUE,lw=1.0)
a.plot([25,75],[34,34],color=BLUE,lw=1.0);arrow(a,25,34,25,29);arrow(a,75,34,75,29)
box(a,2,13,46,16,'Unscreened\nAccept the candidate directly',fill='#f4f5f6',edge=GREY,size=8.7)
box(a,52,13,46,16,'Screened: protected validation\nAccept if accuracy does not fall;\notherwise retain the active model',fill='#eaf5ed',edge=GREEN,size=8.3)
a.text(50,5,'Compare matched prediction, poison admission and legitimate withholding.',ha='center',fontsize=8.2)
records.append(save(f,'experiment1_screening',['05_experiment/engine.py','03_policies/gate.py','02_detectors/monitors.py']))
f,a=canvas(4.45)
a.text(2,97,'Develop and freeze the reset policy',fontsize=9.3,weight='bold')
box(a,2,75,29,17,'Fit Q-tables\nSEA/RBF seed 112\nDelayed accuracy reward',fill='#f4f5f6',edge=GREY,size=8.3)
box(a,36,75,28,17,'Validate frozen tables\nSEA/RBF seed 113\nNo Q-table updates',fill='#f4f5f6',edge=GREY,size=8.3)
box(a,69,75,29,17,'Benchmark evaluation\nSEA / RBF / RADAR\nQ-tables remain frozen',fill='#f4f5f6',edge=GREY,size=8.3)
arrow(a,31,83.5,36,83.5);arrow(a,64,83.5,69,83.5)
a.text(2,66,'At each evaluation block',fontsize=9.3,weight='bold')
labels=[(2,22,'Predict, then ADWIN\nAlarm only, or alarm\n+ recent error change'),(28,21,'Frozen Q-table\nChoose continued\nlearning or reset'),(53,19,'Train candidate\nUpdate current tree\nor fit a fresh tree'),(76,22,'Protected gate\nCompare candidate\nand active accuracy')]
for x,w,t in labels:box(a,x,41,w,19,t,fill='#eaf5ed' if x==76 else '#edf3f9',edge=GREEN if x==76 else BLUE,size=8.0)
for u,v in [(24,28),(49,53),(72,76)]:arrow(a,u,50.5,v,50.5)
a.plot([82,82,45],[41,35,35],color=GREEN,lw=1.1);arrow(a,45,35,45,31,GREEN)
box(a,2,17,66,14,'Accept candidate or retain active model\nUse the resulting learner for the next block',fill='#eaf5ed',edge=GREEN,size=8.4)
box(a,74,17,24,14,'Protected feedback\n5% reserved\nOne-block delay',fill='#eaf5ed',edge=GREEN,size=8.2)
arrow(a,93,31,93,41,GREEN)
a.text(50,8,'Compare with screened confirmed-reset ADWIN and screened no reset.',ha='center',fontsize=8.1)
a.text(50,2,'The same learner and validation gate are used in the matched arms.',ha='center',fontsize=8.1)
records.append(save(f,'experiment2_controller',['04_rl_training/train.py','04_rl_training/controller.py','05_experiment/engine.py','03_policies/gate.py']))
(OUT/'manifest.json').write_text(json.dumps({'figures':records,'builder_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},indent=2)+'\n')
print('Created two vector methodology diagrams.')
