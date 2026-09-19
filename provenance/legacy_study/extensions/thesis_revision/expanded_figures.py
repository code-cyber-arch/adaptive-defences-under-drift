"""Thesis-sized vector diagrams and evidence-bound evaluation figures."""
from pathlib import Path
import hashlib
import json
import os
import shutil
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/methods_revision/figures'
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'results/.matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

DATASETS=['SEA','RBF','RADAR']
STREAMS=['SEA_A','RBF_I','radar']
DETECTORS=['adwin','hddm_w','hellinger','d3_oof']
MONITORS=['ADWIN','HDDM-W','Hellinger','D3-inspired']
CONDITIONS=['Clean','Label\n15%','Label\n25%','Concept\n15%','Concept\n25%','Splice\n15%','Splice\n25%']
manifest={}
plt.rcParams.update({'font.size':9,'axes.titlesize':10,'axes.labelsize':9,'xtick.labelsize':8.5,
                     'ytick.labelsize':8.5,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(rel):return pd.read_csv(ROOT/rel,float_precision='round_trip')
def save(fig,name,sources=(),points=None):
    OUT.mkdir(parents=True,exist_ok=True)
    path=OUT/(name+'.pdf')
    fig.savefig(path,bbox_inches='tight',pad_inches=.06,metadata={'CreationDate':None,'ModDate':None})
    fig.savefig(OUT/(name+'.png'),dpi=180,bbox_inches='tight',pad_inches=.06)
    manifest[path.name]={'sources':{s:sha(ROOT/s) for s in sources},'output_sha256':sha(path),
                         'plotted_values':points or [],'builder_sha256':sha(Path(__file__))}
    plt.close(fig)


def canvas(height):
    fig,ax=plt.subplots(figsize=(6.02,height))
    ax.set_xlim(0,100);ax.set_ylim(-2,100);ax.axis('off')
    fig.subplots_adjust(left=0,right=1,top=1,bottom=0)
    return fig,ax


def box(ax,x,y,w,h,text,color='#eaf1f8',edge='#34536f',size=9):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.4,rounding_size=1.3',
                              linewidth=1.15,edgecolor=edge,facecolor=color))
    ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=size,linespacing=1.25)


def arrow(ax,a,b,color='#34536f',dashed=False):
    ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=10,color=color,lw=1.15,
                               linestyle='--' if dashed else '-'))


def diagrams():
    fig,ax=canvas(4.1)
    box(ax,3,82,94,15,'Shared inputs: SEA, RBF and RADAR\nOne clean + six poisoned conditions; same learner and reserved rows')
    steps=[(59,'Experiment 1: fixed responses','Compare no reset, immediate reset and confirmed reset.\nRepeat each comparison with and without update screening.'),
           (35,'Experiment 2: learned reset control','Replace the fixed rule with two frozen Q-learning policies.\nKeep ADWIN and protected screening unchanged.'),
           (11,'Experiment 3: observation filtering','Filter before monitoring and training; retain validation.\nCompare filters, monitors and matched no-reset controls.')]
    for y,title,body in steps:
        arrow(ax,(50,y+23),(50,y+19))
        box(ax,3,y,94,19,title+'\n'+body,color='#f4f6f8',size=9)
    ax.text(50,2,'Evaluate prediction, poison admission, legitimate withholding and response activity.',ha='center',fontsize=8.5)
    save(fig,'research_overview',sources=['extensions/filter_study/run.py','extensions/filter_no_reset/run.py','common/protocol.py'])

    fig,ax=canvas(4.25)
    box(ax,3,81,94,16,'Attacker access: offline labelled source archive, including future rows\nFixed interventions; no model queries or detector-state access',color='#fff0dc',edge='#a76817')
    attack_rows=[(60,'Label poisoning','(x, y)  →  (x, altered label)','Features remain unchanged.'),
                 (40,'Concept replacement','(x, y)  →  (generated x, generated y)','250-row intervals; marginal sampling and a projection rule.'),
                 (20,'Splice copying','(x, y)  →  (copied x, copied y)','250-row intervals copied from another source segment.')]
    for y,name,example,note in attack_rows:
        box(ax,3,y,27,15,name,color='#fff8eb',edge='#a76817',size=9)
        arrow(ax,(31,y+7.5),(36,y+7.5),'#a76817')
        ax.text(39,y+10,example,fontsize=9,va='center')
        ax.text(39,y+4,note,fontsize=8,va='center')
    box(ax,3,0,94,13,'Protected boundary: original validation rows and evaluation truth\nThe attacker cannot alter released protected feedback or scoring references.',color='#e9f4ed',edge='#33764b',size=9)
    save(fig,'threat_and_attacks',sources=['01_attacks/interventions.py','common/channels.py'])

    fig,ax=canvas(4.25)
    ax.text(2,96,'Time-ordered blocks: 1,000 rows; no temporal shuffling',fontsize=10,weight='bold')
    for x,label,color in [(3,'Block t - 1\n50 protected rows','#f3f5f7'),(36,'Block t\n950 non-protected rows','#eaf1f8'),(69,'Block t + 1\nnext predictions','#f3f5f7')]:
        box(ax,x,77,28,14,label,color=color,size=9)
    arrow(ax,(31,84),(35,84));arrow(ax,(64,84),(68,84))
    ax.text(50,70,'At block t, only protected feedback from t - 1 is released.',ha='center',fontsize=9,color='#33764b')
    labels=['Predict\ncurrent rows','Optional\nrow filter','Monitor +\nresponse rule','Train a\ncandidate','Protected\nvalidation']
    for i,label in enumerate(labels):
        x=2+i*20
        box(ax,x,47,16,15,label,color='#e9f4ed' if i==4 else '#eaf1f8',size=8.5)
        if i<4:arrow(ax,(x+16,54.5),(x+19.5,54.5))
    ax.text(50,39,'Accept candidate if validation accuracy ≥ active accuracy; otherwise retain active model.',ha='center',fontsize=8.5)
    ax.text(2,29,'Confirmed reset: five prior errors → alarm → two following errors',fontsize=9,weight='bold')
    box(ax,3,9,26,13,'Alarm block t',size=9)
    box(ax,36,9,26,13,'Block t + 1',size=9)
    box(ax,69,9,26,13,'Block t + 2',size=9)
    arrow(ax,(29,15.5),(35,15.5));arrow(ax,(62,15.5),(68,15.5))
    ax.text(50,1,'If mean error rises by > 0.02, propose refitting on all three retained blocks.',ha='center',fontsize=8.5)
    save(fig,'temporal_blocks',sources=['05_experiment/engine.py','03_policies/gate.py','extensions/filter_study/engine.py'])

    fig,ax=canvas(4)
    box(ax,3,81,94,16,'Offline filter validation\nFit on development data → calibrate threshold → freeze → assess held-out rows',color='#f3f5f7')
    arrow(ax,(29,80),(29,64),dashed=True)
    labels=['Predict\nstream rows','Poisoning\nfilter','Drift monitor\n+ reset rule','Candidate\nlearning','Validation\ngate']
    for i,label in enumerate(labels):
        x=2+20*i
        box(ax,x,47,16,17,label,color='#e9f4ed' if i==4 else '#eaf1f8',size=8.5)
        if i<4:arrow(ax,(x+16,55.5),(x+19.5,55.5))
    box(ax,10,16,38,17,'Rejected rows\nExcluded from monitoring and training;\nretained in eligible prediction scoring.',color='#f3f5f7',size=8.5)
    arrow(ax,(30,46),(30,34),dashed=True)
    box(ax,57,16,40,17,'Protected validation sample\n5% reserved; released one block later\nNever used for training or final scoring.',color='#e9f4ed',edge='#33764b',size=8.5)
    arrow(ax,(90,34),(90,46),color='#33764b')
    ax.text(50,4,'Accept the candidate or retain the active model for the next block.',ha='center',fontsize=9)
    save(fig,'experiment3_validation',sources=['extensions/filter_study/engine.py','extensions/filter_study/separability.py'])


def heatmap_grid(name,arrays,rowlabels,titles,limit=None,absolute=False,sources=(),points=None):
    fig,axes=plt.subplots(len(arrays),1,figsize=(6.02,1.25*len(arrays)+.85),squeeze=False)
    flat=np.concatenate([np.asarray(x).ravel() for x in arrays])
    lim=limit or max(1,np.nanmax(np.abs(flat)))
    for ax,arr,title in zip(axes.flat,arrays,titles):
        arr=np.asarray(arr)
        if absolute:
            im=ax.imshow(arr,aspect='auto',cmap='YlGnBu',vmin=0,vmax=100)
        else:im=ax.imshow(arr,aspect='auto',cmap='RdBu',vmin=-lim,vmax=lim)
        ax.set_yticks(range(len(rowlabels)),rowlabels)
        ax.set_xticks(range(7),CONDITIONS)
        ax.tick_params(length=0)
        ax.set_title(title,loc='left',pad=5)
        for (i,j),v in np.ndenumerate(arr):
            text='NA' if not np.isfinite(v) else f'{v:.1f}' if absolute else '0.0' if round(float(v),1)==0 else f'{v:+.1f}'
            dark=(v>60 if absolute else abs(v)>lim*.62)
            ax.text(j,i,text,ha='center',va='center',fontsize=8.5,color='white' if dark else '#13202e')
        for spine in ax.spines.values():spine.set_visible(False)
    fig.subplots_adjust(left=.17,right=.86,bottom=.12,top=.93,hspace=.75)
    cax=fig.add_axes([.89,.24,.02,.5]);bar=fig.colorbar(im,cax=cax)
    bar.set_label('Macro-F1 (%)' if absolute else 'Macro-F1 change (pp)')
    save(fig,name,sources,points)


def evaluation_figures():
    # Preserve two original evaluation figures exactly, with terminology explained in captions.
    for name,source in [('evaluation_radar_baseline','01_RADAR_clean_vs_poisoned.pdf'),('evaluation_radar_monitors','02_RADAR_detector_baseline.pdf')]:
        rel='results/evaluation/06_analysis/presentation/experiment_figures/'+source
        OUT.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/rel,OUT/(name+'.pdf'))
        manifest[name+'.pdf']={'sources':{rel:sha(ROOT/rel)},'output_sha256':sha(OUT/(name+'.pdf')),'copied_without_data_changes':True}
    source='results/evaluation/06_analysis/presentation/detector_heatmaps/plot_values.csv'
    data=read(source);data=data[data.profile.eq('confirmed_screened') & data.metric.eq('macro_f1')]
    arrays=[100*data[data.dataset.eq(ds)].pivot(index='series',columns='category',values='mean').reindex(index=DETECTORS,columns=range(7)).to_numpy() for ds in DATASETS]
    heatmap_grid('evaluation_detector_heatmaps',arrays,MONITORS,DATASETS,absolute=True,sources=[source],points=data.to_dict('records'))
    source='results/methods_revision/fixed_no_reset_conditions.csv'
    data=read(source);data=data[data.detector.eq('adwin') & data.policy.eq('confirmed')]
    arrays=[data[data.dataset.eq(ds)].pivot(index='guarded',columns='category',values='macro_f1_difference').reindex(index=[False,True],columns=range(7)).to_numpy() for ds in DATASETS]
    heatmap_grid('fixed_reset_advantage',arrays,['Unscreened','Screened'],DATASETS,sources=[source],points=data.to_dict('records'))
    source='results/evaluation/06_analysis/presentation/rl_comparison/tables/improvement_values.csv'
    data=read(source);data=data[data.metric.eq('macro_f1') & data.reference.eq('confirmed')]
    arrays=[data[data.dataset.eq(ds)].pivot(index='series',columns='category',values='mean').reindex(index=['rl_alarm','rl'],columns=range(7)).to_numpy() for ds in DATASETS]
    heatmap_grid('evaluation_rl_heatmaps',arrays,['Alarm-only','Error-aware'],DATASETS,sources=[source],points=data.to_dict('records'))
    source='results/narrative/tables/filter_condition_differences.csv'
    data=read(source);data=data[data['filter'].eq('learned_features_label')].copy()
    cats={('clean','none'):0,('instance','moderate'):1,('instance','severe'):2,('concept','moderate'):3,('concept','severe'):4,('splice','moderate'):5,('splice','severe'):6}
    data['category']=[cats[(m,l)] for m,l in zip(data['mode'],data.level)]
    arrays=[100*data[data.stream.eq(s)].pivot(index='detector',columns='category',values='macro_f1_difference').reindex(index=DETECTORS,columns=range(7)).to_numpy() for s in STREAMS]
    heatmap_grid('filter_monitor_heatmaps',arrays,MONITORS,DATASETS,sources=[source],points=data.to_dict('records'))


def detector_activity():
    source='results/methods_revision/filter_monitor_overview.csv'
    data=read(source);data=data[data.population.eq('attacked')]
    fig,axes=plt.subplots(3,2,figsize=(6.02,5.8))
    labels=['No filter','Features','Features + label'];colors=['#8a939d','#77a7ca','#1a6492'];kinds=['none','learned_features','learned_features_label']
    for row,(stream,ds) in enumerate(zip(STREAMS,DATASETS)):
        for col,metric in enumerate(['alarm_rate','event_recall' if stream!='radar' else 'resets_per_100k']):
            ax=axes[row,col];x=np.arange(4)
            for i,(kind,label,color) in enumerate(zip(kinds,labels,colors)):
                values=data[data.stream.eq(stream)&data['filter'].eq(kind)].set_index('detector').reindex(DETECTORS)[metric]
                ax.bar(x+(i-1)*.24,values*(100 if metric!='resets_per_100k' else 1),width=.23,label=label,color=color)
            ax.set_xticks(x,['ADWIN','HDDM-W','Hell.','D3'],rotation=25,ha='right')
            ax.set_title(ds+' - '+{'alarm_rate':'alarm blocks (%)','event_recall':'event recall (%)','resets_per_100k':'resets / 100k rows'}[metric],loc='left')
            if metric!='resets_per_100k':ax.set_ylim(0,105)
            ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    fig.legend(*axes[0,0].get_legend_handles_labels(),loc='lower center',ncol=3,frameon=False,bbox_to_anchor=(.53,0))
    fig.subplots_adjust(left=.09,right=.99,top=.95,bottom=.12,hspace=.85,wspace=.35)
    save(fig,'filter_detector_activity',[source],data.to_dict('records'))


def no_reset_figures():
    source='results/filter_no_reset/condition_comparisons.csv'
    if not (ROOT/source).exists():return
    data=read(source);data=data[data.detector.eq('adwin')]
    arrays=[100*data[data.stream.eq(s)].pivot(index='filter',columns='category',values='macro_f1_difference').reindex(index=['none','learned_features','learned_features_label'],columns=range(7)).to_numpy() for s in STREAMS]
    heatmap_grid('filter_reset_advantage',arrays,['No filter','Features','Features + label'],DATASETS,sources=[source],points=data.to_dict('records'))


def response_activity():
    sources=['results/methods_revision/original_response_overview.csv','results/methods_revision/filter_monitor_overview.csv']
    original=read(sources[0]);filtered=read(sources[1])
    policies=['none','immediate','confirmed','rl_alarm','rl']
    kinds=['none','learned_features','learned_features_label']
    original=original[original.population.eq('attacked')&original.guarded&original.detector.isin(['none','adwin'])]
    filtered=filtered[filtered.population.eq('attacked')&filtered.detector.eq('adwin')]
    a=original.pivot(index='stream',columns='policy',values='resets_per_100k').reindex(index=STREAMS,columns=policies).to_numpy()
    b=filtered.pivot(index='stream',columns='filter',values='resets_per_100k').reindex(index=STREAMS,columns=kinds).to_numpy()
    maximum=max(1,float(np.nanmax(a)),float(np.nanmax(b)))
    fig,axes=plt.subplots(2,1,figsize=(6.02,3.9))
    for ax,values,labels,title in zip(axes,[a,b],[['No reset','Immediate','Confirmed','Alarm-only\nRL','Error-aware\nRL'],['No filter','Features','Features + label']],['Original benchmark: screened responses','Experiment 3: screened confirmed reset']):
        ax.imshow(values,aspect='auto',cmap='Blues',vmin=0,vmax=maximum)
        ax.set_xticks(range(len(labels)),labels);ax.set_yticks(range(3),DATASETS)
        ax.set_title(title,loc='left',fontsize=10)
        ax.tick_params(length=0)
        for (i,j),v in np.ndenumerate(values):
            ax.text(j,i,f'{v:.2f}',ha='center',va='center',color='white' if v>maximum*.6 else '#142637',fontsize=9)
        for sp in ax.spines.values():sp.set_visible(False)
    fig.subplots_adjust(left=.12,right=.98,top=.92,bottom=.1,hspace=.7)
    save(fig,'response_activity',sources,[*original.to_dict('records'),*filtered.to_dict('records')])


def main():
    diagrams();evaluation_figures();detector_activity();no_reset_figures();response_activity()
    def finite(value):
        if isinstance(value,dict):return {k:finite(v) for k,v in value.items()}
        if isinstance(value,list):return [finite(v) for v in value]
        if isinstance(value,float) and not np.isfinite(value):return None
        return value
    (OUT/'manifest.json').write_text(json.dumps(finite(manifest),indent=2,allow_nan=False)+'\n')
    print('Figures:',len(manifest))

if __name__=='__main__':main()
