"""Combined comparison map, 170 mm wide for A4; all marks remain vectors."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
OUT=Path(__file__).resolve().parent
plt.rcParams.update({'font.family':'DejaVu Sans','svg.fonttype':'none','pdf.fonttype':42,'text.color':'#243447','axes.labelcolor':'#243447','xtick.color':'#475569','ytick.color':'#475569'})
fig=plt.figure(figsize=(170/25.4,145/25.4),facecolor='white')
blue='#0072B2';green='#009E73';orange='#D55E00';purple='#9565A2'
def matrix(rect,labels,color,title,answer='',rows=None,full=False):
    ax=fig.add_axes(rect);rows=rows or labels;n=len(labels);m=len(rows)
    values=np.ones((m,n)) if full else 1-np.eye(n)
    ax.pcolormesh(np.arange(n+1)-.5,np.arange(m+1)-.5,values,cmap=ListedColormap(['#eef1f4',color]),vmin=0,vmax=1,edgecolors='white',linewidth=1.2)
    ax.set_xlim(-.5,n-.5);ax.set_ylim(m-.5,-.5);ax.set_aspect('equal')
    ax.set_xticks(range(n),labels,fontsize=8.5);ax.set_yticks(range(m),rows,fontsize=8.5)
    ax.tick_params(length=0,pad=3)
    for spine in ax.spines.values():spine.set_visible(False)
    for i in range(m):
        for j in range(n):
            if values[i,j]:ax.plot(j,i,'o',color='white',markersize=3.3,markeredgewidth=0)
    ax.set_title(title,fontsize=10,weight='bold',pad=9,color=color)
    if answer:
        x,y,w,h=rect
        fig.text(x+w/2,y-.08,answer,ha='center',va='top',fontsize=8.5,weight='bold',color=color)
    return ax
fig.text(.035,.962,'COVERAGE',fontsize=9,weight='bold',color='#64748b')
for i,(name,c) in enumerate([('Fixed responses',blue),('RL policies',green),('Filters',orange)]):
    matrix([.10+i*.30,.68,.22,.22],['A','H','He','D'],c,name,rows=['SEA','RBF','RADAR'],full=True)
fig.text(.5,.602,'A: ADWIN   H: HDDM-W   He: Hellinger   D: D3 OOF',ha='center',fontsize=8.5,color='#526477')
fig.text(.035,.532,'PAIRWISE COMPARISONS',fontsize=9,weight='bold',color='#64748b')
matrix([.055,.275,.16,.17],['A','H','He','D'],blue,'Detectors','Which\ndetector?')
matrix([.30,.275,.16,.17],['Fixed','Rule','RL'],green,'Policies','Which\nresponse?')
matrix([.545,.275,.16,.17],['SEA','RBF','RADAR'],orange,'Datasets','Consistent\nbenefits?')
matrix([.79,.275,.16,.17],['Fixed','RL','Filter'],purple,'Experiments','Which\napproach?')
# Preserve the original combined structure while spacing long labels at report size.
for ax in fig.axes[3:]:
    ax.tick_params(axis='x',labelrotation=35)
for x,label,c in [(.18,'PREDICTION',blue),(.50,'PROTECTION',green),(.82,'LEARNING COST',orange)]:
    fig.text(x,.062,label,ha='center',fontsize=9,weight='bold',color=c,bbox=dict(boxstyle='round,pad=.55',facecolor='#f1f6fa',edgecolor='none'))
for ext in ['pdf','svg','png']:
    fig.savefig(OUT/f'comparison_heatmap.{ext}',dpi=220,facecolor='white')
plt.close(fig)
