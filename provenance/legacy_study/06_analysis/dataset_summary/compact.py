"""Shared four-panel PDF layout for the sequential experiment comparisons."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, ScalarFormatter

WIDTH_MM, HEIGHT_MM = 170, 128
LABELS = ['Clean', 'Instance 15%', 'Instance 25%', 'Concept 15%', 'Concept 25%', 'Splice 15%', 'Splice 25%']
COLORS = ['#0072B2', '#D55E00', '#009E73', '#CC79A7']
MARKERS = ['o', 's', '^', 'D']


def draw(data, specs, title, subtitle, series, destination, note, styles=None, limits=None):
    """Draw four wide panels with readable labels and a shared detector legend."""
    if len(specs) != 4:
        raise ValueError('Exactly four metrics are required')
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'pdf.fonttype': 42,
                         'text.color':'#243447','axes.labelcolor':'#243447',
                         'xtick.color':'#475569','ytick.color':'#475569'})
    fig, axes = plt.subplots(2, 2, figsize=(WIDTH_MM/25.4, HEIGHT_MM/25.4))
    fig.subplots_adjust(left=.085, right=.982, top=.80 if len(series)>4 else .87 if len(series)>1 else .94,
                        bottom=.18, wspace=.28, hspace=.96)
    handles = {}
    for ax, (metric, label, scale) in zip(axes.flat, specs):
        finite_values = []
        available_categories=set()
        for index, (key, name) in enumerate(series.items()):
            style=(styles or {}).get(key,{})
            part = data[data.metric.eq(metric) & data.series.eq(key)].sort_values('category')
            values, errors = part['mean'].to_numpy()*scale, part.sd.to_numpy()*scale
            errors = np.nan_to_num(errors, nan=0)
            if not np.isfinite(values).any():
                continue
            handle = ax.errorbar(part.category, values, yerr=errors, color=style.get('color',COLORS[index % len(COLORS)]),
                                marker=style.get('marker',MARKERS[index % len(MARKERS)]), markersize=3.8, linewidth=1.35,
                                linestyle=style.get('linestyle',['-','--','-.',':'][index % 4]),
                                markerfacecolor=style.get('markerfacecolor',style.get('color',COLORS[index % len(COLORS)])),
                                elinewidth=.8, capsize=2, label=name, markeredgewidth=.8)
            handles[key] = handle
            available_categories.update(part.loc[np.isfinite(values),'category'])
            finite_values.extend((values-errors)[np.isfinite(values)])
            finite_values.extend((values+errors)[np.isfinite(values)])
            if len(series)==1 and np.isfinite(values[0]):
                ax.axhline(values[0], color='#94a3b8', linestyle='--', linewidth=.8,zorder=0)
        if finite_values:
            low, high = min(finite_values), max(finite_values)
            pad = max((high-low)*.18, .15 if scale==100 else .0005)
            low, high = max(0, low-pad), high+pad
            if scale==100:
                high = min(100, high)
                if high==100:
                    high=102
                if low==0:
                    low=-2
            elif metric in ['roc_auc', 'average_precision']:
                high = min(1, high)
                if high==1:
                    high=1.02
            if high <= low:
                low, high = 0, max(1, high)
            ax.set_ylim(low, high)
            if limits and metric in limits:
                ax.set_ylim(*limits[metric])
            for category in set(range(7))-available_categories:
                ax.text(category,.07,'N/A',ha='center',va='bottom',rotation=90,
                        transform=ax.get_xaxis_transform(),fontsize=7,color='#666666')
        else:
            ax.text(.5,.5,'Not available', ha='center', va='center',
                    transform=ax.transAxes, fontsize=8.5, color='#555555')
            ax.set_yticks([])
        ax.set_title(label, fontsize=10, pad=8,color='#243447')
        ax.set_xticks(range(7), LABELS, rotation=38, ha='right')
        ax.set_xlim(-.35,6.35)
        ax.tick_params(labelsize=8.5, pad=3, length=2.5, width=.6)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
        formatter = ScalarFormatter(useOffset=False)
        formatter.set_scientific(False)
        ax.yaxis.set_major_formatter(formatter)
        if not finite_values:
            ax.set_yticks([])
        ax.spines[['top','right']].set_visible(False)
        for spine in ['left','bottom']:
            ax.spines[spine].set_color('#a5b0bf')
            ax.spines[spine].set_linewidth(.65)
        ax.set_facecolor('#fcfdff')
        ax.grid(axis='y',color='#d8e0e9',alpha=.8,linewidth=.5)
        # Light attack-family bands separate categorical comparisons from a time axis.
        for left in [.5,4.5]:
            ax.axvspan(left,left+2,color='#edf2f7',alpha=.65,zorder=-2)
    if len(series)>1:
        fig.legend([handles[k] for k in series if k in handles],
                   [series[k] for k in series if k in handles],loc='upper center',
                   bbox_to_anchor=(.5,.995),ncol=3 if len(series)>4 else 4,
                   frameon=False,fontsize=8.5 if len(series)>4 else 9.5,handlelength=2,
                   columnspacing=1.2)
    destination=Path(destination)
    fig.savefig(destination.with_suffix('.pdf'))
    fig.savefig(destination.with_suffix('.png'),dpi=180)
    plt.close(fig)
    return {'pdf':destination.with_suffix('.pdf').name,'png':destination.with_suffix('.png').name,
            'plots':4,'layout':[2,2],'width_mm':WIDTH_MM,'height_mm':HEIGHT_MM,'metrics':[s[0] for s in specs]}
