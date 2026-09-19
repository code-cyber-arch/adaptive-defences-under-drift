"""Render annotated detector-performance heatmaps from verified comparison tables."""
from pathlib import Path
import sys
sys.dont_write_bytecode=True
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.ticker import FixedLocator, FormatStrFormatter
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from common import protocol as p
comparison=p.load_module('heatmap_comparison','06_analysis/dataset_summary/compare_detectors.py')
fixed=comparison.fixed
WIDTH_MM,HEIGHT_MM=170,128


def colour_limits(data):
    """Keep each metric's colour range fixed across response settings in a dataset."""
    limits={}
    for metric,_,scale in fixed.PREDICTIVE:
        values=data.loc[data.metric.eq(metric),'mean'].to_numpy()*scale
        if not np.isfinite(values).all():
            raise ValueError('A heatmap contains unsupported values')
        step=1 if scale==100 else .01
        lower=np.floor(values.min()/step)*step
        upper=np.ceil(values.max()/step)*step
        if upper<=lower:
            lower=max(0,lower-step)
            upper=min(100 if scale==100 else 1,upper+step)
        limits[metric]=[float(lower),float(upper)]
    return limits


def foreground(rgba):
    """Choose dark or white annotation text against the actual cell colour."""
    rgb=np.asarray(rgba[:3])
    linear=np.where(rgb<=.04045,rgb/12.92,((rgb+.055)/1.055)**2.4)
    luminance=float(linear@np.array([.2126,.7152,.0722]))
    return 'white' if luminance<.30 else '#172b3a'


def draw(data,dataset,profile,output,limits):
    """Place four labelled 4-by-7 matrices and their colour keys in one compact PDF."""
    plt.rcParams.update({'font.family':'DejaVu Sans','pdf.fonttype':42,'text.color':'#243447'})
    fig=plt.figure(figsize=(WIDTH_MM/25.4,HEIGHT_MM/25.4))
    cmap=plt.get_cmap('YlGnBu')
    row_names=list(fixed.DETECTORS)
    cells=[]
    for index,(metric,title,scale) in enumerate(fixed.PREDICTIVE):
        left=.12+.50*(index%2)
        bottom=.70 if index<2 else .24
        ax=fig.add_axes([left,bottom,.36,.23])
        table=data[data.metric.eq(metric)].pivot(index='series',columns='category',values='mean')
        table=table.reindex(index=row_names,columns=range(7))
        values=table.to_numpy()*scale
        if values.shape!=(4,7) or not np.isfinite(values).all():
            raise ValueError('Every panel must contain all 28 supported scores')
        norm=Normalize(*limits[metric])
        mesh=ax.pcolormesh(np.arange(8),np.arange(5),values,cmap=cmap,norm=norm,
                           edgecolors='white',linewidth=.7,rasterized=False)
        for row,detector in enumerate(row_names):
            for col in range(7):
                value=values[row,col]
                label=f'{value:.1f}' if scale==100 else f'{value:.3f}'.removeprefix('0')
                ax.text(col+.5,row+.5,label,ha='center',va='center',fontsize=8.2,
                        color=foreground(cmap(norm(value))))
                cells.append({'dataset':dataset,'profile':profile,'metric':metric,
                              'series':detector,'category':col,'display_value':float(value),'annotation':label})
        ax.set_xlim(0,7)
        ax.set_ylim(4,0)
        ax.set_xticks(np.arange(7)+.5,fixed.layout.LABELS,rotation=38,ha='right')
        ax.set_yticks(np.arange(4)+.5,list(fixed.DETECTORS.values()))
        ax.tick_params(axis='both',length=0,labelsize=8.5,pad=3,colors='#334155')
        ax.set_title(title,fontsize=10,pad=9,color='#243447')
        for spine in ax.spines.values():
            spine.set_visible(False)
        # A horizontal colour key leaves room for full detector and condition names.
        bar_ax=fig.add_axes([left+.045,bottom-.14,.27,.015])
        bar=fig.colorbar(mesh,cax=bar_ax,orientation='horizontal')
        low,high=limits[metric]
        bar.locator=FixedLocator([low,(low+high)/2,high])
        bar.formatter=FormatStrFormatter('%.1f' if scale==100 else '%.3f')
        bar.update_ticks()
        bar.ax.tick_params(labelsize=7.5,length=2,pad=2,colors='#475569')
        bar.outline.set_visible(False)
    folder=output/dataset
    previews=output/'previews'/dataset
    folder.mkdir(exist_ok=True)
    previews.mkdir(parents=True,exist_ok=True)
    name=f'{dataset}_{profile}_heatmap'
    fig.savefig(folder/(name+'.pdf'))
    fig.savefig(previews/(name+'.png'),dpi=200)
    plt.close(fig)
    return {'dataset':dataset,'profile':profile,'pdf':f'{dataset}/{name}.pdf',
            'png':f'previews/{dataset}/{name}.png','plots':4,'layout':[2,2],'cells':112,
            'width_mm':WIDTH_MM,'height_mm':HEIGHT_MM,'colour_limits':limits,
            'metrics':[s[0] for s in fixed.PREDICTIVE],'detectors':row_names},cells


def document(output):
    """Keep explanation and uncertainty guidance outside the requested figure area."""
    text='''# Detector-performance heatmaps

Each PDF shows 112 mean scores: four metric panels x four detectors x seven conditions. Read down a column to compare detectors under the same condition; read across a row to see how one detector changes across attack types and severities. Accuracy, macro-F1, macro recall and ROC-AUC are shown. Darker blue means a higher score. Values are printed inside every cell.

Each figure fixes one dataset, response rule and screening setting. Start with immediate reset without screening, then confirmed reset, followed by the screened versions. The figures compare the classifier performance obtained with each detector and response; they are not measurements of drift-detection accuracy itself. All four settings are included without choosing only favourable conditions.

Colour ranges are identical across the four settings within each dataset and metric. Each panel has its own labelled colour bar. Ranges may differ between datasets or metrics and do not necessarily begin at zero. Use the printed numbers and bars when comparing figures. Percentage scores are shown to one decimal place; ROC-AUC is shown to three decimals, omitting the leading zero inside cells (for example, .804 means 0.804). Display rounding does not change the underlying scores. The selected colour map progresses from light yellow through green to dark blue, with contrasting cell text.

The means are copied exactly from the verified detector-comparison tables. No experiment was rerun and no conditions were pooled. All predictive comparisons use the existing common host-row masks, excluding warm-up, protected reservations and concept/splice replacement intervals across all seven conditions. The original full streams, including all 484,753 RADAR rows, remain unchanged.

Synthetic means use three stream realizations. RADAR means use three matched attack assignments on one capture; its clean predictions are reused on the corresponding masks. The heatmaps show means only. Standard deviations, defined counts and unrounded values remain in plot_values.csv. Use the matching detector_comparison line plots to inspect error bars. Neither colour differences nor rounded differences establish statistical significance.

Synthetic ROC-AUC is macro one-versus-rest; RADAR ROC-AUC contrasts malicious with benign scores. No observation-level poison-identification metric is implied. Higher predictive scores must still be considered alongside poisoning admission, legitimate-data withholding and computational cost in the companion fixed-response figures.

All PDFs are individual, vector figures in a 170 x 128 mm, 2 x 2 layout. They have metric names, direct row/column labels, cell values and colour bars, with no surrounding heading or explanatory footer. Dataset folders contain PDFs; PNG previews are separate. No HTML or combined PDF is produced.

Reproduce with `python -B 06_analysis/dataset_summary/detector_heatmaps.py`. plot_values.csv preserves the source mean, SD and sample count. displayed_cells.csv records every displayed number and its dataset, setting, metric, detector and condition. manifest.json fingerprints the source tables, reporting code and outputs.

| Response setting | SEA | RBF | RADAR |
|---|---|---|---|
'''
    for profile,label in comparison.PROFILES.items():
        links=[f'[{dataset} PDF]({dataset}/{dataset}_{profile}_heatmap.pdf)' for dataset in fixed.DATASETS.values()]
        text+='| '+label+' | '+' | '.join(links)+' |\n'
    text+='\n[Matching line plots and uncertainty](../detector_comparison/README.md)\n'
    (output/'README.md').write_text(text,encoding='utf-8')


def main():
    """Verify the source comparison and generate all twelve annotated heatmaps."""
    root=ROOT/'results/evaluation'
    if p.sources()!=p.read(root/'request.json')['sources'] or p.read(root/'verification.json')['status']!='passed':
        raise ValueError('An unchanged verified benchmark is required')
    source=root/'06_analysis/presentation/detector_comparison'
    manifest=p.read(source/'manifest.json')
    source_file=source/'plot_values.csv'
    if p.sha(source_file)!=manifest['outputs']['plot_values.csv']:
        raise ValueError('Source comparison values changed')
    for file,digest in manifest['inputs'].items():
        if p.sha(ROOT/file)!=digest:
            raise ValueError('Upstream comparison input changed')
    points=pd.read_csv(source_file)
    if len(points)!=1344 or not points.defined.eq(3).all():
        raise ValueError('Expected all supported detector comparisons')
    output=root/'06_analysis/presentation/detector_heatmaps'
    output.mkdir(parents=True,exist_ok=True)
    points.to_csv(output/'plot_values.csv',index=False)
    entries,cells=[],[]
    for dataset in fixed.DATASETS.values():
        part=points[points.dataset.eq(dataset)]
        limits=colour_limits(part)
        for profile in comparison.PROFILES:
            entry,annotations=draw(part[part.profile.eq(profile)],dataset,profile,output,limits)
            entries.append(entry)
            cells.extend(annotations)
    pd.DataFrame(cells).to_csv(output/'displayed_cells.csv',index=False)
    document(output)
    scripts=['detector_heatmaps.py','compare_detectors.py','fixed_responses.py','compact.py']
    p.write(output/'manifest.json',{'created_utc':p.now(),'script_sha256':p.sha(__file__),
            'reporting_sources':{name:p.sha(ROOT/'06_analysis/dataset_summary'/name) for name in scripts},
            'request_sha256':p.sha(root/'request.json'),
            'inputs':{source_file.relative_to(ROOT).as_posix():p.sha(source_file),
                      (source/'manifest.json').relative_to(ROOT).as_posix():p.sha(source/'manifest.json')},
            'entries':entries,'outputs':{f.relative_to(output).as_posix():p.sha(f) for f in output.rglob('*')
            if f.is_file() and f.name!='manifest.json'}})
    print('Created 12 heatmap PDFs with 112 annotated scores per figure.',flush=True)


if __name__=='__main__':
    main()
