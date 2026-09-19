"""Create a report-ready overview of the four implemented evaluation stages."""
from pathlib import Path
import sys
sys.dont_write_bytecode=True
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,Circle
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from common import protocol as p


def box(ax,x,y,width,height,fill,border):
    """Draw a restrained rounded card in figure-relative coordinates."""
    patch=FancyBboxPatch((x,y),width,height,boxstyle='round,pad=0.007,rounding_size=0.013',
                        linewidth=.8,edgecolor=border,facecolor=fill)
    ax.add_patch(patch)


def stage(ax,number,title,x,y,colour,lines):
    """Label one study stage without implying that all stages share one model run."""
    box(ax,x,y,.452,.292,'#f8fafc','#d8e1eb')
    ax.add_patch(Circle((x+.032,y+.25),.021,facecolor=colour,edgecolor='none'))
    ax.text(x+.032,y+.25,str(number),ha='center',va='center',color='white',fontsize=9,fontweight='bold')
    ax.text(x+.068,y+.25,title,ha='left',va='center',fontsize=10,fontweight='bold',color='#21354b')
    for index,line in enumerate(lines):
        ax.text(x+.020,y+.192-index*.043,line,ha='left',va='center',fontsize=8.5,color='#334155')


def main():
    """Export one vector PDF, a preview and a matching LaTeX figure environment."""
    root=ROOT/'results/evaluation'
    request=p.read(root/'request.json')
    if p.sources()!=request['sources']:
        raise ValueError('Experiment code differs from the recorded execution')
    if p.read(root/'verification.json')['status']!='passed':
        raise ValueError('Completed benchmark verification is required')
    output=root/'06_analysis/presentation/report_figure'
    output.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','pdf.fonttype':42})
    fig=plt.figure(figsize=(170/25.4,128/25.4),facecolor='white')
    ax=fig.add_axes([0,0,1,1])
    ax.set(xlim=(0,1),ylim=(0,1))
    ax.axis('off')
    box(ax,.025,.855,.95,.117,'#edf3f9','#c7d6e4')
    ax.text(.5,.940,'MATCHED STREAMING EXPERIMENTS',ha='center',va='center',fontsize=9.5,
            fontweight='bold',color='#244764')
    ax.text(.5,.906,'SEA · RBF · RADAR  |  Clean + instance / concept / splice (nominal 15%, 25%)',
            ha='center',va='center',fontsize=8.5,color='#334155')
    ax.text(.5,.874,'Hoeffding Tree  |  Predict each block before learning its observed labels',
            ha='center',va='center',fontsize=8.5,color='#334155')
    stage(ax,1,'Attack damage',.025,.520,'#2874a6',[
        'No reset; unscreened learning',
        'Compare clean with all six poisoned conditions',
        'Measure accuracy, macro-F1 and macro recall',
        'Summarise probability ranking with ROC-AUC'])
    stage(ax,2,'Detector behaviour',.523,.520,'#16877c',[
        'ADWIN · HDDM-W · Hellinger · D3 OOF',
        'Monitor the baseline; do not reset the learner',
        'Measure alarm behaviour and monitoring time',
        'Drift-event recall and delay: SEA / RBF only'])
    stage(ax,3,'Fixed responses',.025,.186,'#b56a28',[
        'Compare all four detectors under matched rules',
        'Immediate reset / confirmed reset',
        'Confirmed: error check after two blocks',
        'Compare screened and unscreened updates',
        'Assess prediction, exposure and cost'])
    stage(ax,4,'Learned responses',.523,.186,'#76519b',[
        'ADWIN + frozen Q-learning policies',
        'Alarm-only / alarm + error deterioration',
        'Choose continued learning or a reset',
        'Both actions use the same screening check',
        'Compare against screened fixed responses'])
    box(ax,.025,.025,.95,.113,'#f0f7f4','#c5ddd3')
    ax.text(.5,.107,'Screening: accept the candidate when its protected-sample accuracy ≥ the active model’s',
            ha='center',va='center',fontsize=8.2,fontweight='bold',color='#285448')
    ax.text(.5,.070,'Simulated protected sample: 5%, delayed one block; excluded from training and final scoring',
            ha='center',va='center',fontsize=8.2,color='#334155')
    ax.text(.5,.040,'Rejected candidate: retain the active learner',ha='center',va='center',fontsize=8.2,color='#334155')
    fig.savefig(output/'experiment_overview.pdf')
    fig.savefig(output/'experiment_overview.png',dpi=220)
    plt.close(fig)
    caption=('Overview of the four matched evaluation stages. Attack damage and passive detector behaviour establish '
             'the references for fixed-response comparisons. The RL extension uses ADWIN and compares two frozen '
             'state representations against screened fixed responses. Screening checks a trained candidate before '
             'it replaces the active learner.')
    latex='''% Add \\usepackage{graphicx} to the report preamble.
% Put experiment_overview.pdf beside your report's .tex file.
\\begin{figure}[htbp]
  \\centering
  \\includegraphics[width=\\linewidth]{experiment_overview.pdf}
  \\caption{'''+caption+'''}
  \\label{fig:experiment-overview}
\\end{figure}
'''
    (output/'figure.tex').write_text(latex,encoding='utf-8')
    (output/'CAPTION.md').write_text(caption+'\n',encoding='utf-8')
    (output/'README.md').write_text('''# Figure for the experiment report

Use experiment_overview.pdf after the Setup section, before the detailed experiment descriptions. It summarises the four completed stages in a 170 x 128 mm vector figure, with readable type at A4 text width. The attached report is background material; descriptions in this figure follow the executed code and documented protocol. The planned upstream discriminator is outside these four completed stages.

Copy the PDF beside the report's LaTeX source, add `\\usepackage{graphicx}` to the preamble, and insert the environment in figure.tex. CAPTION.md contains the same short caption. The original uploaded report has not been edited. experiment_overview.png is a preview.

The panels describe separate matched evaluations, not consecutive processing by one learner. Detector behaviour is passive in Stage 2. Stage 3 compares four detectors, while Stage 4 is the ADWIN-based RL extension. Error confirmation and candidate screening are distinct checks. The protected 5% is an explicit simulation assumption and is reserved in all arms, including the unscreened baseline.

Reproduce with `python -B 06_analysis/dataset_summary/report_overview.py`. The PDF is an overview of methods rather than a numerical-results summary. The data-dependent findings and uncertainty remain in the existing comparison figures and tables.
''',encoding='utf-8')
    p.write(output/'manifest.json',{'created_utc':p.now(),'script_sha256':p.sha(__file__),
            'request_sha256':p.sha(root/'request.json'),'width_mm':170,'height_mm':128,
            'stages':4,'inputs':{name:p.sha(ROOT/name) for name in ['03_policies/gate.py',
                '04_rl_training/controller.py','05_experiment/engine.py','docs/METHODOLOGY.md']},
            'outputs':{file.name:p.sha(file) for file in output.iterdir() if file.is_file() and file.name!='manifest.json'}})
    print(f'Created {output / "experiment_overview.pdf"}',flush=True)


if __name__=='__main__':
    main()
