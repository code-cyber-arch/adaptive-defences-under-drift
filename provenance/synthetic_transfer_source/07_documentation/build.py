"""Compile methodology and write the verified experiment sequence."""
from pathlib import Path
import subprocess
import sys
import pandas as pd
from common import protocol as p

def markdown_tex(source):
    """Convert the short methodology text to editable LaTeX."""
    text = Path(source).read_text(encoding='utf-8')

    def escape(s):
        """Escape characters that have a special meaning in LaTeX."""
        return ''.join(({'\\': '\\textbackslash{}', '&': '\\&', '%': '\\%', '_': '\\_', '#': '\\#', '$': '\\$', '{': '\\{', '}': '\\}'}.get(c, c) for c in s))
    lines = ['\\documentclass[11pt,a4paper]{article}', '\\usepackage[margin=25mm]{geometry}', '\\usepackage[T1]{fontenc}', '\\usepackage{lmodern}', '\\setlength{\\parindent}{0pt}', '\\setlength{\\parskip}{6pt}', '\\begin{document}']
    for paragraph in text.strip().split('\n\n'):
        if paragraph.startswith('# '):
            lines.append('{\\Large\\bfseries ' + escape(paragraph[2:]) + '}\\par')
        elif paragraph.startswith('## '):
            lines.append('\\section*{' + escape(paragraph[3:]) + '}')
        else:
            lines.append(escape(paragraph.replace('**', '').replace('`', '')) + '\\par')
    lines.append('\\end{document}')
    target = Path(source).with_suffix('.tex')
    target.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return target

def compile_pdf(tex):
    """Compile a LaTeX document using the configured compiler."""
    subprocess.run([sys.executable, str(p.ROOT / 'tools/compile_document.py'), '--tex', str(tex)], check=True)

def build(root):
    """Build the documented outputs for this phase."""
    verification = p.read(root / 'verification.json')
    if verification['status'] != 'passed':
        raise ValueError('Results have not passed verification')
    m = pd.read_csv(root / '06_analysis/metrics.csv')
    text = f"# Experiment results\n\nThe complete benchmark grid contains {len(m)} verified policy runs on {m.condition.nunique()} matched observation conditions. Each RADAR arm processes all 484,753 source rows.\n\nBoth RL representations were fitted on synthetic seed 112 and validated on seed 113. Each used controller seeds 7, 17 and 27. Alarm-only RL has two states; persistence RL has four. The two actions and delayed accuracy reward are identical. The gate accepts a candidate only when its accuracy on the previous protected sample is at least the active learner's accuracy. There was no threshold search.\n\nThe six 250,000-row synthetic benchmark streams and the RADAR stream were reused exactly from the fixed input bank. All learners were rerun under the audit reservation. No previous model results were copied into these comparisons. The benchmark was previously inspected; this is evaluation of frozen controllers on a reused benchmark, not a pristine unseen study.\n\nOpen results/evaluation/06_analysis/presentation/index.html for individual figures and catalogue links. Each run has a temporal trace, block table and short narrative. Severity remains separate in summary figures. RADAR ROC and precision-recall curves use saved pre-learning scores. The tables include confusion counts, matched screening and RL differences, and descriptive variation.\n\nProtected observations never train the learner. The frozen RL Q-tables were unchanged during evaluation. These checks validate the execution protocol; they do not establish that accepted updates are poison-free.\n"
    (p.ROOT / 'docs/RESULTS.md').write_text(text, encoding='utf-8')
    compile_pdf(markdown_tex(p.ROOT / 'docs/METHODOLOGY.md'))
    compile_pdf(markdown_tex(p.ROOT / 'docs/RESULTS.md'))
    compile_pdf(root / '06_analysis/presentation/NARRATION.tex')
    p.write(root / 'documentation.json', {'outputs': {f.relative_to(p.ROOT).as_posix(): p.sha(f) for f in [p.ROOT / 'docs/METHODOLOGY.md', p.ROOT / 'docs/METHODOLOGY.tex', p.ROOT / 'docs/METHODOLOGY.pdf', p.ROOT / 'docs/RESULTS.md', p.ROOT / 'docs/RESULTS.pdf']}})
