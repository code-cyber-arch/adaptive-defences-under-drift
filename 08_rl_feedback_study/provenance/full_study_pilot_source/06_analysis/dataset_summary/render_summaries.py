"""Render the learner and passive-detector summaries from verified saved tables."""
from pathlib import Path
import ast
import sys
sys.dont_write_bytecode=True
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from common import protocol as p
baseline=p.load_module('summary_baseline','06_analysis/dataset_summary/baseline.py')
detectors=p.load_module('summary_detectors','06_analysis/dataset_summary/detector_baseline.py')


def description(source,prefix):
    """Use the same documentation as the corresponding complete evaluation script."""
    tree=ast.parse(source.read_text(encoding='utf-8'))
    return next(node.value for node in ast.walk(tree)
                if isinstance(node,ast.Constant) and isinstance(node.value,str) and node.value.startswith(prefix))


def main():
    """Check saved tables, draw six individual PDFs and fingerprint the presentation."""
    root=ROOT/'results/evaluation'
    if p.sources()!=p.read(root/'request.json')['sources']:
        raise ValueError('Original benchmark sources changed')
    parent=root/'06_analysis/presentation'
    for folder,script,prefix in [('baseline_clean_vs_poisoned','baseline.py','# Baseline:'),
                                 ('detector_baseline','detector_baseline.py','# Detector baseline:')]:
        output=parent/folder
        manifest=p.read(output/'manifest.json')
        for name,digest in manifest['outputs'].items():
            if p.sha(output/name)!=digest:
                raise ValueError(f'Changed input: {output/name}')
        points=pd.read_csv(output/'plot_values.csv')
        entries=[]
        for dataset in baseline.DATASETS.values():
            data=points[points.dataset.eq(dataset)]
            if folder=='baseline_clean_vs_poisoned':
                entry=baseline.figure(data,dataset,output)
                data[data.metric.isin(entry['metrics'])].to_csv(output/entry['table'],index=False)
            else:
                specs=detectors.figure_specs(dataset)
                entry=detectors.layout.draw(data,specs,'','',detectors.SERIES,
                                           output/f'{dataset}_detector_baseline','')
                entry['dataset']=dataset
            if any(not data.loc[data.metric.eq(metric),'defined'].gt(0).any() for metric in entry['metrics']):
                raise ValueError(f'An empty panel was selected for {dataset}')
            entries.append(entry)
        source=ROOT/'06_analysis/dataset_summary'/script
        (output/'README.md').write_text(description(source,prefix),encoding='utf-8')
        # Retain the calculation provenance while separately recording this rendering.
        manifest.setdefault('calculation_script_sha256',manifest['script_sha256'])
        manifest.update(entries=entries,script_sha256=p.sha(source),
                        layout_sha256=p.sha(ROOT/'06_analysis/dataset_summary/compact.py'),
                        render_script_sha256=p.sha(__file__),rendered_utc=p.now(),
                        width_mm=170,height_mm=128,layout=[2,2])
        manifest['outputs']={f.relative_to(output).as_posix():p.sha(f) for f in output.rglob('*')
                             if f.is_file() and f.name!='manifest.json'}
        p.write(output/'manifest.json',manifest)
        print(f'Rendered {folder}: three four-panel PDFs.',flush=True)


if __name__=='__main__':
    main()
